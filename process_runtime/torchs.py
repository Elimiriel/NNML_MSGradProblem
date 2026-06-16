from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Sequence, Protocol, runtime_checkable, cast, Mapping, Literal, TYPE_CHECKING
from subprocess import run, CalledProcessError
import numpy as np
import torch
from torch import Tensor
from torch.cuda import (is_available as cuda_available, memory_allocated, memory_reserved, get_device_properties,
                        empty_cache,)
from torch.nn import Module
from torch.optim import Optimizer
from types import RuntimeState, LossFn
if TYPE_CHECKING:
    from config import RuntimeConfig

USE_CUDA = False
DEVICE = torch.device("cpu")
AMP_DEVICE_TYPE = "cpu"
SELECTED_GPUS = ""
_RUNTIME_READY = False

@dataclass
class VramWatermark:
    """dataclass for checking and managing VRAM usage with simple thresholds"""
    high_ratio: float = 0.80
    critical_ratio: float = 0.90
    use_reserved: bool = True  # allocated보다 reserved가 보수적

    def state(self) -> Literal["low", "high", "critical"]:
        if not cuda_available():
            return "low"
        total = get_device_properties(0).total_memory
        used = memory_reserved(0) if self.use_reserved else memory_allocated(0)
        ratio = used / total if total > 0 else 0.0
        if ratio >= self.critical_ratio:
            return "critical"
        if ratio >= self.high_ratio:
            return "high"
        return "low"

    def maybe_release(self) -> None:
        if not cuda_available():
            return
        if self.state() == "critical":
            empty_cache()


def configure_cuda_allocator() -> None:
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_gpu_ids(gpus: str) -> list[int]:
    ids: list[int] = []
    for gpu in str(gpus).split(","):
        gpu = gpu.strip()
        if gpu.isdigit():
            ids.append(int(gpu))
    return ids


def query_gpu_stats() -> list[tuple[int, int, int, int]]:
    try:
        proc = run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, OSError, CalledProcessError):
        return []

    stats: list[tuple[int, int, int, int]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            stats.append(tuple(int(part) for part in parts))
        except ValueError:
            continue
    return stats


def select_visible_gpus(gpus: str) -> str:
    stats = query_gpu_stats()
    if not stats:
        return gpus

    requested_ids = parse_gpu_ids(gpus)
    if requested_ids:
        stats = [stat for stat in stats if stat[0] in requested_ids]
        if not stats:
            return gpus

    idle_threshold_mib = 1024
    idle_threshold_util = 20
    idle = [
        stat for stat in stats
        if stat[1] <= idle_threshold_mib and stat[3] <= idle_threshold_util
    ]
    if len(idle) >= 2:
        selected = sorted(idle, key=lambda stat: (stat[1], stat[3], stat[0]))[:2]
    else:
        selected = sorted(stats, key=lambda stat: (stat[1], stat[3], stat[0]))[:1]
    return ",".join(str(stat[0]) for stat in selected)


def gpu_count_from_config(gpus: str) -> int:
    return len([gpu.strip() for gpu in str(gpus).split(",") if gpu.strip()])


def should_use_dataparallel(use_cuda: bool, gpus: str) -> bool:
    return use_cuda and gpu_count_from_config(gpus) > 1


def to_device_loss(loss_fn: LossFn, use_cuda: bool) -> LossFn:
    return loss_fn.cuda() if use_cuda and hasattr(loss_fn, "cuda") else loss_fn


def safe_run_name(name: str) -> str:
    return name.replace(os.sep, "_").replace("/", "_").strip() or "run"


def make_scaler() -> torch.GradScaler:
    return torch.GradScaler(enabled=USE_CUDA)


def initialize_runtime(
        config: RuntimeConfig,
        deterministic_runtime: bool | None = None,
        *,
        auto_select_gpus: bool = True,
) -> RuntimeState:
    from random import random
    from torch import matmul
    global _RUNTIME_READY, USE_CUDA, DEVICE, AMP_DEVICE_TYPE, SELECTED_GPUS

    deterministic = (
        config.deterministic_runtime
        if deterministic_runtime is None
        else bool(deterministic_runtime)
    )

    if _RUNTIME_READY:
        return RuntimeState(
            use_cuda=USE_CUDA,
            device=DEVICE,
            amp_device_type=AMP_DEVICE_TYPE,
            deterministic_runtime=deterministic,
            selected_gpus=SELECTED_GPUS or str(config.gpus),
        )

    configure_cuda_allocator()
    selected_gpus = (
        select_visible_gpus(str(config.gpus))
        if auto_select_gpus else str(config.gpus)
    )
    SELECTED_GPUS = selected_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
    print(f"[GPU] CUDA_VISIBLE_DEVICES={selected_gpus}")

    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = True
    matmul.allow_tf32 = True
    print(
        f"[GPU] cudnn benchmark={torch.backends.cudnn.benchmark} "
        f"deterministic={torch.backends.cudnn.deterministic}"
    )

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    AMP_DEVICE_TYPE = "cuda" if USE_CUDA else "cpu"
    empty_cache()
    _RUNTIME_READY = True
    return RuntimeState(
        use_cuda=USE_CUDA,
        device=DEVICE,
        amp_device_type=AMP_DEVICE_TYPE,
        deterministic_runtime=deterministic,
        selected_gpus=selected_gpus,
    )


def same_device(tensors: Sequence[Tensor]) -> Sequence[Tensor]:
    """move tensors to the most device at all

    Args:
        tensors (Sequence[Tensor]): tensors to be moved

    Returns:
        Sequence[Tensor]: tensors moved, the same order as inputs
    """
    from collections import Counter
    if not tensors:
        return tensors
    devices = [t.device for t in tensors]
    # 전부 CPU면 그대로 반환
    if all(d.type == "cpu" for d in devices):
        return tensors
    # CUDA tensor만 모아서 가장 많이 쓰인 GPU 선택
    cuda_indices = [d.index for d in devices if d.type == "cuda" and d.index is not None]
    if not cuda_indices:
        return tensors
    dev_id = Counter(cuda_indices).most_common(1)[0][0]
    target = f"cuda:{dev_id}"
    return [t.to(target) for t in tensors]

def scalarize(x: Tensor, pyvar: bool = True)-> Tensor|float|int:
    """return scalar of tensor. calc abs*complex_angle if complex tensor data

    Raises:
        ValueError: non-scalar tensor

    Returns:
        Tensor|float|int: scalar numbers. not Tensor if pyvar is True
    """
    def _commonp(x=x,)->Tensor:
        v = x.detach()
        if v.ndim !=0:
            from torch import ComplexType, float_power, arctan
            if isinstance(v, ComplexType):
                return (float_power(float_power(v.real, 2.0)+float_power(v.imag, 2.0), 0.5)*arctan(v.real/v.imag))
            raise ValueError(f"shape {tuple(v.shape)}: cannot be scalar or complex")
        return v
    return _commonp().item() if pyvar else _commonp()

@runtime_checkable
class HasExtractFeatures(Protocol):
    """
    linter typechecker helper about extract features after model compile
    """
    def extract_features(self, x: Tensor) -> Tensor: ...


def extract_features_from(model: Module, x: Tensor) -> Tensor:
    # DataParallel case: actual model is still nn.Module
    m = cast(Module, model.module) if hasattr(model, "module") else model

    if isinstance(m, HasExtractFeatures):
        return m.extract_features(x)
    # fallback: 메서드명이 다른 모델 대비 (원하면 추가)
    raise AttributeError(f"{type(m).__name__} has no extract_features()")

def get_model_name(model: Module)->str:
    if isinstance(model, Module):
        return model._get_name()
    else:
        raise AttributeError(f"{type(model).__name__} has no get_name()")

def has_nan_or_inf(result: Mapping) -> bool:
    tensors = [v.reshape(-1) for v in result.values() if isinstance(v, Tensor)]

    if not tensors:
        return False

    merged = torch.cat(tensors)
    return not torch.isfinite(merged).all()

def trainloop_inv_guard(valscope: Tensor, optimizer: Optimizer):
    if torch.isfinite(valscope).all():
        print(f"[inv] {valscope.__name__}")
        optimizer.zero_grad(True)
        return True
    else:
        return False
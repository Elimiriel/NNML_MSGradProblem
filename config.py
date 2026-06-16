from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, Callable
from torch import Tensor

from dataframe.gpurun import configure_polars_engine
from logger import RunLoggerConfig
from types import RuntimeConfig
from process_runtime.torchs import VramWatermark


@dataclass(slots=True)
class GlobalConfig:
    run_name: str
    train_real_metafile: Path
    train_spoof_metafile: Path
    eval_metafile: Path
    test_metafile: Path
    train_batch: int
    worker_train: int
    eval_batch: int
    worker_eval: int
    deviceconf: RuntimeConfig
    logconf: RunLoggerConfig
    watermark: VramWatermark | None = None
    eps: float = 1e-6
    lr_init: float = 0.01
    lr_decay: float|Callable[[float|Tensor, ...], Tensor]|None = None

    def __post_init__(self) -> None:
        if self.watermark is None:
            self.watermark = VramWatermark(
                high_ratio=self.deviceconf.vram_high_ratio,
                critical_ratio=self.deviceconf.vram_critical_ratio,
                use_reserved=self.deviceconf.vram_use_reserved,
            )

    def apply_runtime_overrides(self) -> None:
        configure_polars_engine(
            use_gpu=self.deviceconf.polars_use_gpu_engine,
            verbose=self.deviceconf.polars_verbose,
        )

config = GlobalConfig()
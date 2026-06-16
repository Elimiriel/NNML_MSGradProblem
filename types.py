from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, TypeVar
from annotated_types import Gt, Lt
from numpy import number
from torch import Tensor, int as tint, float as tfloat, bool as tbool, device as torch_device

type IntLike = int | tint
type FloatLike = float | tfloat
type BoolLike = bool | tbool
type Numeric = int|float|complex|number|bool
type ComparableNumeric = int|float|bool
type PositiveNumeric = Annotated[Numeric, Gt(0)]
type PositiveReal = Annotated[int|float|number|bool, Gt(0)]
type Uint = Annotated[int, Gt(-1)]
type NonNegativeNumeric = Annotated[Numeric, Gt(-1)]
type NegativeNumeric = Annotated[Numeric, Lt(0)]
type Rate = Annotated[float, Gt(0.0), Lt(1.0)]
type ActivFn = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class RuntimeState:
    use_cuda: bool
    device: torch_device
    amp_device_type: str
    deterministic_runtime: bool
    selected_gpus: str

type LossFn = Callable[[Tensor, ...], Tensor]
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@dataclass(slots=True)
class RuntimeConfig:
    gpus: str = ""
    seed: int = 0
    deterministic_runtime: bool = False
    polars_use_gpu_engine: bool = False
    polars_verbose: bool = False
    vram_high_ratio: float = 0.80
    vram_critical_ratio: float = 0.90
    vram_use_reserved: bool = True

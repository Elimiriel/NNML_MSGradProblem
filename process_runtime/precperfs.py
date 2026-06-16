from __future__ import annotations
from torch import set_default_dtype, float64, complex128, set_printoptions


set_default_dtype(float64)#torch.set_default_device("cpu")
realTType=float64
compTType=complex128
#set_up_backend("torch", "float64")
set_printoptions(precision=20)
vErr=1E-5
cpu_reserve_fraction=0.8


def nameof(variable):
    return variable.__annotations__.keys()
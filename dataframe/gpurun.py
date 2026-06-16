from __future__ import annotations

import os
from typing import Literal

import polars as pl
from polars import GPUEngine

PolarsEngineName = Literal["auto", "gpu"]

# Use this only when an explicit `engine=` argument is needed.
gpu_wrapper: GPUEngine = GPUEngine()


def configure_polars_engine(use_gpu: bool, *, verbose: bool = False) -> PolarsEngineName:
    """
    Configure the default engine Polars will *attempt* to use for LazyFrame.collect().

    Note:
        This only affects lazy execution entrypoints such as `.collect()`.
        Eager `pl.DataFrame` operations are not globally redirected to the GPU engine.
    """
    engine: PolarsEngineName = "gpu" if use_gpu else "auto"
    pl.Config.set_engine_affinity(engine)
    os.environ["POLARS_ENGINE_AFFINITY"] = engine

    if verbose:
        os.environ["POLARS_VERBOSE"] = "1"
    else:
        os.environ.pop("POLARS_VERBOSE", None)
    return engine


def get_collect_engine(use_gpu: bool) -> PolarsEngineName | GPUEngine:
    """
    Return an explicit engine object for call-sites that still pass `engine=`.

    Global affinity can only be set to the string engine names. If you need
    fine-grained GPU options, pass `gpu_wrapper` explicitly to `.collect()`.
    """
    return gpu_wrapper if use_gpu else "auto"

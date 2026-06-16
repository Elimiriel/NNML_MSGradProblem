from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, TypeAlias, TypeVar, TypedDict

import polars as pl

try:
    from .colname import (
        FILEDATA_COL,
        IMG_CODEC_COL,
        IMG_C_COL,
        IMG_DTYPE_COL,
        IMG_H_COL,
        IMG_W_COL,
        LABEL_COL,
        LOCAL_PATH_COL,
        SOURCE_PATH_COL,
        normalize_columns,
    )
    from pyutils.multiprocess import run_in_parallel
except ImportError:
    #relative import fallback
    from dataframe.colname import (
        FILEDATA_COL,
        IMG_CODEC_COL,
        IMG_C_COL,
        IMG_DTYPE_COL,
        IMG_H_COL,
        IMG_W_COL,
        LABEL_COL,
        LOCAL_PATH_COL,
        SOURCE_PATH_COL,
        normalize_columns,
    )
    from pyutils.multiprocess import run_in_parallel

T = TypeVar("T")
LABEL_EXTS = frozenset({".json", ".parquet"})
LabelDataframe: TypeAlias = pl.DataFrame


class LabelRow(TypedDict, total=False):
    label: int
    source_path: str | None
    local_path: str | None
    filedata: bytes | None
    img_codec: str | None
    img_h: int | None
    img_w: int | None
    img_c: int | None
    img_dtype: str | None


try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class _TqdmFallback:
        def __call__(self, iterable=None, *args: Any, **kwargs: Any):
            return iterable

        @staticmethod
        def write(message: str) -> None:
            print(message)

    tqdm = _TqdmFallback()


def _json_default(value: Any) -> Any:
    #convert Path or bytes into re-encodable format
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_stream(records: Sequence[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for row in records:
            if not first:
                f.write(",\n")
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            first = False
        f.write("\n]")


def run_with_tqdm_log(description: str, task: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    apply tqdm to task

    :param description: tqdm description
    :param task: task to be monitored
    :param args: args of the task
    :param kwargs: kwargs of the task
    :return: return of the task
    """
    started_at = perf_counter()
    tqdm.write(f"[IO] {description}")
    result = task(*args, **kwargs)
    tqdm.write(f"[IO] {description} done in {perf_counter() - started_at:.1f}s")
    return result


def load_json_df(path: Path) -> LabelDataframe:
    return pl.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def load_parquet_df(path: Path) -> LabelDataframe:
    return pl.read_parquet(path)


def dump_json_df(path: Path, df: LabelDataframe) -> None:
    """
    atomic dump dataframe into json
    :param path: path to save
    :param df: target dataframe
    :return:
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(df.to_dicts(), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(path)


def dump_parquet_df(path: Path, df: LabelDataframe) -> None:
    """
    atomic dump dataframe into parquet
    :param path: path to save
    :param df: target dataframe
    :return:
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=3)
    tmp.replace(path)


def _resolve_label_files(root_path: Path) -> list[Path]:
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    if root_path.is_file():
        return [root_path] if root_path.suffix.lower() in LABEL_EXTS else []

    parquet_paths = sorted(root_path.rglob("*.parquet"))
    if parquet_paths:
        return parquet_paths

    json_paths = sorted(root_path.rglob("*.json"))
    if json_paths:
        return json_paths

    raise FileNotFoundError(f"No json or parquet files in {root_path}")


def _empty_df() -> LabelDataframe:
    """
    represent empty dataframe in handling format
    :return: colname normalized empty dataframe
    """
    return pl.DataFrame({
        LABEL_COL: pl.Series(LABEL_COL, [], dtype=pl.UInt8),
        SOURCE_PATH_COL: pl.Series(SOURCE_PATH_COL, [], dtype=pl.Utf8),
        LOCAL_PATH_COL: pl.Series(LOCAL_PATH_COL, [], dtype=pl.Utf8),
        FILEDATA_COL: pl.Series(FILEDATA_COL, [], dtype=pl.Binary),
        IMG_CODEC_COL: pl.Series(IMG_CODEC_COL, [], dtype=pl.Utf8),
        IMG_H_COL: pl.Series(IMG_H_COL, [], dtype=pl.Int32),
        IMG_W_COL: pl.Series(IMG_W_COL, [], dtype=pl.Int32),
        IMG_C_COL: pl.Series(IMG_C_COL, [], dtype=pl.Int16),
        IMG_DTYPE_COL: pl.Series(IMG_DTYPE_COL, [], dtype=pl.Utf8),
    })


def _normalize_label_df(df: LabelDataframe) -> LabelDataframe:
    """
    normalize and typecast to dataframe
    :param df: target dataframe
    :return: processed dataframe
    """
    df = normalize_columns(df)

    exprs: list[pl.Expr] = []
    if LABEL_COL in df.columns:
        exprs.append(pl.col(LABEL_COL).cast(pl.UInt8))
    else:
        raise ValueError(f"missing required label column: {LABEL_COL}")

    if SOURCE_PATH_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Utf8).alias(SOURCE_PATH_COL))
    else:
        exprs.append(pl.col(SOURCE_PATH_COL).cast(pl.Utf8))

    if LOCAL_PATH_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Utf8).alias(LOCAL_PATH_COL))
    else:
        exprs.append(pl.col(LOCAL_PATH_COL).cast(pl.Utf8))

    if FILEDATA_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Binary).alias(FILEDATA_COL))
    if IMG_CODEC_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Utf8).alias(IMG_CODEC_COL))

    if IMG_H_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Int32).alias(IMG_H_COL))
    if IMG_W_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Int32).alias(IMG_W_COL))
    if IMG_C_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Int16).alias(IMG_C_COL))
    if IMG_DTYPE_COL not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Utf8).alias(IMG_DTYPE_COL))

    return df.with_columns(exprs)


def _build_dataframe(label_paths: Sequence[Path]) -> LabelDataframe:
    """
    build dataframe from saved formats
    :param label_paths: label paths gathered in sequence
    :return: normalized dataframe
    """
    frames: list[LabelDataframe] = []
    for path in tqdm(label_paths, desc="Loading labels", unit="file", dynamic_ncols=True):
        suffix = path.suffix.lower()
        if suffix == ".json":
            frame = load_json_df(path)
        elif suffix == ".parquet":
            frame = load_parquet_df(path)
        else:
            continue
        frames.append(_normalize_label_df(frame))

    return pl.concat(frames, how="vertical_relaxed") if frames else _empty_df()


def _build_dataframe_from_root(root_path: Path) -> LabelDataframe:
    """
    build dataframe from root path calling sequential subs
    :param root_path: root path of files
    :return: dataframe built from files in root path
    """
    return _build_dataframe(_resolve_label_files(root_path))


def build_dataframe(
    data_path_roots: Path | Sequence[Path],
    flag: Literal[0, 1, 2],
    seed: int = 42,
    n_sample: int = 5000,
) -> LabelDataframe:
    """
    build dataframe from several root paths in parallel
    :param data_path_roots: root paths of files
    :param flag: 0 for normal train, 1 for occulusion train, 2 for test
    :param seed: sampling seed
    :param n_sample: number of samples
    :return: normalized dataframe
    """
    roots = [data_path_roots] if isinstance(data_path_roots, Path) else list(data_path_roots)
    if not roots:
        raise ValueError("data_path_roots must not be empty")

    missing_paths = [path for path in roots if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(missing_paths[0])

    if len(roots) == 1:
        df = _build_dataframe_from_root(roots[0])
    else:
        task_args = [(root,) for root in roots]
        frames = run_in_parallel(
            _build_dataframe_from_root,
            task_args,
            show_progress=True,
            progress_desc="loading label frames",
        )
        valid_frames = [frame for frame in frames if isinstance(frame, pl.DataFrame)]
        df = pl.concat(valid_frames, how="vertical_relaxed") if valid_frames else _empty_df()

    if flag != 2:
        df = df.filter(pl.col(LABEL_COL) == flag)
    if df.is_empty():
        return _empty_df()

    if df.height > n_sample:
        if flag == 2:
            real_df = df.filter(pl.col(LABEL_COL) == 0)
            fake_df = df.filter(pl.col(LABEL_COL) == 1)
            balanced: list[LabelDataframe] = []
            if not real_df.is_empty():
                balanced.append(real_df.sample(min(real_df.height, n_sample // 2), seed=seed))
            if not fake_df.is_empty():
                balanced.append(fake_df.sample(min(fake_df.height, n_sample // 2), seed=seed))
            if balanced:
                df = pl.concat(balanced, how="vertical_relaxed")
        else:
            df = df.sample(n_sample, seed=seed)
    return df

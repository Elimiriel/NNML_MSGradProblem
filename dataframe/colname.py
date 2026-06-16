from __future__ import annotations

from pathlib import Path
from re import IGNORECASE, compile
from typing import Any, Final

import polars as pl

LABEL_COL: Final[str] = "label"
SOURCE_PATH_COL: Final[str] = "source_path"
LOCAL_PATH_COL: Final[str] = "local_path"
FILEDATA_COL: Final[str] = "filedata"
IMG_CODEC_COL: Final[str] = "img_codec"
IMG_H_COL: Final[str] = "img_h"
IMG_W_COL: Final[str] = "img_w"
IMG_C_COL: Final[str] = "img_c"
IMG_DTYPE_COL: Final[str] = "img_dtype"

DsetColName = (
    LABEL_COL,
    SOURCE_PATH_COL,
    LOCAL_PATH_COL,
    FILEDATA_COL,
    IMG_CODEC_COL,
    IMG_H_COL,
    IMG_W_COL,
    IMG_C_COL,
    IMG_DTYPE_COL,
)

lpat = compile(r".*(lab(?:el)|res(?:ult)|det(?:ect(?:ed)?)).*", IGNORECASE)
source_pat = compile(r".*(source|src|origin|orig).*(path|file).*", IGNORECASE)
local_pat = compile(r".*(local|cache|cached|downsample|resized).*(path|file).*", IGNORECASE)
dpat = compile(r".*(file[_|\b]?data|bytes?|binary|blob|content).*", IGNORECASE)
img_h_pat = compile(r".*(img|image).*(height|_h)\b|^imgh$|^height$", IGNORECASE)
img_w_pat = compile(r".*(img|image).*(width|_w)\b|^imgw$|^width$", IGNORECASE)
img_c_pat = compile(r".*(img|image).*(channels?|_c)\b|^imgc$|^channels?$", IGNORECASE)
img_codec_pat = compile(r".*(img|image).*(codec|format|encoding)\b|^imgcodec$|^codec$", IGNORECASE)
img_dtype_pat = compile(r".*(img|image).*(dtype|type)\b|^imgdtype$|^dtype$", IGNORECASE)


def _norm_col(col: str) -> str:
    return "".join(ch for ch in col.lower() if ch.isalnum() or ch == "_")


def _is_existing_path(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, Path):
        return v.exists()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return False
        try:
            return Path(s).exists()
        except (OSError, ValueError):
            return False
    return False


def _is_path_like_str(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return False
    return "/" in s or "\\" in s or s.startswith(".")


def _is_filedata_like(v: Any) -> bool:
    return isinstance(v, (bytes, bytearray, memoryview))


def _decide_path_col(series: pl.Series, fallback: str) -> str | None:
    vals = series.drop_nulls().head(20).to_list()
    if not vals:
        return None
    if sum(_is_existing_path(v) for v in vals) > 0:
        return fallback
    if sum(_is_path_like_str(v) for v in vals) > 0:
        return fallback
    return None


def rename_col(df: pl.DataFrame, col: str) -> str:
    """
    single time column name normalizer, input of dataframe is required
    :param df:
    :param col:
    :return:
    """
    norm = _norm_col(col)
    series = df.get_column(col)

    if lpat.search(norm):
        return LABEL_COL
    if source_pat.search(norm):
        return SOURCE_PATH_COL
    if local_pat.search(norm):
        return LOCAL_PATH_COL
    if dpat.search(norm) or series.dtype == pl.Binary:
        return FILEDATA_COL
    if img_h_pat.search(norm):
        return IMG_H_COL
    if img_w_pat.search(norm):
        return IMG_W_COL
    if img_c_pat.search(norm):
        return IMG_C_COL
    if img_codec_pat.search(norm):
        return IMG_CODEC_COL
    if img_dtype_pat.search(norm):
        return IMG_DTYPE_COL

    decided = _decide_path_col(series, SOURCE_PATH_COL)
    if decided is not None:
        return decided
    return col


def make_rename_mapping(df: pl.DataFrame) -> dict[str, str]:
    """
    dataframe columns normalizer: dataframe required
    :param df: target dataframe
    :return: mapping functions can be used to pl.DataFrame.rename
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in df.columns:
        new_col = rename_col(df, col)
        if new_col in used and new_col != col:
            suffix = 1
            cand = f"{new_col}_{suffix}"
            while cand in used:
                suffix += 1
                cand = f"{new_col}_{suffix}"
            new_col = cand
        mapping[col] = new_col
        used.add(new_col)
    return mapping


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    normalize column names of input dataframe
    :param df: target dataframe
    :return: normalized dataframe
    """
    return df.rename(make_rename_mapping(df))

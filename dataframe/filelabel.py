from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import polars as pl


def read_meta_file(p: Path) -> pl.DataFrame | None:
    ext = p.suffix.lower()
    try:
        if ext == ".json":
            df = pl.read_json(p)
        elif ext == ".ndjson":
            df = pl.read_ndjson(p)
        elif ext == ".csv":
            df = pl.read_csv(p)
        elif ext in [".xlsx", ".xls"]:
            df = pl.read_excel(p)
        elif ext == ".avro":
            df = pl.read_avro(p)
        elif ext in [".parquet", ".parq"]:
            df = pl.read_parquet(p)
        else:
            return None
        # source track col
        df = df.with_columns(pl.lit(str(p)).alias("meta_source"))
        return df
    
    except Exception as e:
        print(f"[WARN] failed to read {p}: {e}")
        return None

def find_meta_files(root: Path):
    keywords = ["meta", "mark", "info"]
    valid_ext = {".json", ".csv", ".ndjson", ".xlsx", ".parquet", ".avro"}

    return [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in valid_ext
        and any(k in p.stem.lower() for k in keywords)
    ]
    
def pick_first_existing(cols: Sequence[str], candidates: Sequence[str]) -> str | None:
    """search 1st matching pattern str"""
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None
    
def normalize_meta_df(df: pl.DataFrame) -> pl.DataFrame | None:
    """dataframe into col with file_path, folder, group_id"""
    
    cols = df.columns
    path_col = pick_first_existing(cols, [
        "path", "file", "filepath", "file_path", "image_path", "img_path"
    ])
    label_col = pick_first_existing(cols, [
        "label", "res", "detect", "target", "class"
    ])
    id_col = pick_first_existing(cols, [
        "id", "fid", "video_id", "subject_id", "clip_id", "source_id", "source"
    ])

    if path_col is None:
        return None
    exprs = [
        pl.col(path_col).cast(pl.Utf8).alias("file_path"),
    ]
    if label_col is not None:
        exprs.append(pl.col(label_col).alias("label"))
    else:
        exprs.append(pl.lit(None).alias("label"))

    if id_col is not None:
        exprs.append(pl.col(id_col).cast(pl.Utf8).alias("raw_id"))
    else:
        exprs.append(pl.lit(None).alias("raw_id"))
    if "meta_source" in cols:
        exprs.append(pl.col("meta_source"))
    else:
        exprs.append(pl.lit(None).alias("meta_source"))

    out = df.select(exprs).with_columns([
        pl.col("file_path")
        .str.replace_all(r"\\", "/")
        .alias("file_path"),
        pl.col("file_path")
        .str.split("/")
        .list.get(0)
        .alias("folder"),
    ]).with_columns([
        pl.col("folder")
        .str.replace(r"_\d+$", "")
        .alias("group_id")
    ])
    return out
    
def read_metafiles(rootpath: Path) -> pl.DataFrame:
    """get metainfo from metadata file"""
    meta_paths = find_meta_files(rootpath)
    meta_dfs = [read_meta_file(p) for p in meta_paths]
    meta_dfs = [df for df in meta_dfs if df is not None]

    normalized = []
    for df in meta_dfs:
        nd = normalize_meta_df(df)
        if nd is not None:
            normalized.append(nd)

    if not normalized:
        return pl.DataFrame({
            "file_path": [],
            "folder": [],
            "group_id": [],
            "label": [],
            "raw_id": [],
            "meta_source": [],
        })

    return pl.concat(normalized, how="diagonal_relaxed")


from __future__ import annotations
from collections.abc import Sequence
from ..mathematics.varrange import Uint
from .colname import DsetColName
import polars as pl
from pandas import DataFrame as pdDataFrame


def eval_test_split_keep_ratio(
    data: pl.DataFrame|pdDataFrame, target_col: str = DsetColName[0],
    eval_ratio: float = 0.5, seed: int = 42,) -> tuple[pl.DataFrame, pl.DataFrame]:
    """split polars.DataFrame as 2 frames in similar distribution"""
    if target_col not in data.columns:
        raise ValueError(f"'{target_col}' not in columns: {data.columns}")
    if isinstance(data, pdDataFrame):
        data = pl.from_pandas(data)
    df = data.filter(pl.col(target_col).is_not_null()).with_row_index("rid")
    N = df.height
    if N == 0:
        raise ValueError("No rows after filtering null labels.")

    # counts for each labels
    counts = df.group_by(target_col).agg(pl.len().alias("cnt"))

    # cnt<2 case: eval/test both is impossible
    bad = counts.filter(pl.col("cnt") < 2)
    if bad.height > 0:
        labs = bad.select(target_col).to_series().to_list()
        raise ValueError(
            f"These labels have <2 samples, can't appear in BOTH eval and test: {labs}"
        )
    # target eval size in whole
    target_eval = round(N * eval_ratio)
    # raw = cnt*ratio, base=floor(raw), residual=raw-base for each class
    # eval_n -> [1, cnt-1] clip, least 1 for eval/test
    counts = counts.with_columns(
        (pl.col("cnt") * pl.lit(eval_ratio)).alias("raw"),
    ).with_columns(
        pl.col("raw").floor().cast(pl.Int64).alias("base"),
        (pl.col("raw") -pl.col("raw").floor()).alias("residual"),
    ).with_columns(
        pl.col("base").clip(1, pl.col("cnt") - 1).alias("eval_n")
    )
    # 현재 eval 합계
    cur_eval = counts.select(pl.col("eval_n").sum()).item()
    delta = target_eval - int(cur_eval)  # +면 더 넣어야, -면 덜 넣어야
    # delta만큼 클래스별 eval_n을 +1/-1 보정 (가능한 범위 내에서)
    # +1은 residual 큰 클래스부터, -1은 residual 작은 클래스부터
    if delta > 0:
        counts = counts.with_columns(
            (pl.col("cnt") - 1 - pl.col("eval_n")).alias("cap_up")  # 더 늘릴 수 있는 여지
        ).with_columns(
            pl.when(pl.col("cap_up") > 0)
              .then(pl.col("residual").rank(method="dense", descending=True))
              .otherwise(None)
              .alias("rk")
        ).with_columns(
            pl.when((pl.col("rk").is_not_null()) & (pl.col("rk") <= pl.lit(delta)))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .alias("adj")
        ).with_columns(
            (pl.col("eval_n") + pl.col("adj")).alias("eval_n")
        ).drop(["cap_up", "rk", "adj"])
        
    elif delta < 0:
        need = -delta
        counts = counts.with_columns(
            (pl.col("eval_n") - 1).alias("cap_down")  # 더 줄일 수 있는 여지
        ).with_columns(
            pl.when(pl.col("cap_down") > 0)
              .then(pl.col("residual").rank(method="dense", descending=False))
              .otherwise(None)
              .alias("rk")
        ).with_columns(
            pl.when((pl.col("rk").is_not_null()) & (pl.col("rk") <= pl.lit(need)))
              .then(pl.lit(1))
              .otherwise(pl.lit(0))
              .alias("adj")
        ).with_columns(
            (pl.col("eval_n") - pl.col("adj")).alias("eval_n")
        ).drop(["cap_down", "rk", "adj"])

    # 원본에 eval_n join
    df = df.join(counts.select([target_col, "eval_n"]), on=target_col, how="left")

    # 라벨별로 shuffle -> 클래스 내 순번(k) 부여 -> k < eval_n 이면 eval
    df = (
        df.with_columns(
            pl.col("rid").shuffle(seed=seed).over(target_col).alias("_shuf")
        )
        .sort(["_shuf"])
        .with_columns(
            pl.cum_count("rid").over(target_col).alias("_k"),
        ).with_columns(
            (pl.col("_k") < pl.col("eval_n")).alias("_is_eval"),
        )
    )
    eval_df = df.filter(pl.col("_is_eval")).drop(["rid", "_shuf", "_k", "eval_n", "_is_eval"])
    test_df = df.filter(~pl.col("_is_eval")).drop(["rid", "_shuf", "_k", "eval_n", "_is_eval"])
    return eval_df, test_df


def df_splits_with_same_ratio(
    data: pl.DataFrame|pdDataFrame, target_col: str, n_split: Uint, seed: int = 42,
    drop_null_target: bool = True,
) -> Sequence[pl.DataFrame]:
    """splitting dataframe into dataframes keeping distribution ratio

    Args:
        data (DataFrame): dataframe to split. notice: pandas will be converted to polars
        target_col (str): target col to keep distribution
        n_split (int): number of splits
        seed (int, optional): shuffling seed. Defaults to 42.
        drop_null_target (bool, optional): drop null data. Defaults to True.

    Raises:
        ValueError: target colname mismatch
        

    Returns:
        Sequence[pl.DataFrame]: splitted dataframes in length of n_split sequence
    """
    if isinstance(data, pdDataFrame):
        data = pl.from_pandas(data)

    if target_col not in data.columns:
        raise ValueError(f"{target_col} not in dataframe")
    
    df = data.filter(pl.col(target_col).is_not_null()) if drop_null_target else data

    if df.height == 0:
        raise ValueError("data is empty after filtering")

    df = df.with_row_index("rid")

    df = df.with_columns(
        pl.int_range(0, pl.len()).shuffle(seed=seed).alias("_shuffle_key")
    )

    df = df.with_columns(
        (
            pl.col("_shuffle_key")
            .rank("ordinal")
            .over(target_col)
            .sub(1)
            .cast(pl.UInt32)
        ).alias("_target_order")
    )

    df = df.with_columns(
        (pl.col("_target_order") % n_split).alias("split_id")
    )

    df = df.drop(["_shuffle_key", "_target_order"])

    return [
        df.filter(pl.col("split_id") == split_id)
          .sort("rid")
          .drop(["rid", "split_id"])
        for split_id in range(n_split)
    ]
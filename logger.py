from __future__ import annotations
import json
import math
import queue
import threading
import time
from typing import TypedDict, Literal, Any, NotRequired, Required, TypeAlias
from collections.abc import Mapping
from torch import Tensor, int as tint, float as tfloat, bool as tbool
from dataclasses import dataclass
from pathlib import Path
import polars as pl
from torch.utils.tensorboard import SummaryWriter
from types import IntLike, FloatLike, BoolLike


class ModelResult(TypedDict, total=False):
    """
    Per-phase metrics row for logging.

    Requireds: phase, run_id, hter, tn, fp, fn, tp, loss\n
    optionals: seed, epoch, iter, n_samples, n_batches,
    time_epoch_sec, label_t, lr, loss_is_best, acc(Accurate),
    top1(top-1 Accurate), eer(Equal/Crossover Error Rate),
    auc(Area under Curve), thr, acc_thr(Accu at thr),
    apcer, bpcer, f1, acer, err
    """
    phase: Required[Literal["train", "eval", "test"]]
    run_id: Required[str]
    seed: NotRequired[IntLike]
    epoch: NotRequired[IntLike]
    iter: NotRequired[IntLike]
    n_samples: NotRequired[IntLike]
    n_batches: NotRequired[IntLike]
    time_epoch_sec: NotRequired[FloatLike]
    lr: NotRequired[FloatLike]
    loss: Required[FloatLike]
    loss_is_best: NotRequired[BoolLike]
    acc: NotRequired[Tensor]
    top1: NotRequired[FloatLike]
    eer: NotRequired[FloatLike]
    hter: Required[FloatLike]
    auc: NotRequired[FloatLike]
    thr: NotRequired[FloatLike]
    acc_thr: NotRequired[FloatLike]
    apcer: NotRequired[FloatLike]
    bpcer: NotRequired[FloatLike]
    f1: NotRequired[FloatLike]
    acer: NotRequired[FloatLike]
    tn: Required[IntLike]
    fp: Required[IntLike]
    fn: Required[IntLike]
    tp: Required[IntLike]
    err: NotRequired[Exception]


def _safe_float(x: Any) -> float|None:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            x = x.item()
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> int|None:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            x = x.item()
        return int(x)
    except Exception:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class AsyncArtifactWriter:
    def __init__(self, max_queue: int = 256) -> None:
        self._q: queue.Queue[tuple[str, pl.DataFrame, Path, dict[str, Any]] | None] = queue.Queue(maxsize=max_queue)
        self._errors: list[Exception] = []
        self._worker = threading.Thread(target=self._run, name="artifact-writer", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            kind, df, path, kwargs = item
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if kind == "parquet":
                    df.write_parquet(path, **kwargs)
                elif kind == "csv":
                    df.write_csv(path, **kwargs)
                else:
                    raise ValueError(f"Unsupported artifact kind: {kind}")
            except Exception as exc:
                self._errors.append(exc)
            finally:
                self._q.task_done()

    def submit_parquet(self, df: pl.DataFrame, path: Path, **kwargs: Any) -> None:
        self._q.put(("parquet", df, path, kwargs))

    def submit_csv(self, df: pl.DataFrame, path: Path, **kwargs: Any) -> None:
        self._q.put(("csv", df, path, kwargs))

    def flush(self) -> None:
        self._q.join()
        if self._errors:
            raise RuntimeError(f"artifact writer failed: {self._errors[-1]}")

    def close(self) -> None:
        self.flush()
        self._q.put(None)
        self._worker.join(timeout=10.0)
        if self._errors:
            raise RuntimeError(f"artifact writer failed: {self._errors[-1]}")


@dataclass(slots=True)
class RunLoggerConfig:
    logger_root: Path
    run_id: str
    seed: int = 0
    enable_step_parquet: bool = False
    step_parquet_every: int = 100
    write_csv_on_finalize: bool = True


class RunLogger:
    def __init__(self, cfg: RunLoggerConfig):
        self.cfg = cfg
        self.base = cfg.logger_root / "Res" / cfg.run_id
        self.tb_dir = self.base
        self.parquet_dir = self.base / "parquet"
        self.reports_dir = self.base / "reports"
        self.artifacts_dir = self.base / "artifacts"

        _ensure_dir(self.base)
        _ensure_dir(self.tb_dir)
        _ensure_dir(self.parquet_dir)
        _ensure_dir(self.reports_dir)
        _ensure_dir(self.artifacts_dir)

        self.epoch_dir = self.parquet_dir / "epoch_metrics"
        self.step_dir = self.parquet_dir / "step_metrics"
        self.err_dir = self.parquet_dir / "errors"
        for p in [self.epoch_dir, self.step_dir, self.err_dir]:
            _ensure_dir(p)

        self.tb = SummaryWriter(log_dir=str(self.tb_dir))
        self.best_loss = float("inf")
        self.writer = AsyncArtifactWriter()


    def save_config(self, config: Mapping[str, Any]) -> None:
        out = self.artifacts_dir / "config.json"
        out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def _tensorboard_items(self, scalars: dict[str, Any]) -> dict[str, float]:
        aliases = {
            "top1": "acc",
            "acc_thr": "acc_thr",
            "na_acc": "na_acc",
            "loss_cls": "loss_cls",
            "loss_diff": "loss_diff",
            "loss_lsr_easy": "loss_lsr_easy",
            "loss_lsr_hard": "loss_lsr_hard",
        }
        out: dict[str, float] = {}
        for k, v in scalars.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            out[aliases.get(k, k)] = fv
        return out

    def tb_step(self, phase: str, global_step: int, scalars: dict[str, Any]) -> None:
        for k, v in self._tensorboard_items(scalars).items():
            try:
                self.tb.add_scalar(f"{phase}/{k}", v, int(global_step))
            except Exception:
                pass

    def tb_epoch(self, phase: str, epoch: int, metrics: ModelResult) -> None:
        for k, v in self._tensorboard_items(metrics).items():
            try:
                self.tb.add_scalar(f"{phase}/{k}", v, int(epoch))
            except Exception:
                pass
        loss = _safe_float(metrics.get("loss"))
        if bool(metrics.get("loss_is_best", False)) and loss is not None:
            self.tb.add_scalar("best/loss", loss, int(epoch))
            self.tb.add_scalar("best/epoch", float(epoch), int(epoch))

    def log_epoch_parquet(self, metrics: ModelResult) -> None:
        phase = str(metrics["phase"])
        epoch = int(_safe_int(metrics["epoch"]) or 0)
        out_dir = self.epoch_dir / f"p{phase}"
        out_path = out_dir / f"epoch={epoch:04d}.parquet"
        df = pl.DataFrame([self._normalize_epoch_metrics(metrics)])
        self.writer.submit_parquet(df, out_path, compression="lz4")

    def log_step_parquet(self, row: ModelResult) -> None:
        if not self.cfg.enable_step_parquet:
            return
        phase = str(row.get("phase", "step"))
        step = int(_safe_int(row.get("global_step", row.get("iter", 0))) or 0)
        if step % int(self.cfg.step_parquet_every) != 0:
            return
        out_dir = self.step_dir / f"phase={phase}"
        out_path = out_dir / f"step={step:07d}.parquet"
        self.writer.submit_parquet(pl.DataFrame([self._normalize_step_metrics(row)]), out_path, compression="lz4")

    def log_error(self, row: ModelResult) -> None:
        ts = int(time.time() * 1000)
        out_path = self.err_dir / f"error={ts}.parquet"
        self.writer.submit_parquet(pl.DataFrame([self._normalize_step_metrics(row)]), out_path, compression="lz4")

    def update_best_loss(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def finalize_reports(self) -> None:
        self.writer.flush()
        epoch_glob = str(self.epoch_dir / "p*/epoch=*.parquet")
        if not list(self.epoch_dir.glob("p*/epoch=*.parquet")):
            return
        df = pl.scan_parquet(epoch_glob, missing_columns="insert").collect().sort(["phase", "epoch"])
        if self.cfg.write_csv_on_finalize:
            df.write_csv(self.reports_dir / "summary.csv")
        self._write_xlsx(df, self.reports_dir / "summary.xlsx")

    def close(self) -> None:
        self.tb.flush()
        self.tb.close()
        self.writer.close()

    def _normalize_step_metrics(self, m: ModelResult) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in m.items():
            if k in {"phase", "run_id", "thr_policy"}:
                out[k] = str(v)
                continue
            iv = _safe_int(v)
            fv = _safe_float(v)
            out[k] = fv if fv is not None else iv if iv is not None else str(v)
        return out

    def _normalize_epoch_metrics(self, m: ModelResult) -> dict[str, Any]:
        out = {
            "run_id": str(m.get("run_id", self.cfg.run_id)),
            "seed": _safe_int(m.get("seed", self.cfg.seed)),
            "epoch": _safe_int(m.get("epoch")),
            "phase": str(m.get("phase", "unknown")),
            "n_samples": _safe_int(m.get("n_samples")),
            "n_batches": _safe_int(m.get("n_batches")),
            "time_epoch_sec": _safe_float(m.get("time_epoch_sec")),
            "lr": _safe_float(m.get("lr")),
            "loss": _safe_float(m.get("loss")),
            "loss_is_best": bool(m.get("loss_is_best", False)),
            "acc": _safe_float(m.get("acc", m.get("top1"))),
            "top1": _safe_float(m.get("top1", m.get("acc"))),
            "eer": _safe_float(m.get("eer")),
            "auc": _safe_float(m.get("auc")),
            "acc_thr": _safe_float(m.get("acc_thr")),
            "f1": _safe_float(m.get("f1")),
            "fp": _safe_float(m.get("fp")), "fn": _safe_float(m.get("fn")),
            "tp": _safe_float(m.get("tp")), "tn": _safe_float(m.get("tn")),
        }
        for k in ["hter", "apcer", "bpcer", "acer"]:
            out[k] = _safe_float(m.get(k))
        for k in ["tn", "fp", "fn", "tp"]:
            out[k] = _safe_int(m.get(k))
        if "thr" in m:
            out["thr"] = _safe_float(m.get("thr"))
        for k in ["loss_diff", "loss_cls", "loss_lsr_easy", "loss_lsr_hard"]:
            if k in m:
                out[k] = _safe_float(m.get(k))
        return out

    def _write_xlsx(self, df: pl.DataFrame, path: Path) -> None:
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows

        wb = Workbook()
        ws1 = wb.active
        ws1.title = "epoch_metrics"
        pdf = df.to_pandas()
        for r in dataframe_to_rows(pdf, index=False, header=True):
            ws1.append(r)

        ws2 = wb.create_sheet("best_summary")
        if not pdf.empty and "phase" in pdf.columns and "loss" in pdf.columns:
            best_rows = []
            for ph in pdf["phase"].dropna().unique():
                sub = pdf[pdf["phase"] == ph].dropna(subset=["loss"])
                if len(sub) == 0:
                    continue
                best_rows.append(sub.loc[sub["loss"].idxmin()])
            if best_rows:
                import pandas as pd
                best_df = pd.DataFrame(best_rows)
                for r in dataframe_to_rows(best_df, index=False, header=True):
                    ws2.append(r)
        wb.save(path)

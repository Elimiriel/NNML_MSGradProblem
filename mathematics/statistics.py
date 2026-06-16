from __future__ import annotations
from collections.abc import Sequence
from torch import Tensor, float32, float64, nan, uint8, zeros, as_tensor, empty, nan_to_num_
from process_runtime.torchs import same_device


def eval_state(probs: Tensor, labels: Tensor, thr: float | Sequence[float] | Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """calculate TN, FP, FN, TP values

    Args:
        probs (Tensor): prediction probabilities
        labels (Tensor): input labels on ISO format(0: real)
        thr (float): threshold values

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: TN, FP, FN, TP
    """
    from torch import as_tensor, sum as tsum
    probs, labels = same_device((probs, labels))
    probs = probs.detach().flatten()
    labels = labels.detach().to(dtype=uint8).flatten()
    label0 = labels == 0
    label1 = labels == 1

    thr_t = thr if isinstance(thr, Tensor) else as_tensor(thr, device=probs.device, dtype=probs.dtype)

    if thr_t.ndim == 0:
        predict = probs >= thr_t
        return (
            tsum(label0 & (~predict)),
            tsum(label0 & predict),
            tsum(label1 & (~predict)),
            tsum(label1 & predict),
        )

    predict = probs.unsqueeze(1) >= thr_t.reshape(1, -1)
    label0 = label0.unsqueeze(1)
    label1 = label1.unsqueeze(1)
    return (
        tsum(label0 & (~predict), dim=0),
        tsum(label0 & predict, dim=0),
        tsum(label1 & (~predict), dim=0),
        tsum(label1 & predict, dim=0),
    )


def calculate(probs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """statistical calculates based on TN, FP, FN, TP

    Returns:
        tuple[float|Tensor]: APCER, BPCER, ACER, ACC, F1score
    """
    tn, fp, fn, tp = eval_state(probs, labels, 0.5)
    bpcer = nan_to_num_(fp/(fp + tn), 1.0)
    apcer = nan_to_num_(fn/(fn + tp), 1.0)
    acer = 0.5 * (apcer + bpcer)
    acc = nan_to_num_((tp + tn)/(tn + fp + fn + tp), 0.0)
    precision = nan_to_num_(tp/(tp + fp), 0.0)
    recall = nan_to_num_(tp/(tp + fn), 0.0)
    f1 = nan_to_num_((2 * precision * recall)/(precision + recall), 0.0)
    return apcer, bpcer, acer, acc, f1


def auc_from_roc(fpr: Tensor, tpr: Tensor) -> Tensor:
    """AUC calc from confusion matrix via FPR, TPR

    Args:
        fpr (Tensor): FP rates
        tpr (Tensor): TP rates

    Returns:
        Tensor: AUC
    """
    from torch import argsort, clip, trapz

    fpr, tpr = same_device((fpr, tpr))
    order = argsort(fpr)
    fpr = clip(fpr[order], 0.0, 1.0)
    tpr = clip(tpr[order], 0.0, 1.0)
    return trapz(tpr, fpr)


def metrics_at_thr(probs: Tensor, labels: Tensor,
                   thr: float | Sequence[float] | Tensor = 0.5) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """statisticals on confusion matrix

    Args:
        probs (Tensor): prediction probabilities
        labels (Tensor): input labels
        thr (float, optional): threshold values. Defaults to 0.5.

    Returns:
        tuple[Tensor, ...]: TN, FP, FN, TP, APCER, BPCER, ACER(=HTER), ACC, precision, recall, F1, TNR, balanced ACC, thresholds
    """
    from torch import tensor

    thr_t = thr if isinstance(thr, Tensor) else tensor(thr)
    tn, fp, fn, tp = eval_state(probs, labels, thr_t)
    n = labels.shape[0]
    bpcer = nan_to_num_(fp/(fp + tn), 1.0)
    apcer = nan_to_num_(fn/(fn + tp), 1.0)
    acer = 0.5 * (apcer + bpcer)
    acc = nan_to_num_((tp + tn)/n, 0.0)
    precision = nan_to_num_(tp/(tp + fp), 0.0)
    recall = nan_to_num_(tp/(tp + fn), 0.0)
    f1 = nan_to_num_(2 * precision * recall/(precision + recall), 0.0)
    tnr = nan_to_num_(tn/(tn + fp), 0.0)
    bal_acc = 0.5 * (recall + tnr)
    return tn, fp, fn, tp, apcer, bpcer, acer, acc, precision, recall, f1, tnr, bal_acc, thr_t


def roc_points(probs: Tensor, labels: Tensor, thresholds: float | Sequence[float] | Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from torch import tensor

    if not isinstance(thresholds, Tensor):
        thresholds = tensor(list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds], dtype=probs.dtype, device=probs.device)
    else:
        thresholds = thresholds.to(device=probs.device, dtype=probs.dtype)

    probs, labels = same_device((probs, labels))
    tn, fp, fn, tp = eval_state(probs, labels, thresholds)
    fpr = fp / (fp + tn).clamp_min(1)
    fnr = fn / (fn + tp).clamp_min(1)
    tpr = tp / (tp + fn).clamp_min(1)
    tnr = tn / (tn + fp).clamp_min(1)
    return fnr, fpr, tnr, tpr


def calculate_threshold(probs: Tensor, labels: Tensor, threshold: float | Tensor):
    """simple calc about accurate on thresholds

    Args:
        probs (Tensor): prediction p
        labels (Tensor): input labels
        threshold (float|Tensor): threshold values

    Returns:
        Tensor: ACCurate on thresholds
    """
    tn, _, _, tp = eval_state(probs, labels, threshold)
    return (tp + tn) / labels.shape[0]


def _as_1d_score_or_2d_probs(probs: Tensor) -> tuple[Tensor | None, Tensor | None]:
    if probs.ndim == 1:
        return probs.reshape(-1), None
    if probs.ndim == 2:
        return None, probs
    raise ValueError(f"probs must be 1D or 2D, got shape {probs.shape}")


def _make_threshold_grid(score_1d: Tensor, steps: int = 128) -> Tensor:
    """build confusion matrix space including thresholds

    Args:
        score_1d (Tensor): scores in 1D tensor
        steps (int, optional): steps to divide. Defaults to 128.

    Returns:
        Tensor: confusion matrix axis
    """
    from torch import linspace

    score = score_1d.detach().to(dtype=float32).reshape(-1)
    if score.numel() == 0:
        return linspace(0.0, 1.0, 2, dtype=float32)
    lo = float(score.min().item())
    hi = float(score.max().item())
    lo = min(lo, 0.0)
    hi = max(hi, 1.0)
    return linspace(lo, hi, steps, dtype=float32, device=score.device)


def _eer_from_score(score_1d: Tensor, labels_1d: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """calc EER from 1D score

    Args:
        score_1d (Tensor): 1D score
        labels_1d (Tensor): 1D input labels

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: EER, thresholds, FRR, FAR
    """
    from torch import argmin

    thresholds = _make_threshold_grid(score_1d)
    score = score_1d.detach().to(dtype=float32).reshape(-1)
    labels = labels_1d.detach().to(dtype=uint8).reshape(-1)
    frr, far, _, _ = roc_points(score, labels, thresholds)
    idx = argmin((frr - far).abs())
    eer = 0.5 * (far[idx] + frr[idx])
    return eer, thresholds[idx], frr, far


def auc_from_score(score_1d: Tensor, labels_1d: Tensor) -> Tensor:
    """AUC calc through ROCs

    Args:
        score_1d (Tensor): 1D predictions
        labels_1d (Tensor): 1D input labels

    Returns:
        Tensor: AUC
    """
    thresholds = _make_threshold_grid(score_1d)
    score = score_1d.detach().to(dtype=float32).reshape(-1)
    labels = labels_1d.detach().to(dtype=uint8).reshape(-1)
    _, fpr, _, tpr = roc_points(score, labels, thresholds)
    return auc_from_roc(fpr, tpr)


def get_EER_states(probs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """get EER status using confusion matrix

    Args:
        probs (Tensor): prediction probabilities
        labels (Tensor): input labels

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: EER, threshold, AUC, FRRs, FARs
    """
    from torch import nanmean, stack, tensor, unique

    probs, labels = same_device((probs, labels))
    y = labels.reshape(-1).to(dtype=uint8)
    score_1d, probs_2d = _as_1d_score_or_2d_probs(probs)
    n_classes = len(unique(y))
    if n_classes < 2:
        return as_tensor(nan), as_tensor(nan), as_tensor(nan), empty(0), empty(0)

    if score_1d is not None:
        if n_classes != 2:
            return as_tensor(nan), as_tensor(nan), as_tensor(nan), empty(0), empty(0)
        score = score_1d.to(dtype=float64).detach()
        eer, thr, frr_list, far_list = _eer_from_score(score, y)
        auc = auc_from_score(score, y)
        return eer, thr, auc, frr_list, far_list

    if probs_2d is not None:
        p = probs_2d.detach()
        n, c = p.shape
        if c == 2:
            score = p[:, 1].reshape(-1)
            eer, thr, frr_list, far_list = _eer_from_score(score, y)
            auc = auc_from_score(score, y)
            return eer, thr, auc, frr_list, far_list

        eers: list[Tensor] = []
        aucs: list[Tensor] = []
        for k in range(c):
            yk = (y == k).to(dtype=uint8)
            if len(unique(yk)) < 2:
                continue
            score_k = p[:, k].reshape(-1)
            eer_k, _, _, _ = _eer_from_score(score_k, yk)
            eers.append(eer_k)
            aucs.append(tensor(auc_from_score(score_k, yk), dtype=float32))
        eer_macro = nanmean(tensor(eers)) if eers else as_tensor(nan)
        auc_macro = nanmean(stack(aucs)) if aucs else as_tensor(nan)
        return eer_macro, as_tensor(nan), auc_macro, empty(0), empty(0)

    return as_tensor(nan), as_tensor(nan), as_tensor(nan), empty(0), empty(0)


def get_HTER_at_thr(probs: Tensor, labels: Tensor, thr: float | Tensor) -> Tensor:
    """calc HTER on thresholds

    Args:
        probs (Tensor): predictions
        labels (Tensor): input labels
        thr (float): thresholds

    Returns:
        Tensor: HTER
    """
    tn, fp, fn, tp = eval_state(probs, labels, thr)
    far = nan_to_num_(fp/(fp + tn), 1.0)
    frr = nan_to_num_(fn/(fn + tp), 1.0)
    return 0.5 * (far + frr)


class EpochPredBuffer:
    """storage about prediction probabilities and labels in averaged CPU, about epoch
    """
    def __init__(self, store_on_cpu: bool = True, dtype=float32):
        """storage about prediction probabilities and labels in averaged CPU, about epoch

        Args:
            store_on_cpu (bool, optional): device choice to CPU. Defaults to True.
            dtype ([type], optional): dtypes for probabilities. Defaults to torch.float32.
        """
        self.store_on_cpu = store_on_cpu
        self.dtype = dtype
        self.ps: list[Tensor] = []
        self.ys: list[Tensor] = []

    @staticmethod
    def _maybe_cpu(x: Tensor, *, dtype) -> Tensor:
        out = x.detach().reshape(-1)
        if out.is_cuda:
            out = out.to(device="cpu", dtype=dtype, non_blocking=True)
        else:
            out = out.to(dtype=dtype)
        return out.contiguous()

    def add(self, probs: Tensor, y: Tensor) -> None:
        """add storage to be averaged

        Args:
            probs (Tensor): probability
            y (Tensor): input label
        """
        if self.store_on_cpu:
            self.ps.append(self._maybe_cpu(probs, dtype=self.dtype))
            self.ys.append(self._maybe_cpu(y, dtype=uint8))
        else:
            self.ps.append(probs.detach().reshape(-1).to(dtype=self.dtype))
            self.ys.append(y.detach().reshape(-1).to(dtype=uint8))


    def get_means(self)-> tuple[Tensor, Tensor]:
        """avg/mean to components

        Returns:
            tuple[Tensor, Tensor]: pmean, ymean
        """
        from torch import cat
        if not self.ps:
            z = zeros((), dtype=self.dtype)
            return z, z
        ps = cat(self.ps, 0).mean()
        ys = cat(self.ys, 0).float().mean()
        return ps, ys

    def cat(self) -> tuple[Tensor, Tensor]:
        from torch import cat
        if not self.ps:
            z = zeros((), dtype=self.dtype)
            return z, z
        return cat(self.ps, dim=0), cat(self.ys, dim=0)

    def reset(self) -> None:
        self.ps.clear()
        self.ys.clear()

class RunningScalarBuffer:
    """storage about scalars incl loss and accuracy and return means
    """
    def __init__(self, store_on_cpu: bool = True, dtype=float32):
        """storage about prediction probabilities and labels in averaged CPU, about iter window

        Args:
            store_on_cpu (bool, optional): device choice to CPU. Defaults to True.
            dtype ([type], optional): dtypes for probabilities. Defaults to torch.float32.
        """
        self.store_on_cpu = store_on_cpu
        self.dtype = dtype
        self.values: list[Tensor] = []

    @staticmethod
    def _maybe_cpu(x: Tensor, *, dtype) -> Tensor:
        out = x.detach().reshape(-1)
        if out.is_cuda:
            out = out.to(device="cpu", dtype=dtype, non_blocking=True)
        else:
            out = out.to(dtype=dtype)
        return out.contiguous()

    def add(self, x: Tensor) -> None:
        if self.store_on_cpu:
            self.values.append(self._maybe_cpu(x, dtype=self.dtype))
        else:
            self.values.append(x.detach().reshape(-1).to(dtype=self.dtype))

    def get_means(self) -> Tensor:
        """avg/mean to components

        Returns:
            Tensor: mean value scalar
        """
        if not self.values:
            return zeros((), dtype=self.dtype)
        from torch import cat
        return cat(self.values, dim=0).mean()

    def reset(self) -> None:
        self.values.clear()
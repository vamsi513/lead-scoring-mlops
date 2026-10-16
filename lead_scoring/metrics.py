"""Evaluation metrics with bootstrap confidence intervals.

Ranking quality (AUROC, AUPRC) and calibration (Brier score, expected
calibration error) are reported together on purpose: a model can rank well and
still produce probabilities that are badly scaled, and on this data a lead
score is only useful if the number means something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

#: Bootstrap resamples. 1000 gives percentile interval endpoints stable to
#: about the third decimal, which is finer than any difference reported here.
DEFAULT_BOOTSTRAP: Final = 1000

#: Equal-width bins for expected calibration error.
ECE_BINS: Final = 10


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap confidence interval."""

    value: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.low:.4f}, {self.high:.4f}]"


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Weighted mean gap between predicted and observed rate, over equal-width bins.

    Bins are fixed on [0, 1] rather than taken from quantiles so that the value
    is comparable between two models scoring the same data.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``digitize`` with right=True puts 0.0 in bin 0 and 1.0 in the last bin.
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(float(y_prob[in_bin].mean()) - float(y_true[in_bin].mean()))
        total += gap * count / y_true.shape[0]
    return total


def max_calibration_gap(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Largest predicted-vs-observed gap over bins holding at least 1% of rows.

    ECE can hide a badly miscalibrated tail behind a well-populated middle;
    this reports the worst bin that is big enough to be worth believing.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    floor = max(1, y_true.shape[0] // 100)

    worst = 0.0
    for b in range(n_bins):
        in_bin = bin_index == b
        if int(in_bin.sum()) < floor:
            continue
        gap = abs(float(y_prob[in_bin].mean()) - float(y_true[in_bin].mean()))
        worst = max(worst, gap)
    return worst


#: Metric name -> callable. Every one takes (y_true, y_prob); higher is better
#: for the first two and worse for the rest.
METRICS: Final = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "brier": brier_score_loss,
    "ece": expected_calibration_error,
    "max_calibration_gap": max_calibration_gap,
}

HIGHER_IS_BETTER: Final = frozenset({"auroc", "auprc"})


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 12345,
) -> dict[str, Interval]:
    """Point estimates and percentile bootstrap intervals for every metric.

    Resampling is done on rows with replacement, which is the right unit here
    because each row is one independent lead at one snapshot. Resamples that
    happen to contain a single class are skipped, since AUROC is undefined
    there; with the positive rates in this project that is rare, and the count
    of usable resamples bounds how much the interval can be trusted.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if len(np.unique(y_true)) < 2:
        raise ValueError("need both classes present to evaluate")

    point = {name: float(fn(y_true, y_prob)) for name, fn in METRICS.items()}

    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    draws: dict[str, list[float]] = {name: [] for name in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled_true = y_true[idx]
        if len(np.unique(resampled_true)) < 2:
            continue
        resampled_prob = y_prob[idx]
        for name, fn in METRICS.items():
            draws[name].append(float(fn(resampled_true, resampled_prob)))

    low_q, high_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        name: Interval(
            value=point[name],
            low=float(np.percentile(draws[name], low_q)),
            high=float(np.percentile(draws[name], high_q)),
        )
        for name in METRICS
    }


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS
) -> list[tuple[float, float, float, int]]:
    """Rows of (bin_low, mean_predicted, observed_rate, count), empty bins dropped.

    Returned as a list so it can be printed or logged without pandas.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        rows.append(
            (
                float(edges[b]),
                float(y_prob[in_bin].mean()),
                float(y_true[in_bin].mean()),
                count,
            )
        )
    return rows

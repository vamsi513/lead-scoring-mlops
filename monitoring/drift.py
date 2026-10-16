"""Population stability index per feature.

PSI compares how a feature's distribution is spread across a fixed set of bins
in a reference sample against a later one:

    PSI = sum over bins of (actual_share - expected_share) * ln(actual_share / expected_share)

Bin edges come from quantiles of the *reference* only, so the reference is the
fixed yardstick and the comparison sample is scored against it. The usual
reading, which this project keeps, is that below 0.1 is no meaningful change,
0.1 to 0.25 is worth a look, and above 0.25 is a real shift. Those thresholds
are convention, not a calibrated test: ``tests/test_drift.py`` measures what
they actually do on this data, over repeated trials, rather than assuming.

PSI is a distance between distributions. It says a feature moved; it does not
say the model got worse, and it has no idea whether the label distribution
moved with it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from lead_scoring import cli
from lead_scoring.config import (
    CATEGORICAL_FEATURES,
    FEATURES_DIR,
    NUMERIC_FEATURES,
    TEST_SNAPSHOT,
    TRAIN_SNAPSHOT,
)

#: Default bin count for numeric features.
DEFAULT_BINS: Final = 10

#: Conventional reading of a PSI value.
WATCH_THRESHOLD: Final = 0.10
SHIFT_THRESHOLD: Final = 0.25

#: Floor applied to a bin's share before the log, so that a bin which is empty
#: in one sample gives a large but finite contribution instead of infinity.
#: 1e-6 means a bin that empties out entirely contributes about 12 per unit of
#: share, which is enough to dominate the sum without making it meaningless.
EPSILON: Final = 1e-6


@dataclass(frozen=True)
class FeatureDrift:
    """PSI for one feature, with the per-bin detail behind it."""

    feature: str
    psi: float
    bins: list[dict[str, float]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.psi >= SHIFT_THRESHOLD:
            return "shift"
        if self.psi >= WATCH_THRESHOLD:
            return "watch"
        return "stable"


def _numeric_edges(reference: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges from the reference sample.

    Duplicate edges are collapsed, which matters here because several count
    features are zero for a large share of leads and would otherwise produce
    several identical cut points.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    # Open the outer edges so values beyond the reference range still land in
    # the first or last bin rather than falling outside every bin.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        raise ValueError("cannot compute shares for an empty sample")
    return counts / total


def psi_numeric(
    reference: np.ndarray, current: np.ndarray, n_bins: int = DEFAULT_BINS
) -> tuple[float, list[dict[str, float]]]:
    """PSI for a numeric feature, with per-bin shares."""
    reference = np.asarray(reference, dtype="float64")
    current = np.asarray(current, dtype="float64")
    edges = _numeric_edges(reference, n_bins)
    if edges.shape[0] < 3:
        # The reference is effectively constant, so there is nothing to bin.
        # Report movement only if the current sample left that single value.
        same = float(np.mean(current == reference[0])) if current.size else 1.0
        return (0.0 if same == 1.0 else float(abs(1.0 - same)), [])

    expected = _shares(np.histogram(reference, bins=edges)[0].astype("float64"))
    actual = _shares(np.histogram(current, bins=edges)[0].astype("float64"))

    expected_safe = np.clip(expected, EPSILON, None)
    actual_safe = np.clip(actual, EPSILON, None)
    contributions = (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)

    detail = [
        {
            "bin_low": float(edges[i]),
            "bin_high": float(edges[i + 1]),
            "expected": float(expected[i]),
            "actual": float(actual[i]),
            "contribution": float(contributions[i]),
        }
        for i in range(len(expected))
    ]
    return float(contributions.sum()), detail


def psi_categorical(
    reference: pd.Series, current: pd.Series
) -> tuple[float, list[dict[str, float]]]:
    """PSI for a categorical feature, one bin per level.

    Levels are the union of both samples, so a category that appears only in
    the current sample is counted rather than silently dropped.
    """
    levels = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    expected = _shares(
        np.array([float((reference == level).sum()) for level in levels])
    )
    actual = _shares(np.array([float((current == level).sum()) for level in levels]))

    expected_safe = np.clip(expected, EPSILON, None)
    actual_safe = np.clip(actual, EPSILON, None)
    contributions = (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)

    detail = [
        {
            "level": level,
            "expected": float(expected[i]),
            "actual": float(actual[i]),
            "contribution": float(contributions[i]),
        }
        for i, level in enumerate(levels)
    ]
    return float(contributions.sum()), detail


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
    n_bins: int = DEFAULT_BINS,
) -> list[FeatureDrift]:
    """PSI for every feature present in both frames, worst first."""
    results: list[FeatureDrift] = []
    for name in numeric:
        if name not in reference.columns or name not in current.columns:
            continue
        value, detail = psi_numeric(
            reference[name].to_numpy(), current[name].to_numpy(), n_bins
        )
        results.append(FeatureDrift(name, value, detail))
    for name in categorical:
        if name not in reference.columns or name not in current.columns:
            continue
        value, detail = psi_categorical(reference[name], current[name])
        results.append(FeatureDrift(name, value, detail))
    return sorted(results, key=lambda r: r.psi, reverse=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Population stability index between two feature snapshots (synthetic data)."
    )
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument(
        "--reference", default=TRAIN_SNAPSHOT.cutoff_date, help="reference snapshot cutoff"
    )
    parser.add_argument(
        "--current", default=TEST_SNAPSHOT.cutoff_date, help="snapshot to score"
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args(argv)

    reference = pd.read_parquet(args.features / f"snapshot={args.reference}")
    current = pd.read_parquet(args.features / f"snapshot={args.current}")

    results = compare(reference, current, n_bins=args.bins)

    print(f"PSI  reference={args.reference} ({len(reference):,} rows)  "
          f"current={args.current} ({len(current):,} rows)")
    print(f"{'feature':<24s} {'psi':>8s}  verdict")
    for result in results:
        print(f"{result.feature:<24s} {result.psi:>8.4f}  {result.verdict}")

    shifted = [r.feature for r in results if r.verdict == "shift"]
    print(f"\n{len(shifted)} feature(s) above {SHIFT_THRESHOLD}" + (
        f": {', '.join(shifted)}" if shifted else ""))

    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {"feature": r.feature, "psi": r.psi, "verdict": r.verdict, "bins": r.bins}
                    for r in results
                ],
                indent=2,
            )
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    # Not SystemExit: see lead_scoring.cli for the Arrow shutdown deadlock.
    cli.run(main)

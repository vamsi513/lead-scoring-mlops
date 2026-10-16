"""Does the PSI detector actually behave like a detector?

Two questions, each answered by repeated trials rather than a single draw,
because a one-shot assertion on a random sample tells you almost nothing:

  (a) how often does it flag a fresh sample drawn from the same distribution?
  (b) how often does it flag a sample that really was shifted?

Both counts are printed, so the numbers in the README come from a run of these
tests and not from an estimate. The trial counts are deliberately modest so the
suite stays quick; they are large enough to catch a detector that is badly
miscalibrated in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    SHIFT_THRESHOLD,
    compare,
    psi_categorical,
    psi_numeric,
)

#: Trials per direction.
N_TRIALS = 100

#: Rows per sample. Comparable to a daily scoring batch for this project, and
#: small enough that PSI has real sampling noise.
SAMPLE_SIZE = 3000

NUMERIC = ("n_total_7d", "n_total_30d", "recency_days", "trend_ratio")
CATEGORICAL = ("industry", "source")


def _draw(rng: np.random.Generator, n: int, shift: float = 0.0) -> pd.DataFrame:
    """A feature-like sample. ``shift`` moves every feature's distribution.

    The shapes mimic the real feature table: counts are overdispersed and
    zero-heavy, recency is long-tailed, the ratio is centred near one.
    """
    scale = 1.0 + shift
    frame = pd.DataFrame(
        {
            "n_total_7d": rng.poisson(4.5 * scale, n).astype("float64"),
            "n_total_30d": rng.poisson(19.5 * scale, n).astype("float64"),
            "recency_days": rng.exponential(5.0 * scale, n),
            "trend_ratio": rng.lognormal(0.0 + 0.5 * shift, 0.5, n),
        }
    )
    industry_p = np.array([0.30, 0.20, 0.15, 0.20, 0.15])
    if shift:
        # Move mass toward the first level by the same proportion.
        industry_p = industry_p + shift * (np.array([1.0, 0, 0, 0, 0]) - industry_p)
        industry_p = industry_p / industry_p.sum()
    frame["industry"] = rng.choice(
        ["saas", "manufacturing", "healthcare", "retail", "finance"], n, p=industry_p
    )
    source_p = np.array([0.22, 0.24, 0.16, 0.26, 0.12])
    if shift:
        source_p = source_p + shift * (np.array([0, 0, 0, 0, 1.0]) - source_p)
        source_p = source_p / source_p.sum()
    frame["source"] = rng.choice(
        ["organic", "paid_search", "webinar", "outbound", "referral"], n, p=source_p
    )
    return frame


@dataclass
class TrialResult:
    flagged: int
    trials: int
    max_psi: float
    mean_psi: float

    def __str__(self) -> str:
        return (
            f"{self.flagged}/{self.trials} flagged  "
            f"(mean psi {self.mean_psi:.4f}, max {self.max_psi:.4f})"
        )


def _run_trials(shift: float, seed: int) -> TrialResult:
    """Run N_TRIALS comparisons at a given shift; count how many flag."""
    rng = np.random.default_rng(seed)
    flagged = 0
    worst_per_trial: list[float] = []
    for _ in range(N_TRIALS):
        reference = _draw(rng, SAMPLE_SIZE)
        current = _draw(rng, SAMPLE_SIZE, shift=shift)
        results = compare(reference, current, numeric=NUMERIC, categorical=CATEGORICAL)
        worst = max(r.psi for r in results)
        worst_per_trial.append(worst)
        if worst >= SHIFT_THRESHOLD:
            flagged += 1
    return TrialResult(
        flagged=flagged,
        trials=N_TRIALS,
        max_psi=float(np.max(worst_per_trial)),
        mean_psi=float(np.mean(worst_per_trial)),
    )


@pytest.fixture(scope="module")
def same_distribution() -> TrialResult:
    return _run_trials(shift=0.0, seed=2024)


@pytest.fixture(scope="module")
def shifted_distribution() -> TrialResult:
    return _run_trials(shift=0.25, seed=2025)


@pytest.fixture(scope="module")
def mildly_shifted_distribution() -> TrialResult:
    """A 25% shift is an easy case. This one characterises where the detector
    actually sits, rather than only testing it where it cannot fail."""
    return _run_trials(shift=0.05, seed=2026)


def test_does_not_flag_a_fresh_sample_from_the_same_distribution(
    same_distribution: TrialResult, capsys: pytest.CaptureFixture[str]
) -> None:
    """False positives: a fresh draw from the reference distribution.

    Allowing a couple out of 100 rather than demanding zero: PSI on 3000 rows
    has real sampling noise, and a detector tuned to never fire on noise would
    be too dull to fire on anything.
    """
    with capsys.disabled():
        print(f"\n  same distribution:    {same_distribution}")
    assert same_distribution.flagged <= 2, (
        f"{same_distribution.flagged}/{N_TRIALS} false positives at threshold {SHIFT_THRESHOLD}"
    )


def test_flags_deliberately_shifted_data(
    shifted_distribution: TrialResult, capsys: pytest.CaptureFixture[str]
) -> None:
    """True positives: every feature's distribution moved by 25%."""
    with capsys.disabled():
        print(f"  shifted by 25%:       {shifted_distribution}")
    assert shifted_distribution.flagged >= 95, (
        f"only {shifted_distribution.flagged}/{N_TRIALS} shifted samples flagged"
    )


def test_sensitivity_to_a_mild_shift(
    same_distribution: TrialResult,
    mildly_shifted_distribution: TrialResult,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 5% shift is near the edge of what PSI resolves at this sample size.

    The assertion is only that a real shift moves PSI more than noise does;
    the flag count is reported for information, because whether 5% *should*
    trip the 0.25 threshold is a judgement about the use case and not a
    property of the detector.
    """
    with capsys.disabled():
        print(f"  shifted by 5%:        {mildly_shifted_distribution}")
    assert mildly_shifted_distribution.mean_psi > same_distribution.mean_psi


def test_the_two_regimes_are_clearly_separated(
    same_distribution: TrialResult, shifted_distribution: TrialResult
) -> None:
    """The quiet case's worst PSI should sit well below the shifted case's
    typical PSI, otherwise the threshold is doing no work."""
    assert same_distribution.max_psi < shifted_distribution.mean_psi


def test_psi_of_a_sample_against_itself_is_zero() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    value, _ = psi_numeric(x, x)
    assert value == pytest.approx(0.0, abs=1e-12)


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(9)
    reference = rng.normal(0.0, 1.0, 20000)
    values = [
        psi_numeric(reference, rng.normal(delta, 1.0, 20000))[0] for delta in (0.0, 0.25, 0.5, 1.0)
    ]
    assert values == sorted(values), values
    assert values[0] < 0.01 < values[-1]


def test_categorical_psi_notices_a_new_level() -> None:
    """A category absent from the reference must register, not be dropped."""
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 400 + ["b"] * 400 + ["c"] * 200)
    value, detail = psi_categorical(reference, current)
    assert {row["level"] for row in detail} == {"a", "b", "c"}
    assert value > SHIFT_THRESHOLD


def test_values_outside_the_reference_range_still_bin() -> None:
    """Outer edges are open, so a current sample that exceeds the reference
    range is counted in the end bins rather than vanishing."""
    rng = np.random.default_rng(4)
    reference = rng.uniform(0.0, 1.0, 5000)
    current = rng.uniform(5.0, 6.0, 5000)
    value, detail = psi_numeric(reference, current)
    assert sum(row["actual"] for row in detail) == pytest.approx(1.0)
    assert value > SHIFT_THRESHOLD


def test_constant_reference_does_not_raise() -> None:
    value, _ = psi_numeric(np.zeros(1000), np.zeros(1000))
    assert value == 0.0

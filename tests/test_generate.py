"""The generator must be reproducible, and its output must be hashable.

Reproducibility is not a nicety here: the data hash logged with every MLflow
run is only a useful provenance record if regenerating from the same seed
produces the same digest.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.generate import generate, write
from lead_scoring.config import DAY_ZERO, EVENT_TYPES, OBSERVATION_DAYS
from lead_scoring.provenance import hash_path, hash_paths

SMALL = {"n_leads": 800, "target_events": 30_000, "n_days": OBSERVATION_DAYS}


@pytest.fixture(scope="module")
def generated() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return generate(seed=99, **SMALL)


def test_same_seed_gives_identical_tables(generated) -> None:
    again = generate(seed=99, **SMALL)
    for first, second in zip(generated, again, strict=True):
        pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_data(generated) -> None:
    leads, events, _ = generated
    other_leads, other_events, _ = generate(seed=100, **SMALL)
    assert len(events) != len(other_events) or not events.equals(other_events)
    assert not leads["industry"].equals(other_leads["industry"])


def test_written_dataset_hash_is_stable_across_runs(tmp_path) -> None:
    """Two generations of the same seed must hash identically on disk.

    This caught a real problem: the Parquet writer's default file-name template
    embeds a fresh UUID per run, so the digest moved even though every byte of
    data was the same.
    """
    digests = []
    for attempt in ("first", "second"):
        root = tmp_path / attempt
        root.mkdir()
        leads, events, conversions = generate(seed=7, **SMALL)
        write(
            leads,
            events,
            conversions,
            raw_dir=root,
            events_dir=root / "events",
            leads_path=root / "leads.parquet",
            conversions_path=root / "conversions.parquet",
        )
        digests.append(
            hash_paths([root / "events", root / "leads.parquet", root / "conversions.parquet"])
        )
    assert digests[0] == digests[1]


def test_hash_changes_when_a_byte_changes(tmp_path) -> None:
    """Guard the guard: a hash that never changes would prove nothing."""
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    before = hash_path(path)
    path.write_bytes(b"abd")
    assert hash_path(path) != before


def test_event_volume_is_near_the_requested_target(generated) -> None:
    _, events, _ = generated
    # Poisson draws plus the drop of post-conversion activity mean this lands
    # near, not on, the target.
    assert 0.75 * SMALL["target_events"] <= len(events) <= 1.1 * SMALL["target_events"]


def test_events_stay_inside_the_observation_window(generated) -> None:
    _, events, _ = generated
    start = pd.Timestamp(DAY_ZERO)
    end = start + pd.Timedelta(days=SMALL["n_days"])
    assert events["event_ts"].min() >= start
    assert events["event_ts"].max() < end


def test_every_event_type_appears(generated) -> None:
    _, events, _ = generated
    assert set(events["event_type"].astype(str).unique()) == set(EVENT_TYPES)


def test_event_lead_ids_are_all_known_leads(generated) -> None:
    leads, events, conversions = generated
    known = set(leads["lead_id"])
    assert set(events["lead_id"]) <= known
    assert set(conversions["lead_id"]) <= known


def test_conversion_rate_is_plausible(generated) -> None:
    """A generator that converted nobody, or everybody, would make every
    downstream metric meaningless."""
    leads, _, conversions = generated
    rate = len(conversions) / len(leads)
    assert 0.05 < rate < 0.60, rate


def test_activity_predicts_conversion(generated) -> None:
    """The signal the whole project depends on must actually be there.

    Converting leads should be more active before they convert than
    non-converting leads are over the same stretch. Without this the AUROC
    numbers would be measuring nothing.
    """
    leads, events, conversions = generated
    cutoff = pd.Timestamp(DAY_ZERO) + pd.Timedelta(days=30)
    early = events.loc[events["event_ts"] < cutoff]
    counts = early.groupby("lead_id").size()

    converted_after = set(conversions.loc[conversions["converted_at"] >= cutoff, "lead_id"])
    never = set(leads["lead_id"]) - set(conversions["lead_id"])

    converted_mean = counts.reindex(sorted(converted_after)).fillna(0).mean()
    never_mean = counts.reindex(sorted(never)).fillna(0).mean()
    assert converted_mean > never_mean, (converted_mean, never_mean)


def test_conversions_have_no_duplicate_leads(generated) -> None:
    _, _, conversions = generated
    assert conversions["lead_id"].is_unique


def test_leads_predate_the_observation_window(generated) -> None:
    """Every lead exists before day zero, so no lead's count windows are
    truncated by its own age."""
    leads, _, _ = generated
    assert (leads["created_at"] < pd.Timestamp(DAY_ZERO)).all()

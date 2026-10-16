"""The feature job must be blind to anything at or after the cutoff.

The check is adversarial rather than structural: take the sample, bolt on a
large volume of fabricated post-cutoff activity, rerun the job, and require
byte-for-byte identical features. A pipeline that drifted its filter, or that
computed recency from the whole event log, would fail here.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.generate import to_parquet_timestamps
from lead_scoring.config import FEATURE_NAMES
from pipelines.features import build_features
from tests.conftest import SampleData

pytestmark = pytest.mark.spark


def _loud_post_cutoff_events(sample: SampleData, per_lead: int) -> pd.DataFrame:
    """Fabricate dense activity strictly after the cutoff for every lead.

    Every lead gets the same burst, so if any of it reached the features, the
    affected columns would move for essentially every row.
    """
    rng = np.random.default_rng(7)
    lead_ids = np.repeat(sample.leads["lead_id"].to_numpy(), per_lead)
    offsets = rng.integers(0, 10 * 86_400, size=lead_ids.shape[0])
    timestamps = pd.to_datetime(
        np.int64(sample.cutoff.timestamp()) + offsets, unit="s", utc=True
    )
    types = rng.choice(["email_open", "call", "demo_request", "page_visit"], size=lead_ids.shape[0])
    frame = pd.DataFrame(
        {"lead_id": lead_ids, "event_type": types, "event_ts": timestamps}
    )
    frame["event_date"] = frame["event_ts"].dt.date.astype("string")
    return to_parquet_timestamps(frame)


@pytest.fixture(scope="module")
def features_with_and_without_future(spark, sample_data: SampleData, tmp_path_factory):
    """Features from the clean log, and from the log plus post-cutoff noise."""
    clean_events = spark.read.parquet(str(sample_data.root / "events"))
    leads = spark.read.parquet(str(sample_data.root / "leads.parquet"))

    baseline = (
        build_features(spark, clean_events, leads, sample_data.cutoff)
        .toPandas()
        .sort_values("lead_id", ignore_index=True)
    )

    # Write a second event dataset: the original rows plus the fabricated burst.
    polluted_root = tmp_path_factory.mktemp("polluted_events")
    extra = _loud_post_cutoff_events(sample_data, per_lead=25)
    original = sample_data.events[["lead_id", "event_type", "event_ts", "event_date"]].copy()
    original["event_type"] = original["event_type"].astype(str)
    combined = pd.concat([original, extra], ignore_index=True)
    pq.write_to_dataset(
        pa.Table.from_pandas(to_parquet_timestamps(combined), preserve_index=False),
        root_path=str(polluted_root),
        partition_cols=["event_date"],
    )

    polluted_events = spark.read.parquet(str(polluted_root))
    polluted = (
        build_features(spark, polluted_events, leads, sample_data.cutoff)
        .toPandas()
        .sort_values("lead_id", ignore_index=True)
    )
    return baseline, polluted, len(extra)


def test_post_cutoff_events_were_actually_added(features_with_and_without_future) -> None:
    """Guard the guard: if the burst were empty, the test below would pass trivially."""
    _, _, added = features_with_and_without_future
    assert added == 400 * 25


def test_features_identical_despite_future_events(features_with_and_without_future) -> None:
    baseline, polluted, _ = features_with_and_without_future
    pd.testing.assert_frame_equal(
        baseline[["lead_id", *FEATURE_NAMES]],
        polluted[["lead_id", *FEATURE_NAMES]],
        check_exact=True,
    )


def test_recency_never_negative(features_with_and_without_future) -> None:
    """A negative recency would mean an event at or after the cutoff was used."""
    baseline, _, _ = features_with_and_without_future
    assert baseline["recency_days"].min() >= 0.0


def test_generator_stops_activity_at_conversion(sample_data: SampleData) -> None:
    """A converted lead produces no further events, which is what makes
    post-cutoff volume a consequence of the label rather than a predictor."""
    converted = sample_data.conversions.set_index("lead_id")["converted_at"]
    joined = sample_data.events.join(converted, on="lead_id", how="inner")
    after = joined.loc[joined["event_ts"] >= joined["converted_at"]]
    assert after.empty, f"{len(after)} events on or after their lead's conversion"


def test_cutoffs_are_ordered_and_windows_do_not_overlap_labels() -> None:
    """The train label window must close before the test label window opens."""
    from lead_scoring.config import TEST_SNAPSHOT, TRAIN_SNAPSHOT

    assert TRAIN_SNAPSHOT.cutoff < TEST_SNAPSHOT.cutoff
    assert TRAIN_SNAPSHOT.label_end <= TEST_SNAPSHOT.cutoff
    assert TEST_SNAPSHOT.label_end - TEST_SNAPSHOT.cutoff == dt.timedelta(days=15)

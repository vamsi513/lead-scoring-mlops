"""PySpark job: build per-lead activity features as of a snapshot cutoff.

Reads the synthetic event log, aggregates one row per lead, writes Parquet.

The cutoff is the whole point of this job. Only events with
``event_ts < cutoff`` are read, so no feature can depend on anything that
happened at or after the cutoff. ``tests/test_leakage.py`` enforces that by
appending post-cutoff events and asserting the output does not move.

The feature definitions here are mirrored by an independent pandas
implementation in ``lead_scoring.reference_features``; the two are compared on
a small sample by ``tests/test_features_parity.py``. The duplication is
deliberate, since a shared implementation would make that test vacuous.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import TYPE_CHECKING

from lead_scoring import cli
from lead_scoring.config import (
    CATEGORICAL_FEATURES,
    COUNT_WINDOWS,
    EVENT_TYPES,
    EVENTS_DIR,
    FEATURES_DIR,
    LEADS_PATH,
    NO_EVENT_RECENCY,
    NUMERIC_FEATURES,
    SNAPSHOTS,
    Snapshot,
)
from lead_scoring.spark import session_scope

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession

#: Seconds in a day, as a float, for timestamp arithmetic.
_DAY_SECONDS = 86_400.0


def build_features(
    events: DataFrame,
    leads: DataFrame,
    cutoff: dt.datetime,
) -> DataFrame:
    """Aggregate events into one feature row per lead, as of ``cutoff``.

    Every lead in ``leads`` appears in the output. Leads with no event before
    the cutoff get zero counts and ``recency_days = NO_EVENT_RECENCY``.
    """
    from pyspark.sql import functions as F  # noqa: N812 - the PySpark convention

    cutoff_epoch = F.lit(cutoff.timestamp())

    # All time comparisons are done on epoch seconds. Comparing timestamps
    # against literals would depend on the session timezone and on how the
    # Parquet writer recorded the column; epoch seconds have neither problem.
    event_epoch = F.col("event_ts").cast("double")

    # Only events strictly before the cutoff. The partition-column predicate
    # lets Spark skip whole date directories; the epoch predicate is the one
    # that actually defines the window.
    before = events.where(
        (F.col("event_date").cast("string") < F.lit(cutoff.date().isoformat()))
        & (event_epoch < cutoff_epoch)
    )

    def in_window(days_from: int, days_to: int) -> Column:
        """Events in [cutoff - days_from, cutoff - days_to)."""
        lower = F.lit((cutoff - dt.timedelta(days=days_from)).timestamp())
        upper = F.lit((cutoff - dt.timedelta(days=days_to)).timestamp())
        return (event_epoch >= lower) & (event_epoch < upper)

    aggregations = []
    for window in COUNT_WINDOWS:
        window_cond = in_window(window, 0)
        for event_type in EVENT_TYPES:
            aggregations.append(
                F.sum(
                    F.when(window_cond & (F.col("event_type") == event_type), 1).otherwise(0)
                ).alias(f"n_{event_type}_{window}d")
            )
        aggregations.append(F.sum(F.when(window_cond, 1).otherwise(0)).alias(f"n_total_{window}d"))

    aggregations.extend(
        [
            F.sum(F.when(in_window(14, 7), 1).otherwise(0)).alias("n_total_prev7d"),
            F.count_distinct(F.when(in_window(30, 0), F.to_date("event_ts"))).alias(
                "active_days_30d"
            ),
            F.count_distinct(F.when(in_window(30, 0), F.col("event_type"))).alias(
                "distinct_types_30d"
            ),
            F.max(event_epoch).alias("_last_epoch"),
        ]
    )

    per_lead = before.groupBy("lead_id").agg(*aggregations)

    joined = leads.join(per_lead, on="lead_id", how="left")

    # Leads with no qualifying event have nulls from the left join.
    zero_filled = joined
    for column in (
        [f"n_{t}_{w}d" for w in COUNT_WINDOWS for t in EVENT_TYPES]
        + [f"n_total_{w}d" for w in COUNT_WINDOWS]
        + ["n_total_prev7d", "active_days_30d", "distinct_types_30d"]
    ):
        zero_filled = zero_filled.withColumn(
            column, F.coalesce(F.col(column), F.lit(0)).cast("int")
        )

    recency = F.when(F.col("_last_epoch").isNull(), F.lit(NO_EVENT_RECENCY)).otherwise(
        (cutoff_epoch - F.col("_last_epoch")) / F.lit(_DAY_SECONDS)
    )

    derived = (
        zero_filled.withColumn("recency_days", recency.cast("double"))
        .withColumn(
            "trend_ratio",
            (F.col("n_total_7d") + F.lit(1.0)) / (F.col("n_total_prev7d") + F.lit(1.0)),
        )
        .withColumn(
            "trend_excess",
            F.col("n_total_7d").cast("double")
            - (F.lit(7.0) / F.lit(30.0)) * F.col("n_total_30d").cast("double"),
        )
        .withColumn(
            "tenure_days",
            (cutoff_epoch - F.col("created_at").cast("double")) / F.lit(_DAY_SECONDS),
        )
        .withColumn("snapshot_cutoff", F.lit(cutoff.date().isoformat()))
    )

    columns = ["lead_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "snapshot_cutoff"]
    return derived.select(*columns)


def run_snapshot(
    spark: SparkSession,
    snapshot: Snapshot,
    events_dir: Path = EVENTS_DIR,
    leads_path: Path = LEADS_PATH,
    out_dir: Path = FEATURES_DIR,
) -> tuple[int, float]:
    """Build and write features for one snapshot. Returns (rows, seconds)."""
    events = spark.read.parquet(str(events_dir))
    leads = spark.read.parquet(str(leads_path))

    started = time.perf_counter()
    features = build_features(events, leads, snapshot.cutoff)
    destination = out_dir / f"snapshot={snapshot.cutoff_date}"
    # One row per lead is a small output; writing it from every shuffle
    # partition would produce dozens of tiny files and make the Parquet writer
    # contend for heap. Four files keep the read parallel without that.
    features.coalesce(4).write.mode("overwrite").parquet(str(destination))
    elapsed = time.perf_counter() - started

    rows = spark.read.parquet(str(destination)).count()
    return rows, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=EVENTS_DIR)
    parser.add_argument("--leads", type=Path, default=LEADS_PATH)
    parser.add_argument("--out", type=Path, default=FEATURES_DIR)
    parser.add_argument("--master", default="local[*]")
    args = parser.parse_args(argv)

    with session_scope("lead-scoring-features", master=args.master) as spark:
        for snapshot in SNAPSHOTS:
            rows, elapsed = run_snapshot(spark, snapshot, args.events, args.leads, args.out)
            print(
                f"snapshot {snapshot.name:5s} cutoff={snapshot.cutoff_date} "
                f"rows={rows:,} wrote in {elapsed:.1f}s"
            )
    return 0


if __name__ == "__main__":
    # Not SystemExit: see lead_scoring.cli for the Arrow shutdown deadlock.
    cli.run(main)

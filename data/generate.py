"""Generate a SYNTHETIC CRM activity dataset.

THE DATA THIS SCRIPT WRITES IS ENTIRELY SYNTHETIC. It is produced by the random
process described below and has never been near a real CRM, real company or
real person. Any model score measured on it says something about this process
and nothing about real lead behaviour.

How a lead's outcome is decided
-------------------------------
Each lead gets a latent quality ``q`` from its segment plus lead-level noise.
``q`` drives a daily activity intensity, and activity is drawn per day and per
event type from a Poisson distribution. Conversion is then a *daily hazard*
process: on each day the chance of converting depends on ``q`` and on how much
activity the lead produced in the preceding 7 days, plus per-lead noise. The
first day the hazard fires is the conversion day.

Because the hazard on day ``d`` only ever reads activity strictly before day
``d``, activity before any cutoff is predictive of conversion after that cutoff
without the label being a function of post-cutoff events. That is what makes a
leakage-free split possible at all: see ``lead_scoring.config`` for the two
snapshot cutoffs, and ``tests/test_leakage.py`` for the test that holds the
feature pipeline to it.

Outputs (all under data/raw/, all synthetic):
    leads.parquet         one row per lead: segment attributes and created_at
    conversions.parquet   one row per converting lead: lead_id, converted_at
    events/date=.../      activity events, Hive-partitioned by event date
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lead_scoring import cli
from lead_scoring.config import (
    CONVERSIONS_PATH,
    DAY_ZERO,
    DEFAULT_SEED,
    EVENT_TYPES,
    EVENTS_DIR,
    LEADS_PATH,
    N_LEADS,
    OBSERVATION_DAYS,
    RAW_DIR,
    TARGET_EVENTS,
)

# --- segment definitions ------------------------------------------------------
# Each segment level carries an additive effect on latent lead quality. The
# numbers are invented; they only need to produce a learnable signal.

INDUSTRIES: Final = ("saas", "manufacturing", "healthcare", "retail", "finance")
INDUSTRY_WEIGHTS: Final = (0.30, 0.20, 0.15, 0.20, 0.15)
INDUSTRY_EFFECT: Final = (0.45, -0.20, 0.10, -0.35, 0.30)

COMPANY_SIZES: Final = ("1-10", "11-50", "51-200", "201-1000", "1000+")
COMPANY_SIZE_WEIGHTS: Final = (0.25, 0.28, 0.22, 0.15, 0.10)
COMPANY_SIZE_EFFECT: Final = (-0.30, -0.05, 0.20, 0.35, 0.25)

SOURCES: Final = ("organic", "paid_search", "webinar", "outbound", "referral")
SOURCE_WEIGHTS: Final = (0.22, 0.24, 0.16, 0.26, 0.12)
SOURCE_EFFECT: Final = (0.10, -0.15, 0.30, -0.40, 0.60)

REGIONS: Final = ("na", "emea", "apac", "latam")
REGION_WEIGHTS: Final = (0.45, 0.30, 0.17, 0.08)
REGION_EFFECT: Final = (0.12, 0.00, -0.10, -0.18)

#: Relative share of each event type in the overall event mix. Demo requests are
#: rare and page visits common, as in a real activity feed.
EVENT_TYPE_SHARE: Final = {
    "email_open": 0.42,
    "call": 0.10,
    "demo_request": 0.03,
    "page_visit": 0.45,
}

#: How strongly each event type's recent volume feeds the conversion hazard.
#: A demo request is worth far more than a page visit.
EVENT_TYPE_INTENT: Final = {
    "email_open": 0.25,
    "call": 0.70,
    "demo_request": 2.10,
    "page_visit": 0.10,
}

#: Hazard coefficients. Tuned once by hand so the conversion rate per 15-day
#: window lands in the low single-digit percent and the signal is learnable but
#: far from separable.
HAZARD_INTERCEPT: Final = -6.30
HAZARD_QUALITY_COEF: Final = 0.55
HAZARD_INTENT_COEF: Final = 0.22
HAZARD_NOISE_SD: Final = 0.90

#: Spread of the per-lead multiplicative activity level (lognormal sigma).
ACTIVITY_SIGMA: Final = 0.75
#: Per-lead day-over-day drift in activity, as a log-slope across the window.
TREND_SD: Final = 0.022
#: Share of weekly volume that lands on weekends (business activity dips).
WEEKEND_FACTOR: Final = 0.35


#: Parquet timestamp precision. pandas defaults to nanoseconds, which Spark
#: refuses to read (PARQUET_TYPE_ILLEGAL on TIMESTAMP(NANOS)). Microseconds are
#: the widest precision both sides agree on, and the generator only ever
#: produces whole seconds, so nothing is lost.
TIMESTAMP_UNIT: Final = "us"


def to_parquet_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with every datetime column at microsecond precision."""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.as_unit(TIMESTAMP_UNIT)
    return out


def _as_categorical(codes: np.ndarray, categories: tuple[str, ...]) -> pd.Categorical:
    """Expand integer codes into a categorical carrying every category.

    ``Categorical.from_codes`` is the direct call, but its type stubs reject a
    NumPy code array. Expanding by fancy indexing is equivalent and checks.
    The full category list is passed explicitly because a small sample may not
    contain every level and the column should still declare them all.
    """
    return pd.Categorical(np.asarray(categories)[codes], categories=pd.Index(categories))


def _draw_segments(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw categorical lead attributes and the latent quality they imply."""
    industry_idx = rng.choice(len(INDUSTRIES), size=n, p=INDUSTRY_WEIGHTS)
    size_idx = rng.choice(len(COMPANY_SIZES), size=n, p=COMPANY_SIZE_WEIGHTS)
    source_idx = rng.choice(len(SOURCES), size=n, p=SOURCE_WEIGHTS)
    region_idx = rng.choice(len(REGIONS), size=n, p=REGION_WEIGHTS)

    quality = (
        np.asarray(INDUSTRY_EFFECT)[industry_idx]
        + np.asarray(COMPANY_SIZE_EFFECT)[size_idx]
        + np.asarray(SOURCE_EFFECT)[source_idx]
        + np.asarray(REGION_EFFECT)[region_idx]
        + rng.normal(0.0, 0.60, size=n)
    )

    return pd.DataFrame(
        {
            "lead_id": np.arange(1, n + 1, dtype=np.int64),
            "industry": _as_categorical(industry_idx, INDUSTRIES),
            "company_size": _as_categorical(size_idx, COMPANY_SIZES),
            "source": _as_categorical(source_idx, SOURCES),
            "region": _as_categorical(region_idx, REGIONS),
            "_quality": quality,
        }
    )


def _daily_event_counts(
    rng: np.random.Generator,
    quality: np.ndarray,
    n_days: int,
    events_per_lead: float,
) -> dict[str, np.ndarray]:
    """Draw a (n_leads, n_days) count matrix per event type.

    Activity intensity rises with latent quality, drifts per lead over the
    window (which is what makes a trend feature meaningful) and dips at
    weekends.
    """
    n_leads = quality.shape[0]
    days = np.arange(n_days)

    # Per-lead activity level: lognormal around a quality-driven mean.
    level = np.exp(0.40 * quality + rng.normal(0.0, ACTIVITY_SIGMA, size=n_leads))
    # Per-lead drift, so some leads heat up across the window and some cool off.
    slope = np.asarray(rng.normal(0.0, TREND_SD, size=n_leads))
    drift = np.exp(slope[:, None] * days[None, :])

    weekday = (DAY_ZERO.weekday() + days) % 7
    weekly = np.where(weekday >= 5, WEEKEND_FACTOR, 1.0)

    # Scale so the whole dataset lands near the requested event count.
    base = events_per_lead / n_days
    scale = base / level.mean() / weekly.mean()

    intensity = scale * level[:, None] * drift * weekly[None, :]

    counts: dict[str, np.ndarray] = {}
    for event_type in EVENT_TYPES:
        share = EVENT_TYPE_SHARE[event_type]
        counts[event_type] = rng.poisson(intensity * share).astype(np.int32)
    return counts


def _conversion_day(
    rng: np.random.Generator,
    quality: np.ndarray,
    counts: dict[str, np.ndarray],
    n_days: int,
) -> np.ndarray:
    """Return the 0-based day each lead converts, or -1 for never.

    The hazard for day ``d`` reads activity from days ``[d-7, d)`` only. No
    quantity computed from day ``d`` or later enters the decision, which is the
    property the leakage test depends on.
    """
    n_leads = quality.shape[0]

    # Intent-weighted activity per lead per day.
    intent = np.zeros((n_leads, n_days), dtype=np.float64)
    for event_type, matrix in counts.items():
        intent += EVENT_TYPE_INTENT[event_type] * matrix

    # Trailing 7-day sum, shifted so day d sees days [d-7, d).
    padded = np.concatenate([np.zeros((n_leads, 1)), intent], axis=1)
    cumulative = np.cumsum(padded, axis=1)
    trailing = np.zeros((n_leads, n_days), dtype=np.float64)
    for day in range(n_days):
        start = max(0, day - 7)
        trailing[:, day] = cumulative[:, day] - cumulative[:, start]

    noise = np.asarray(rng.normal(0.0, HAZARD_NOISE_SD, size=n_leads))[:, None]
    logit = (
        HAZARD_INTERCEPT
        + HAZARD_QUALITY_COEF * quality[:, None]
        + HAZARD_INTENT_COEF * trailing
        + noise
    )
    hazard = 1.0 / (1.0 + np.exp(-logit))

    fired = rng.random((n_leads, n_days)) < hazard
    any_fired = fired.any(axis=1)
    first = np.where(any_fired, fired.argmax(axis=1), -1)
    return first.astype(np.int64)


def _expand_events(
    rng: np.random.Generator,
    counts: dict[str, np.ndarray],
    conversion_day: np.ndarray,
    n_days: int,
) -> pd.DataFrame:
    """Turn count matrices into one row per event, with intra-day timestamps.

    A lead stops producing activity once it converts: events on or after the
    conversion day are dropped. This is the realistic behaviour and it also
    means post-cutoff event volume is itself a consequence of the label, so a
    pipeline that leaked post-cutoff events would score suspiciously well.
    """
    frames: list[pd.DataFrame] = []
    n_leads = conversion_day.shape[0]
    lead_ids = np.arange(1, n_leads + 1, dtype=np.int64)

    # A lead's events are kept only for days strictly before its conversion day.
    last_day = np.where(conversion_day < 0, n_days, conversion_day)

    for type_code, event_type in enumerate(EVENT_TYPES):
        matrix = counts[event_type].copy()
        day_index = np.arange(n_days)[None, :]
        matrix[day_index >= last_day[:, None]] = 0

        flat = matrix.reshape(-1)
        nonzero = np.flatnonzero(flat)
        if nonzero.size == 0:
            continue
        repeats = flat[nonzero]
        cells = np.repeat(nonzero, repeats)
        lead_pos, day = np.divmod(cells, n_days)

        total = int(repeats.sum())
        # Business-hours-ish timestamps: 7am to 9pm, triangular toward midday.
        second_of_day = (
            7 * 3600 + (rng.triangular(0.0, 0.45, 1.0, size=total) * 14 * 3600)
        ).astype(np.int64)

        frames.append(
            pd.DataFrame(
                {
                    "lead_id": lead_ids[lead_pos],
                    "event_type": np.full(total, type_code, dtype=np.int8),
                    "_day": day.astype(np.int32),
                    "_sec": second_of_day,
                }
            )
        )

    events = pd.concat(frames, ignore_index=True)
    events["event_ts"] = (
        events["_day"].astype("int64") * 86_400 + events["_sec"] + int(DAY_ZERO.timestamp())
    )
    events = events.drop(columns=["_sec"])
    events["event_type"] = _as_categorical(events["event_type"].to_numpy(), EVENT_TYPES)
    return events.sort_values(["event_ts", "lead_id"], kind="stable", ignore_index=True)


def generate(
    seed: int = DEFAULT_SEED,
    n_leads: int = N_LEADS,
    target_events: int = TARGET_EVENTS,
    n_days: int = OBSERVATION_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the synthetic leads, events and conversions tables in memory."""
    rng = np.random.default_rng(seed)

    leads = _draw_segments(rng, n_leads)
    quality = leads["_quality"].to_numpy()

    counts = _daily_event_counts(rng, quality, n_days, target_events / n_leads)
    conversion_day = _conversion_day(rng, quality, counts, n_days)
    events = _expand_events(rng, counts, conversion_day, n_days)

    # Leads pre-date the observation window, so every lead has a full event
    # history and the count windows are never truncated by the lead's own age.
    age_days = rng.integers(1, 181, size=n_leads)
    created_at = np.int64(DAY_ZERO.timestamp()) - age_days.astype(np.int64) * 86_400

    leads_out = leads.drop(columns=["_quality"]).copy()
    leads_out["created_at"] = pd.to_datetime(created_at, unit="s", utc=True)

    converted = conversion_day >= 0
    conversions = pd.DataFrame(
        {
            "lead_id": leads["lead_id"].to_numpy()[converted],
            "converted_at": pd.to_datetime(
                np.int64(DAY_ZERO.timestamp())
                + conversion_day[converted] * 86_400
                # Conversions land mid-afternoon; the exact hour is immaterial
                # because every cutoff in config.py is at UTC midnight.
                + 15 * 3600,
                unit="s",
                utc=True,
            ),
        }
    )

    events_out = events.drop(columns=["_day"]).copy()
    events_out["event_ts"] = pd.to_datetime(events_out["event_ts"], unit="s", utc=True)
    events_out["event_date"] = events_out["event_ts"].dt.date.astype("string")
    return (
        to_parquet_timestamps(leads_out),
        to_parquet_timestamps(events_out),
        to_parquet_timestamps(conversions),
    )


def write(
    leads: pd.DataFrame,
    events: pd.DataFrame,
    conversions: pd.DataFrame,
    raw_dir: Path = RAW_DIR,
    events_dir: Path = EVENTS_DIR,
    leads_path: Path = LEADS_PATH,
    conversions_path: Path = CONVERSIONS_PATH,
) -> None:
    """Write the three tables to Parquet, replacing any previous output."""
    if events_dir.exists():
        shutil.rmtree(events_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(pa.Table.from_pandas(leads, preserve_index=False), leads_path)
    pq.write_table(pa.Table.from_pandas(conversions, preserve_index=False), conversions_path)
    pq.write_to_dataset(
        pa.Table.from_pandas(events, preserve_index=False),
        root_path=str(events_dir),
        partition_cols=["event_date"],
        # Deterministic file names. The default template embeds a fresh UUID per
        # run, which made the dataset hash in lead_scoring.provenance differ
        # between two generations of byte-identical data, and a provenance hash
        # that changes when the data has not is worse than no hash at all.
        basename_template="part-{i}.parquet",
        # Arrow's dataset writer thread pool can deadlock in its destructor at
        # interpreter exit: the data is fully written, then the process hangs in
        # ThreadPool::Shutdown and never returns. Writing single-threaded avoids
        # the pool altogether; the cost is under a second at this data size.
        use_threads=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic CRM dataset (no real data involved)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--leads", type=int, default=N_LEADS)
    parser.add_argument("--events", type=int, default=TARGET_EVENTS, help="target event count")
    parser.add_argument("--days", type=int, default=OBSERVATION_DAYS)
    parser.add_argument("--out", type=Path, default=RAW_DIR)
    args = parser.parse_args(argv)

    leads, events, conversions = generate(args.seed, args.leads, args.events, args.days)
    out: Path = args.out
    write(
        leads,
        events,
        conversions,
        raw_dir=out,
        events_dir=out / "events",
        leads_path=out / "leads.parquet",
        conversions_path=out / "conversions.parquet",
    )

    window_end = DAY_ZERO + dt.timedelta(days=args.days)
    print("SYNTHETIC dataset written to", out)
    print(f"  leads            {len(leads):>10,}")
    print(f"  events           {len(events):>10,}")
    print(f"  converting leads {len(conversions):>10,}  ({len(conversions) / len(leads):.2%})")
    print(f"  window           {DAY_ZERO.date()} .. {window_end.date()} ({args.days} days)")
    return 0


if __name__ == "__main__":
    # Not SystemExit: see lead_scoring.cli for the Arrow shutdown deadlock.
    cli.run(main)

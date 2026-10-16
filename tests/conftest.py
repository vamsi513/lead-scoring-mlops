"""Shared fixtures. All data used by tests is synthetic, generated on the fly."""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _vendored_java_home() -> str | None:
    """JAVA_HOME for the JDK vendored under .jdk, if it is there."""
    mac = REPO_ROOT / ".jdk" / "Contents" / "Home"
    linux = REPO_ROOT / ".jdk"
    if (mac / "bin" / "java").exists():
        return str(mac)
    if (linux / "bin" / "java").exists():
        return str(linux)
    return None


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """One local SparkSession for the whole test session.

    Starting a JVM costs seconds, so this is session-scoped. Tests that use it
    must not change session-level config.
    """
    if "JAVA_HOME" not in os.environ:
        java_home = _vendored_java_home()
        if java_home is None:
            pytest.skip("no JAVA_HOME and no vendored JDK in .jdk; run scripts/setup_jdk.sh")
        os.environ["JAVA_HOME"] = java_home

    from lead_scoring.spark import session_scope

    with session_scope("lead-scoring-tests", master="local[2]", shuffle_partitions=4) as session:
        yield session


class SampleData(NamedTuple):
    """A small synthetic dataset, both in memory and written to Parquet."""

    leads: pd.DataFrame
    events: pd.DataFrame
    conversions: pd.DataFrame
    root: Path
    cutoff: dt.datetime


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory: pytest.TempPathFactory) -> SampleData:
    """A few hundred leads of synthetic activity, written out as Parquet.

    Small enough that the pandas reference runs instantly, large enough that
    every feature branch is exercised: leads with no events, leads with events
    only outside the 7-day window, and leads active on the cutoff boundary.
    """
    from data.generate import generate, write
    from lead_scoring.config import DAY_ZERO

    leads, events, conversions = generate(seed=404, n_leads=400, target_events=12_000, n_days=60)
    root = tmp_path_factory.mktemp("sample_raw")
    write(
        leads,
        events,
        conversions,
        raw_dir=root,
        events_dir=root / "events",
        leads_path=root / "leads.parquet",
        conversions_path=root / "conversions.parquet",
    )
    return SampleData(
        leads=leads,
        events=events,
        conversions=conversions,
        root=root,
        cutoff=DAY_ZERO + dt.timedelta(days=30),
    )

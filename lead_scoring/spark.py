"""Local-mode SparkSession factory.

Everything in this project runs Spark in ``local[*]`` on one machine. There is
no cluster, and nothing here has been tried at cluster scale.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _pin_worker_interpreter() -> None:
    """Make Spark's Python workers use this interpreter.

    The JVM launches workers via whatever ``python3`` resolves to on PATH. If
    that is a different minor version from the driver, Spark fails with
    PYTHON_VERSION_MISMATCH part-way into a job. Setting both variables to
    ``sys.executable`` removes the dependence on PATH ordering entirely.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def build_session(
    app_name: str = "lead-scoring",
    master: str = "local[*]",
    shuffle_partitions: int | None = None,
) -> SparkSession:
    """Create (or reuse) a local SparkSession configured for this project."""
    _pin_worker_interpreter()
    from pyspark.sql import SparkSession

    if shuffle_partitions is None:
        shuffle_partitions = min(32, (os.cpu_count() or 4) * 2)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # The Spark UI is pure overhead for a batch job on one machine.
        .config("spark.ui.enabled", "false")
    )
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel("WARN")
    return session


@contextmanager
def session_scope(
    app_name: str = "lead-scoring",
    master: str = "local[*]",
    shuffle_partitions: int | None = None,
) -> Iterator[SparkSession]:
    """Yield a SparkSession and stop it afterwards, even on failure."""
    session = build_session(app_name, master, shuffle_partitions)
    try:
        yield session
    finally:
        session.stop()

"""Data hashing and git commit lookup, for run provenance in MLflow."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

#: Read files in chunks so hashing a multi-hundred-megabyte Parquet tree does
#: not need the whole file in memory.
_CHUNK_BYTES = 1 << 20


def _file_digest(path: Path) -> str:
    """sha256 over one file's bytes, with no reference to its name."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: Path) -> str:
    """Return a sha256 identifying the data at ``path``.

    For a directory, every file underneath contributes its content digest
    paired with the *directory* it sits in, and the pairs are sorted before
    being combined.

    File names are deliberately excluded while directory names are kept, and
    both halves of that matter:

    - Writers choose file names, and two of the writers used here choose them
      randomly. Spark embeds a fresh job UUID in every output file name, so
      including names made this digest change between two builds of identical
      data, which is worse than having no digest at all.
    - Directory names are not an artifact. In a Hive-partitioned dataset the
      partition value lives only in the directory name
      (``event_date=2025-01-01/``), so dropping those would let two datasets
      with the same files under different partitions hash alike.

    The result therefore identifies the dataset's contents and its partition
    layout, and is stable across rebuilds.
    """
    if path.is_file():
        return _file_digest(path)

    if not path.is_dir():
        raise FileNotFoundError(f"cannot hash missing path: {path}")

    entries = sorted(
        (child.parent.relative_to(path).as_posix(), _file_digest(child))
        for child in path.rglob("*")
        if child.is_file()
    )

    digest = hashlib.sha256()
    for directory, content in entries:
        digest.update(directory.encode())
        digest.update(b"\0")
        digest.update(content.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def hash_paths(paths: list[Path]) -> str:
    """Combine the digests of several paths into one dataset hash."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hash_path(path).encode())
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    """Current commit sha, with ``-dirty`` appended if the tree has changes.

    Returns ``"unknown"`` when git is unavailable or the directory is not a
    repository, so that logging provenance never fails a training run.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def hash_table(path: Path, sort_by: list[str] | None = None) -> str:
    """Return a sha256 over the *contents* of a Parquet file or dataset.

    Why this exists alongside ``hash_path``: Spark's Parquet output is not
    byte-reproducible, and not for any reason this project can fix. Two runs
    over identical input produce files of identical length holding identical
    data whose footers differ, because parquet-mr serialises each column
    chunk's set of encodings in JVM hash-set order. Measured here: three
    consecutive builds of the same 50,000-row table gave three different
    digests, differing in 46 bytes of footer.

    So the data hash logged with a training run is taken over the data rather
    than over the file bytes. Rows are sorted, columns are ordered by name, and
    each column contributes its name, its dtype and its raw values. The result
    is independent of how many files the table was split into, what they were
    named, and what order the writer happened to emit rows in.

    The whole table is read into memory, which is fine for the tables here
    (tens of thousands of rows) and would not be for a large one.
    """
    frame = pd.read_parquet(path)
    key = sort_by if sort_by is not None else sorted(frame.columns)
    frame = frame.sort_values(key, kind="stable").reset_index(drop=True)

    digest = hashlib.sha256()
    digest.update(str(len(frame)).encode())
    digest.update(b"\0")
    for column in sorted(frame.columns):
        values = frame[column]
        digest.update(column.encode())
        digest.update(b"\0")
        digest.update(str(values.dtype).encode())
        digest.update(b"\0")
        if pd.api.types.is_datetime64_any_dtype(values.dtype):
            # A timezone-aware column's to_numpy() is an object array of
            # Timestamps, whose tobytes() is a block of pointer addresses that
            # differ on every run. Normalising to epoch integers first is what
            # makes a table with a timestamp column hashable at all.
            digest.update(values.astype("int64").to_numpy().tobytes())
        elif values.to_numpy().dtype.kind in "iufb":
            # Real numeric buffer: hash the values directly.
            digest.update(values.to_numpy().tobytes())
        else:
            # Strings, categories, anything object-backed. Joined with a
            # separator that cannot appear in a value, so ["a", "bc"] and
            # ["ab", "c"] cannot collide.
            digest.update(b"\0".join(str(v).encode() for v in values))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_tables(paths: list[Path]) -> str:
    """Combine the content digests of several tables into one dataset hash."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hash_table(path).encode())
    return digest.hexdigest()

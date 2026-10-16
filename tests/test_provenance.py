"""The data hash has to identify data, and nothing else.

The point of logging a digest with every training run is to be able to say
later which bytes produced a model. That only works if the digest is stable
across rebuilds of the same data and moves when the data moves. Both of those
were broken at different times in this project, by two different writers
picking random file names, so both are tested here.
"""

from __future__ import annotations

import pytest

from lead_scoring.provenance import git_commit, hash_path, hash_paths


def _write(root, relative: str, content: bytes) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def test_identical_trees_hash_alike(tmp_path) -> None:
    for name in ("a", "b"):
        _write(tmp_path / name, "part=1/data.parquet", b"rows")
        _write(tmp_path / name, "part=2/data.parquet", b"more")
    assert hash_path(tmp_path / "a") == hash_path(tmp_path / "b")


def test_writer_chosen_file_names_do_not_affect_the_hash(tmp_path) -> None:
    """Spark names output files ``part-00000-<fresh uuid>-c000.snappy.parquet``,
    so a digest that included names changed on every rebuild."""
    _write(tmp_path / "run1", "part=1/part-00000-aaaaaaaa-c000.snappy.parquet", b"rows")
    _write(tmp_path / "run2", "part=1/part-00000-bbbbbbbb-c000.snappy.parquet", b"rows")
    assert hash_path(tmp_path / "run1") == hash_path(tmp_path / "run2")


def test_partition_directory_names_do_affect_the_hash(tmp_path) -> None:
    """The partition value lives only in the directory name, so it is data."""
    _write(tmp_path / "x", "event_date=2025-01-01/f.parquet", b"rows")
    _write(tmp_path / "y", "event_date=2025-01-02/f.parquet", b"rows")
    assert hash_path(tmp_path / "x") != hash_path(tmp_path / "y")


def test_changed_content_changes_the_hash(tmp_path) -> None:
    _write(tmp_path / "x", "part=1/f.parquet", b"rows")
    before = hash_path(tmp_path / "x")
    _write(tmp_path / "x", "part=1/f.parquet", b"rowz")
    assert hash_path(tmp_path / "x") != before


def test_extra_file_changes_the_hash(tmp_path) -> None:
    _write(tmp_path / "x", "part=1/f.parquet", b"rows")
    before = hash_path(tmp_path / "x")
    _write(tmp_path / "x", "part=1/g.parquet", b"rows")
    assert hash_path(tmp_path / "x") != before


def test_moving_a_file_between_partitions_changes_the_hash(tmp_path) -> None:
    _write(tmp_path / "x", "part=1/a.parquet", b"rows")
    _write(tmp_path / "x", "part=2/b.parquet", b"more")
    before = hash_path(tmp_path / "x")
    (tmp_path / "x" / "part=1" / "a.parquet").unlink()
    _write(tmp_path / "x", "part=2/a.parquet", b"rows")
    assert hash_path(tmp_path / "x") != before


def test_single_file_hash_is_content_only(tmp_path) -> None:
    (tmp_path / "one.parquet").write_bytes(b"rows")
    (tmp_path / "two.parquet").write_bytes(b"rows")
    assert hash_path(tmp_path / "one.parquet") == hash_path(tmp_path / "two.parquet")


def test_hash_paths_is_order_sensitive(tmp_path) -> None:
    """Combining is positional, so the caller's list order is part of the key."""
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    first = hash_paths([tmp_path / "a", tmp_path / "b"])
    assert first != hash_paths([tmp_path / "b", tmp_path / "a"])


def test_missing_path_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_path(tmp_path / "nope")


def test_git_commit_is_a_sha_when_the_tree_is_a_checkout() -> None:
    """Skipped when it is not. A source export (``git archive``, a release
    tarball, the Docker image) carries no .git, and provenance correctly
    reports "unknown" there rather than failing; the test below covers that
    path. Asserting a sha unconditionally made this test fail in a clean
    export even though the code was behaving as designed."""
    from lead_scoring.config import REPO_ROOT

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout, so there is no commit to report")

    commit = git_commit(REPO_ROOT)
    base = commit.removesuffix("-dirty")
    assert len(base) == 40, commit
    assert all(c in "0123456789abcdef" for c in base), commit


def test_git_commit_reports_unknown_outside_a_repo(tmp_path) -> None:
    """Provenance logging must never be the thing that fails a training run."""
    assert git_commit(tmp_path) == "unknown"


def _write_parquet(path, frame, parts: int = 1) -> None:
    """Write ``frame`` as ``parts`` Parquet files under ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    size = -(-len(frame) // parts)  # ceiling division
    for i in range(parts):
        chunk = frame.iloc[i * size : (i + 1) * size]
        chunk.to_parquet(path / f"chunk-{i}.parquet", index=False)


def _sample_frame():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "lead_id": np.arange(60, dtype="int64"),
            "count": rng.integers(0, 50, 60).astype("int32"),
            "score": rng.random(60),
            "segment": rng.choice(["saas", "retail", "finance"], 60),
        }
    )


def test_content_hash_ignores_how_the_table_was_split(tmp_path) -> None:
    """The same rows written as one file or as five must hash alike.

    This is the property byte-level hashing could not give: Spark splits rows
    across output files differently on every run.
    """
    frame = _sample_frame()
    _write_parquet(tmp_path / "one", frame, parts=1)
    _write_parquet(tmp_path / "five", frame, parts=5)
    from lead_scoring.provenance import hash_table

    assert hash_table(tmp_path / "one") == hash_table(tmp_path / "five")


def test_content_hash_ignores_row_order(tmp_path) -> None:
    frame = _sample_frame()
    _write_parquet(tmp_path / "asc", frame, parts=1)
    _write_parquet(tmp_path / "shuffled", frame.sample(frac=1.0, random_state=1), parts=1)
    from lead_scoring.provenance import hash_table

    assert hash_table(tmp_path / "asc") == hash_table(tmp_path / "shuffled")


def test_content_hash_notices_a_changed_value(tmp_path) -> None:
    from lead_scoring.provenance import hash_table

    frame = _sample_frame()
    _write_parquet(tmp_path / "a", frame)
    before = hash_table(tmp_path / "a")

    changed = frame.copy()
    changed.loc[0, "score"] = changed.loc[0, "score"] + 1e-9
    _write_parquet(tmp_path / "b", changed)
    assert hash_table(tmp_path / "b") != before


def test_content_hash_notices_a_dropped_row(tmp_path) -> None:
    from lead_scoring.provenance import hash_table

    frame = _sample_frame()
    _write_parquet(tmp_path / "a", frame)
    _write_parquet(tmp_path / "b", frame.iloc[:-1])
    assert hash_table(tmp_path / "a") != hash_table(tmp_path / "b")


def test_content_hash_notices_a_renamed_column(tmp_path) -> None:
    from lead_scoring.provenance import hash_table

    frame = _sample_frame()
    _write_parquet(tmp_path / "a", frame)
    _write_parquet(tmp_path / "b", frame.rename(columns={"count": "tally"}))
    assert hash_table(tmp_path / "a") != hash_table(tmp_path / "b")


def test_content_hash_cannot_be_fooled_by_string_boundaries(tmp_path) -> None:
    """["ab", "c"] must not hash the same as ["a", "bc"]."""
    import pandas as pd

    from lead_scoring.provenance import hash_table

    _write_parquet(tmp_path / "a", pd.DataFrame({"s": ["ab", "c"]}))
    _write_parquet(tmp_path / "b", pd.DataFrame({"s": ["a", "bc"]}))
    assert hash_table(tmp_path / "a") != hash_table(tmp_path / "b")


def test_content_hash_is_stable_for_a_timezone_aware_column(tmp_path) -> None:
    """A tz-aware column's ``to_numpy()`` is an object array of Timestamps.

    Hashing its ``tobytes()`` hashes pointer addresses, which change on every
    run. This caught exactly that: the conversions table has a tz-aware
    ``converted_at``, and its digest moved between two reads of the same file.
    """
    import pandas as pd

    from lead_scoring.provenance import hash_table

    frame = pd.DataFrame(
        {
            "lead_id": [1, 2, 3],
            "converted_at": pd.to_datetime(
                ["2025-02-03", "2025-02-07", "2025-02-11"], utc=True
            ).as_unit("us"),
        }
    )
    _write_parquet(tmp_path / "t", frame)
    digests = {hash_table(tmp_path / "t") for _ in range(5)}
    assert len(digests) == 1, "digest differs between reads of the same file"


def test_content_hash_notices_a_changed_timestamp(tmp_path) -> None:
    import pandas as pd

    from lead_scoring.provenance import hash_table

    base = pd.DataFrame({"lead_id": [1], "converted_at": pd.to_datetime(["2025-02-03"], utc=True)})
    shifted = pd.DataFrame(
        {"lead_id": [1], "converted_at": pd.to_datetime(["2025-02-04"], utc=True)}
    )
    _write_parquet(tmp_path / "a", base)
    _write_parquet(tmp_path / "b", shifted)
    assert hash_table(tmp_path / "a") != hash_table(tmp_path / "b")

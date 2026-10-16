"""Data hashing and git commit lookup, for run provenance in MLflow."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: Read files in chunks so hashing a multi-hundred-megabyte Parquet tree does
#: not need the whole file in memory.
_CHUNK_BYTES = 1 << 20


def hash_path(path: Path) -> str:
    """Return a sha256 over the contents of ``path``.

    For a directory, every file underneath is hashed in sorted relative-path
    order, and the relative path itself is mixed in, so that renaming or moving
    a file changes the digest. The result identifies a dataset's bytes and
    layout, not its location on disk.
    """
    digest = hashlib.sha256()
    if path.is_file():
        _feed_file(digest, path, path.name)
        return digest.hexdigest()

    if not path.is_dir():
        raise FileNotFoundError(f"cannot hash missing path: {path}")

    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        _feed_file(digest, child, str(child.relative_to(path)))
    return digest.hexdigest()


def _feed_file(digest: "hashlib._Hash", path: Path, label: str) -> None:
    digest.update(label.encode())
    digest.update(b"\0")
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)


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

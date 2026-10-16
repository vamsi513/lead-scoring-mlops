"""The built image must contain every first-party module the app imports.

This exists because the usual way a serving image breaks is not a bad
Dockerfile line but a missing COPY: the app imports a package that was never
added to the image, and nothing notices until the container starts in an
environment where the source tree is not also mounted.

The check is in two halves, and both matter:

  1. every first-party module reachable from ``serving.app`` imports cleanly
     *inside the container*, which catches a missing COPY directly;
  2. the module list the test derives is non-trivial and really does span both
     packages, so the test cannot quietly pass by checking nothing.

The image is built by the fixture if it is not already present, so the test is
self-contained.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from lead_scoring.config import REPO_ROOT
from lead_scoring.importgraph import first_party_modules

pytestmark = pytest.mark.docker

IMAGE_TAG = "lead-scoring-serving:test"
ENTRY_MODULE = "serving.app"
BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 300


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return probe.returncode == 0


@pytest.fixture(scope="session")
def image() -> str:
    """Build the serving image once, or reuse one already tagged."""
    if not _docker_available():
        pytest.skip("docker is not available")

    exists = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if exists.returncode != 0:
        build = subprocess.run(
            ["docker", "build", "-t", IMAGE_TAG, "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
            check=False,
        )
        if build.returncode != 0:
            pytest.fail(f"docker build failed:\n{build.stdout[-4000:]}\n{build.stderr[-4000:]}")
    return IMAGE_TAG


@pytest.fixture(scope="session")
def expected_modules() -> set[str]:
    """First-party modules reachable from the app's entry point."""
    return first_party_modules(ENTRY_MODULE, REPO_ROOT)


def test_the_module_list_is_not_trivially_empty(expected_modules: set[str]) -> None:
    """Guard the guard. If the import walk returned nothing, or only the entry
    point, the real test below would pass against an empty image."""
    assert ENTRY_MODULE in expected_modules
    assert len(expected_modules) >= 5, sorted(expected_modules)
    assert any(m.startswith("lead_scoring") for m in expected_modules)
    assert any(m.startswith("serving") for m in expected_modules)


def test_every_first_party_module_imports_inside_the_image(
    image: str, expected_modules: set[str]
) -> None:
    """Import each module in the container and report every failure at once."""
    script = (
        "import importlib, json, sys\n"
        f"mods = {sorted(expected_modules)!r}\n"
        "failures = {}\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as exc:\n"
        "        failures[m] = f'{type(exc).__name__}: {exc}'\n"
        "print('RESULT' + json.dumps({'checked': len(mods), 'failures': failures}))\n"
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image, "-c", script],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, f"container failed:\n{result.stdout}\n{result.stderr}"

    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("RESULT"))
    payload = json.loads(line.removeprefix("RESULT"))
    assert payload["failures"] == {}, (
        f"{len(payload['failures'])} of {payload['checked']} modules missing or broken "
        f"in the image: {payload['failures']}"
    )
    assert payload["checked"] == len(expected_modules)


def test_the_app_object_is_importable_in_the_image(image: str) -> None:
    """Beyond the modules existing, the ASGI app must actually construct."""
    script = (
        "from serving.app import app\n"
        "print('ROUTES' + ','.join(sorted(r.path for r in app.routes)))\n"
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image, "-c", script],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    routes = next(ln for ln in result.stdout.splitlines() if ln.startswith("ROUTES"))
    for path in ("/health", "/model-info", "/predict"):
        assert path in routes, f"{path} missing from {routes}"


def test_container_runs_as_a_non_root_user(image: str) -> None:
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image,
         "-c", "import os; print(f'UID{os.getuid()}:{os.geteuid()}')"],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("UID"))
    uid, euid = line.removeprefix("UID").split(":")
    assert int(uid) != 0, "container runs as root"
    assert int(euid) != 0, "container has an effective root uid"


def test_pyspark_is_not_in_the_serving_image(image: str) -> None:
    """The API scores prebuilt features; shipping Spark would add hundreds of
    megabytes and a JVM for nothing."""
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image,
         "-c", "import importlib.util as u; print('HAS' + str(u.find_spec('pyspark') is not None))"],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "HASFalse" in result.stdout

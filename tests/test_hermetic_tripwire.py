"""The conftest tripwire (_refuse_unhermetic_run) must never regress.

It is the last line of defence the 2026-07-20 incident demanded: a plain
``pytest`` invocation on a machine with a live, writable ``~/.hermes`` has to
die in ``pytest_configure`` — before a single test or fixture runs. These
tests spawn a real pytest subprocess against a fake writable store to prove
both directions: refusal without attestation, acceptance with it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TARGET = "tests/tools/test_store_guard.py"


def _collect(env_extra: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    home = tmp_path / "fake-home"
    (home / ".hermes").mkdir(parents=True, exist_ok=True)
    (home / ".hermes" / "state.db").touch()
    env = {**os.environ, "HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("HERMES_HERMETIC", None)
    env.pop("HERMES_ALLOW_UNHERMETIC", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", _TARGET],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_tripwire_refuses_writable_real_store(tmp_path):
    proc = _collect({}, tmp_path)
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "ABGEBROCHEN" in proc.stdout + proc.stderr


def test_tripwire_accepts_runner_attestation(tmp_path):
    proc = _collect({"HERMES_HERMETIC": "1"}, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_tripwire_accepts_explicit_override(tmp_path):
    proc = _collect(
        {"HERMES_ALLOW_UNHERMETIC": "yes-i-accept-the-risk"}, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr

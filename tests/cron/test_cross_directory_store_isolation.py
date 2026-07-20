"""Cron isolation must hold across a cross-directory collection.

Regression guard for the 2026-07-20 incident. The isolation fixture used to
live in ``tests/cron/conftest.py``, so it only applied to the ``tests/cron``
subtree — but the leak happened elsewhere: collecting
``tests/hermes_cli/test_cron.py`` (which imports ``cron.jobs`` at module
scope) together with ``tests/hermes_cli/test_console_engine.py`` created job
``alpha`` (``say hello``, every 1h) in the production
``~/.hermes/cron/jobs.json``, where the live scheduler then executed it
hourly. A same-directory test cannot catch that class of bug.

The check here runs the real cross-directory scope in a child pytest whose
``HOME`` points at a seeded decoy ``~/.hermes``, with ``HERMES_HOME`` unset.
That reproduces the production failure mode exactly — ``cron/jobs.py``
freezes ``HERMES_DIR = get_hermes_home().resolve()`` at import time, so the
decoy store is what the module constants bind to, and only a fixture that
overrides them at call time can keep jobs out of it.

Using a decoy rather than the operator's real store is deliberate: a
regression must be *detected*, not *performed*. If this test ever fails, the
damage is confined to a tempdir. ``test_live_store_isolation.py`` covers the
real store non-destructively alongside it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The exact pair from the incident: a module-scope ``from cron.jobs import ...``
# in one directory, and the job-creating console test in the same directory,
# collected together in one process.
CROSS_DIRECTORY_SCOPE = [
    "tests/hermes_cli/test_cron.py",
    "tests/hermes_cli/test_console_engine.py::test_cron_pause_resume_and_run_require_confirmation",
]

DECOY_JOBS = {
    "jobs": [
        {
            "id": "decoyjob00001",
            "name": "decoy-keepalive",
            "prompt": "noop",
            "schedule": {"display": "every 60m", "kind": "interval", "minutes": 60},
            "state": "scheduled",
        },
        {
            "id": "decoyjob00002",
            "name": "decoy-second",
            "prompt": "noop",
            "schedule": {"display": "every 30m", "kind": "interval", "minutes": 30},
            "state": "paused",
        },
    ],
    "updated_at": "2026-07-20T00:00:00+00:00",
}


def _job_identities(store: Path) -> set[str]:
    """Identity set of a job store, ignoring volatile bookkeeping fields.

    A byte hash would be the wrong assertion even here: marking a run stamps
    ``last_run_at`` on an untouched job. What a test may never do is change
    *which jobs exist*.
    """
    raw = json.loads(store.read_text(encoding="utf-8"))
    jobs = raw if isinstance(raw, list) else raw.get("jobs", [])
    return {str(job.get("id")) for job in jobs}


def test_cross_directory_collection_does_not_touch_the_default_store(tmp_path):
    """Running the incident's own scope must leave the default store intact."""
    decoy_home = tmp_path / "decoy_home"
    decoy_store = decoy_home / ".hermes" / "cron" / "jobs.json"
    decoy_store.parent.mkdir(parents=True)
    decoy_store.write_text(json.dumps(DECOY_JOBS, indent=2), encoding="utf-8")
    before = _job_identities(decoy_store)

    env = dict(os.environ)
    # HOME is what ``get_hermes_home()`` falls back to, and HERMES_HOME must be
    # absent so the child resolves the decoy exactly the way production
    # resolves the operator's real store.
    env["HOME"] = str(decoy_home)
    env.pop("HERMES_HOME", None)
    # Inherit the parent's resolved import path. Moving HOME also moves
    # Python's per-user site-packages, so without this the child could lose
    # dependencies the parent has and fail for reasons unrelated to isolation.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [entry for entry in sys.path if entry]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *CROSS_DIRECTORY_SCOPE,
            "-q",
            "--no-header",
            "-rf",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert decoy_store.exists(), "child run deleted the default cron store"
    after = _job_identities(decoy_store)

    created = after - before
    assert not created, (
        "cross-directory pytest scope wrote into the default cron store — the "
        f"suite-root _isolated_cron_store fixture is not effective. New job ids: {created}\n"
        f"--- child stdout ---\n{result.stdout[-4000:]}"
    )
    assert after == before, f"default job set changed: {after ^ before}"

    # A clean store proves nothing if the scope never ran. Exit codes 2-5 are
    # interrupted / internal error / usage error / nothing-collected — any of
    # them means the assertion above passed vacuously.
    assert result.returncode in (0, 1), (
        f"child scope did not execute (exit {result.returncode})\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
    )

    # The job-creating test in particular must have run and passed: it is the
    # one that produced the live `alpha` job. We assert on it by name rather
    # than on the whole scope's exit code, because unrelated pre-existing
    # failures there (e.g. a missing optional `croniter`) are not this
    # fixture's business and must not turn this guard red.
    failed = {
        line.split("::", 1)[-1].split()[0]
        for line in result.stdout.splitlines()
        if line.startswith("FAILED ")
    }
    alpha_test = CROSS_DIRECTORY_SCOPE[1].split("::", 1)[1]
    assert alpha_test not in failed, (
        f"{alpha_test} failed — isolation cannot be judged from this run\n"
        f"--- stdout ---\n{result.stdout[-4000:]}"
    )
    assert " passed" in result.stdout, (
        f"child scope collected no passing tests\n--- stdout ---\n{result.stdout[-4000:]}"
    )

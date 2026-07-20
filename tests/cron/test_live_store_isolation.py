"""The cron test suite must never touch the real job store.

Regression guard for the 2026-07-19 incident: a full-suite run created
seven dummy jobs in the production ``~/.hermes/cron/jobs.json`` (one of
them was then executed by the live scheduler every 60 minutes). Root
cause: ``cron/jobs.py`` resolves ``HERMES_DIR`` at import time, so the
global ``HERMES_HOME`` redirect could not reach it — only the ContextVar
override in ``use_cron_store()`` can.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cron.jobs import _current_cron_store, create_job, load_jobs


def _real_store() -> Path:
    """The production store, resolved independently of any override."""
    return Path.home() / ".hermes" / "cron" / "jobs.json"


def test_active_store_is_not_the_real_one(tmp_path):
    """The autouse fixture must have redirected the store away from ~/.hermes."""
    active = Path(_current_cron_store().jobs_file).resolve()
    assert active != _real_store().resolve()
    # ...and it must live under the per-test tempdir, not somewhere else.
    assert tmp_path in active.parents


def _real_job_ids() -> set[str]:
    """Job identities in the production store, ignoring volatile bookkeeping.

    A byte hash would be flaky here: a live scheduler rewrites the file
    whenever it stamps ``last_run_at`` on an unrelated job. The identity
    set is the property that a test must never change.
    """
    raw = json.loads(_real_store().read_text(encoding="utf-8"))
    jobs = raw if isinstance(raw, list) else raw.get("jobs", [])
    return {str(job.get("id")) for job in jobs}


def test_creating_a_job_does_not_reach_the_real_store():
    """A job created in a test may not appear in the production store."""
    if not _real_store().exists():
        pytest.skip("no production cron store on this machine")
    before = _real_job_ids()

    created = create_job(name="isolation probe", schedule="every 60m", prompt="noop")

    assert any(j["name"] == "isolation probe" for j in load_jobs())
    after = _real_job_ids()
    assert str(created["id"]) not in after, (
        "cron test wrote into the production job store — the isolation "
        "fixture in tests/cron/conftest.py is not effective"
    )
    assert after == before, f"production job set changed: {after ^ before}"

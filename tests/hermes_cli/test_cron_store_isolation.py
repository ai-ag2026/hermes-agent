"""The suite-root cron isolation must reach ``tests/hermes_cli`` too.

``tests/cron/test_live_store_isolation.py`` asserts the same property from
inside the ``tests/cron`` subtree — where the old fixture already worked. The
2026-07-20 leak happened *here*: ``test_console_engine.py`` created job
``alpha`` in the production ``~/.hermes/cron/jobs.json`` because the guard was
scoped to a directory this file is not in.

Keeping the assertion in this directory is the point. It fails the moment
``_isolated_cron_store`` is narrowed back to a subtree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cron.jobs import _current_cron_store, create_job, load_jobs


def _real_store() -> Path:
    """The production store, resolved independently of any override."""
    return Path.home() / ".hermes" / "cron" / "jobs.json"


def _real_job_ids() -> set[str]:
    """Job identities in the production store, ignoring volatile bookkeeping."""
    raw = json.loads(_real_store().read_text(encoding="utf-8"))
    jobs = raw if isinstance(raw, list) else raw.get("jobs", [])
    return {str(job.get("id")) for job in jobs}


def test_active_store_is_not_the_real_one(tmp_path):
    """The suite-root fixture must redirect the store away from ~/.hermes."""
    active = Path(_current_cron_store().jobs_file).resolve()
    assert active != _real_store().resolve()
    assert tmp_path in active.parents


def test_ticker_markers_are_redirected(tmp_path):
    """The ticker markers bypass _current_cron_store() and need their own patch.

    ``record_ticker_heartbeat()`` writes ``TICKER_HEARTBEAT_FILE`` straight
    from the module globals, so a fixture that redirects only the job-store
    paths still let the suite touch ``~/.hermes/cron/ticker_heartbeat``.
    """
    from cron import jobs as cron_jobs

    for marker in ("TICKER_HEARTBEAT_FILE", "TICKER_SUCCESS_FILE"):
        active = Path(getattr(cron_jobs, marker)).resolve()
        assert tmp_path in active.parents, f"{marker} still points at {active}"

    cron_jobs.record_ticker_heartbeat(success=True)
    assert Path(cron_jobs.TICKER_HEARTBEAT_FILE).exists()
    assert Path(cron_jobs.TICKER_SUCCESS_FILE).exists()


def test_creating_the_alpha_job_does_not_reach_the_real_store():
    """The exact job from the incident may not appear in the production store."""
    if not _real_store().exists():
        pytest.skip("no production cron store on this machine")
    before = _real_job_ids()

    created = create_job(prompt="say hello", schedule="every 1h", name="alpha")

    assert any(j["name"] == "alpha" for j in load_jobs())
    after = _real_job_ids()
    assert str(created["id"]) not in after, (
        "a tests/hermes_cli test wrote into the production job store — the "
        "suite-root _isolated_cron_store fixture in tests/conftest.py is not "
        "effective for this directory"
    )
    assert after == before, f"production job set changed: {after ^ before}"

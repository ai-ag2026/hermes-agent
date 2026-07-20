"""Cron-test fixtures.

Provides a default ``HERMES_MODEL`` for cron run_job tests so each one
doesn't have to spell out a model. The global conftest blanks
HERMES_MODEL hermetically; without this autouse fixture every cron test
that exercises ``run_job`` would hit the fail-fast guard added in
``cron/scheduler.py`` (see issue #23979) and have to be rewritten.

Tests that specifically need ``HERMES_MODEL`` unset — model-resolution
edge cases — call ``monkeypatch.delenv("HERMES_MODEL", raising=False)``
inside the test, which overrides this fixture's value for that scope.

It also routes the cron store to a per-test tempdir. The global
``_hermetic_environment`` fixture redirects ``HERMES_HOME``, but that is
not enough here: ``cron/jobs.py`` freezes ``HERMES_DIR`` at *import* time
(``HERMES_DIR = get_hermes_home().resolve()``), long before any fixture
runs, so every test creating a job wrote into the real
``~/.hermes/cron/jobs.json``. On 2026-07-19 a full-suite run left seven
dummy jobs in the production store — one of them ("alpha", every 60m)
was picked up and executed by the live scheduler. ``use_cron_store()``
overrides the paths through a ContextVar without mutating those module
globals, which is exactly what an isolated test needs.
"""

import pytest

from cron import jobs as cron_jobs


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
    yield


@pytest.fixture(autouse=True)
def _isolated_cron_store(tmp_path, monkeypatch):
    """Point the default cron store at a per-test tempdir.

    ``cron/jobs.py`` freezes its paths at *import* time
    (``HERMES_DIR = get_hermes_home().resolve()``), so the global
    ``HERMES_HOME`` redirect in the root conftest never reaches it — every
    test that created a job wrote into the real
    ``~/.hermes/cron/jobs.json``. On 2026-07-19 a full-suite run left seven
    dummy jobs there and the live scheduler executed one of them ("alpha",
    every 60m) on schedule.

    Patching the module constants (rather than entering
    ``use_cron_store()``) keeps the existing precedence intact:
    ``_current_cron_store()`` prefers an explicit ContextVar override and
    only falls back to these constants. Tests that pin their own paths —
    ``@patch("cron.jobs.CRON_DIR", ...)`` or their own ``use_cron_store()``
    scope — therefore still win over this fixture, while everything else
    lands in the tempdir.
    """
    cron_dir = tmp_path / "cron_home" / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cron_jobs, "HERMES_DIR", cron_dir.parent, raising=False)
    monkeypatch.setattr(cron_jobs, "CRON_DIR", cron_dir, raising=False)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", cron_dir / "jobs.json", raising=False)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", cron_dir / "output", raising=False)
    yield cron_dir.parent

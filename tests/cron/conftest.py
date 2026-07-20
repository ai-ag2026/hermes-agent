"""Cron-test fixtures.

Provides a default ``HERMES_MODEL`` for cron run_job tests so each one
doesn't have to spell out a model. The global conftest blanks
HERMES_MODEL hermetically; without this autouse fixture every cron test
that exercises ``run_job`` would hit the fail-fast guard added in
``cron/scheduler.py`` (see issue #23979) and have to be rewritten.

Tests that specifically need ``HERMES_MODEL`` unset — model-resolution
edge cases — call ``monkeypatch.delenv("HERMES_MODEL", raising=False)``
inside the test, which overrides this fixture's value for that scope.

Cron-store isolation is NOT here. It used to be — an autouse fixture in
this file redirected the store — but a ``tests/cron``-scoped guard misses
the place the leak actually happened: on 2026-07-20
``tests/hermes_cli/test_console_engine.py`` created job ``alpha`` in the
real ``~/.hermes/cron/jobs.json`` and the live scheduler ran it hourly.
The isolation therefore moved to the suite-root ``_isolated_cron_store``
fixture in ``tests/conftest.py``, which covers every directory. This file
keeps only what is genuinely cron-subtree-specific.
"""

import pytest


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
    yield

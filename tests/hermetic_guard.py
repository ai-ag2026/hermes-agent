"""Pytest plugin: refuse an unhermetic run from OUTSIDE conftest.

TARS review R9, **P0-RUN-5**: until now the tripwire existed only as a
``pytest_configure`` hook in ``tests/conftest.py``. A conftest hook cannot
defend itself — pytest can be told not to load conftests at all:

* ``python -m pytest --noconftest tests/…`` — exit 0 against a writable
  live store (TARS' reproducible counter-proof);
* ``PYTEST_ADDOPTS=--noconftest`` — the same lever through the environment;
* ``--confcutdir=/somewhere/else`` — conftest discovery cut above the suite.

This module is loaded through the ini ``addopts`` (``-p tests.hermetic_guard``
in ``pyproject.toml``), which pytest applies *before* collection and which
none of the three levers above disables. The rule itself is not duplicated
here — it comes from :mod:`tests.hermetic_policy`, the single source of truth
that the runner and the conftest tripwire also use.

Failure to import this module is itself fail-closed: pytest aborts with a
usage error (exit 4) instead of running the suite unguarded.

**Exact scope of the guarantee** — stated plainly because a half-true claim
is worse than none (TARS R9 §3, acceptance point 4):

Covered (mechanically, with regression tests):
  ``--noconftest``, ``PYTEST_ADDOPTS=--noconftest``, ``--confcutdir``,
  and any combination of them.

NOT covered — pytest lets the caller unload any plugin, and no in-repo code
can prevent that:
  ``-p no:hermetic_guard`` · ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` combined with
  a hand-built config · ``-c <other-ini>`` that drops ``addopts`` ·
  ``pytest.main()`` called from a Python process that never reads the ini ·
  a different interpreter/checkout altogether.

So: this closes the *accidental* and *casually adversarial* direct path, and
it is defence in depth — **not** a security boundary against a caller who
controls the command line. The only path with a kernel-enforced guarantee
remains ``scripts/run_tests_hermetic.py`` (bubblewrap ``--ro-bind``), where
a write to a live root fails with ``EROFS`` no matter what any Python-level
guard did or did not do.
"""

from __future__ import annotations

import sys


def _refusal_reason() -> str | None:
    """The shared rule — imported defensively so a broken import fails closed."""
    try:
        from tests.hermetic_policy import refusal_reason
    except ImportError:  # loaded outside the package (e.g. -p by file path)
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tests.hermetic_policy import refusal_reason
    return refusal_reason()


def pytest_configure(config):  # noqa: D401 — pytest hook
    """Abort before collection when a live store is writable.

    Registered from ``addopts``, so this fires even when every conftest is
    skipped. ``pytest.UsageError`` exits with code 4 before any test module
    is imported.
    """
    import pytest

    reason = _refusal_reason()
    if reason is None:
        return
    raise pytest.UsageError(
        f"{reason}\n"
        "(erzwungen von tests/hermetic_guard.py — dieser Schutz liegt "
        "außerhalb der conftest-Hooks und überlebt --noconftest, "
        "PYTEST_ADDOPTS und --confcutdir.)"
    )

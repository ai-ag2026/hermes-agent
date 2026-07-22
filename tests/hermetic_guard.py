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

**Classification: BEST EFFORT — the direct pytest path is NOT fail-closed.**

TARS review R10 reopened P0-RUN-5 and was right to. An earlier version of this
docstring listed the residual levers honestly but the surrounding documents
still called the P0 *closed*. Both cannot be true: a guard the caller can
unload is defence in depth, not a boundary. So, per TARS' acceptance option 2,
the claim is withdrawn rather than the levers being papered over:

* **P0-RUN-5 is NOT closed.** Direct ``pytest`` invocation against a live store
  is not mechanically prevented.
* The only path with a kernel-enforced guarantee is
  ``scripts/run_tests_hermetic.py`` (bubblewrap ``--ro-bind``), where a write
  to a live root fails with ``EROFS`` regardless of what any Python-level guard
  did or did not do.
* Closing it for real requires a host-/agent-level gate (TARS' alternative 1b)
  — a live-system change, and therefore a decision for Manfred, not a code
  change that can be smuggled in here.

**Exact scope of what this DOES buy** — stated plainly because a half-true
claim is worse than none (TARS R9 §3, acceptance point 4):

Covered (mechanically, with regression tests):
  ``--noconftest``, ``PYTEST_ADDOPTS=--noconftest``, ``--confcutdir``,
  and any combination of them.

NOT covered — every one of these was reproduced by TARS (R10 §2) and is
pinned by a test below, so this list cannot silently drift:
  ``-p no:tests.hermetic_guard`` · ``-c <other-ini>`` that drops ``addopts`` ·
  ``pytest.main([...])`` from a process that never reads this ini ·
  a test target outside the repo, where no repo ini applies at all ·
  ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` with a hand-built config ·
  a different interpreter/checkout altogether.

So: this closes the *accidental* and *casually adversarial* direct path. It
is worth having for exactly that reason — the 2026-07-20 incident was an
accident, not an attack — but it must never be cited as the reason a live
store is safe.
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

"""The conftest tripwire must never regress — end to end, in a real pytest.

``tests/test_hermetic_policy.py`` pins the decision rule. This file proves the
rule is actually *wired into* ``pytest_configure``: a real pytest subprocess
against a live-looking store has to die before collection (``UsageError`` →
exit 4), and a protected one has to proceed.

Every case injects its own throwaway store through the ambient
``HERMES_HOME`` or the additive ``HERMES_HERMETIC_PROTECT``, so the verdict
never depends on what this particular machine happens to have.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests import hermetic_policy as policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TARGET = "tests/tools/test_store_guard.py"


@pytest.fixture(autouse=True)
def _requires_protected_real_store():
    """These tests need the canonical store to be protected already.

    That is true under ``scripts/run_tests_hermetic.py`` (bwrap --ro-bind)
    and on machines without a live store. Anywhere else the tripwire would
    fire for the machine's own reasons and the assertions would be
    meaningless — so skip rather than lie.
    """
    real = policy.passwd_home() / ".hermes"
    if policy.is_live(real) and not policy.is_readonly_mount(real):
        pytest.skip("canonical store is live and writable — not inside the runner")


def _live_store(tmp_path: Path, name: str, marker: str = "state.db") -> Path:
    root = tmp_path / name
    (root / marker).parent.mkdir(parents=True, exist_ok=True)
    (root / marker).touch()
    return root


def _collect(env_extra: dict, *extra_args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for var in ("HERMES_HERMETIC", "HERMES_ALLOW_UNHERMETIC", policy.PROTECT_ENV,
                "PYTEST_ADDOPTS"):
        env.pop(var, None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", *extra_args, _TARGET],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def _assert_refused(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "ABGEBROCHEN" in proc.stdout + proc.stderr


def test_refuses_writable_live_store(tmp_path):
    _assert_refused(_collect({policy.PROTECT_ENV: str(_live_store(tmp_path, "s"))}))


def test_refuses_ambient_custom_hermes_home(tmp_path):
    """P0 §3.3: a direct run against a custom/profile store, with nobody
    passing HERMES_HERMETIC_PROTECT, must still be caught."""
    _assert_refused(_collect({"HERMES_HOME": str(_live_store(tmp_path, "custom"))}))


def test_refuses_partial_store_without_state_db(tmp_path):
    """P1 §3.4: after a partial loss the store keeps config/profiles and is
    still worth protecting — state.db alone must not be the sentinel."""
    store = _live_store(tmp_path, "damaged", "config.yaml")
    assert not (store / "state.db").exists()
    _assert_refused(_collect({"HERMES_HOME": str(store)}))


def test_refuses_despite_spoofed_attestation_flag(tmp_path):
    """The original P0: the environment flag must change nothing."""
    _assert_refused(_collect({
        "HERMES_HOME": str(_live_store(tmp_path, "spoof")),
        "HERMES_HERMETIC": "1",
    }))


def test_refuses_chmod_only_directory(tmp_path):
    """A 0555 directory is not protection — the database inside stays
    writable, so the run must still be refused."""
    store = _live_store(tmp_path, "chmod-only")
    store.chmod(0o555)
    try:
        _assert_refused(_collect({policy.PROTECT_ENV: str(store)}))
    finally:
        store.chmod(0o755)


def test_accepts_when_nothing_live_is_exposed(tmp_path):
    empty = tmp_path / "empty-sandbox"
    empty.mkdir()
    proc = _collect({"HERMES_HOME": str(empty)})
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_legacy_override_flag_no_longer_excuses_a_live_store(tmp_path):
    """P0-RUN-1 end to end: the abolished escape hatch must not bring a
    writable live store back to green."""
    _assert_refused(_collect({
        "HERMES_HOME": str(_live_store(tmp_path, "forensic")),
        "HERMES_ALLOW_UNHERMETIC": "yes-i-accept-the-risk",
    }))


# ---- conftest cannot defend itself (TARS R9, P0-RUN-5) -------------------
#
# TARS' counter-proof: with the guard living only in ``tests/conftest.py``,
# ``--noconftest`` collected and ran the suite against a writable fake live
# store with exit 0. The guard now also loads as a plugin through the ini
# ``addopts``, which none of these levers disables.


def test_noconftest_does_not_disable_the_tripwire(tmp_path):
    """``pytest --noconftest`` — TARS' exact reproduction, now refused."""
    _assert_refused(_collect(
        {"HERMES_HOME": str(_live_store(tmp_path, "noconftest"))},
        "--noconftest"))


def test_pytest_addopts_noconftest_does_not_disable_the_tripwire(tmp_path):
    """Same lever through the environment, where no command line shows it."""
    _assert_refused(_collect({
        "HERMES_HOME": str(_live_store(tmp_path, "addopts")),
        "PYTEST_ADDOPTS": "--noconftest",
    }))


def test_confcutdir_does_not_disable_the_tripwire(tmp_path):
    """Conftest discovery cut above the suite must not lose the guard."""
    _assert_refused(_collect(
        {"HERMES_HOME": str(_live_store(tmp_path, "confcut"))},
        f"--confcutdir={tmp_path}"))


def test_all_conftest_levers_combined_still_refuse(tmp_path):
    _assert_refused(_collect({
        "HERMES_HOME": str(_live_store(tmp_path, "combined")),
        "PYTEST_ADDOPTS": "--noconftest",
    }, "--noconftest", f"--confcutdir={tmp_path}"))


def test_noconftest_still_collects_when_nothing_live_is_exposed(tmp_path):
    """The guard must refuse the *store*, not the flag: no live root, no veto.

    Without this, ``--noconftest`` would look "fixed" simply because every
    run died — and the official runner path would die with it.
    """
    empty = tmp_path / "empty-sandbox"
    empty.mkdir()
    proc = _collect({"HERMES_HOME": str(empty)}, "--noconftest")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_plugin_unload_is_the_documented_residual_lever(tmp_path):
    """``-p no:…`` still disables it — pinned so the doc cannot drift.

    pytest lets the caller unload any plugin; no in-repo code can prevent
    that. ``tests/hermetic_guard.py`` says so explicitly, and this test is
    what keeps that statement true (or fails loudly if pytest ever changes).
    The kernel-enforced guarantee lives in the bwrap runner, not here.
    """
    proc = _collect(
        {"HERMES_HOME": str(_live_store(tmp_path, "unloaded"))},
        "--noconftest", "-p", "no:tests.hermetic_guard")
    assert proc.returncode == 0, (
        "pytest no longer honours -p no:… — the residual-lever documentation "
        "in tests/hermetic_guard.py needs revisiting")

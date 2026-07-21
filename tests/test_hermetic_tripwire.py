"""The conftest tripwire (_refuse_unhermetic_run) must never regress.

It is the last line of defence the 2026-07-20 incident demanded: a plain
``pytest`` invocation on a machine with a live, writable store has to die in
``pytest_configure`` — before a single test or fixture runs.

Hardened contract (2026-07-21 TARS review, P0): there is NO trusted
environment flag. ``HERMES_HERMETIC=1`` is dead — the tripwire only accepts
what it can mechanically observe: every live store non-writable for the test
process. The subprocess tests below prove all four directions end-to-end by
pointing ``HERMES_HERMETIC_PROTECT`` (additive-only) at a throwaway store,
which works identically inside and outside the runner's bwrap sandbox. The
unit tests pin the decision core, including that the passwd-derived real
home — not a redirected ``$HOME`` — is what the wired-up hook checks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import _hermetic_refusal, _tripwire_real_home

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TARGET = "tests/tools/test_store_guard.py"


def _fake_store(tmp_path: Path) -> Path:
    store = tmp_path / "fake-hermes"
    store.mkdir()
    (store / "state.db").touch()
    return store


def _collect(store: Path, env_extra: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           # Additive-only: our throwaway store is the ONLY extra root, so
           # the verdict on it is isolated from whatever this host has.
           "HERMES_HERMETIC_PROTECT": str(store)}
    env.pop("HERMES_HERMETIC", None)
    env.pop("HERMES_ALLOW_UNHERMETIC", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-p", "no:cacheprovider", _TARGET],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_tripwire_refuses_writable_live_store(tmp_path):
    proc = _collect(_fake_store(tmp_path), {})
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "ABGEBROCHEN" in proc.stdout + proc.stderr


def test_tripwire_ignores_spoofed_attestation(tmp_path):
    """The exact P0 bypass from the TARS review: the flag must be dead."""
    proc = _collect(_fake_store(tmp_path), {"HERMES_HERMETIC": "1"})
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "ABGEBROCHEN" in proc.stdout + proc.stderr


def test_tripwire_accepts_kernel_readonly_store(tmp_path):
    """A store the process cannot write is the ONLY accepted normal state —
    exactly what the runner's bwrap read-only bind produces."""
    store = _fake_store(tmp_path)
    store.chmod(0o555)
    try:
        proc = _collect(store, {})
        assert proc.returncode == 0, proc.stdout + proc.stderr
    finally:
        store.chmod(0o755)


def test_tripwire_accepts_explicit_override(tmp_path):
    proc = _collect(_fake_store(tmp_path),
                    {"HERMES_ALLOW_UNHERMETIC": "yes-i-accept-the-risk"})
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---- decision core, unit level -------------------------------------------


def test_refusal_on_writable_home_store(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "state.db").touch()
    reason = _hermetic_refusal(tmp_path, {})
    assert reason is not None and "ABGEBROCHEN" in reason


def test_flag_does_not_soften_the_verdict(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "state.db").touch()
    assert _hermetic_refusal(tmp_path, {"HERMES_HERMETIC": "1"}) is not None


def test_no_live_store_means_no_refusal(tmp_path):
    assert _hermetic_refusal(tmp_path, {}) is None


def test_protect_var_is_additive(tmp_path):
    """A clean home does not excuse a writable custom store."""
    custom = tmp_path / "custom-hermes"
    custom.mkdir()
    (custom / "state.db").touch()
    reason = _hermetic_refusal(
        tmp_path, {"HERMES_HERMETIC_PROTECT": str(custom)})
    assert reason is not None and str(custom) in reason


def test_real_home_comes_from_passwd_not_env(monkeypatch, tmp_path):
    """Redirecting $HOME must not move what the tripwire protects."""
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        import pwd
        expected = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        pytest.skip("no passwd database on this platform")
    assert _tripwire_real_home() == expected
    assert _tripwire_real_home() != tmp_path

"""The adapter-antipattern guard must survive an unwritable cache.

Background: the guard caches its scan verdict under a directory that used to
be hardcoded to ``Path.cwd()/.pytest-cache``. Under the mandated hermetic
runner the checkout is bound read-only, so ``mkdir`` failed and collection
died before the first gateway test — the whole suite was unrunnable on the
prescribed path.

The repair made the guard scan UNCACHED instead of returning early (returning
would have silently disabled the anti-pattern check entirely).

Why this test exists in-process: the first "verification" of that repair set
``HERMES_GATEWAY_TEST_CACHE`` from outside the runner — which strips every
inherited ``HERMES_*`` variable from the child environment, so the value never
reached the test process and the green run proved nothing (false green, TARS
re-review 2026-07-27). The state must be produced INSIDE the sandboxed process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_conftest_module():
    """Import tests/gateway/conftest.py as a standalone module."""
    path = Path(__file__).with_name("conftest.py")
    spec = importlib.util.spec_from_file_location("gateway_conftest_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConfig:
    """Minimal pytest-config stand-in (no workerinput → controller path)."""

    def __init__(self, basetemp=None):
        self._basetemp = basetemp

    def getoption(self, name, default=None):
        if name == "basetemp":
            return self._basetemp
        return default


def test_cache_dir_prefers_explicit_override(tmp_path, monkeypatch):
    mod = _load_conftest_module()
    monkeypatch.setenv("HERMES_GATEWAY_TEST_CACHE", str(tmp_path / "explicit"))
    assert mod._gateway_guard_cache_dir(_FakeConfig()) == tmp_path / "explicit"


def test_cache_dir_falls_back_to_basetemp(tmp_path, monkeypatch):
    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)
    got = mod._gateway_guard_cache_dir(_FakeConfig(basetemp=tmp_path))
    assert got == tmp_path / ".pytest-cache"


def test_unwritable_cache_still_runs_the_scan(tmp_path, monkeypatch):
    """The decisive one: mkdir fails → scan runs anyway, verdict unchanged."""
    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)

    real_mkdir = Path.mkdir

    def refusing_mkdir(self, *args, **kwargs):
        if self.name == ".pytest-cache":
            raise OSError(30, "Read-only file system")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refusing_mkdir)

    scanned = []
    monkeypatch.setattr(
        mod, "_run_adapter_antipattern_scan",
        lambda: scanned.append(True) or [],
    )
    mod.pytest_configure(_FakeConfig(basetemp=tmp_path))
    assert scanned == [True], "an unwritable cache must not disable the guard"


def test_unwritable_cache_still_reports_violations(tmp_path, monkeypatch):
    """And an uncached scan must still FAIL collection on a violation."""
    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)

    real_mkdir = Path.mkdir

    def refusing_mkdir(self, *args, **kwargs):
        if self.name == ".pytest-cache":
            raise OSError(30, "Read-only file system")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refusing_mkdir)
    monkeypatch.setattr(
        mod, "_run_adapter_antipattern_scan",
        lambda: ["tests/gateway/test_bogus.py: bare adapter import"],
    )
    with pytest.raises(pytest.UsageError, match="anti-pattern"):
        mod.pytest_configure(_FakeConfig(basetemp=tmp_path))


# ---------------------------------------------------------------------------
# Repair round 5 (2026-07-28): the fallback covered mkdir ONLY
# ---------------------------------------------------------------------------
#
# TARS re-review: everything else about the cache — taking the lock, reading a
# stale entry, writing the verdict — still propagated its OSError and killed
# collection. The cache is a speed-up; none of those failures says anything
# about the anti-pattern, so they must degrade to the uncached scan exactly
# like mkdir does.


def _fake_filelock_module(exc: Exception):
    """A ``filelock`` stand-in whose lock cannot be entered."""
    import types

    module = types.ModuleType("filelock")

    class _FailingLock:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            raise exc

        def __exit__(self, *a):
            return False

    module.FileLock = _FailingLock
    module.Timeout = RuntimeError
    return module


def test_lock_timeout_still_runs_the_scan(tmp_path, monkeypatch):
    """Lock contention/timeout must not fail collection — scan uncached."""
    import sys

    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)
    monkeypatch.setitem(
        sys.modules, "filelock", _fake_filelock_module(TimeoutError("lock busy")),
    )

    scanned = []
    monkeypatch.setattr(
        mod, "_run_adapter_antipattern_scan", lambda: scanned.append(True) or [],
    )
    mod.pytest_configure(_FakeConfig(basetemp=tmp_path))
    assert scanned == [True], "an unusable lock must not disable the guard"


def test_lock_timeout_still_reports_violations(tmp_path, monkeypatch):
    """And the degraded path must still FAIL collection on a violation."""
    import sys

    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)
    monkeypatch.setitem(
        sys.modules, "filelock", _fake_filelock_module(TimeoutError("lock busy")),
    )
    monkeypatch.setattr(
        mod, "_run_adapter_antipattern_scan",
        lambda: ["tests/gateway/test_bogus.py: bare adapter import"],
    )
    with pytest.raises(pytest.UsageError, match="anti-pattern"):
        mod.pytest_configure(_FakeConfig(basetemp=tmp_path))


def test_unreadable_cache_entry_still_runs_the_scan(tmp_path, monkeypatch):
    """A cache entry that exists but cannot be read → re-scan, don't die."""
    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)

    config = _FakeConfig(basetemp=tmp_path)
    cache_dir = mod._gateway_guard_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = mod._fingerprint_gateway_tests()
    (cache_dir / f"gw-adapter-guard-{fp}").write_text("clean", encoding="utf-8")

    real_read_text = Path.read_text

    def refusing_read_text(self, *args, **kwargs):
        if self.name.startswith("gw-adapter-guard-"):
            raise OSError(5, "Input/output error")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refusing_read_text)

    scanned = []
    monkeypatch.setattr(
        mod, "_run_adapter_antipattern_scan", lambda: scanned.append(True) or [],
    )
    mod.pytest_configure(config)
    assert scanned == [True], "an unreadable cache entry must trigger a re-scan"


def test_unwritable_cache_file_keeps_the_clean_verdict(tmp_path, monkeypatch):
    """Memoising may fail; the verdict it was memoising must still stand."""
    mod = _load_conftest_module()
    monkeypatch.delenv("HERMES_GATEWAY_TEST_CACHE", raising=False)

    real_write_text = Path.write_text

    def refusing_write_text(self, *args, **kwargs):
        if self.name.startswith("gw-adapter-guard-"):
            raise OSError(30, "Read-only file system")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refusing_write_text)

    scans = []
    monkeypatch.setattr(
        mod, "_run_adapter_antipattern_scan", lambda: scans.append(True) or [],
    )
    mod.pytest_configure(_FakeConfig(basetemp=tmp_path))
    assert len(scans) == 1, "a failed cache write must not re-run the scan"

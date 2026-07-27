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

"""O-1 + A2/A3 (2026-07-14): the dispatch freeze is a GLOBAL kill-switch enforced at
the single choke point dispatch_once(), so every caller honours it — and it reads the
ROOT config via a test-injectable seam so the operator's real freeze stays hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from gateway.kanban_watchers import _root_dispatch_frozen


def _set_root_cfg(tmp_path, monkeypatch, body: str | None):
    """Point the freeze seam at a controlled root config.yaml (or a missing file)."""
    root = tmp_path / "rootcfg"
    root.mkdir(exist_ok=True)
    cfg = root / "config.yaml"
    if body is not None:
        cfg.write_text(body, encoding="utf-8")
    monkeypatch.setattr(kb, "_ROOT_CONFIG_PATH_OVERRIDE", str(cfg), raising=False)
    return cfg


def test_root_freeze_true_when_root_config_disables(tmp_path, monkeypatch):
    _set_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: false\n")
    assert _root_dispatch_frozen() is True
    assert kb._root_dispatch_frozen() is True


def test_root_freeze_false_when_root_config_enables(tmp_path, monkeypatch):
    _set_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: true\n")
    assert _root_dispatch_frozen() is False


def test_root_freeze_false_when_no_kanban_block(tmp_path, monkeypatch):
    _set_root_cfg(tmp_path, monkeypatch, "model: x\n")
    assert _root_dispatch_frozen() is False


def test_root_freeze_false_when_config_missing(tmp_path, monkeypatch):
    _set_root_cfg(tmp_path, monkeypatch, None)  # no file -> fail open
    assert _root_dispatch_frozen() is False


def test_root_freeze_fails_open_on_broken_yaml(tmp_path, monkeypatch):
    _set_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: [unterminated\n")
    assert _root_dispatch_frozen() is False


# --- A2: dispatch_frozen() + dispatch_once() choke point ------------------------

def test_dispatch_frozen_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "off")
    assert kb.dispatch_frozen() is True


def test_dispatch_frozen_via_root(tmp_path, monkeypatch):
    _set_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: false\n")
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    assert kb.dispatch_frozen() is True


def test_dispatch_once_short_circuits_when_frozen(tmp_path, monkeypatch):
    # A frozen root config must make dispatch_once a no-op for ANY caller (CLI /
    # dashboard / daemon all funnel here) — returns skipped_frozen, does no DB writes.
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    _set_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: false\n")
    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    with kb.connect() as conn:
        # A ready task exists; a non-frozen dispatch would try to spawn it.
        kb.create_task(conn, title="ready-work", assignee="worker")
        spawned = []
        res = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: spawned.append(a) or 12345)
        assert res.skipped_frozen is True
        assert res.spawned == [] and spawned == []

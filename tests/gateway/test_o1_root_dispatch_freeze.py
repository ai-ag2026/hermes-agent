"""O-1 (2026-07-14): the dispatch freeze must be a GLOBAL kill-switch.

The per-gateway dispatch gate reads load_config() (profile-scoped). Setting
kanban.dispatch_in_gateway: false only in the ROOT config left a profile gateway
free to dispatch. _root_dispatch_frozen() reads the root config directly, independent
of HERMES_HOME/profile, so one root setting freezes every gateway lane.
"""

from __future__ import annotations

from pathlib import Path

from gateway.kanban_watchers import _root_dispatch_frozen


def _write_root_cfg(tmp_path, monkeypatch, body: str | None):
    home = tmp_path
    hermes = home / ".hermes"
    hermes.mkdir(parents=True, exist_ok=True)
    if body is not None:
        (hermes / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    # Even a profile-style HERMES_HOME must not change the result.
    monkeypatch.setenv("HERMES_HOME", str(hermes / "profiles" / "work"))


def test_root_freeze_true_when_root_config_disables(tmp_path, monkeypatch):
    _write_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: false\n")
    assert _root_dispatch_frozen() is True


def test_root_freeze_false_when_root_config_enables(tmp_path, monkeypatch):
    _write_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: true\n")
    assert _root_dispatch_frozen() is False


def test_root_freeze_false_when_no_kanban_block(tmp_path, monkeypatch):
    _write_root_cfg(tmp_path, monkeypatch, "model: x\n")
    assert _root_dispatch_frozen() is False


def test_root_freeze_fails_open_when_config_missing(tmp_path, monkeypatch):
    _write_root_cfg(tmp_path, monkeypatch, None)  # no config.yaml
    assert _root_dispatch_frozen() is False


def test_root_freeze_fails_open_on_broken_yaml(tmp_path, monkeypatch):
    _write_root_cfg(tmp_path, monkeypatch, "kanban:\n  dispatch_in_gateway: [unterminated\n")
    assert _root_dispatch_frozen() is False

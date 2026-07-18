"""Gate-notification umbau: ntfy is out of the kanban gate path by default,
and every gated card is subscribed to a durable operator channel so a gate
reaches the operator on ANY board (not only boards a browser cockpit watches).
"""
from __future__ import annotations

import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_gate_ops_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    module_names = [
        n for n in sys.modules
        if n.startswith("hermes_cli") or n.startswith("hermes_state") or n == "hermes_constants"
    ]
    original = {n: sys.modules[n] for n in module_names}
    for n in module_names:
        del sys.modules[n]
    from hermes_cli import kanban_db
    try:
        yield kanban_db
    finally:
        for n in [
            n for n in sys.modules
            if n.startswith("hermes_cli") or n.startswith("hermes_state") or n == "hermes_constants"
        ]:
            del sys.modules[n]
        sys.modules.update(original)


def _make_gated_card(kb, conn, title="Edit ~/.hermes/config.yaml gateway settings"):
    tid = kb.create_task(conn, title=title, assignee="backend-eng")
    kb.block_task(conn, tid, reason="self-modification gate: test", trusted_internal=True)
    kb.set_human_gate(conn, tid, on=True, actor="test")
    return tid


def test_ntfy_disabled_by_default(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    assert kb.gate_notify_ntfy_enabled() is False
    called = {"ntfy": False}
    monkeypatch.setattr(kb, "send_gate_token_ntfy",
                        lambda *a, **k: called.__setitem__("ntfy", True) or True)
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        # Token still issued (for the CLI --token path) but ntfy is NOT pushed.
        assert kb.issue_and_notify_gate_token(conn, tid) is True
    assert called["ntfy"] is False


def test_ntfy_still_works_when_reenabled(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setattr(kb, "_kanban_notify_config", lambda: {"gate_notify_ntfy": True})
    seen = {}
    monkeypatch.setattr(kb, "send_gate_token_ntfy",
                        lambda task_id, token, **k: seen.update(task_id=task_id) or True)
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        assert kb.issue_and_notify_gate_token(conn, tid) is True
    assert seen.get("task_id") == tid


def test_ops_channel_subscribes_gated_card(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL", "-1001234567890")
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        n = kb.ensure_ops_channel_gate_subs(conn, board="default")
        assert n == 1
        subs = kb.list_notify_subs(conn, task_id=tid)
        ops = [s for s in subs if s["platform"] == "telegram" and s["chat_id"] == "-1001234567890"]
        assert len(ops) == 1
        assert int(ops[0]["escalate_after_seconds"]) == 120  # WebUI first, ops escalates
        # idempotent: a second tick does not duplicate
        kb.ensure_ops_channel_gate_subs(conn, board="default")
        subs2 = kb.list_notify_subs(conn, task_id=tid)
        ops2 = [s for s in subs2 if s["platform"] == "telegram" and s["chat_id"] == "-1001234567890"]
        assert len(ops2) == 1


def test_ops_channel_noop_without_channel(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.delenv("TELEGRAM_OPS_CHANNEL", raising=False)
    monkeypatch.setattr(kb, "_kanban_notify_config", lambda: {})
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        assert kb.ensure_ops_channel_gate_subs(conn, board="default") == 0
        assert list(kb.list_notify_subs(conn, task_id=tid)) == []


def test_ops_channel_disabled_by_config(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL", "-1001234567890") if hasattr(monkeypatch, "setenv") else None
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL", "-1001234567890")
    monkeypatch.setattr(kb, "_kanban_notify_config",
                        lambda: {"gate_ops_channel_enabled": False})
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        assert kb.ensure_ops_channel_gate_subs(conn, board="default") == 0


def test_ops_channel_escalate_seconds_configurable(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setattr(kb, "_kanban_notify_config",
                        lambda: {"gate_ops_chat_id": "-1009999999999",
                                 "gate_ops_escalate_seconds": 0})
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        kb.ensure_ops_channel_gate_subs(conn, board="default")
        subs = kb.list_notify_subs(conn, task_id=tid)
        ops = [s for s in subs if s["chat_id"] == "-1009999999999"]
        assert len(ops) == 1
        assert int(ops[0]["escalate_after_seconds"]) == 0  # immediate ops delivery

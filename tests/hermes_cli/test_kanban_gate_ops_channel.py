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


# ── Weg 2: operator channels on every board (leak-free rework) ──

_OP_CH = [{"platform": "telegram", "chat_id": "-100home", "thread_id": "",
           "escalate_after_seconds": 120}]


def test_operator_channels_subscribed_on_other_board_gated_card(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setattr(kb, "_operator_gate_channels", lambda: list(_OP_CH))
    with kb.connect(board="tars-ops") as conn:
        tid = _make_gated_card(kb, conn)
        assert list(kb.list_notify_subs(conn, task_id=tid)) == []
        n = kb.mirror_default_board_subs_to_gated(conn, board="tars-ops")
        assert n == 1
        got = {(s["platform"], s["chat_id"]) for s in kb.list_notify_subs(conn, task_id=tid)}
        assert ("telegram", "-100home") in got
        # idempotent
        kb.mirror_default_board_subs_to_gated(conn, board="tars-ops")
        assert len(kb.list_notify_subs(conn, task_id=tid)) == 1


def test_mirror_does_not_leak_arbitrary_default_subs(isolated_kanban_home, monkeypatch):
    """Finding 1 regression guard: a random subscriber on an unrelated default
    task must NOT be propagated to another board's gated card."""
    kb = isolated_kanban_home
    with kb.connect(board="default") as dconn:
        base = kb.create_task(dconn, title="anchor", assignee="pm")
        kb.add_notify_sub(dconn, task_id=base, platform="telegram", chat_id="-100stranger")
    # no operator channels configured
    monkeypatch.setattr(kb, "_operator_gate_channels", lambda: [])
    with kb.connect(board="tars-ops") as conn:
        tid = _make_gated_card(kb, conn)
        assert kb.mirror_default_board_subs_to_gated(conn, board="tars-ops") == 0
        subs = kb.list_notify_subs(conn, task_id=tid)
        assert not any(s["chat_id"] == "-100stranger" for s in subs)


def test_mirror_is_noop_on_default_board(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setattr(kb, "_operator_gate_channels", lambda: list(_OP_CH))
    with kb.connect(board="default") as conn:
        tid = _make_gated_card(kb, conn)
        assert kb.mirror_default_board_subs_to_gated(conn, board="default") == 0


def test_mirror_disabled_by_config(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    monkeypatch.setattr(kb, "_operator_gate_channels", lambda: list(_OP_CH))
    monkeypatch.setattr(kb, "_kanban_notify_config",
                        lambda: {"inherit_default_board_subs": False})
    with kb.connect(board="tars-ops") as conn:
        tid = _make_gated_card(kb, conn)
        assert kb.mirror_default_board_subs_to_gated(conn, board="tars-ops") == 0
        assert list(kb.list_notify_subs(conn, task_id=tid)) == []


def test_gate_subs_respect_deliberate_unsubscribe(isolated_kanban_home, monkeypatch):
    """Finding 2 regression guard: a removed sub is NOT revived every tick."""
    kb = isolated_kanban_home
    monkeypatch.setattr(kb, "_operator_gate_channels", lambda: list(_OP_CH))
    with kb.connect(board="tars-ops") as conn:
        tid = _make_gated_card(kb, conn)
        kb.mirror_default_board_subs_to_gated(conn, board="tars-ops")
        # operator removes it
        kb.remove_notify_sub(conn, task_id=tid, platform="telegram", chat_id="-100home", thread_id="")
        active = [s for s in kb.list_notify_subs(conn, task_id=tid) if s["platform"] == "telegram"]
        assert active == []
        # next notifier tick must NOT resurrect it
        kb.mirror_default_board_subs_to_gated(conn, board="tars-ops")
        active2 = [s for s in kb.list_notify_subs(conn, task_id=tid) if s["platform"] == "telegram"]
        assert active2 == []


def test_string_false_config_flag_disables(isolated_kanban_home, monkeypatch):
    """Finding 5 regression guard: 'false' string must not read as truthy."""
    kb = isolated_kanban_home
    assert kb._cfg_flag({"x": "false"}, "x", True) is False
    assert kb._cfg_flag({"x": "true"}, "x", False) is True
    assert kb._cfg_flag({}, "x", True) is True

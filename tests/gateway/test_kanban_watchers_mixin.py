"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
from pathlib import Path

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_commit_delivery",
    "_kanban_unsub",
    "_kanban_release_lease",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


def test_issue_pending_gate_tokens_issues_and_pushes_for_gated_blocked_cards(
    tmp_path, monkeypatch,
):
    """Human-Gate v1's designated hook point: the notifier tick's per-board
    gate-token scan must issue + push a token for every blocked,
    human_gate=1 card missing one — independent of any chat subscription
    (a card blocked from a bare `kanban block --human-gate` CLI call, with
    nobody subscribed via a bot chat, must still get gated)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    pushed: list[tuple[str, str]] = []
    # Legacy ntfy delivery path — opt-in since gate_notify_ntfy defaulted off
    # (the durable ops channel replaced it); this test pins the legacy push.
    monkeypatch.setattr(kb, "gate_notify_ntfy_enabled", lambda: True)
    monkeypatch.setattr(
        kb, "send_gate_token_ntfy",
        lambda task_id, token, **kw: (pushed.append((task_id, token)) or True),
    )

    conn = kb.connect()
    try:
        gated = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, gated, reason="x", kind="needs_input", human_gate=True)
        ungated = kb.create_task(conn, title="plain", assignee="worker")
        kb.block_task(conn, ungated, reason="y", kind="needs_input")

        GatewayKanbanWatchersMixin._kanban_issue_pending_gate_tokens(
            None, conn, board="default",
        )

        assert kb.get_task(conn, gated).human_gate is True
        gated_row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (gated,)
        ).fetchone()
        assert gated_row["gate_token_hash"] is not None
        assert len(pushed) == 1 and pushed[0][0] == gated

        # Ungated card: never touched, no push.
        ungated_row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (ungated,)
        ).fetchone()
        assert ungated_row["gate_token_hash"] is None

        # A second scan is a no-op for the already-issued card — one push
        # total, not a duplicate on every tick.
        GatewayKanbanWatchersMixin._kanban_issue_pending_gate_tokens(
            None, conn, board="default",
        )
        assert len(pushed) == 1
    finally:
        conn.close()

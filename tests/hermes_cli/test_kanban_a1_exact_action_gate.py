"""A1 (2026-07-14 full audit): complete_task / schedule_task must not launder a card
with a live PENDING exact action out of `blocked` without going through approval.

The redesign kept ONE gate class — the exact-action approval ("short list":
upstream/prod/credentials/publish), enforced by the pending_action state machine.
unblock_task/promote_task refuse while it's live; complete_task/schedule_task did
NOT, so an unscoped orchestrator could flip a blocked+pending card to done/scheduled
and orphan the action — the 2026-07-13 incident's shape at a second door.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)  # unscoped = orchestrator
    kb.init_db()
    return home


def _card_with_pending_exact_action(conn):
    t = kb.create_task(conn, title="danger", assignee="worker")
    kb.claim_task(conn, t)
    rid = kb.get_task(conn, t).current_run_id
    kb.record_pending_action_and_block(
        conn, task_id=t, run_id=rid, command="git push origin main",
        summary="push to prod", profile="worker", workspace="/tmp",
        expires_at=int(time.time()) + 3600,
    )
    assert kb.get_task(conn, t).status == "blocked"
    return t


def test_complete_refused_while_exact_action_pending(kanban_home):
    with kb.connect() as conn:
        t = _card_with_pending_exact_action(conn)
        assert kb.complete_task(conn, t, result="approved, done") is False
        assert kb.get_task(conn, t).status == "blocked"
        # the pending action is NOT cancelled/orphaned
        row = conn.execute(
            "SELECT state FROM task_pending_actions WHERE task_id=? ORDER BY id DESC LIMIT 1", (t,)
        ).fetchone()
        assert row["state"] == "pending"


def test_schedule_refused_while_exact_action_pending(kanban_home):
    with kb.connect() as conn:
        t = _card_with_pending_exact_action(conn)
        assert kb.schedule_task(conn, t) is False
        assert kb.get_task(conn, t).status == "blocked"


def test_complete_allowed_after_approval_resolves_action(kanban_home):
    with kb.connect() as conn:
        t = _card_with_pending_exact_action(conn)
        a = kb.get_current_attention(conn, t)
        kb.approve_pending_action_and_unblock_versioned(
            conn, t, a.action_id, expected_version=a.version, actor="op"
        )
        # No pending row remains -> the card is out of the exact-action hold.
        pend = conn.execute(
            "SELECT 1 FROM task_pending_actions WHERE task_id=? AND state='pending' LIMIT 1", (t,)
        ).fetchone()
        assert pend is None
        assert kb.get_task(conn, t).status in ("ready", "todo")


def test_normal_card_completes_freely(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="ok", assignee="worker")
        kb.claim_task(conn, t)
        assert kb.complete_task(conn, t, result="done") is True
        assert kb.get_task(conn, t).status == "done"

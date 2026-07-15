"""WS2 (2026-07-15): reopen_task -- the missing way back out of a terminal state.

Field failure: card t_bccbacc4 was auto-completed by a rerun against the explicit
advice of its own analyst. Getting it back was impossible through any verb:
block_task takes only running/ready, reclaim_task refuses anything not running,
unarchive_task starts from `archived`. Both the operator and, independently, a
later agent hit the same wall -- the agent's own words:

    could not block t_bccbacc4 (unknown id or not in running/ready)

The only way back was a raw status write, which skips the `reopened` event, the
completed_at/result cleanup and the child demotion.

NOTE: HERMES_HOME must stay off ~/.hermes -- a board rooted there writes into
the live kanban DB (get_default_hermes_root mapping).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    assert str(kb.kanban_db_path()).startswith(str(tmp_path)), "must not touch the live board"
    return home


def _completed_card(conn, *, title="audit", result="fertig"):
    task_id = kb.create_task(conn, title=title, assignee="analyst")
    kb.claim_task(conn, task_id)
    assert kb.complete_task(conn, task_id, result=result)
    assert kb.get_task(conn, task_id).status == "done"
    return task_id


def test_reopen_brings_a_done_card_back_to_blocked(kanban_home):
    with kb.connect() as conn:
        task_id = _completed_card(conn)

        assert kb.reopen_task(conn, task_id, actor="operator", reason="Auto-Completion zurückgenommen")

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.completed_at is None, "a reopened card must not still claim a completion time"
        assert task.result is None
        assert task.current_run_id is None


def test_reopened_card_is_not_handed_to_the_dispatcher(kanban_home):
    """The whole point of defaulting to `blocked`.

    recompute_ready promotes blocked cards too and only spares gated /
    attention-bearing / STICKY ones. A reopened card's last blocked-vs-unblocked
    event is whatever preceded its completion, so without a real block event the
    next readiness pass would promote it to `ready` and the dispatcher would
    re-run the very work the operator just took back -- which is exactly what
    happened on 2026-07-15.
    """
    with kb.connect() as conn:
        task_id = _completed_card(conn)
        # Give it the history of a real card: blocked once, then released.
        kb.reopen_task(conn, task_id, actor="operator", reason="zurückgeholt")
        assert kb.get_task(conn, task_id).status == "blocked"

        kb.recompute_ready(conn)
        kb.recompute_ready(conn)

        assert kb.get_task(conn, task_id).status == "blocked", (
            "a reopened card must never auto-promote to ready"
        )


def test_reopen_after_an_unblock_history_still_sticks(kanban_home):
    """The regression the sticky predicate makes subtle.

    A card that was blocked and then unblocked before completing has `unblocked`
    as its most recent block-ish event. That is the common shape (it is exactly
    t_bccbacc4's), and it is the one where a naive reopen silently un-sticks.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="wie t_bccbacc4", assignee="analyst")
        kb.claim_task(conn, task_id)
        kb.block_task(conn, task_id, kind="needs_input", reason="braucht Menschen")
        assert kb.unblock_task(conn, task_id, actor="operator")
        kb.claim_task(conn, task_id)
        assert kb.complete_task(conn, task_id, result="fertig")

        assert kb.reopen_task(conn, task_id, actor="operator", reason="doch nicht fertig")
        kb.recompute_ready(conn)

        assert kb.get_task(conn, task_id).status == "blocked"


def test_reopen_preserves_the_old_result_as_a_comment(kanban_home):
    """The result is evidence -- clearing it must not destroy it."""
    with kb.connect() as conn:
        task_id = _completed_card(conn, result="Report liegt in REPORT.md, 21kB, Readback PASS")

        kb.reopen_task(conn, task_id, actor="operator", reason="DoD-Abweichung")

        assert kb.get_task(conn, task_id).result is None
        bodies = [r[0] for r in conn.execute(
            "SELECT body FROM task_comments WHERE task_id=? ORDER BY id", (task_id,)
        ).fetchall()]
        assert any("REPORT.md, 21kB, Readback PASS" in b for b in bodies)


def test_reopen_emits_an_attributable_event(kanban_home):
    with kb.connect() as conn:
        task_id = _completed_card(conn)

        kb.reopen_task(conn, task_id, actor="manfred", reason="Analyst riet ab")

        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='reopened'", (task_id,)
        ).fetchone()
        assert row is not None, "a reopen must be attributable after the fact"
        import json
        payload = json.loads(row[0])
        assert payload["actor"] == "manfred"
        assert payload["reason"] == "Analyst riet ab"
        assert payload["previous_status"] == "done"
        assert payload["status"] == "blocked"


def test_reopening_a_parent_demotes_its_ready_children(kanban_home):
    """recompute_ready can only promote, never demote.

    Without this a child stays `ready` -- dispatchable -- on a dependency that is
    no longer satisfied. The logic existed only in the dashboard, so a reopen
    through any other door left the board lying.
    """
    with kb.connect() as conn:
        parent = _completed_card(conn, title="parent")
        child = kb.create_task(conn, title="child", assignee="w")
        kb.link_tasks(conn, parent, child)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready", "precondition: child is ready"

        kb.reopen_task(conn, parent, actor="operator", reason="parent doch nicht fertig")

        assert kb.get_task(conn, child).status == "todo", "child must not stay dispatchable"
        assert kb.get_task(conn, parent).status == "blocked"


def test_reopening_a_parent_does_not_touch_unrelated_ready_cards(kanban_home):
    """Behavioural neutrality: demotion follows links, not the whole board."""
    with kb.connect() as conn:
        parent = _completed_card(conn, title="parent")
        stranger = kb.create_task(conn, title="stranger", assignee="w")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, stranger).status == "ready"

        kb.reopen_task(conn, parent, actor="operator", reason="x")

        assert kb.get_task(conn, stranger).status == "ready"


def test_reopen_to_todo_is_allowed_and_requeues(kanban_home):
    """`todo` is the explicit "put it back in the queue" choice."""
    with kb.connect() as conn:
        task_id = _completed_card(conn)

        assert kb.reopen_task(conn, task_id, actor="operator", reason="nochmal", to_status="todo")
        kb.recompute_ready(conn)

        # No parents -> the normal readiness pass promotes it, as `todo` implies.
        assert kb.get_task(conn, task_id).status == "ready"


def test_reopen_rejects_a_ready_target(kanban_home):
    """`ready` would hand the card straight to the dispatcher -- not offered."""
    with kb.connect() as conn:
        task_id = _completed_card(conn)
        with pytest.raises(ValueError):
            kb.reopen_task(conn, task_id, actor="operator", reason="x", to_status="ready")
        assert kb.get_task(conn, task_id).status == "done"


def test_reopen_only_applies_to_terminal_cards(kanban_home):
    """A running or blocked card is not this verb's business."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="laeuft", assignee="w")
        kb.claim_task(conn, task_id)
        assert kb.reopen_task(conn, task_id, actor="operator", reason="x") is False
        assert kb.get_task(conn, task_id).status == "running"

        assert kb.reopen_task(conn, "does-not-exist", actor="operator", reason="x") is False


def test_reopen_works_from_archived(kanban_home):
    with kb.connect() as conn:
        task_id = _completed_card(conn)
        assert kb.archive_task(conn, task_id)

        assert kb.reopen_task(conn, task_id, actor="operator", reason="doch gebraucht")

        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.completed_at is None

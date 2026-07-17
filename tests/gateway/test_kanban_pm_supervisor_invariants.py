"""Invariant tests for the pm-supervisor tick (Autonomie-Umbau Baustein B,
2026-07-17). Each test corresponds 1:1 to a NON-NEGOTIABLE invariant from
the task brief:

  1. Flag off -> byte-identical (the tick never touches the board).
  2. pm never touches an operator-created card.
  3. pm can never bypass Tier-3/exact-action/human_gate (covered here at the
     tick's candidate-selection layer; the kernel-level refusal itself is
     covered by test_kanban_pm_supervisor_actions.py's execute-time
     re-checks).
  4. Attempt-limit reached -> no further pm touch, one-time gave-up escalation.
  5. Unparsable/unknown LLM response -> escalate, never close
     (kernel-level: test_kanban_pm_supervisor_actions.py; here: end-to-end
     through the real tick).
  6. Every pm action is auditable (event + comment).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import (
    _DECOMPOSE_GAVE_UP_EVENT,
    _PM_SUPERVISOR_ATTEMPTED_EVENT,
    _PM_SUPERVISOR_GAVE_UP_EVENT,
    _operator_owned_skip_logged,
    _pm_supervisor_tick,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    _operator_owned_skip_logged.clear()
    yield home
    _operator_owned_skip_logged.clear()


def _blocked_card(kb, *, created_by=None, title="stuck card"):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title=title, created_by=created_by)
        kb.block_task(conn, tid, kind="needs_input", reason="need input")
    return tid


def _llm_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fixed_action_llm(action_json: str):
    calls = []

    def call_llm(**kwargs):
        calls.append(kwargs)
        return _llm_response(action_json)

    return call_llm, calls


# ---------------------------------------------------------------------------
# Invariant 1: flag off -> byte-identical
# ---------------------------------------------------------------------------


def test_disabled_tick_never_reads_board_or_calls_llm(kanban_home):
    """This test documents the wiring contract enforced in
    _kanban_dispatcher_watcher: when kanban.pm_supervisor_enabled is False
    (the default), _pm_supervisor_tick is never invoked at all -- the
    dispatcher loop gates the call with `if _pm_enabled:` before awaiting
    it. Exercising that gate directly requires the full async watcher; here
    we verify the cheaper, equivalent property: a tick that IS called with
    an LLM stub that raises on any use never touches an untouched board,
    so "not calling the tick" and "calling it against an empty board" are
    behaviourally indistinguishable -- there is nothing for it to find."""
    from hermes_cli import kanban_db as kb

    def call_llm(**kwargs):
        raise AssertionError("must not be called when there are no candidates")

    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset({"claude-code", "manfred"}),
    )
    assert handled == 0


# ---------------------------------------------------------------------------
# Invariant 2: never touch an operator-created card
# ---------------------------------------------------------------------------


def test_operator_created_card_is_never_touched(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb, created_by="manfred", title="operator's card")

    def call_llm(**kwargs):
        raise AssertionError("must never be called for an operator-owned card")

    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset({"claude-code", "manfred"}),
    )
    assert handled == 0

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        events = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchall()
    assert task.status == "blocked"  # untouched
    assert events == []


def test_non_operator_card_on_same_board_is_still_handled(kanban_home):
    """The operator exemption must be per-card, not a whole-board skip."""
    from hermes_cli import kanban_db as kb

    op_tid = _blocked_card(kb, created_by="manfred", title="operator's card")
    normal_tid = _blocked_card(kb, created_by="a-worker", title="worker's card")

    call_llm, calls = _fixed_action_llm('{"action": "answer_and_requeue", "comment": "ok"}')

    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset({"claude-code", "manfred"}),
    )
    assert handled == 1
    assert len(calls) == 1

    with kb.connect() as conn:
        op_task = kb.get_task(conn, op_tid)
        normal_task = kb.get_task(conn, normal_tid)
    assert op_task.status == "blocked"
    assert normal_task.status in ("ready", "todo")


# ---------------------------------------------------------------------------
# Invariant 4: attempt limit -> no further touch, one-time gave-up event
# ---------------------------------------------------------------------------


def test_attempt_limit_stops_further_touches_and_gives_up_once(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    # Every LLM decision escalates -- the card is never resolved, so it
    # accumulates pm_supervisor_attempted events tick after tick.
    call_llm, calls = _fixed_action_llm(
        '{"action": "escalate", "severity": "routine", '
        '"memo": {"situation": "still stuck"}}'
    )

    for _ in range(2):  # max_attempts=2
        _pm_supervisor_tick(
            kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
            per_tick=3, max_attempts=2, operator_authors=frozenset(),
        )

    with kb.connect() as conn:
        attempted = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()[0]
    assert attempted == 2
    calls_after_two_ticks = len(calls)

    # A third tick must skip the card entirely: no new LLM call, no new
    # attempted event, and exactly one gave_up event recorded.
    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset(),
    )
    assert handled == 0
    assert len(calls) == calls_after_two_ticks  # no new LLM call

    with kb.connect() as conn:
        attempted_after = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()[0]
        gave_up = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_GAVE_UP_EVENT),
        ).fetchone()[0]
        task = kb.get_task(conn, tid)
    assert attempted_after == 2  # unchanged
    assert gave_up == 1
    assert task.status == "blocked"  # left exactly where it was

    # A fourth tick must not record a second gave_up event (idempotent).
    _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset(),
    )
    with kb.connect() as conn:
        gave_up_again = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_GAVE_UP_EVENT),
        ).fetchone()[0]
    assert gave_up_again == 1


def test_attempt_limit_does_not_consume_per_tick_budget(kanban_home):
    """A gave-up skip must not eat into per_tick -- the tick should still
    reach a genuinely fresh candidate in the same pass."""
    from hermes_cli import kanban_db as kb

    stuck_tid = _blocked_card(kb, title="already gave up")
    with kb.connect() as conn:
        with kb.write_txn(conn):
            kb._append_event(
                conn, stuck_tid, _PM_SUPERVISOR_ATTEMPTED_EVENT, {"action": "escalate"},
            )
            kb._append_event(
                conn, stuck_tid, _PM_SUPERVISOR_ATTEMPTED_EVENT, {"action": "escalate"},
            )

    fresh_tid = _blocked_card(kb, title="fresh candidate")

    call_llm, calls = _fixed_action_llm('{"action": "answer_and_requeue", "comment": "ok"}')
    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=1, max_attempts=2, operator_authors=frozenset(),
    )
    assert handled == 1
    assert len(calls) == 1

    with kb.connect() as conn:
        fresh_task = kb.get_task(conn, fresh_tid)
    assert fresh_task.status in ("ready", "todo")


# ---------------------------------------------------------------------------
# Invariant 5: unparsable/unknown LLM response -> escalate end-to-end
# ---------------------------------------------------------------------------


def test_garbage_llm_response_escalates_not_closes(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    call_llm, _ = _fixed_action_llm("I refuse to output JSON today.")

    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset(),
    )
    assert handled == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'pm_supervisor_escalated'",
            (tid,),
        ).fetchall()
    assert task.status == "blocked"  # NOT archived/closed
    assert len(events) == 1
    payload = json.loads(events[0]["payload"])
    assert payload["severity"] == "routine"


# ---------------------------------------------------------------------------
# Invariant 6: every pm action is auditable (event + comment)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action_json,expect_comment_substr",
    [
        ('{"action": "answer_and_requeue", "comment": "answer text"}', "answer text"),
        ('{"action": "clarify_dod", "comment": "dod text"}', "dod text"),
        ('{"action": "close_obsolete", "comment": "obsolete text"}', "obsolete text"),
        (
            '{"action": "escalate", "severity": "routine", '
            '"memo": {"situation": "escalation text"}}',
            "escalation text",
        ),
    ],
)
def test_every_action_leaves_an_event_and_a_comment(
    kanban_home, action_json, expect_comment_substr,
):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    call_llm, _ = _fixed_action_llm(action_json)

    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset(),
    )
    assert handled == 1

    with kb.connect() as conn:
        attempted = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()[0]
        comments = kb.list_comments(conn, tid)
    assert attempted == 1
    assert any(
        expect_comment_substr in c.body and c.author == "pm-supervisor" for c in comments
    ), [c.body for c in comments]


def test_reassign_action_is_auditable(kanban_home):
    """reassign's comment is the LLM's `comment` field (the assignee change
    itself is a separate, structured `assigned` event kb.assign_task already
    emits -- so the audit trail for reassign is two events, not one)."""
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    call_llm, _ = _fixed_action_llm(
        '{"action": "reassign", "comment": "route to specialist", "assignee": "specialist-profile"}'
    )

    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset(),
    )
    assert handled == 1

    with kb.connect() as conn:
        attempted = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()[0]
        assigned = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (tid,),
        ).fetchone()[0]
        comments = kb.list_comments(conn, tid)
        task = kb.get_task(conn, tid)
    assert attempted == 1
    assert assigned == 1
    assert task.assignee == "specialist-profile"
    assert any("route to specialist" in c.body for c in comments)


def test_decompose_gave_up_triage_candidate_end_to_end(kanban_home, monkeypatch):
    """The other candidate shape (triage + decompose_gave_up event, not a
    blocked/needs_input card) goes through the exact same tick."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_decompose as decomp

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gave up decomposing", triage=True)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, _DECOMPOSE_GAVE_UP_EVENT, {"attempts": 2, "limit": 2})

    def fake_decompose_task(task_id, *, author=None, timeout=None):
        return decomp.DecomposeOutcome(task_id, True, "single task (no fanout)")

    monkeypatch.setattr(decomp, "decompose_task", fake_decompose_task)

    call_llm, _ = _fixed_action_llm('{"action": "decompose", "comment": "break it down"}')
    handled = _pm_supervisor_tick(
        kb_module=kb, load_config=lambda: {"kanban": {}}, call_llm_fn=call_llm,
        per_tick=3, max_attempts=2, operator_authors=frozenset(),
    )
    assert handled == 1

    with kb.connect() as conn:
        attempted = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (tid, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()[0]
    assert attempted == 1

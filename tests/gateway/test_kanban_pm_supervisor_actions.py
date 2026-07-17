"""Tests for pm-supervisor decision parsing and action execution
(Autonomie-Umbau Baustein B, 2026-07-17).

``_decide_pm_action`` is the LLM boundary: it must degrade to
``escalate``/``routine`` on every failure mode (API error, malformed JSON,
unknown action) and never let a bad response silently do nothing or close a
card (invariant 5).

``_execute_pm_decision`` is the trusted execution boundary: the LLM only
ever selects an action + supplies text; this function is what actually
touches ``kb.*``. Each bound action is tested against a REAL test DB (no
mocking of the kb layer) so the assertions are about actual board state,
not about what we told a mock to return.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import (
    PmSupervisorDecision,
    _decide_pm_action,
    _execute_pm_decision,
    _pm_extract_json_blob,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    yield home


def _blocked_card(kb, *, title="stuck card"):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title=title)
        kb.block_task(conn, tid, kind="needs_input", reason="need input")
    return tid


def _triage_gave_up_card(kb, *, title="gave up decomposing"):
    from gateway.kanban_watchers import _DECOMPOSE_GAVE_UP_EVENT

    with kb.connect() as conn:
        tid = kb.create_task(conn, title=title, triage=True)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, _DECOMPOSE_GAVE_UP_EVENT, {"attempts": 2, "limit": 2})
    return tid


def _llm_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# ---------------------------------------------------------------------------
# _pm_extract_json_blob
# ---------------------------------------------------------------------------


def test_extract_plain_json():
    assert _pm_extract_json_blob('{"action": "escalate"}') == {"action": "escalate"}


def test_extract_fenced_json():
    raw = '```json\n{"action": "escalate"}\n```'
    assert _pm_extract_json_blob(raw) == {"action": "escalate"}


def test_extract_empty_or_garbage_returns_none():
    assert _pm_extract_json_blob("") is None
    assert _pm_extract_json_blob("not json at all") is None
    assert _pm_extract_json_blob("[1, 2, 3]") is None  # not a dict


# ---------------------------------------------------------------------------
# _decide_pm_action -- fail-safe degradation (invariant 5)
# ---------------------------------------------------------------------------


def test_decide_returns_valid_action_on_clean_response():
    def call_llm(**kwargs):
        return _llm_response('{"action": "answer_and_requeue", "comment": "here is the answer"}')

    decision = _decide_pm_action(
        call_llm, task=SimpleNamespace(id="t_x", title="t", body="b", status="blocked",
                                        block_kind="needs_input", assignee="p1"),
        comments=[], profiles_roster=[], default_assignee="p1",
    )
    assert decision.action == "answer_and_requeue"
    assert decision.comment == "here is the answer"


def test_decide_llm_call_raises_degrades_to_escalate_routine():
    def call_llm(**kwargs):
        raise RuntimeError("quota exceeded")

    decision = _decide_pm_action(
        call_llm, task=SimpleNamespace(id="t_x", title="t", body="b", status="blocked",
                                        block_kind="needs_input", assignee=None),
        comments=[], profiles_roster=[], default_assignee="",
    )
    assert decision.action == "escalate"
    assert decision.severity == "routine"
    assert decision.memo is not None


def test_decide_malformed_json_degrades_to_escalate_routine():
    def call_llm(**kwargs):
        return _llm_response("I cannot help with that, sorry.")

    decision = _decide_pm_action(
        call_llm, task=SimpleNamespace(id="t_x", title="t", body="b", status="blocked",
                                        block_kind="needs_input", assignee=None),
        comments=[], profiles_roster=[], default_assignee="",
    )
    assert decision.action == "escalate"
    assert decision.severity == "routine"


def test_decide_unknown_action_degrades_to_escalate_routine():
    def call_llm(**kwargs):
        return _llm_response('{"action": "delete_everything"}')

    decision = _decide_pm_action(
        call_llm, task=SimpleNamespace(id="t_x", title="t", body="b", status="blocked",
                                        block_kind="needs_input", assignee=None),
        comments=[], profiles_roster=[], default_assignee="",
    )
    assert decision.action == "escalate"
    assert decision.severity == "routine"


def test_decide_severity_only_accepts_critical_or_routine():
    def call_llm(**kwargs):
        return _llm_response(
            '{"action": "escalate", "severity": "URGENT!!", '
            '"memo": {"situation": "x"}}'
        )

    decision = _decide_pm_action(
        call_llm, task=SimpleNamespace(id="t_x", title="t", body="b", status="blocked",
                                        block_kind="needs_input", assignee=None),
        comments=[], profiles_roster=[], default_assignee="",
    )
    assert decision.severity == "routine"


def test_decide_critical_severity_is_preserved():
    def call_llm(**kwargs):
        return _llm_response(
            '{"action": "escalate", "severity": "critical", '
            '"memo": {"situation": "prod is down"}}'
        )

    decision = _decide_pm_action(
        call_llm, task=SimpleNamespace(id="t_x", title="t", body="b", status="blocked",
                                        block_kind="needs_input", assignee=None),
        comments=[], profiles_roster=[], default_assignee="",
    )
    assert decision.severity == "critical"


# ---------------------------------------------------------------------------
# _execute_pm_decision -- answer_and_requeue / clarify_dod
# ---------------------------------------------------------------------------


def test_answer_and_requeue_comments_and_requeues_blocked_card(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    decision = PmSupervisorDecision(action="answer_and_requeue", comment="the answer is 42")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome == "answer_and_requeue"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert task.status in ("ready", "todo")
    assert any(c.body == "the answer is 42" and c.author == "pm-supervisor" for c in comments)


def test_clarify_dod_comments_and_requeues_triage_gave_up_card(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _triage_gave_up_card(kb)
    decision = PmSupervisorDecision(action="clarify_dod", comment="DoD: tests green + doc updated")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome == "clarify_dod"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status in ("ready", "todo")


# ---------------------------------------------------------------------------
# _execute_pm_decision -- reassign
# ---------------------------------------------------------------------------


def test_reassign_sets_new_assignee_and_requeues(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    decision = PmSupervisorDecision(
        action="reassign", comment="better fit", assignee="other-profile",
    )
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome == "reassign"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.assignee == "other-profile"
    assert task.status in ("ready", "todo")


def test_reassign_without_assignee_fails_safe_no_mutation(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    decision = PmSupervisorDecision(action="reassign", comment="better fit", assignee=None)
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is False
    assert outcome == "reassign_missing_assignee"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    # Card stays blocked -- an incomplete reassign decision must not leave
    # the card in limbo (requeued but still assigned to whoever gave up).
    assert task.status == "blocked"


def test_reassign_to_human_driven_profile_refused(kanban_home):
    """The trusted boundary must never route a card to a human-driven profile
    (default/work): the dispatcher would refuse to spawn it, silently parking
    the card in 'ready' with no attention signal -- worse than the block."""
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    # 'work' is human-driven by the built-in default set.
    decision = PmSupervisorDecision(action="reassign", comment="hand to work", assignee="work")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is False
    assert outcome == "reassign_to_human_driven"
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"  # untouched, not stranded in ready


# ---------------------------------------------------------------------------
# _execute_pm_decision -- decompose
# ---------------------------------------------------------------------------


def test_decompose_on_triage_card_invokes_decomposer(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_decompose as decomp

    tid = _triage_gave_up_card(kb)

    called = {}

    def fake_decompose_task(task_id, *, author=None, timeout=None):
        called["task_id"] = task_id
        called["author"] = author
        return decomp.DecomposeOutcome(task_id, True, "single task (no fanout)")

    monkeypatch.setattr(decomp, "decompose_task", fake_decompose_task)

    decision = PmSupervisorDecision(action="decompose", comment="break it down")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome == "decompose"
    assert called == {"task_id": tid, "author": "pm-supervisor"}


def test_decompose_on_blocked_card_fails_safe_to_escalate(kanban_home):
    """decompose_task only accepts triage-status cards. A blocked candidate
    that the LLM chose 'decompose' for is a structurally invalid decision
    -- must fail safe to escalate, not silently no-op or crash."""
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    decision = PmSupervisorDecision(action="decompose", comment="break it down")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome.startswith("escalate:")

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"  # untouched, per the escalate contract


# ---------------------------------------------------------------------------
# _execute_pm_decision -- close_obsolete
# ---------------------------------------------------------------------------


def test_close_obsolete_archives_the_card(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    decision = PmSupervisorDecision(action="close_obsolete", comment="superseded by t_other")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome == "close_obsolete"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        comments = kb.list_comments(conn, tid)
    assert task.status == "archived"
    assert any("superseded by t_other" in c.body for c in comments)


# ---------------------------------------------------------------------------
# _execute_pm_decision -- escalate
# ---------------------------------------------------------------------------


def test_escalate_leaves_status_untouched_and_records_memo(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    memo = {
        "situation": "the worker asked something outside its scope",
        "options": ["reassign to X", "close as won't-fix"],
        "recommendation": "reassign to X",
        "cost_of_ignoring": "task stays stuck",
    }
    decision = PmSupervisorDecision(action="escalate", memo=memo, severity="critical")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is True
    assert outcome == "escalate:critical"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'pm_supervisor_escalated'",
            (tid,),
        ).fetchall()
    assert task.status == "blocked"  # untouched
    assert len(events) == 1
    import json as _json
    payload = _json.loads(events[0]["payload"])
    assert payload["severity"] == "critical"
    assert payload["pm_memo"]["recommendation"] == "reassign to X"


def test_escalate_comment_is_deduplicated_across_repeated_passes(kanban_home):
    """A card stuck across multiple pm-supervisor attempts with a similar
    memo each time must not accumulate duplicate identical comments."""
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    decision = PmSupervisorDecision(
        action="escalate",
        memo={"situation": "same situation every time", "recommendation": ""},
        severity="routine",
    )
    _execute_pm_decision(kb, tid, decision, board_slug="default")
    _execute_pm_decision(kb, tid, decision, board_slug="default")

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
    matching = [c for c in comments if "same situation every time" in c.body]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# _execute_pm_decision -- defense-in-depth re-checks at execution time
# ---------------------------------------------------------------------------


def test_execute_refuses_when_card_no_longer_a_candidate_status(kanban_home):
    """The LLM decision round-trip takes real time; if a human already
    moved the card out of blocked/triage before execution, the action must
    be refused, not forced through."""
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    with kb.connect() as conn:
        kb.unblock_task(conn, tid, actor="a-human")  # card is 'ready' now

    decision = PmSupervisorDecision(action="answer_and_requeue", comment="too late")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is False
    assert outcome == "status_changed"


def test_close_obsolete_refuses_when_pending_exact_action_present(kanban_home):
    """kb.archive_task itself does NOT check task_pending_actions (it only
    guards human_gate) -- so close_obsolete's safety against a live Tier-3
    approval comes entirely from _execute_pm_decision's own
    get_pending_action() re-check, not from the kernel call it makes.
    This test pins that guarantee directly against the real DB rather than
    relying on candidate-selection having already filtered the card out."""
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    with kb.connect() as conn:
        conn.execute(
            "INSERT INTO task_pending_actions "
            "(task_id, run_id, command_hash, fingerprint, mutation_kind, summary, "
            " profile, workspace, created_at, state, expires_at, version, updated_at) "
            "VALUES (?, NULL, 'h', 'f', 'shell', 's', 'p', 'w', 0, 'pending', 9999999999, 1, 0)",
            (tid,),
        )
        conn.commit()

    decision = PmSupervisorDecision(action="close_obsolete", comment="obsolete")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is False
    assert outcome == "pending_action_appeared"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"  # NOT archived


def test_execute_refuses_when_human_gate_appeared(kanban_home):
    from hermes_cli import kanban_db as kb

    tid = _blocked_card(kb)
    with kb.connect() as conn:
        kb.set_human_gate(conn, tid, on=True)

    decision = PmSupervisorDecision(action="close_obsolete", comment="obsolete")
    ok, outcome = _execute_pm_decision(kb, tid, decision, board_slug="default")
    assert ok is False
    assert outcome == "human_gate_set"

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"  # still there, un-archived

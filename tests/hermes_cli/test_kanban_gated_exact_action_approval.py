"""2026-07-15: exact-action approval on a HUMAN-GATED card.

Field failure: card `Audit 2026-07-15 A` (human_gate=1) hit an exact terminal
action and became unreleasable. Approving an exact action also unblocks the
card, so the Core enforces the gate on it; every caller that passed no token was
refused, and the refusal was reported as `conflict` -- indistinguishable from a
lost CAS race. The operator saw "conflicted with newer state" for a race that
never happened, while the card's only other exit (unblock) refuses precisely
because an exact action is pending.

No test combined `human_gate=1` with exact-action approval before this file:
test_kanban_a1_exact_action_gate.py and test_kanban_action_approval.py only ever
build ungated cards, which is why the deadlock shipped.

NOTE: HERMES_HOME must stay off ~/.hermes -- a board rooted there writes into
the live kanban DB (get_default_hermes_root mapping).
"""

from __future__ import annotations

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
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    assert str(kb.kanban_db_path()).startswith(str(tmp_path)), "must not touch the live board"
    return home


def _gated_card_with_pending_exact_action(conn, *, gated: bool = True):
    task_id = kb.create_task(conn, title="danger", assignee="worker")
    kb.claim_task(conn, task_id)
    run_id = kb.get_task(conn, task_id).current_run_id
    kb.record_pending_action_and_block(
        conn, task_id=task_id, run_id=run_id, command="git push origin main",
        summary="push to prod", profile="worker", workspace="/tmp",
        expires_at=int(time.time()) + 3600,
    )
    if gated:
        kb.set_human_gate(conn, task_id, on=True, actor="test")
    assert kb.get_task(conn, task_id).status == "blocked"
    action = kb.get_current_attention(conn, task_id)
    return task_id, action


def test_gated_approval_without_token_is_gate_refused_not_conflict(kanban_home):
    """The core regression: a gate refusal must not masquerade as a lost race.

    'conflict' means "someone else moved first, refresh and retry"; a gate
    refusal means "you hold no token, retrying is futile". Reporting the latter
    as the former is what made the live incident undiagnosable.
    """
    with kb.connect() as conn:
        task_id, att = _gated_card_with_pending_exact_action(conn)

        result = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, att.action_id, expected_version=att.version, actor="op", token=None,
        )

        assert result.status == "gate_refused"
        assert not result
        # Refusal is inert: nothing was consumed, nothing moved.
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.get_pending_action(conn, task_id).state == "pending"


def test_gated_approval_with_token_succeeds(kanban_home):
    with kb.connect() as conn:
        task_id, att = _gated_card_with_pending_exact_action(conn)
        token = kb.issue_gate_token(conn, task_id, action="unblock")
        assert token

        result = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, att.action_id, expected_version=att.version, actor="op", token=token,
        )

        assert result.status == "approved"
        assert kb.get_task(conn, task_id).status in ("ready", "todo")
        assert kb.get_pending_action_by_id(conn, task_id, att.action_id).state == "approved"


def test_real_version_conflict_is_still_reported_as_conflict(kanban_home):
    """gate_refused must not swallow genuine CAS races -- they stay retryable."""
    with kb.connect() as conn:
        task_id, att = _gated_card_with_pending_exact_action(conn)
        token = kb.issue_gate_token(conn, task_id, action="unblock")

        result = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, att.action_id, expected_version=att.version + 99,
            actor="op", token=token,
        )

        assert result.status == "conflict"


def test_ungated_approval_needs_no_token(kanban_home):
    """Behavioural neutrality: the gate is a pure no-op for ungated cards."""
    with kb.connect() as conn:
        task_id, att = _gated_card_with_pending_exact_action(conn, gated=False)

        result = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, att.action_id, expected_version=att.version, actor="op", token=None,
        )

        assert result.status == "approved"
        assert kb.get_task(conn, task_id).status in ("ready", "todo")


def test_failed_gated_approval_does_not_burn_the_operators_token(kanban_home):
    """A one-shot grant must survive a failure that isn't the gate's fault.

    The token is single-use; if a losing CAS consumed it, the operator would be
    left holding a spent token and no way to approve -- a second deadlock.
    """
    with kb.connect() as conn:
        task_id, att = _gated_card_with_pending_exact_action(conn)
        token = kb.issue_gate_token(conn, task_id, action="unblock")

        # Lose the CAS on version, while presenting a perfectly valid token.
        assert kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, att.action_id, expected_version=att.version + 99,
            actor="op", token=token,
        ).status == "conflict"

        # The same token must still work.
        result = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, att.action_id, expected_version=att.version, actor="op", token=token,
        )
        assert result.status == "approved", "a lost CAS must not consume the one-shot grant"


def test_status_transition_gate_refusal_is_also_not_a_conflict(kanban_home):
    """Same conflation, second seam: transition_task_status_with_attention.

    Reachable through the TOCTOU window between the dashboard's human_gate
    pre-check and this transaction. No token is minted for this route on
    purpose -- gated cards are meant to go through exact-action approval, which
    test_kanban_a1_exact_action_gate.py protects. Only the diagnosis changes.

    The card here is gated but has NO pending exact action: with one, the
    exact-action guard above the gate check refuses first (correctly, and still
    as `conflict`), so the gate branch would never be reached.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="gated, no action", assignee="worker")
        kb.claim_task(conn, task_id)
        kb.block_task(conn, task_id, kind="needs_input", reason="needs a human", human_gate=True)
        assert kb.get_current_attention(conn, task_id) is None

        result = kb.transition_task_status_with_attention(
            conn, task_id=task_id, status="ready", actor="dashboard",
        )

        assert result.status == "gate_refused"
        assert not result
        assert kb.get_task(conn, task_id).status == "blocked", "refusal must be inert"


def test_live_exact_action_still_refuses_status_transition_as_conflict(kanban_home):
    """The A1 guard sits above the gate check and must keep firing first.

    Pins the branch order: a card with a live exact action is refused for THAT
    reason, not the gate -- otherwise gate_refused would mask the A1 protection.
    """
    with kb.connect() as conn:
        task_id, att = _gated_card_with_pending_exact_action(conn)

        result = kb.transition_task_status_with_attention(
            conn, task_id=task_id, status="ready", actor="dashboard",
            expected_attention_id=att.id, expected_attention_version=att.version,
        )

        assert result.status == "conflict"
        assert kb.get_task(conn, task_id).status == "blocked"

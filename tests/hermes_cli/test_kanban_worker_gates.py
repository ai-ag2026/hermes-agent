"""Kernel worker gates (audit 2026-07-11 C1).

`hermes kanban complete/block` from a worker's own terminal used to bypass
every tool-layer gate (ownership, needs_input, goal judge, goal-mode block
kinds). The gates now live in the kernel (complete_task/block_task) keyed on
the PROCESS being scoped to a task via HERMES_KANBAN_TASK/_RUN_ID, so CLI,
tool and any surface inside a worker process share one gate. Operator CLI
(no scope) and the gateway process stay ungated.
"""
from __future__ import annotations

import sys
import tempfile

import pytest


@pytest.fixture()
def kb_env(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_workergates_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db, monkeypatch


def _claimed_card(kb, conn, **kwargs):
    tid = kb.create_task(conn, title=kwargs.pop("title", "do work"), assignee="backend-eng", **kwargs)
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run_id = kb.get_task(conn, tid).current_run_id
    assert run_id is not None
    return tid, run_id


def _scope(monkeypatch, tid, run_id=None):
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    if run_id is None:
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    else:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------

def test_worker_cannot_complete_foreign_task(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        t1, r1 = _claimed_card(kb, conn, title="mine")
        t2, _ = _claimed_card(kb, conn, title="foreign")
        _scope(mp, t1, r1)
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.complete_task(conn, t2, summary="hijack")
        assert exc.value.gate == "ownership"
        assert kb.get_task(conn, t2).status == "running"


def test_worker_without_run_identity_refused(kb_env):
    """Delegate_task subagent subprocesses keep HERMES_KANBAN_TASK but have
    RUN_ID stripped — exactly this shape must be refused."""
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, _ = _claimed_card(kb, conn)
        _scope(mp, tid, run_id=None)
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.complete_task(conn, tid, summary="from delegated subagent shell")
        assert exc.value.gate == "run_identity"
        assert kb.get_task(conn, tid).status == "running"


def test_worker_with_stale_run_identity_refused(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, run_id = _claimed_card(kb, conn)
        _scope(mp, tid, run_id=run_id + 999)
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.complete_task(conn, tid, summary="stale worker")
        assert exc.value.gate == "run_identity"


def test_worker_cannot_complete_needs_input_card(kb_env):
    """The t_614c91e9 pattern, now closed at the kernel: a worker completing
    its own needs_input-blocked card would dissolve the human gate."""
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, run_id = _claimed_card(kb, conn)
        assert kb.block_task(conn, tid, reason="needs human", kind="needs_input")
        _scope(mp, tid, run_id)
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.complete_task(conn, tid, summary="dissolving the gate")
        assert exc.value.gate == "needs_input"
        assert kb.get_task(conn, tid).status == "blocked"


def test_worker_happy_path_still_completes(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, run_id = _claimed_card(kb, conn)
        _scope(mp, tid, run_id)
        assert kb.complete_task(conn, tid, summary="done, evidence: x")
        assert kb.get_task(conn, tid).status == "done"


def test_operator_cli_unaffected(kb_env):
    """No worker scope in env → gates are a no-op (operator affordance:
    completing a needs_input-blocked card via CLI stays allowed)."""
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, _ = _claimed_card(kb, conn)
        assert kb.block_task(conn, tid, reason="needs human", kind="needs_input")
        assert kb.complete_task(conn, tid, summary="operator closes it out")
        assert kb.get_task(conn, tid).status == "done"


def test_goal_judge_gates_worker_completion(kb_env, monkeypatch):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="goal card", assignee="backend-eng", goal_mode=True
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.get_task(conn, tid).current_run_id
        _scope(mp, tid, run_id)
        monkeypatch.setattr(kb, "_goal_judge_available", lambda: True)
        import hermes_cli.goals as goals
        monkeypatch.setattr(
            goals, "judge_goal",
            # Voller judge_goal-Vertrag: (verdict, reason, parse_failed,
            # wait_directive, transport_failed) — ein kürzerer Mock löst im
            # Gate ValueError aus, der defensiv geschluckt wird und die
            # Sperrsemantik still deaktiviert (TARS-Zweitreview, Stopplinie 1).
            lambda goal, last_response: ("continue", "no acceptance evidence", False, None, False),
        )
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.complete_task(conn, tid, summary="trust me it's done")
        assert exc.value.gate == "goal_judge"
        assert kb.get_task(conn, tid).status == "running"
        # Judge satisfied → completes.
        monkeypatch.setattr(
            goals, "judge_goal",
            lambda goal, last_response: ("done", "", False, None, False),
        )
        assert kb.complete_task(conn, tid, summary="evidence attached")


def test_gate_refusal_is_audited(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, run_id = _claimed_card(kb, conn)
        assert kb.block_task(conn, tid, reason="needs human", kind="needs_input")
        _scope(mp, tid, run_id)
        with pytest.raises(kb.WorkerGateError):
            kb.complete_task(conn, tid, summary="x")
        ev = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completion_blocked_gate'",
            (tid,),
        ).fetchone()[0]
        assert ev == 1


# ---------------------------------------------------------------------------
# block_task
# ---------------------------------------------------------------------------

def test_worker_cannot_block_foreign_task(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        t1, r1 = _claimed_card(kb, conn, title="mine")
        t2, _ = _claimed_card(kb, conn, title="foreign")
        _scope(mp, t1, r1)
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.block_task(conn, t2, reason="hijack", kind="needs_input")
        assert exc.value.gate == "ownership"
        assert kb.get_task(conn, t2).status == "running"


def test_goal_mode_block_kind_restricted_in_kernel(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="goal card", assignee="backend-eng", goal_mode=True
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.get_task(conn, tid).current_run_id
        _scope(mp, tid, run_id)
        # Untyped block = goal-loop escape → refused.
        with pytest.raises(kb.WorkerGateError) as exc:
            kb.block_task(conn, tid, reason="escaping the loop")
        assert exc.value.gate == "goal_mode_block"
        assert kb.get_task(conn, tid).status == "running"
        # Genuine external blocker kinds stay allowed.
        assert kb.block_task(
            conn, tid, reason="waiting on operator", kind="needs_input"
        )


def test_trusted_internal_bypasses_block_gates(kb_env):
    """Harness code (goal-loop escape in cli.py) blocks in-process inside the
    worker and must keep working."""
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="goal card", assignee="backend-eng", goal_mode=True
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = kb.get_task(conn, tid).current_run_id
        _scope(mp, tid, run_id)
        assert kb.block_task(
            conn, tid, reason="loop breaker", trusted_internal=True
        )
        assert kb.get_task(conn, tid).status == "blocked"


def test_operator_block_unaffected(kb_env):
    kb, mp = kb_env
    with kb.connect_closing() as conn:
        tid, _ = _claimed_card(kb, conn)
        assert kb.block_task(conn, tid, reason="operator parks it")
        assert kb.get_task(conn, tid).status == "blocked"

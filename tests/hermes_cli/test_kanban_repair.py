"""Tests for the Kanban invariant reconciler (hermes_cli.kanban_repair).

Card t_0cb8668d ("S5 Invariant-Reconciler"): a read-only audit over
task/run/event/claim/artifact-manifest state that classifies drift into
typed findings, plus a strictly-allowlisted, idempotent, CAS-bound repair
path (dry-run by default). The fixture matrix below exercises every
invariant named in the card, both the violating case (finding + repair
where safe) and the compliant case (must NOT be flagged).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_repair as kr


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_ARTIFACTS_ROOT", raising=False)
    kb.init_db()
    return home


def _findings_by_kind(report: kr.AuditReport, kind: str) -> list[kr.Finding]:
    return [f for f in report.findings if f.kind == kind]


def _kinds(report: kr.AuditReport) -> set[str]:
    return {f.kind for f in report.findings}


# ---------------------------------------------------------------------------
# 1. task status / current_run_id / claim / worker PID
# ---------------------------------------------------------------------------

def test_running_task_with_dangling_run_pointer_is_flagged(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        # Simulate a corrupted invariant: the run already ended but the
        # task's pointer + claim state were never cleared (e.g. a crash
        # between two statements outside write_txn, or manual DB surgery).
        conn.execute(
            "UPDATE task_runs SET ended_at = ?, status='crashed', outcome='crashed' "
            "WHERE id = ?",
            (int(time.time()), run_id),
        )
        report = kr.run_audit(conn)
    findings = _findings_by_kind(report, "running_task_missing_live_run")
    assert len(findings) == 1
    assert findings[0].task_id == task_id
    assert findings[0].safe_repair == "reclaim_orphaned_running_task"


def test_running_task_with_live_run_is_not_flagged(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        kb.claim_task(conn, task_id)
        report = kr.run_audit(conn)
    assert "running_task_missing_live_run" not in _kinds(report)


def test_repair_reclaims_orphaned_running_task_idempotently(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        claimed = kb.claim_task(conn, task_id)
        run_id = claimed.current_run_id
        conn.execute(
            "UPDATE task_runs SET ended_at = ?, status='crashed', outcome='crashed' "
            "WHERE id = ?",
            (int(time.time()), run_id),
        )

        # Dry run changes nothing.
        dry = kr.run_repair(conn, dry_run=True)
        assert dry.dry_run is True
        assert any(r.finding.kind == "running_task_missing_live_run" for r in dry.results)
        task = kb.get_task(conn, task_id)
        assert task.status == "running"

        applied = kr.run_repair(conn, dry_run=False, actor="tester", reason="fix")
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.claim_lock is None

        # Idempotent: a second apply finds nothing left to do.
        second = kr.run_repair(conn, dry_run=False, actor="tester", reason="fix")
        assert not any(
            r.applied and r.finding.kind == "running_task_missing_live_run"
            for r in second.results
        )


def test_apply_requires_actor_and_reason(kanban_home):
    with kb.connect() as conn:
        kb.create_task(conn, title="t")
        with pytest.raises(ValueError):
            kr.run_repair(conn, dry_run=False)
        with pytest.raises(ValueError):
            kr.run_repair(conn, dry_run=False, actor="tester")
        with pytest.raises(ValueError):
            kr.run_repair(conn, dry_run=False, reason="fix")


def test_dry_run_never_mutates_db(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        claimed = kb.claim_task(conn, task_id)
        run_id = claimed.current_run_id
        conn.execute(
            "UPDATE task_runs SET ended_at = ?, status='crashed', outcome='crashed' "
            "WHERE id = ?",
            (int(time.time()), run_id),
        )
        before = json.dumps(dict(kb.get_task(conn, task_id).__dict__), default=str, sort_keys=True)
        kr.run_repair(conn, dry_run=True)
        after = json.dumps(dict(kb.get_task(conn, task_id).__dict__), default=str, sort_keys=True)
    assert before == after


# ---------------------------------------------------------------------------
# 2. maximal ein aktiver Run pro Task
# ---------------------------------------------------------------------------

def test_multiple_open_runs_unambiguous_case_is_repaired(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        claimed = kb.claim_task(conn, task_id)
        real_run_id = claimed.current_run_id
        # Inject a second, bogus "open" run row (data corruption / a
        # duplicate insert that should never happen).
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) "
            "VALUES (?, 'running', ?)",
            (task_id, int(time.time())),
        )
        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "multiple_open_runs")
        assert len(findings) == 1
        assert findings[0].safe_repair == "close_duplicate_open_runs"

        kr.run_repair(conn, dry_run=False, actor="tester", reason="dup run")
        open_runs = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            (task_id,),
        ).fetchall()
        assert [r["id"] for r in open_runs] == [real_run_id]


def test_multiple_open_runs_ambiguous_case_is_triage_only(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        # Two open runs, neither referenced by current_run_id (task was
        # never actually claimed through the normal path in this fixture).
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', ?)",
            (task_id, int(time.time())),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at) VALUES (?, 'running', ?)",
            (task_id, int(time.time())),
        )
        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "multiple_open_runs")
        assert len(findings) == 1
        assert findings[0].safe_repair is None
        assert findings[0].bucket == "triage"

        kr.run_repair(conn, dry_run=False, actor="tester", reason="x")
        open_runs = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            (task_id,),
        ).fetchall()
        assert len(open_runs) == 2  # untouched


# ---------------------------------------------------------------------------
# 3. terminale Eltern vs stale todo/blocked (+ sticky/review block negatives)
# ---------------------------------------------------------------------------

def test_stale_todo_with_done_parents_is_flagged_and_repaired(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"
        kb.complete_task(conn, parent, summary="done")
        # Simulate a missed recompute_ready tick: force child back to todo
        # (it may already have been promoted by complete_task's own call,
        # so make the drift explicit and deterministic for the test).
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "stale_promotable_task")
        assert any(f.task_id == child for f in findings)

        kr.run_repair(conn, dry_run=False, actor="tester", reason="promote")
        assert kb.get_task(conn, child).status == "ready"


def test_sticky_blocked_task_is_never_flagged_as_stale_promotable(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")  # no parents -> ready
        kb.claim_task(conn, task_id)
        kb.block_task(conn, task_id, reason="review-required: needs a human", kind="needs_input")

        report = kr.run_audit(conn)
        assert not any(
            f.task_id == task_id for f in _findings_by_kind(report, "stale_promotable_task")
        )
        # Repair must not touch it either.
        kr.run_repair(conn, dry_run=False, actor="tester", reason="x")
        assert kb.get_task(conn, task_id).status == "blocked"


def test_task_over_failure_limit_is_never_flagged_as_stale_promotable(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent], max_retries=1)
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1 WHERE id=?",
            (child,),
        )
        kb.complete_task(conn, parent, summary="done")

        report = kr.run_audit(conn)
        assert not any(
            f.task_id == child for f in _findings_by_kind(report, "stale_promotable_task")
        )


# ---------------------------------------------------------------------------
# Task-/Runstatus-Mismatch bei dependency wait: must NOT be misclassified
# ---------------------------------------------------------------------------

def test_dependency_wait_task_run_status_divergence_is_not_flagged(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        task_id = kb.create_task(conn, title="t")  # no parents yet -> ready
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        # A genuine, still-open dependency discovered mid-run (this is what
        # makes 'todo' the durable resting state instead of an immediate
        # re-promotion).
        kb.link_tasks(conn, parent, task_id)
        # By design: block_task(kind="dependency") lands the task in
        # 'todo' while the closed run row is stamped status='blocked'.
        assert kb.block_task(conn, task_id, reason="waiting", kind="dependency")
        assert kb.get_task(conn, task_id).status == "todo"

        report = kr.run_audit(conn)
        # None of the run/task coherence or stale-promotion findings
        # should fire for this intentional, by-design divergence — the
        # parent is still open, so there is nothing stale to promote.
        assert task_id not in {f.task_id for f in report.findings}


# ---------------------------------------------------------------------------
# 4. done ohne completed Run oder Completion-Event
# ---------------------------------------------------------------------------

def test_done_without_any_completion_evidence_is_fail_closed(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        # Force straight to done, bypassing complete_task entirely (as if
        # a bug or manual SQL flipped the column directly).
        conn.execute("UPDATE tasks SET status='done', completed_at=? WHERE id=?", (int(time.time()), task_id))

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "done_missing_completion_evidence")
        assert len(findings) == 1
        assert findings[0].bucket == "triage"
        assert findings[0].safe_repair is None

        kr.run_repair(conn, dry_run=False, actor="tester", reason="x")
        # Never "fixed" by fabricating evidence or reverting status.
        assert kb.get_task(conn, task_id).status == "done"


def test_done_via_complete_task_has_evidence_and_is_not_flagged(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done")
        report = kr.run_audit(conn)
    assert "done_missing_completion_evidence" not in _kinds(report)


def test_done_via_review_accept_has_evidence_and_is_not_flagged(kanban_home):
    # decide_task_review's ACCEPT path ends the run with outcome="accept"
    # (not "completed") but appends a "completed" event — the reconciler
    # must recognise this second, legitimate shape of completion evidence.
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert kb.request_task_review(conn, task_id, reviewer="reviewer", summary="please review")
        assert kb.claim_review_task(conn, task_id) is not None
        assert kb.decide_task_review(conn, task_id, decision="ACCEPT", summary="looks good")
        assert kb.get_task(conn, task_id).status == "done"
        report = kr.run_audit(conn)
    assert "done_missing_completion_evidence" not in _kinds(report)


# ---------------------------------------------------------------------------
# 5/6. done mit fehlendem/invalidem durable Artifact-Manifest (+ reattach)
# ---------------------------------------------------------------------------

def test_done_with_corrupted_artifact_file_is_fail_closed(kanban_home):
    content = b"evidence\n"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        source = workspace / "proof.txt"
        source.write_bytes(content)
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(source)]})

        row = conn.execute(
            "SELECT durable_path FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()
        durable = Path(row["durable_path"])
        durable.write_bytes(b"tampered")

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "done_artifact_manifest_invalid")
        assert len(findings) == 1
        assert findings[0].safe_repair is None
        assert findings[0].bucket == "triage"


def test_done_with_missing_artifact_file_is_fail_closed(kanban_home):
    content = b"evidence\n"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        source = workspace / "proof.txt"
        source.write_bytes(content)
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(source)]})

        row = conn.execute(
            "SELECT durable_path FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()
        Path(row["durable_path"]).unlink()

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "done_artifact_manifest_invalid")
        assert len(findings) == 1


def test_valid_completion_artifact_is_not_flagged(kanban_home):
    content = b"evidence\n"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        source = workspace / "proof.txt"
        source.write_bytes(content)
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(source)]})
        report = kr.run_audit(conn)
    assert "done_artifact_manifest_invalid" not in _kinds(report)
    assert "done_artifact_manifest_lost" not in _kinds(report)


def test_lost_manifest_row_is_safely_reattached_from_event_and_file(kanban_home):
    content = b"evidence\n"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        source = workspace / "proof.txt"
        source.write_bytes(content)
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(source)]})

        row = conn.execute(
            "SELECT durable_path FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()
        durable_path = row["durable_path"]
        expected_sha = hashlib.sha256(content).hexdigest()
        assert Path(durable_path).read_bytes() == content

        # Simulate a lost DB row (e.g. a migration/rebuild bug) while the
        # durable file and its immutable event-log record survive intact.
        conn.execute("DELETE FROM task_artifacts WHERE task_id=?", (task_id,))
        assert not conn.execute(
            "SELECT 1 FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "done_artifact_manifest_lost")
        assert len(findings) == 1
        assert findings[0].safe_repair == "reattach_artifact_manifest"

        applied = kr.run_repair(conn, dry_run=False, actor="tester", reason="reattach")
        assert any(
            r.applied and r.finding.kind == "done_artifact_manifest_lost"
            for r in applied.results
        )
        rows = conn.execute(
            "SELECT * FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["durable_path"] == durable_path
        assert rows[0]["sha256"] == expected_sha

        # Idempotent: a second apply is a no-op (row already restored).
        second = kr.run_repair(conn, dry_run=False, actor="tester", reason="reattach")
        assert not any(
            r.applied and r.finding.kind == "done_artifact_manifest_lost"
            for r in second.results
        )
        rows_after = conn.execute(
            "SELECT * FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchall()
        assert len(rows_after) == 1


# ---------------------------------------------------------------------------
# eindeutige Run-/Taskabschluss-Reconciliation
# ---------------------------------------------------------------------------

def test_orphaned_completed_run_reconciles_task_to_done(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done")
        # Simulate a lost tasks-row write in an otherwise atomic
        # completion (e.g. a partial table-rebuild bug): the run and the
        # "completed" event are durable and consistent with each other,
        # but the tasks row reverted to a pre-completion state.
        conn.execute(
            "UPDATE tasks SET status='running', completed_at=NULL, "
            "current_run_id=(SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1) "
            "WHERE id=?",
            (task_id, task_id),
        )

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "nonterminal_task_completed_run_orphaned")
        assert len(findings) == 1
        assert findings[0].safe_repair == "reconcile_task_done_from_completed_run"

        kr.run_repair(conn, dry_run=False, actor="tester", reason="reconcile")
        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        assert task.current_run_id is None


def test_orphaned_completed_run_with_invalid_artifacts_is_triage_only(kanban_home):
    content = b"evidence\n"
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        source = workspace / "proof.txt"
        source.write_bytes(content)
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(source)]})

        row = conn.execute(
            "SELECT durable_path FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()
        Path(row["durable_path"]).unlink()

        conn.execute(
            "UPDATE tasks SET status='running', completed_at=NULL, "
            "current_run_id=(SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1) "
            "WHERE id=?",
            (task_id, task_id),
        )

        report = kr.run_audit(conn)
        assert not _findings_by_kind(report, "nonterminal_task_completed_run_orphaned")
        findings = _findings_by_kind(
            report, "nonterminal_task_completed_run_orphaned_evidence_invalid"
        )
        assert len(findings) == 1
        assert findings[0].safe_repair is None

        kr.run_repair(conn, dry_run=False, actor="tester", reason="x")
        # This fixture's status='running' + current_run_id pointing at the
        # now-ended run is ALSO, independently, a dangling live-run-pointer
        # violation (a completely separate, legitimate invariant) — that
        # part is safe to repair (reclaim to 'ready') regardless of the
        # artifact evidence. What must never happen is a fabricated 'done'.
        assert kb.get_task(conn, task_id).status == "ready"


# ---------------------------------------------------------------------------
# gave_up / timed_out mit verifizierter Closeout-Evidenz
# ---------------------------------------------------------------------------

def test_blocked_goal_mode_with_closeout_evidence_is_flagged_triage_only(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="t", goal_mode=True,
        )
        claimed = kb.claim_task(conn, task_id)
        run_id = claimed.current_run_id
        kb.record_goal_progress_event(
            conn, task_id,
            {
                "task_id": task_id, "phase": "closeout", "turns_used": 5,
                "max_turns": 5, "nudged_to_finalize": True,
                "closeout_turns_used": 1, "reason": "judge said done",
            },
            run_id=run_id,
        )
        kb.block_task(
            conn, task_id,
            reason="Goal-mode worker exhausted its reserved closeout corridor",
            kind=None,
        )

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "blocked_with_verified_closeout_evidence")
        assert len(findings) == 1
        assert findings[0].task_id == task_id
        assert findings[0].safe_repair is None
        assert findings[0].bucket == "triage"

        # Must never be auto-completed — that would be exactly the
        # forbidden "heuristic completion from prose".
        kr.run_repair(conn, dry_run=False, actor="tester", reason="x")
        assert kb.get_task(conn, task_id).status == "blocked"


def test_blocked_without_closeout_evidence_is_not_flagged(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t", goal_mode=True)
        kb.claim_task(conn, task_id)
        kb.block_task(conn, task_id, reason="normal block", kind="needs_input")
        report = kr.run_audit(conn)
    assert "blocked_with_verified_closeout_evidence" not in _kinds(report)


# ---------------------------------------------------------------------------
# archive-vs-complete terminal conflict
# ---------------------------------------------------------------------------

def test_archived_task_with_pending_review_is_flagged(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t", assignee="worker")
        kb.claim_task(conn, task_id)
        assert kb.request_task_review(conn, task_id, reviewer="reviewer", summary="pls review")
        assert kb.archive_task(conn, task_id)

        report = kr.run_audit(conn)
        findings = _findings_by_kind(report, "archived_with_pending_review")
        assert len(findings) == 1
        assert findings[0].safe_repair is None
        assert findings[0].bucket == "triage"


def test_archived_task_without_pending_review_is_not_flagged(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        kb.claim_task(conn, task_id)
        kb.complete_task(conn, task_id, summary="done")
        assert kb.archive_task(conn, task_id)
        report = kr.run_audit(conn)
    assert "archived_with_pending_review" not in _kinds(report)


# ---------------------------------------------------------------------------
# Concurrency: at most one winner, no claim/run violation (DoD 6)
# ---------------------------------------------------------------------------

def test_concurrent_repair_and_claim_yield_single_winner_no_violation(kanban_home, tmp_path):
    db_path = kb.kanban_db_path()
    task_id = None
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        claimed = kb.claim_task(conn, task_id)
        run_id = claimed.current_run_id
        conn.execute(
            "UPDATE task_runs SET ended_at = ?, status='crashed', outcome='crashed' "
            "WHERE id = ?",
            (int(time.time()), run_id),
        )

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def _repair_once():
        try:
            barrier.wait(timeout=5)
            with kb.connect_closing(db_path=db_path) as conn:
                kr.run_repair(conn, dry_run=False, actor="repair-a", reason="race")
        except BaseException as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    def _repair_again():
        try:
            barrier.wait(timeout=5)
            with kb.connect_closing(db_path=db_path) as conn:
                kr.run_repair(conn, dry_run=False, actor="repair-b", reason="race")
        except BaseException as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)

    t1 = threading.Thread(target=_repair_once)
    t2 = threading.Thread(target=_repair_again)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, errors
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        # Exactly one reclaim landed; task ends up ready with a clean
        # claim/run pointer — no double-application, no dangling state.
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.claim_lock is None
        open_runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id=? AND ended_at IS NULL",
            (task_id,),
        ).fetchone()["n"]
        assert open_runs == 0
        # Exactly one repair-applied audit event for this finding kind —
        # the second racer observed the precondition already gone.
        applied_events = [
            e for e in kb.list_events(conn, task_id)
            if e.kind == "kanban_repair_applied"
            and (e.payload or {}).get("finding_kind") == "running_task_missing_live_run"
        ]
        assert len(applied_events) == 1


# ---------------------------------------------------------------------------
# Report shape / structured output (audit + metrics extension)
# ---------------------------------------------------------------------------

def test_audit_report_to_dict_is_json_serializable(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        report = kr.run_audit(conn)
    payload = report.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["counts_by_kind"].get("done_missing_completion_evidence") == 1


def test_repair_report_to_dict_is_json_serializable(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        report = kr.run_repair(conn, dry_run=True)
    json.dumps(report.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# CLI wiring: `hermes kanban audit` / `hermes kanban repair`
# ---------------------------------------------------------------------------

def test_cli_audit_json_reports_finding(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
    out = kc.run_slash("audit --json")
    payload = json.loads(out)
    assert payload["counts_by_kind"].get("done_missing_completion_evidence") == 1


def test_cli_repair_dry_run_then_apply_requires_actor_and_reason(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="t")
        claimed = kb.claim_task(conn, task_id)
        run_id = claimed.current_run_id
        conn.execute(
            "UPDATE task_runs SET ended_at = ?, status='crashed', outcome='crashed' "
            "WHERE id = ?",
            (int(time.time()), run_id),
        )

    dry = kc.run_slash("repair --json")
    dry_payload = json.loads(dry)
    assert dry_payload["dry_run"] is True
    assert dry_payload["applied_count"] == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "running"

    usage_error = kc.run_slash("repair --apply")
    assert "usage error" in usage_error or "actor" in usage_error

    applied = kc.run_slash("repair --apply --actor tester --reason fix --json")
    applied_payload = json.loads(applied)
    assert applied_payload["dry_run"] is False
    assert applied_payload["applied_count"] == 1
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


# ---------------------------------------------------------------------------
# H8 (Audit 2026-07-11): referential integrity of side tables
# ---------------------------------------------------------------------------

def test_orphaned_notify_sub_is_flagged(kanban_home):
    with kb.connect() as conn:
        # A sub whose task never existed (the 2026-07-09 corruption residue).
        conn.execute(
            "INSERT INTO kanban_notify_subs "
            "(task_id, platform, chat_id, thread_id, created_at) "
            "VALUES ('t_ghost', 'webui', 'c1', '', 0)"
        )
        conn.commit()
        report = kr.run_audit(conn)
    orphans = _findings_by_kind(report, "orphaned_reference")
    assert len(orphans) == 1
    assert orphans[0].task_id == "t_ghost"
    assert orphans[0].data["table"] == "kanban_notify_subs"
    assert orphans[0].bucket == "triage"


def test_orphaned_artifact_and_link_are_flagged(kanban_home):
    with kb.connect() as conn:
        real = kb.create_task(conn, title="real")
        conn.execute(
            "INSERT INTO task_artifacts "
            "(task_id, producer_run_id, original_path, durable_path, sha256, "
            " size, validated_at, retention_class) "
            "VALUES ('t_gone', 0, '/tmp/a', '/tmp/d', 'x', 1, 0, 'keep')"
        )
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, 't_missing_child')",
            (real,),
        )
        conn.commit()
        report = kr.run_audit(conn)
    orphans = _findings_by_kind(report, "orphaned_reference")
    tables = {f.data["table"] for f in orphans}
    assert tables == {"task_artifacts", "task_links"}


def test_healthy_references_not_flagged(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="p")
        child = kb.create_task(conn, title="c", parents=[parent])
        conn.execute(
            "INSERT INTO kanban_notify_subs "
            "(task_id, platform, chat_id, thread_id, created_at) "
            "VALUES (?, 'telegram', 'c1', '', 0)", (parent,),
        )
        conn.commit()
        report = kr.run_audit(conn)
    assert not _findings_by_kind(report, "orphaned_reference")

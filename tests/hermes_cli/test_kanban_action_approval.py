from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def isolated_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = home / "kanban.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_PROFILE", "backend-eng")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(db_path)
    return workspace


def _create_running_task(monkeypatch: pytest.MonkeyPatch) -> str:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="publish exact ref", assignee="backend-eng")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
        return task_id


def _atomic_snapshot(conn, task_id: str, run_id: int) -> dict[str, list[tuple]]:
    return {
        "task": [tuple(row) for row in conn.execute("SELECT status, current_run_id, claim_lock, claim_expires, worker_pid, block_kind, block_recurrences FROM tasks WHERE id=?", (task_id,))],
        "run": [tuple(row) for row in conn.execute("SELECT status, outcome, ended_at, claim_lock, claim_expires, worker_pid FROM task_runs WHERE id=?", (run_id,))],
        "actions": [tuple(row) for row in conn.execute("SELECT * FROM task_pending_actions WHERE task_id=? ORDER BY id", (task_id,))],
        "attentions": [tuple(row) for row in conn.execute("SELECT * FROM task_attentions WHERE task_id=? ORDER BY id", (task_id,))],
        "events": [tuple(row) for row in conn.execute("SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,))],
    }


def test_exact_action_hash_preserves_raw_unicode_and_newline_bytes() -> None:
    nfd = "rm -- /workspace/e\u0301"
    nfc = "rm -- /workspace/é"
    assert nfd.encode("utf-8") != nfc.encode("utf-8")
    assert kb._pending_action_hash(nfd) != kb._pending_action_hash(nfc)
    assert kb._pending_action_hash("printf x\r\n") != kb._pending_action_hash("printf x\n")


def test_action_grant_is_exact_expiring_and_consume_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abc1234 fork HEAD:topic"

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command=command,
            summary="git push --force-with-lease to topic", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        row = conn.execute(
            "SELECT * FROM task_pending_actions WHERE id = ?", (action.id,)
        ).fetchone()
        assert command not in "|".join(str(value) for value in row)
        assert action.mutation_kind == "git-push-force-with-lease"
        assert len(action.fingerprint) == 64
        assert row["fingerprint"] == action.fingerprint
        origin_run = task.current_run_id
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="terminal approval required",
            expected_run_id=origin_run,
        )
        assert kb.approve_pending_action_and_unblock(
            conn, task_id, action.id, now=1_900_000_000,
        )
        resumed = kb.claim_task(conn, task_id, claimer="resumed")
        assert resumed is not None
        run_id = resumed.current_run_id

        context = dict(
            task_id=task_id, run_id=run_id, profile="backend-eng",
            workspace=str(isolated_board),
        )
        assert not kb.consume_approved_action(
            conn, command=command + " --dry-run", now=1_900_000_001, **context,
        )
        assert not kb.consume_approved_action(
            conn, command=command, profile="reviewer", task_id=task_id,
            run_id=run_id, workspace=str(isolated_board), now=1_900_000_001,
        )
        conn.execute(
            "UPDATE task_pending_actions SET fingerprint=? WHERE id=?",
            ("0" * 64, action.id),
        )
        assert not kb.consume_approved_action(
            conn, command=command, now=1_900_000_001, **context,
        )
        conn.execute(
            "UPDATE task_pending_actions SET fingerprint=? WHERE id=?",
            (action.fingerprint, action.id),
        )
        assert kb.consume_approved_action(
            conn, command=command, now=1_900_000_001, **context,
        )
        assert not kb.consume_approved_action(
            conn, command=command, now=1_900_000_002, **context,
        )
        event_payloads = [
            event.payload for event in kb.list_events(conn, task_id=task_id)
        ]
        assert action.command_hash not in repr(event_payloads)
        assert action.fingerprint not in repr(event_payloads)


def test_pending_action_persists_only_constant_operator_summary(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    untrusted_summary = "credential-fragment=do-not-persist"

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(
            conn,
            task_id=task_id,
            run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic",
            summary=untrusted_summary,
            profile="backend-eng",
            workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        row = conn.execute(
            "SELECT summary FROM task_pending_actions WHERE id = ?", (action.id,),
        ).fetchone()
        events = kb.list_events(conn, task_id=task_id)

    assert action.summary == kb.PENDING_ACTION_OPERATOR_SUMMARY
    assert row["summary"] == kb.PENDING_ACTION_OPERATOR_SUMMARY
    assert untrusted_summary not in repr(events)
    assert kb.PENDING_ACTION_OPERATOR_SUMMARY in repr(events)


def test_tampered_fingerprint_cannot_be_approved(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task and task.current_run_id is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/main:abcdef1 origin HEAD:main",
            summary="publish exact reviewed ref", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        assert kb.block_task(
            conn, task_id, reason="approval required", kind="needs_input",
            expected_run_id=task.current_run_id,
        )
        conn.execute(
            "UPDATE task_pending_actions SET fingerprint = ? WHERE id = ?",
            ("0" * 64, action.id),
        )
        conn.commit()
        assert not kb.approve_pending_action_and_unblock(
            conn, task_id, action.id, now=1_900_000_001,
        )
        current = kb.get_task(conn, task_id)
        assert current and current.status == "blocked"


def test_plain_force_push_is_never_recorded_as_approvable(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        for unsafe_command in (
            "git push --force fork HEAD:topic",
            "git push --force-with-lease=refs/heads/topic:abcdef1 --force fork HEAD:topic",
            "git push --force-with-lease fork HEAD:topic",
        ):
            with pytest.raises(ValueError, match="force-with-lease"):
                kb.record_pending_action(
                    conn, task_id=task_id, run_id=task.current_run_id,
                    command=unsafe_command, summary="unsafe push",
                    profile="backend-eng", workspace=str(isolated_board),
                    expires_at=2_000_000_000,
                )
        count = conn.execute(
            "SELECT COUNT(*) FROM task_pending_actions WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        assert count == 0


def test_expired_grant_is_rejected(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb.time, "time", lambda: 50)
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="git push",
            profile="backend-eng", workspace=str(isolated_board), expires_at=100,
        )
        assert not kb.approve_pending_action(conn, task_id, action.id, now=101)


def test_pending_action_keeps_repeated_needs_input_block_sticky(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="git push",
            profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        assert kb.block_task(
            conn, task_id, reason="terminal approval required", kind="needs_input",
            expected_run_id=task.current_run_id,
        )
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
            conn.commit()
            assert kb.block_task(
                conn, task_id, reason="terminal approval required", kind="needs_input",
            )
            current = kb.get_task(conn, task_id)
            assert current is not None and current.status == "blocked"
            assert kb.recompute_ready(conn) == 0


def test_terminal_guard_round_trip_records_and_consumes_exact_action(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abc1234 fork HEAD:topic"
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(
        approval, "detect_dangerous_command",
        lambda value: (True, "git_force_push", "force push")
        if value == command else (False, None, None),
    )

    first = approval.check_all_command_guards(command, "local")
    assert first["status"] == "pending_approval"
    with kb.connect() as conn:
        action = kb.get_pending_action(conn, task_id)
        assert action is not None
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked" and task.current_run_id is None
        assert kb.approve_pending_action_and_unblock(conn, task_id, action.id)

    with kb.connect() as conn:
        resumed = kb.claim_task(conn, task_id, claimer="resumed-worker")
        assert resumed is not None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(resumed.current_run_id))

    second = approval.check_all_command_guards(command, "local")
    assert second["approved"] is True
    assert second["kanban_action_grant"] is True
    with kb.connect() as conn:
        assert kb.get_pending_action(conn, task_id) is None


def _park_and_approve(conn, task_id: str, command: str, workspace: Path):
    task = kb.get_task(conn, task_id)
    action = kb.record_pending_action(
        conn, task_id=task_id, run_id=task.current_run_id, command=command,
        summary="git push", profile="backend-eng", workspace=str(workspace),
        expires_at=2_000_000_000,
    )
    assert kb.block_task(
        conn, task_id, kind="needs_input", reason="approval required",
        expected_run_id=task.current_run_id,
    )
    assert kb.approve_pending_action_and_unblock(conn, task_id, action.id)
    resumed = kb.claim_task(conn, task_id, claimer="resumed")
    assert resumed is not None
    return action, resumed.current_run_id


def test_plain_unblock_promote_and_claim_refuse_unapproved_action(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="git push",
            profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="approval required",
            expected_run_id=task.current_run_id,
        )
        assert kb.unblock_task(conn, task_id) is False
        ok, reason = kb.promote_task(conn, task_id, actor="operator", force=True)
        assert ok is False
        assert "terminal action" in (reason or "").lower()
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.claim_task(conn, task_id, claimer="bypass") is None
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.get_pending_action(conn, task_id).id == action.id


def test_consume_requires_current_resumed_run_and_is_once_only(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        origin_run = kb.get_task(conn, task_id).current_run_id
        _action, resumed_run = _park_and_approve(conn, task_id, command, isolated_board)
        context = dict(
            task_id=task_id, command=command, profile="backend-eng",
            workspace=str(isolated_board), now=1_900_000_000,
        )
        assert not kb.consume_approved_action(conn, run_id=origin_run, **context)
        assert not kb.consume_approved_action(conn, run_id=resumed_run + 1000, **context)
        assert kb.consume_approved_action(conn, run_id=resumed_run, **context)
        assert not kb.consume_approved_action(conn, run_id=resumed_run, **context)


def test_reblocked_resumed_run_cancels_unused_grant_and_restores_normal_unblock(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        action, resumed_run = _park_and_approve(
            conn, task_id, command, isolated_board,
        )
        conn.execute("DELETE FROM task_attentions WHERE task_id=?", (task_id,))
        conn.execute("INSERT INTO task_attentions (task_id, action_id, type, cause_fingerprint, summary, created_at) VALUES (?, ?, 'transient', 'reblock-cleanup', 'technical', 1)", (task_id, action.id))
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="different human decision",
            expected_run_id=resumed_run,
        )
        resolved = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert resolved is not None
        assert resolved.cancelled_at is not None
        assert resolved.consumed_at is None
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 0
        assert kb.get_pending_action(conn, task_id) is None
        assert kb.unblock_task(conn, task_id)
        current = kb.get_task(conn, task_id)
        assert current and current.status == "ready"
        events = kb.list_events(conn, task_id=task_id)
        assert any(
            event.kind == "terminal_approval_cancelled"
            and (event.payload or {}).get("reason") == "resumed_run_reblocked"
            for event in events
        )


def test_duplicate_pending_request_reuses_one_record(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        kwargs = dict(
            task_id=task_id, run_id=task.current_run_id, command=command,
            summary="git push", profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        first = kb.record_pending_action(conn, **kwargs)
        second = kb.record_pending_action(conn, **kwargs)
        assert second.id == first.id
        assert conn.execute(
            "SELECT COUNT(*) FROM task_pending_actions WHERE task_id=?", (task_id,),
        ).fetchone()[0] == 1


def test_approved_retry_is_stable_and_does_not_emit_second_pending_event(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        first = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                         summary="publish", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.block_task(conn, task_id, kind="needs_input", reason="approval", expected_run_id=task.current_run_id)
        assert kb.approve_pending_action(conn, task_id, first.id, now=1_900_000_000)
        before = kb.get_pending_action_by_id(conn, task_id, first.id)
        attention = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert before is not None and attention is not None
        second = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                          summary="publish", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_100)
        after = kb.get_pending_action_by_id(conn, task_id, first.id)
        current = kb.get_current_attention(conn, task_id, now=1_900_000_001)
        assert after is not None and current is not None
        assert (second.id, second.state, after.version, after.expires_at, current.id) == (first.id, "approved", before.version, before.expires_at, attention.id)
        assert sum(e.kind == "terminal_approval_pending" for e in kb.list_events(conn, task_id=task_id)) == 1


def test_terminal_same_identity_creates_fresh_attention_after_projection_cleanup(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        first = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                         summary="publish", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        old_attention = kb.get_current_attention(conn, task_id)
        assert old_attention is not None
        assert kb.resolve_pending_action(conn, task_id, first.id, expected_version=first.version, now=1_900_000_000)
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 0
        second = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                          summary="publish", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_100)
        current = kb.get_current_attention(conn, task_id)
        historic = kb.get_pending_action_by_id(conn, task_id, first.id)
        assert current is not None and historic is not None
        assert historic.state == "resolved"
        assert (current.id != old_attention.id, current.action_id) == (True, second.id)


def test_parallel_consumers_only_one_wins(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        _action, run_id = _park_and_approve(conn, task_id, command, isolated_board)

    barrier = threading.Barrier(2)
    results: list[bool] = []

    def consume() -> None:
        with kb.connect() as conn:
            barrier.wait()
            results.append(kb.consume_approved_action(
                conn, task_id=task_id, run_id=run_id, command=command,
                profile="backend-eng", workspace=str(isolated_board),
                now=1_900_000_000,
            ))

    threads = [threading.Thread(target=consume) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert sorted(results) == [False, True]
    with kb.connect() as conn:
        assert sum(e.kind == "terminal_approval_consumed" for e in kb.list_events(conn, task_id=task_id)) == 1


def test_expired_pending_rerequest_creates_fresh_attention_after_projection_cleanup(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"value": 50}
    monkeypatch.setattr(kb.time, "time", lambda: now["value"])
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        first = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                         summary="ignored", profile="backend-eng", workspace=str(isolated_board), expires_at=100)
        attention = kb.get_current_attention(conn, task_id, now=50)
        assert attention is not None
        now["value"] = 101
        second = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                          summary="ignored", profile="backend-eng", workspace=str(isolated_board), expires_at=200)
        old = kb.get_pending_action_by_id(conn, task_id, first.id)
        current = kb.get_current_attention(conn, task_id, now=101)
        assert old is not None and current is not None
        assert (old.state, second.id != first.id, current.id != attention.id, current.action_id) == ("expired", True, True, second.id)


def test_supersession_only_cancels_current_origin_action(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        first = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                         summary="ignored", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        other_run = conn.execute("INSERT INTO task_runs (task_id,status,started_at,profile) VALUES (?, 'done', 1, 'backend-eng')", (task_id,)).lastrowid
        other_hash = kb._pending_action_hash("historical-origin-marker")
        other_fp = kb._pending_action_fingerprint(board_identity=kb._pending_action_board_identity(conn), task_id=task_id,
            run_id=other_run, command_hash=other_hash, mutation_kind="terminal-command", profile="backend-eng", workspace=str(isolated_board.resolve()))
        conn.execute("INSERT INTO task_pending_actions (task_id,run_id,command_hash,fingerprint,mutation_kind,summary,profile,workspace,created_at,expires_at,state,version,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?, 'pending',1,1)",
            (task_id, other_run, other_hash, other_fp, "terminal-command", kb.PENDING_ACTION_OPERATOR_SUMMARY, "backend-eng", str(isolated_board.resolve()), 1, 2_000_000_000))
        second = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command + " --new-identity",
                                          summary="ignored", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.get_pending_action_by_id(conn, task_id, first.id).state == "cancelled"
        assert kb.get_pending_action_by_id(conn, task_id, second.id).state == "pending"
        untouched = conn.execute("SELECT state FROM task_pending_actions WHERE run_id=?", (other_run,)).fetchone()
        assert untouched["state"] == "pending"


def test_parallel_same_identity_record_returns_one_action_and_event(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        run_id = kb.get_task(conn, task_id).current_run_id
    barrier, ids, errors = threading.Barrier(2), [], []
    def record() -> None:
        try:
            with kb.connect() as conn:
                barrier.wait()
                ids.append(kb.record_pending_action(conn, task_id=task_id, run_id=run_id,
                    command="parallel-record-marker", summary="ignored", profile="backend-eng",
                    workspace=str(isolated_board), expires_at=2_000_000_000).id)
        except Exception as exc:  # test records unhandled SQLite races explicitly
            errors.append(exc)
    threads = [threading.Thread(target=record) for _ in range(2)]
    [thread.start() for thread in threads]
    [thread.join(timeout=5) for thread in threads]
    assert not errors and sorted(ids) == [ids[0], ids[0]]
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_pending_actions WHERE task_id=? AND state='pending'", (task_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 1
        assert sum(e.kind == "terminal_approval_pending" for e in kb.list_events(conn, task_id=task_id)) == 1


def test_parallel_approval_grants_exactly_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command="parallel-approve-marker",
                                          summary="ignored", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.block_task(conn, task_id, kind="needs_input", reason="approval", expected_run_id=task.current_run_id)
    barrier, results, errors = threading.Barrier(2), [], []
    def approve() -> None:
        try:
            with kb.connect() as conn:
                barrier.wait()
                results.append(kb.approve_pending_action(conn, task_id, action.id, now=1_900_000_000))
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=approve) for _ in range(2)]
    [thread.start() for thread in threads]
    [thread.join(timeout=5) for thread in threads]
    assert not errors and sorted(results) == [False, True]
    with kb.connect() as conn:
        assert sum(e.kind == "terminal_approval_granted" for e in kb.list_events(conn, task_id=task_id)) == 1


@pytest.mark.parametrize("field,changed", [
    ("run_id", 2), ("command_hash", "b" * 64), ("mutation_kind", "different-kind"),
    ("profile", "identity-profile-two"), ("workspace", "/tmp/identity-workspace-two"),
    ("board_identity", "/tmp/other-board.db"),
])
def test_pending_action_fingerprint_identity_fields_and_ttl(field, changed) -> None:
    base = dict(board_identity="/tmp/identity-board.db", task_id="identity-task", run_id=1,
                command_hash="a" * 64, mutation_kind="terminal-command", profile="identity-profile-one",
                workspace="/tmp/identity-workspace-one")
    original = kb._pending_action_fingerprint(**base)
    assert original == kb._pending_action_fingerprint(**base, expires_at=9_999_999)
    varied = dict(base)
    varied[field] = changed
    assert kb._pending_action_fingerprint(**varied) != original


def test_public_attention_and_events_do_not_leak_exact_bindings(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    raw = "RAW_COMMAND_SECRET_MARKER_8a2f"
    profile = "PROFILE_SECRET_MARKER_8a2f"
    workspace = str((isolated_board / "WORKSPACE_SECRET_MARKER_8a2f").resolve())
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=raw,
                                          summary="ignored", profile=profile, workspace=workspace, expires_at=2_000_000_000)
        attention = kb.get_current_attention(conn, task_id)
        public = repr(attention) + repr(kb.list_events(conn, task_id=task_id))
        for marker in (raw, action.command_hash, action.fingerprint, profile, workspace):
            assert marker not in public


def test_expiry_boundary_archive_and_workspace_drift_fail_closed(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb.time, "time", lambda: 50)
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command=command,
            summary="git push", profile="backend-eng", workspace=str(isolated_board),
            expires_at=100,
        )
        assert not kb.approve_pending_action(conn, task_id, action.id, now=100)
        assert kb.get_pending_action(conn, task_id, now=100) is None
        assert kb.archive_task(conn, task_id)
        assert not kb.approve_pending_action(conn, task_id, action.id, now=99)


def test_secret_like_text_is_not_persisted_in_summary_or_events(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    secret = "password=" + "supersecretmustberedacted"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command=f"curl -H 'Authorization: token {secret}' https://example.invalid",
            summary=f"publish with token {secret}", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        rows = conn.execute(
            "SELECT summary, command_hash FROM task_pending_actions WHERE task_id=?",
            (task_id,),
        ).fetchall()
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=?", (task_id,),
        ).fetchall()
        assert secret not in repr(rows)
        assert secret not in repr(events)
        assert action.command_hash not in repr(events)
        assert action.fingerprint not in repr(events)


def test_stale_triage_projection_cannot_specify_or_auto_decompose(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_decompose, kanban_specify
    from plugins.kanban.dashboard.plugin_api import _set_status_direct

    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="git push",
            profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="approval required",
            expected_run_id=task.current_run_id,
        )
        assert not _set_status_direct(conn, task_id, "triage")
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (task_id,))
        conn.commit()
        assert not kb.specify_triage_task(conn, task_id, body="new body")
        assert kb.decompose_triage_task(
            conn, task_id, root_assignee="pm",
            children=[{"title": "must not exist", "assignee": "backend-eng"}],
        ) is None
        child_count = conn.execute(
            "SELECT COUNT(*) FROM task_links WHERE parent_id=?", (task_id,),
        ).fetchone()[0]
        assert child_count == 0

    assert task_id not in kanban_specify.list_triage_ids()
    assert task_id not in kanban_decompose.list_triage_ids()
    outcome = kanban_decompose.decompose_task(task_id)
    assert not outcome.ok
    assert "terminal action" in outcome.reason


def test_pending_action_block_preserves_latest_human_guidance(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic",
            summary="git push", profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        assert kb.block_task(
            conn, task_id, reason="approval required", kind="needs_input",
            expected_run_id=task.current_run_id,
            human_summary="Exact publication approval is pending.",
            human_action="Approve the displayed action, then resume.",
        )
        blocked = [
            event for event in kb.list_events(conn, task_id=task_id)
            if event.kind == "blocked"
        ][-1]
        assert blocked.payload["human_summary"] == "Exact publication approval is pending."
        assert blocked.payload["human_action"] == "Approve the displayed action, then resume."


def test_package1_attention_projection_and_terminal_states(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command=command,
            summary="ignored", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        attention = kb.get_current_attention(conn, task_id, now=1)
        assert attention is not None
        assert attention.action_id == action.id and attention.type == "exact_action"
        assert attention.summary == kb.PENDING_ACTION_OPERATOR_SUMMARY
        assert action.command_hash not in repr(attention)
        assert kb.resolve_pending_action(conn, task_id, action.id, now=2)
        assert kb.get_current_attention(conn, task_id, now=2) is None
        assert not kb.approve_pending_action(conn, task_id, action.id, now=3)


# Package 0A — desired core contracts for the approval/attention repair.
# These are deliberately red against ddacab180: they describe the atomic
# operator-attention lifecycle that the repair package must introduce.
# Package-2 test gap: this baseline has no existing combined core transition
# with a fault-injection seam, so rollback atomicity cannot be asserted without
# inventing a production API. Cover it when Package 2 exposes that transition.


def test_duplicate_exact_action_after_one_second_keeps_same_approval_id(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    now = {"value": 1_900_000_000}
    monkeypatch.setattr(kb.time, "time", lambda: now["value"])
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        first = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command=command,
            summary="publish", profile="backend-eng", workspace=str(isolated_board),
            expires_at=now["value"] + 86400,
        )
        now["value"] += 2
        second = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command=command,
            summary="publish", profile="backend-eng", workspace=str(isolated_board),
            expires_at=now["value"] + 86400,
        )
    # An approval identity is the exact action, not its refreshed deadline.
    # Repair may retain or extend expires_at, but must not rotate ID/fingerprint.
    assert second.id == first.id
    assert second.fingerprint == first.fingerprint
    assert second.expires_at >= first.expires_at


def _configure_terminal_pending_guard(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    from tools import approval

    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(
        approval, "detect_dangerous_command",
        lambda value: (True, "git_force_push", "force push")
        if value == command else (False, None, None),
    )


def test_terminal_pending_action_atomically_ends_run_blocks_and_surfaces_one_attention(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    _configure_terminal_pending_guard(monkeypatch, command)
    smart_llm = Mock()
    monkeypatch.setattr(approval, "_smart_approve", smart_llm)

    result = approval.check_all_command_guards(command, "local")
    assert result["status"] == "pending_approval"
    smart_llm.assert_not_called()
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked"
        assert task.current_run_id is None
        run = kb.latest_run(conn, task_id)
        assert run is not None and run.status == run.outcome == "blocked" and run.ended_at is not None
        assert (task.claim_lock, task.claim_expires, task.worker_pid) == (None, None, None)
        assert (run.claim_lock, run.claim_expires, run.worker_pid) == (None, None, None)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_pending_actions WHERE task_id=? AND state='pending' "
            "AND consumed_at IS NULL AND cancelled_at IS NULL", (task_id,),
        ).fetchone()[0] == 1
        attention_rows = conn.execute(
            "SELECT action_id FROM task_attentions WHERE task_id=? AND type='exact_action'", (task_id,),
        ).fetchall()
        assert len(attention_rows) == 1
        events = kb.list_events(conn, task_id=task_id)
        pending = [event for event in events if event.kind == "terminal_approval_pending"]
        blocked = [event for event in events if event.kind == "blocked"]
        assert len(pending) == len(blocked) == 1
        assert pending[0].payload["action_id"] == blocked[0].payload["action_id"] == attention_rows[0]["action_id"]


def test_atomic_pending_action_clears_claims_and_cannot_be_reclaimed_or_claimed(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        origin_run_id = task.current_run_id
        # These fields are in-memory claim-spawn metadata. block_task does not
        # clear them, so this equivalent terminal transition must not either.
        task.worker_session_id = "worker-session-must-survive"
        task.resume_requested = True
        conn.execute(
            "UPDATE tasks SET claim_lock=?, claim_expires=?, worker_pid=? WHERE id=?",
            ("claimed-before-terminal-action", 1, 424242, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET claim_lock=?, claim_expires=?, worker_pid=? WHERE id=?",
            ("claimed-before-terminal-action", 1, 424242, origin_run_id),
        )
        result = kb.record_pending_action_and_block(
            conn, task_id=task_id, run_id=origin_run_id, command=command,
            summary="untrusted", profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        current, run = kb.get_task(conn, task_id), kb.latest_run(conn, task_id)
        assert current is not None and current.status == "blocked" and current.current_run_id is None
        assert (current.claim_lock, current.claim_expires, current.worker_pid) == (None, None, None)
        assert run is not None and run.id == origin_run_id
        assert run.status == run.outcome == "blocked" and run.ended_at is not None
        assert (run.claim_lock, run.claim_expires, run.worker_pid) == (None, None, None)
        assert task.worker_session_id == "worker-session-must-survive"
        assert task.resume_requested is True
        assert kb.claim_task(conn, task_id, claimer="bypass") is None
        assert kb.recompute_ready(conn) == 0
        assert kb.release_stale_claims(conn, signal_fn=lambda *_args, **_kwargs: None) == 0
        final = kb.get_task(conn, task_id)
        assert final is not None and final.status == "blocked" and final.current_run_id is None
        assert kb.get_pending_action(conn, task_id) is not None
        attention = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert attention is not None and attention.id == result["attention_id"]


def test_atomic_guard_result_and_durable_payloads_are_redacted(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    raw_command = "git push --force-with-lease=refs/heads/topic:abcdef1 RAW_COMMAND_MARKER fork HEAD:topic"
    profile_marker = "PROFILE_MARKER_UNIQUE"
    workspace = isolated_board / "ABSOLUTE_WORKSPACE_MARKER"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_PROFILE", profile_marker)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        conn.execute("UPDATE tasks SET assignee=? WHERE id=?", (profile_marker, task_id))
        conn.execute("UPDATE task_runs SET profile=? WHERE id=?", (profile_marker, task.current_run_id))
    _configure_terminal_pending_guard(monkeypatch, raw_command)

    result = approval.check_all_command_guards(raw_command, "local")
    assert result["status"] == "pending_approval"
    assert set(result["kanban_approval"]) == {"action_id", "attention_id", "attention_status", "reused"}
    assert result["kanban_approval"]["attention_status"] == "pending"
    with kb.connect() as conn:
        action = kb.get_pending_action(conn, task_id)
        attention = kb.get_current_attention(conn, task_id)
        events = kb.list_events(conn, task_id=task_id)
        assert action is not None and attention is not None
        durable_public = json.dumps(
            {
                "kanban_approval": result["kanban_approval"],
                "attention": repr(attention),
                "events": [{"kind": event.kind, "payload": event.payload} for event in events],
            },
            sort_keys=True,
        )
        for secret in (raw_command, action.command_hash, action.fingerprint, profile_marker, str(workspace.resolve())):
            assert secret not in durable_public
        pending = next(event for event in events if event.kind == "terminal_approval_pending")
        blocked = next(event for event in events if event.kind == "blocked")
        assert set(pending.payload) == {"action_id", "mutation_kind", "summary", "expires_at"}
        assert set(blocked.payload) == {"kind", "reason", "action_id"}


def test_guard_persistence_failure_never_returns_pending_or_queues_request(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    _configure_terminal_pending_guard(monkeypatch, command)
    queued = Mock()
    monkeypatch.setattr(approval, "submit_pending", queued)
    monkeypatch.setattr(approval, "_record_kanban_pending_action", lambda *_args, **_kwargs: None)

    result = approval.check_all_command_guards(command, "local")
    assert result["status"] == "blocked"
    assert result.get("approval_pending") is not True
    queued.assert_not_called()
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert task is not None and task.status == "running" and task.current_run_id is not None
        assert run is not None and run.status == "running" and run.ended_at is None
        assert conn.execute("SELECT COUNT(*) FROM task_pending_actions WHERE task_id=?", (task_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 0
        assert not [event for event in kb.list_events(conn, task_id=task_id) if event.kind in {"terminal_approval_pending", "blocked"}]


def test_guard_response_crash_boundary_leaves_actionable_blocked_operator_entity(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning pending_approval is the simulated crash boundary: no caller cleanup."""
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    _configure_terminal_pending_guard(monkeypatch, command)
    assert approval.check_all_command_guards(command, "local")["status"] == "pending_approval"

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked"
        assert kb.get_pending_action(conn, task_id) is not None
        assert any(
            event.kind == "blocked" and (event.payload or {})["kind"] == "needs_input"
            for event in kb.list_events(conn, task_id=task_id)
        )


def test_atomic_pending_action_wrong_current_run_is_complete_global_noop(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    monkeypatch.setattr(kb.time, "time", lambda: 1_000)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task and task.current_run_id
        origin_run_id = task.current_run_id
        other_task = kb.create_task(conn, title="foreign origin", assignee="backend-eng")
        foreign = kb.claim_task(conn, other_task, claimer="foreign-worker")
        assert foreign and foreign.current_run_id
        expiry_task = kb.create_task(conn, title="independent expiry", assignee="backend-eng")
        expired_origin = kb.claim_task(conn, expiry_task, claimer="expiry-worker")
        assert expired_origin and expired_origin.current_run_id
        expired_action = kb.record_pending_action(
            conn, task_id=expiry_task, run_id=expired_origin.current_run_id,
            command="git push --force-with-lease=refs/heads/expiry:abcdef1 fork HEAD:expiry",
            summary="untrusted", profile="backend-eng", workspace=str(isolated_board), expires_at=1_001,
        )
        before = _atomic_snapshot(conn, task_id, origin_run_id)
        expiry_before = _atomic_snapshot(conn, expiry_task, expired_origin.current_run_id)
        materialize = Mock(wraps=kb._materialize_expired_actions)
        hook = Mock()
        monkeypatch.setattr(kb, "_materialize_expired_actions", materialize)
        monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", hook)
        monkeypatch.setattr(kb.time, "time", lambda: 1_002)
        with pytest.raises(ValueError, match="stale"):
            kb.record_pending_action_and_block(conn, task_id=task_id, run_id=foreign.current_run_id,
                command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="untrusted", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert _atomic_snapshot(conn, task_id, origin_run_id) == before
        assert _atomic_snapshot(conn, expiry_task, expired_origin.current_run_id) == expiry_before
        assert conn.execute("SELECT * FROM task_pending_actions WHERE id=?", (expired_action.id,)).fetchone()["state"] == "pending"
    materialize.assert_not_called()
    hook.assert_not_called()


def test_atomic_pending_action_fault_after_persist_rolls_back_everything(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    lifecycle = Mock()
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", lifecycle)
    monkeypatch.setattr(kb, "_pending_action_after_persist_hook", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task and task.current_run_id
        origin_run_id = task.current_run_id
        before = _atomic_snapshot(conn, task_id, origin_run_id)
        with pytest.raises(RuntimeError, match="boom"):
            kb.record_pending_action_and_block(conn, task_id=task_id, run_id=origin_run_id,
                command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="untrusted", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert _atomic_snapshot(conn, task_id, origin_run_id) == before
    lifecycle.assert_not_called()


def test_atomic_pending_action_identical_retry_reuses_ids_without_extra_events(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task and task.current_run_id
        origin_run_id = task.current_run_id
        first = kb.record_pending_action_and_block(conn, task_id=task_id, run_id=origin_run_id, command=command, summary="untrusted", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        before_retry = _atomic_snapshot(conn, task_id, origin_run_id)
        second = kb.record_pending_action_and_block(conn, task_id=task_id, run_id=origin_run_id, command=command, summary="untrusted", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert second["action_id"] == first["action_id"] and second["attention_id"] == first["attention_id"]
        assert first["reused"] is False and second["reused"] is True
        assert _atomic_snapshot(conn, task_id, origin_run_id) == before_retry


def test_atomic_pending_action_block_hook_fires_once_after_commit_and_retry_is_silent(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    observed: list[tuple[str, str, dict[str, object]]] = []

    def capture(event: str, hooked_task_id: str, **fields: object) -> None:
        with kb.connect() as observer:
            task = kb.get_task(observer, hooked_task_id)
            run = kb.latest_run(observer, hooked_task_id)
            action = kb.get_pending_action(observer, hooked_task_id)
            attention = kb.get_current_attention(observer, hooked_task_id, now=1_900_000_000)
        assert task is not None and task.status == "blocked" and task.current_run_id is None
        assert run is not None and run.status == run.outcome == "blocked" and run.ended_at is not None
        assert action is not None and action.state == "pending"
        assert attention is not None and attention.action_id == action.id
        observed.append((event, hooked_task_id, fields))

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", capture)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        origin_run_id = task.current_run_id
        first = kb.record_pending_action_and_block(
            conn, task_id=task_id, run_id=origin_run_id, command=command, summary="untrusted",
            profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        second = kb.record_pending_action_and_block(
            conn, task_id=task_id, run_id=origin_run_id, command=command, summary="untrusted",
            profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000,
        )
    assert second == {**first, "reused": True}
    assert len(observed) == 1
    event, hooked_task_id, fields = observed[0]
    assert (event, hooked_task_id) == ("kanban_task_blocked", task_id)
    assert "assignee" not in fields
    assert fields["run_id"] == origin_run_id
    assert fields["reason"] == "terminal_approval_required"


def test_atomic_pending_action_parallel_calls_converge_without_partial_state(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task and task.current_run_id
        origin_run_id = task.current_run_id
    barrier, results, failures = threading.Barrier(2), [], []
    def call() -> None:
        try:
            with kb.connect() as conn:
                barrier.wait(timeout=5)
                results.append(kb.record_pending_action_and_block(conn, task_id=task_id, run_id=origin_run_id, command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="untrusted", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000))
        except BaseException as exc:
            failures.append(exc)
    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not failures and len(results) == 2
    assert {result["reused"] for result in results} == {False, True}
    assert len({result["action_id"] for result in results}) == len({result["attention_id"] for result in results}) == 1
    with kb.connect() as conn:
        task, run = kb.get_task(conn, task_id), kb.latest_run(conn, task_id)
        assert task and task.status == "blocked" and task.current_run_id is None
        assert run and run.id == origin_run_id and run.status == run.outcome == "blocked" and run.ended_at
        assert conn.execute("SELECT COUNT(*) FROM task_pending_actions WHERE task_id=? AND state IN ('pending','approved')", (task_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=? AND type='exact_action'", (task_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='terminal_approval_pending'", (task_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='blocked'", (task_id,)).fetchone()[0] == 1


def test_human_gate_token_survives_other_unblock_precondition_failure(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command=command,
            summary="publish", profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="approval required",
            expected_run_id=task.current_run_id, human_gate=True,
        )
        token = kb.issue_gate_token(conn, task_id, action="unblock")
        assert token
        assert kb.unblock_task(conn, task_id, token=token) is False
        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id=?", (task_id,),
        ).fetchone()
        assert row["gate_token_hash"] == kb.hash_gate_token(token)


def test_same_persistent_cause_fingerprint_increments_recurrence_chain(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="need credential choice",
                             attention_type="decision", reason_code="credential_choice",
                             expected_run_id=task.current_run_id)
        attention = kb.get_current_attention(conn, task_id)
        assert attention is not None
        assert kb.transition_task_status_with_attention(
            conn, task_id=task_id, status="ready", expected_attention_id=attention.id,
            expected_attention_version=attention.version,
        ).status == "transitioned"
        assert kb.claim_task(conn, task_id, claimer="second-worker") is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="need credential choice",
                             attention_type="decision", reason_code="credential_choice")
        current = kb.get_task(conn, task_id)
        assert current is not None and current.block_recurrences == 2


def test_different_persistent_cause_fingerprint_starts_new_recurrence_chain(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="need credential choice",
                             attention_type="decision", reason_code="credential_choice",
                             expected_run_id=task.current_run_id)
        attention = kb.get_current_attention(conn, task_id)
        assert attention is not None
        assert kb.transition_task_status_with_attention(
            conn, task_id=task_id, status="ready", expected_attention_id=attention.id,
            expected_attention_version=attention.version,
        ).status == "transitioned"
        assert kb.claim_task(conn, task_id, claimer="second-worker") is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="need publication approval",
                             attention_type="decision", reason_code="publication_approval")
        current = kb.get_task(conn, task_id)
        # This is deliberately exact persistent-cause identity, not NLP similarity.
        assert current is not None and current.block_recurrences == 1


def test_pending_action_goal_finalizer_keeps_single_current_operator_attention(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real CLI goal-finalizer must not add generic attention to approval."""
    import cli as cli_mod
    from hermes_cli import goals

    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic",
            summary="publish", profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000,
        )

    # Drive cli.py's actual run_kanban_goal_loop closeout route: a first DONE
    # verdict produces the finalizer prompt; the second detects no lifecycle
    # call and invokes the real _block closure.
    monkeypatch.setattr(
        goals, "judge_goal", lambda *_args, **_kwargs: ("done", "ready to finalize", False, None),
    )
    fake_cli = SimpleNamespace(
        agent=SimpleNamespace(
            session_id="goal-finalizer-test",
            run_conversation=lambda **_kwargs: {"final_response": "still open"},
        ),
        conversation_history=[],
        session_id="goal-finalizer-test",
    )
    cli_mod._run_kanban_goal_loop_q(fake_cli, "work is complete")  # type: ignore[arg-type]

    with kb.connect() as conn:
        attention_kinds = {"terminal_approval_pending", "blocked", "block_loop_detected"}
        assert len([
            event for event in kb.list_events(conn, task_id=task_id)
            if event.kind in attention_kinds
        ]) == 1



def test_legacy_null_cause_fingerprint_never_continues_recurrence(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        conn.execute("UPDATE tasks SET block_recurrences=99, block_cause_fingerprint=NULL WHERE id=?", (task_id,))
        assert kb.block_task(conn, task_id, kind="needs_input", reason="credential",
                             attention_type="decision", reason_code="credential_choice",
                             expected_run_id=task.current_run_id)
        row = conn.execute("SELECT block_recurrences, block_cause_fingerprint, block_reason_code FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["block_recurrences"] == 1 and row["block_cause_fingerprint"] and row["block_reason_code"] == "credential_choice"


def test_cause_transition_fault_rolls_back_counter_and_fingerprint(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        before = dict(conn.execute("SELECT status, block_recurrences, block_cause_fingerprint FROM tasks WHERE id=?", (task_id,)).fetchone())
        events_before = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0]
        monkeypatch.setattr(kb, "_block_cause_after_task_update_hook", lambda: (_ for _ in ()).throw(RuntimeError("fault")))
        with pytest.raises(RuntimeError, match="fault"):
            kb.block_task(conn, task_id, kind="needs_input", reason="credential",
                          attention_type="decision", reason_code="credential_choice",
                          expected_run_id=task.current_run_id)
        assert dict(conn.execute("SELECT status, block_recurrences, block_cause_fingerprint FROM tasks WHERE id=?", (task_id,)).fetchone()) == before
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0] == events_before


def test_real_legacy_tasks_schema_migrates_cause_columns_idempotently(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A representative old blocked row upgrades through the public initializer."""
    db_path = Path(os.environ["HERMES_KANBAN_DB"])
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy blocked", assignee="backend-eng")
        conn.execute("UPDATE tasks SET status='blocked', block_kind='needs_input', block_recurrences=41 WHERE id=?", (task_id,))
        conn.execute("ALTER TABLE tasks RENAME TO tasks_modern")
        legacy_schema = kb.SCHEMA_SQL[kb.SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS tasks"):kb.SCHEMA_SQL.index("CREATE TABLE IF NOT EXISTS task_links")]
        legacy_schema = legacy_schema.replace("    block_cause_fingerprint TEXT,\n    block_reason_code     TEXT,\n    block_cause_version   INTEGER,\n", "")
        conn.executescript(legacy_schema)
        modern_cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks_modern)")}
        legacy_cols = [row["name"] for row in conn.execute("PRAGMA table_info(tasks)")]
        fields = ", ".join(col for col in legacy_cols if col in modern_cols)
        conn.execute(f"INSERT INTO tasks ({fields}) SELECT {fields} FROM tasks_modern")
        conn.execute("DROP TABLE tasks_modern")
        assert {"block_cause_fingerprint", "block_reason_code", "block_cause_version"}.isdisjoint({row["name"] for row in conn.execute("PRAGMA table_info(tasks)")})
    kb.init_db(db_path)
    with kb.connect() as conn:
        first = tuple(conn.execute("SELECT status, block_recurrences, block_cause_fingerprint, block_reason_code, block_cause_version FROM tasks WHERE id=?", (task_id,)).fetchone())
        assert first == ("blocked", 41, None, None, None)
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    kb.init_db(db_path)
    with kb.connect() as conn:
        assert tuple(conn.execute("SELECT status, block_recurrences, block_cause_fingerprint, block_reason_code, block_cause_version FROM tasks WHERE id=?", (task_id,)).fetchone()) == first
        assert kb.unblock_task(conn, task_id)
        claimed = kb.claim_task(conn, task_id, claimer="migrated-worker")
        assert claimed is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="migration secret must not persist",
                             attention_type="decision", reason_code="credential_choice",
                             cause_scope={"required_decision": "credential"}, expected_run_id=claimed.current_run_id)
        migrated = kb.get_task(conn, task_id)
        cause = conn.execute("SELECT block_cause_fingerprint, block_reason_code FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert migrated is not None and migrated.status == "blocked" and migrated.block_recurrences == 1
        assert cause["block_cause_fingerprint"] and cause["block_reason_code"] == "credential_choice"


def test_parallel_same_typed_cause_has_one_transition_and_one_event(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        run_id = kb.get_task(conn, task_id).current_run_id
    barrier = threading.Barrier(2)
    results: list[bool] = []
    errors: list[Exception] = []

    def blocker() -> None:
        try:
            with kb.connect() as other:
                barrier.wait(timeout=5)
                results.append(kb.block_task(other, task_id, kind="needs_input", reason="SECRET_NEVER_PUBLIC", attention_type="decision", reason_code="credential_choice", cause_scope={"required_decision": "credential"}, expected_run_id=run_id))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=blocker) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=10)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert sorted(results) == [False, True]
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.block_recurrences == 1
        assert [event.kind for event in kb.list_events(conn, task_id=task_id) if event.kind == "blocked"] == ["blocked"]


def test_finalizer_write_lock_blocks_concurrent_action_until_fallback_commits(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two connections prove the fallback read/transition has no action-creation gap."""
    task_id = _create_running_task(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original_append = kb._append_event

    def pause_after_fallback(conn, event_task_id, kind, payload, **kwargs):
        result = original_append(conn, event_task_id, kind, payload, **kwargs)
        if event_task_id == task_id and kind == "blocked":
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(kb, "_append_event", pause_after_fallback)
    finalizer: list[bool] = []
    contender: list[object] = []

    def closeout() -> None:
        with kb.connect() as conn:
            finalizer.append(kb.finalize_goal_block_or_reuse_current_attention(conn, task_id, reason="SECRET_NEVER_PUBLIC"))

    attempt, finished = threading.Event(), threading.Event()

    def create_action() -> None:
        assert entered.wait(5)
        try:
            with kb.connect() as conn:
                task = kb.get_task(conn, task_id)
                attempt.set()  # immediately before the competing write call
                contender.append(kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="x", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000))
        except Exception as exc:
            contender.append(exc)
        finally:
            finished.set()

    first, second = threading.Thread(target=closeout), threading.Thread(target=create_action)
    first.start(); assert entered.wait(5); second.start(); assert attempt.wait(5)
    # The contender has reached its write attempt but cannot complete while the
    # finalizer owns BEGIN IMMEDIATE; this is an interleave proof, not a sleep.
    assert not finished.wait(0.2) and contender == []
    release.set()
    first.join(10); second.join(10)
    assert finalizer == [True] and not first.is_alive() and not second.is_alive()
    assert finished.is_set() and len(contender) == 1
    assert isinstance(contender[0], Exception)
    with kb.connect() as conn:
        assert kb.get_pending_action(conn, task_id) is None
        assert [event.kind for event in kb.list_events(conn, task_id=task_id) if event.kind in {"blocked", "terminal_approval_pending"}] == ["blocked"]


def test_goal_finalizer_exact_action_precedence_retry_is_event_silent(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    observed: list[dict[str, object]] = []

    def hook(event: str, hooked_task_id: str, **fields: object) -> None:
        # Fresh connection proves callbacks run only after the commit.
        run_id = fields["run_id"]
        assert isinstance(run_id, int)
        with kb.connect() as fresh:
            task = kb.get_task(fresh, hooked_task_id)
            run = kb.get_run(fresh, run_id)
            attention = kb.get_current_attention(fresh, hooked_task_id)
            action = kb.get_pending_action(fresh, hooked_task_id)
        assert task is not None and task.status == "blocked" and task.current_run_id is None
        assert run is not None and run.status == run.outcome == "blocked" and run.ended_at is not None
        assert action is not None and attention is not None and attention.action_id == action.id
        observed.append({"event": event, "task_id": hooked_task_id, **fields})

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", hook)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        origin_run_id = task.current_run_id
        action = kb.record_pending_action(conn, task_id=task_id, run_id=origin_run_id, command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="x", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.finalize_goal_block_or_reuse_current_attention(conn, task_id, reason="SECRET_NEVER_PUBLIC")
        attention_id = kb.get_current_attention(conn, task_id).id
        before = _atomic_snapshot(conn, task_id, origin_run_id)
        assert kb.finalize_goal_block_or_reuse_current_attention(conn, task_id, reason="SECRET_NEVER_PUBLIC")
        assert _atomic_snapshot(conn, task_id, origin_run_id) == before
        attention = kb.get_current_attention(conn, task_id)
        assert attention.action_id == action.id and attention.id == attention_id
    assert observed == [{"event": "kanban_task_blocked", "task_id": task_id,
                         "board": kb.get_current_board(), "assignee": "backend-eng",
                         "run_id": origin_run_id,
                         "reason": "terminal_approval_required"}]


def test_typed_cause_redacts_reason_from_event_hook_and_public_projection(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    markers = ("CAUSE_SECRET_MARKER", "PROFILE_MARKER", "WORKSPACE_MARKER")
    raw_reason = " ".join(markers)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", lambda _event, _task, **fields: captured.append(fields))
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason=raw_reason,
            attention_type="decision", reason_code="credential_choice",
            cause_scope={"required_decision": "credential"}, expected_run_id=task.current_run_id)
        latest = kb.latest_run(conn, task_id)
        raw_runs = conn.execute("SELECT summary, error, metadata FROM task_runs WHERE task_id=?", (task_id,)).fetchall()
        public = (
            repr(kb.get_task(conn, task_id)) + repr(kb.get_current_attention(conn, task_id))
            + repr(kb.list_events(conn, task_id=task_id)) + repr(captured)
            + repr(latest) + repr(kb.latest_summary(conn, task_id)) + repr([tuple(row) for row in raw_runs])
        )
        event = next(e for e in kb.list_events(conn, task_id=task_id) if e.kind == "blocked")
        assert latest is not None and latest.summary == "credential_choice"
        assert kb.latest_summary(conn, task_id) == "credential_choice"
        assert len(raw_runs) == 1 and raw_runs[0]["summary"] == "credential_choice"
        assert set(event.payload or {}) == {"kind", "attention_type", "reason_code", "recurrences"}
    assert all(marker not in public for marker in markers)
    assert captured == [{"board": kb.get_current_board(), "assignee": "backend-eng", "run_id": task.current_run_id, "reason": "credential_choice"}]


def test_typed_dependency_cause_redacts_reason_from_event_hook_and_public_projection(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    markers = ("DEPENDENCY_SECRET_MARKER", "DEPENDENCY_PROFILE_MARKER", "DEPENDENCY_WORKSPACE_MARKER")
    raw_reason = " ".join(markers)
    captured: list[dict[str, object]] = []

    def hook(_event: str, _task_id: str, **fields: object) -> None:
        # A new connection observes committed task/run/event state, not caller state.
        with kb.connect() as fresh:
            task = kb.get_task(fresh, task_id)
            run = kb.latest_run(fresh, task_id)
            attention = kb.get_current_attention(fresh, task_id)
            events = kb.list_events(fresh, task_id=task_id)
            rows = fresh.execute("SELECT summary, error, metadata FROM task_runs WHERE task_id=?", (task_id,)).fetchall()
        captured.append({**fields, "task": task, "run": run, "attention": attention,
                         "events": events, "rows": [tuple(row) for row in rows]})

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", hook)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        origin_run_id = task.current_run_id
        assert kb.block_task(conn, task_id, kind="dependency", reason=raw_reason,
                             attention_type="decision", reason_code="credential_choice",
                             cause_scope={"required_decision": "credential"},
                             expected_run_id=origin_run_id)
        latest = kb.latest_run(conn, task_id)
        event = next(e for e in kb.list_events(conn, task_id=task_id) if e.kind == "dependency_wait")
        public = repr(kb.get_task(conn, task_id)) + repr(latest) + repr(kb.latest_summary(conn, task_id)) + repr(kb.get_current_attention(conn, task_id)) + repr(kb.list_events(conn, task_id=task_id)) + repr(captured)
        assert latest is not None and latest.summary == "credential_choice"
        assert kb.latest_summary(conn, task_id) == "credential_choice"
        assert kb.get_current_attention(conn, task_id) is None
        assert event.payload == {"kind": "dependency", "attention_type": "decision",
                                 "reason_code": "credential_choice", "recurrences": 0}
    assert all(marker not in public for marker in markers)
    assert len(captured) == 1
    assert captured[0]["reason"] == "credential_choice"


def test_typed_cause_scope_rejects_unknown_values_and_different_threshold_starts_one(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        with pytest.raises(ValueError, match="scope"):
            kb.block_task(conn, task_id, kind="needs_input", reason="safe", attention_type="decision",
                reason_code="credential_choice", cause_scope={"required_decision": "PROFILE_MARKER"})
        assert kb.block_task(conn, task_id, kind="needs_input", reason="safe", attention_type="decision",
            reason_code="credential_choice", cause_scope={"required_decision": "credential"}, expected_run_id=task.current_run_id)
        conn.execute("UPDATE tasks SET block_recurrences=? WHERE id=?", (kb.BLOCK_RECURRENCE_LIMIT - 1, task_id))
        attention = kb.get_current_attention(conn, task_id)
        assert attention is not None
        assert kb.transition_task_status_with_attention(
            conn, task_id=task_id, status="ready", expected_attention_id=attention.id,
            expected_attention_version=attention.version,
        ).status == "transitioned"
        assert kb.claim_task(conn, task_id, claimer="again") is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="safe", attention_type="decision",
            reason_code="publication_approval", cause_scope={"required_decision": "publication"})
        current = kb.get_task(conn, task_id)
        assert current is not None and current.status == "blocked" and current.block_recurrences == 1


def test_goal_finalizer_stale_origin_is_event_silent_and_exact_retry_is_silent(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    hooks: list[object] = []
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", lambda *_args, **_kwargs: hooks.append(True))
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="x",
            profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        # Binding no longer matches current run: finalizer must not fallback.
        other = conn.execute("INSERT INTO task_runs (task_id,status,started_at,profile) VALUES (?, 'running', 1, 'backend-eng')", (task_id,)).lastrowid
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (other, task_id))
        before = _atomic_snapshot(conn, task_id, other)
        assert not kb.finalize_goal_block_or_reuse_current_attention(conn, task_id, reason="CAUSE_SECRET_MARKER")
        assert _atomic_snapshot(conn, task_id, other) == before
        assert kb.get_current_attention(conn, task_id) is not None
        assert action.id
    assert hooks == []


def test_goal_finalizer_fallback_closes_origin_run_redacts_and_hooks_after_commit(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    marker = "FINALIZER_SECRET_PROFILE_WORKSPACE"
    observed: list[tuple[object, object, object]] = []

    def hook(_event, hooked_task_id, **fields):
        # A fresh connection is the committed-state contract, not a view of the
        # finalizer's still-open transaction.
        with kb.connect() as fresh:
            task = kb.get_task(fresh, hooked_task_id)
            run = kb.get_run(fresh, fields["run_id"])
            observed.append((task.status if task else None, run.outcome if run else None, kb.latest_summary(fresh, hooked_task_id)))

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", hook)
    with kb.connect() as conn:
        origin = kb.get_task(conn, task_id).current_run_id
        assert origin is not None
        assert kb.finalize_goal_block_or_reuse_current_attention(conn, task_id, reason=marker)
        task = kb.get_task(conn, task_id)
        runs = conn.execute("SELECT id, status, outcome, summary, ended_at FROM task_runs WHERE task_id=?", (task_id,)).fetchall()
        public = repr(task) + repr(kb.latest_run(conn, task_id)) + repr(kb.latest_summary(conn, task_id)) + repr(kb.list_events(conn, task_id=task_id))
        assert task is not None and task.current_run_id is None and task.status == "blocked"
        assert len(runs) == 1 and runs[0]["id"] == origin and runs[0]["ended_at"] is not None
        assert runs[0]["status"] == runs[0]["outcome"] == "blocked" and runs[0]["summary"] == "goal_closeout_missing"
        assert marker not in public
    assert observed == [("blocked", "blocked", "goal_closeout_missing")]


def test_goal_finalizer_fault_rolls_back_everything_and_is_hook_silent(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    hooks: list[object] = []
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", lambda *_args, **_kwargs: hooks.append(True))
    with kb.connect() as conn:
        origin = kb.get_task(conn, task_id).current_run_id
        assert origin is not None
        before = _atomic_snapshot(conn, task_id, origin)
        monkeypatch.setattr(kb, "_goal_finalizer_after_task_update_hook", lambda: (_ for _ in ()).throw(RuntimeError("finalizer fault")))
        with pytest.raises(RuntimeError, match="finalizer fault"):
            kb.finalize_goal_block_or_reuse_current_attention(conn, task_id, reason="never durable")
        assert _atomic_snapshot(conn, task_id, origin) == before
    assert hooks == []


def test_current_attention_excludes_done_archived_and_expired_tasks(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic", summary="x",
            profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.get_current_attention(conn, task_id)
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        assert kb.get_current_attention(conn, task_id) is None
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (task_id,))
        assert kb.get_current_attention(conn, task_id) is None
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
        conn.execute("UPDATE task_pending_actions SET expires_at=1 WHERE id=?", (action.id,))
        assert kb.get_current_attention(conn, task_id, now=2) is None

def test_execute_code_and_terminal_pending_paths_create_same_durable_approval_contract(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    from tools import approval, code_execution_tool, terminal_tool

    terminal_task = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    _configure_terminal_pending_guard(monkeypatch, command)
    assert approval.check_all_command_guards(command, "local")["status"] == "pending_approval"

    with kb.connect() as conn:
        code_task = kb.create_task(conn, title="execute code", assignee="backend-eng")
        claimed = kb.claim_task(conn, code_task, claimer="code-worker")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", code_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    # Call the real public wrapper. Its real guard returns pending approval, so
    # no script is executed; mocked dispatch makes accidental execution loud.
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    monkeypatch.setattr(code_execution_tool, "_execute_remote", Mock(side_effect=AssertionError("must not execute")))
    result = json.loads(code_execution_tool.execute_code("import os; os.unlink('x')"))
    assert result["status"] == "pending_approval"
    assert result["outcome"] == "approval_required"
    assert set(result["kanban_approval"]) == {"action_id", "attention_id", "attention_status", "reused"}

    with kb.connect() as conn:
        terminal_action = kb.get_pending_action(conn, terminal_task)
        code_action = kb.get_pending_action(conn, code_task)
        terminal_state = kb.get_task(conn, terminal_task)
        code_state = kb.get_task(conn, code_task)
        terminal_run = kb.latest_run(conn, terminal_task)
        code_run = kb.latest_run(conn, code_task)
        assert terminal_action is not None
        assert code_action is not None
        # Mutation kinds can differ, but lifecycle/approval invariants cannot.
        for action, state, run in (
            (terminal_action, terminal_state, terminal_run),
            (code_action, code_state, code_run),
        ):
            assert state is not None and state.status == "blocked"
            assert state.current_run_id is None
            assert run is not None and run.outcome == "blocked"
            assert action.fingerprint
            assert action.approved_at is None and action.consumed_at is None
            assert conn.execute(
                "SELECT COUNT(*) FROM task_pending_actions WHERE task_id=? "
                "AND consumed_at IS NULL AND cancelled_at IS NULL", (action.task_id,),
            ).fetchone()[0] == 1
            assert len([
                event for event in kb.list_events(conn, task_id=action.task_id)
                if event.kind == "terminal_approval_pending"
            ]) == 1


@pytest.mark.parametrize("bypass", ["yolo", "mode_off", "session", "always", "smart"])
def test_kanban_execute_code_requires_durable_exact_grant_despite_broad_bypasses(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, bypass: str,
) -> None:
    """An arbitrary worker script is never authorized by broad policy state."""
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    code = "print('arbitrary worker script')"
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    if bypass == "yolo":
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    elif bypass == "mode_off":
        monkeypatch.setattr(approval, "_get_approval_mode", lambda: "off")
    elif bypass in {"session", "always"}:
        monkeypatch.setattr(approval, "is_approved", lambda *_args: True)
    else:
        monkeypatch.setattr(approval, "_get_approval_mode", lambda: "smart")
        monkeypatch.setattr(approval, "_smart_approve", lambda *_args: "approve")

    result = approval.check_execute_code_guard(code, "local")
    assert result["guard_outcome"] == "approval_required"
    assert result["status"] == "pending_approval"
    with kb.connect() as conn:
        action = kb.get_pending_action(conn, task_id)
        assert action is not None and action.mutation_kind == "execute-code-arbitrary"


def test_kanban_execute_code_exact_grant_is_byte_bound_and_consumed_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line endings, Unicode normalization, and whitespace are distinct scripts."""
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    code_a = "print('é')\n"
    variants = ("print('é')\r\n", "print('e\u0301')\n", "print('é')\n ")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    first = approval.check_execute_code_guard(code_a, "local")
    assert first["guard_outcome"] == "approval_required"
    action_id = first["kanban_approval"]["action_id"]
    with kb.connect() as conn:
        assert kb.approve_pending_action_and_unblock(conn, task_id, action_id)
        resumed = kb.claim_task(conn, task_id, claimer="resumed-worker")
        assert resumed is not None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(resumed.current_run_id))
    assert approval.check_execute_code_guard(code_a, "local")["guard_outcome"] == "allow"
    for variant in variants:
        assert variant.encode("utf-8") != code_a.encode("utf-8")
        assert kb._pending_action_hash(variant) != kb._pending_action_hash(code_a)
    # The exact grant was consumed; retrying A or offering B cannot reuse it.
    assert approval.check_execute_code_guard(code_a, "local")["guard_outcome"] == "approval_required"
    with kb.connect() as conn:
        action = conn.execute("SELECT state, consumed_at FROM task_pending_actions WHERE id=?", (action_id,)).fetchone()
        assert action["state"] == "consumed" and action["consumed_at"] is not None


@pytest.mark.parametrize("env_type", ["local", "ssh"])
def test_public_execute_code_pending_blocks_before_local_or_remote_spawn(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, env_type: str,
) -> None:
    """F: the real wrapper blocks before either selected dispatch seam."""
    from tools import code_execution_tool, terminal_tool

    task_id = _create_running_task(monkeypatch)
    code = "import pathlib; pathlib.Path('MUST_NOT_RUN').write_text('x')"
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": env_type})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    monkeypatch.setattr(code_execution_tool.tempfile, "mkdtemp", Mock(side_effect=AssertionError("local spawn")))
    monkeypatch.setattr(code_execution_tool, "_execute_remote", Mock(side_effect=AssertionError("remote spawn")))
    result = json.loads(code_execution_tool.execute_code(code))
    assert (result["status"], result["outcome"], result["tool_calls_made"]) == ("pending_approval", "approval_required", 0)
    assert set(result["kanban_approval"]) == {"action_id", "attention_id", "attention_status", "reused"}
    assert code not in json.dumps(result)
    with kb.connect() as conn:
        events = kb.list_events(conn, task_id=task_id)
        assert sum(e.kind == "terminal_approval_pending" for e in events) == 1
        assert sum(e.kind == "blocked" for e in events) == 1


@pytest.mark.parametrize("env_type", ["docker", "vercel_sandbox"])
def test_public_execute_code_isolated_backends_still_require_kanban_exact_grant(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, env_type: str,
) -> None:
    """Isolation shortcuts never bypass a worker's durable exact grant."""
    from tools import code_execution_tool, terminal_tool

    task_id = _create_running_task(monkeypatch)
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": env_type})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    monkeypatch.setattr(code_execution_tool, "_execute_remote", Mock(side_effect=AssertionError("must not spawn")))
    result = json.loads(code_execution_tool.execute_code("print('guarded')"))
    assert (result["status"], result["outcome"]) == ("pending_approval", "approval_required")
    with kb.connect() as conn:
        assert kb.get_pending_action(conn, task_id) is not None
        assert len([event for event in kb.list_events(conn, task_id=task_id) if event.kind == "terminal_approval_pending"]) == 1


@pytest.mark.parametrize("markers", [
    {"HERMES_KANBAN_TASK": "task-only"},
    {"HERMES_KANBAN_TASK": "task", "HERMES_KANBAN_WORKSPACE": "/workspace"},
    {"HERMES_KANBAN_TASK": "task", "HERMES_KANBAN_RUN_ID": "1"},
    {"HERMES_KANBAN_RUN_ID": "1", "HERMES_KANBAN_WORKSPACE": "/workspace"},
])
@pytest.mark.parametrize("bypass", ["yolo", "mode_off"])
def test_partial_kanban_markers_fail_closed_without_durable_side_effects(
    monkeypatch: pytest.MonkeyPatch, markers: dict[str, str], bypass: str,
) -> None:
    from tools import approval

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    for key, value in markers.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(approval, "_record_kanban_pending_action", Mock(side_effect=AssertionError("must not persist")))
    monkeypatch.setattr(approval, "_consume_kanban_action_grant", Mock(side_effect=AssertionError("must not consume")))
    if bypass == "yolo":
        monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    else:
        monkeypatch.setattr(approval, "_get_approval_mode", lambda: "off")
    result = approval.check_execute_code_guard("print('partial')", "vercel_sandbox")
    assert result["guard_outcome"] == "deny_hard"
    assert result["status"] == "blocked"


def test_terminal_safe_heredoc_user_deny_wins_even_with_force_before_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import terminal_tool

    command = "sh <<'SAFE_READ'\ncat -- '/etc/hosts'\nSAFE_READ\n"
    denied = {"approved": False, "guard_outcome": "deny_hard", "description": "user deny", "message": "BLOCKED by user deny."}
    guard = Mock(return_value=denied)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", guard)
    monkeypatch.setattr(terminal_tool, "_create_environment", Mock(side_effect=AssertionError("must not create environment")))
    result = json.loads(terminal_tool.terminal_tool(command, force=True))
    assert result["status"] == "error"
    assert result["error"] == "BLOCKED by user deny."
    guard.assert_called_once()


def test_public_terminal_pending_has_one_correlated_durable_lifecycle(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G: the real terminal wrapper surfaces IDs without double persistence."""
    from tools import terminal_tool

    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    _configure_terminal_pending_guard(monkeypatch, command)
    fake_env = SimpleNamespace(execute=Mock(side_effect=AssertionError("must not execute")), cwd=str(isolated_board))
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local", "cwd": str(isolated_board), "timeout": 30})
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **_kwargs: fake_env)
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    first = json.loads(terminal_tool.terminal_tool(command))
    second = json.loads(terminal_tool.terminal_tool(command))
    assert first["outcome"] == "approval_required"
    assert first["mutation_kind"]
    assert set(first["kanban_approval"]) == {"action_id", "attention_id", "attention_status", "reused"}
    assert second["kanban_approval"] == {**first["kanban_approval"], "reused": True}
    with kb.connect() as conn:
        events = kb.list_events(conn, task_id=task_id)
        action_id = first["kanban_approval"]["action_id"]
        action = kb.get_pending_action(conn, task_id)
        assert action is not None and first["mutation_kind"] == action.mutation_kind
        assert conn.execute("SELECT COUNT(*) FROM task_pending_actions WHERE task_id=?", (task_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 1
        pending = [e for e in events if e.kind == "terminal_approval_pending"]
        blocked = [e for e in events if e.kind == "blocked"]
        assert len(pending) == len(blocked) == 1
        assert pending[0].payload["action_id"] == blocked[0].payload["action_id"] == action_id


@pytest.mark.parametrize("env_type", ["local", "ssh"])
def test_public_execute_code_persist_fault_is_full_rollback_and_no_spawn(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, env_type: str,
) -> None:
    """I: injected post-persist failure returns hard-deny with an unchanged snapshot."""
    from tools import approval, code_execution_tool, terminal_tool

    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        run_id = kb.get_task(conn, task_id).current_run_id
        before = _atomic_snapshot(conn, task_id, run_id)
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": env_type})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    monkeypatch.setattr(approval, "submit_pending", Mock(side_effect=AssertionError("must not queue")))
    monkeypatch.setattr(kb, "_pending_action_after_persist_hook", lambda: (_ for _ in ()).throw(RuntimeError("fault")))
    monkeypatch.setattr(code_execution_tool.tempfile, "mkdtemp", Mock(side_effect=AssertionError("local spawn")))
    monkeypatch.setattr(code_execution_tool, "_execute_remote", Mock(side_effect=AssertionError("remote spawn")))
    result = json.loads(code_execution_tool.execute_code("print('fault')"))
    assert result["status"] == "error" and result["outcome"] == "deny_hard"
    with kb.connect() as conn:
        assert _atomic_snapshot(conn, task_id, run_id) == before


def test_public_execute_code_redacts_operator_surfaces_hook_and_logs(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """J: canonical private bindings remain private on every operator surface."""
    import logging
    from tools import code_execution_tool, terminal_tool

    task_id = _create_running_task(monkeypatch)
    source, profile = "credential='RAW_SOURCE_SECRET_MARKER'", "PROFILE_SECRET_MARKER"
    workspace = isolated_board / "ABSOLUTE_WORKSPACE_SECRET_MARKER"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_PROFILE", profile)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    with kb.connect() as conn:
        run_id = kb.get_task(conn, task_id).current_run_id
        conn.execute("UPDATE tasks SET assignee=? WHERE id=?", (profile, task_id))
        conn.execute("UPDATE task_runs SET profile=? WHERE id=?", (profile, run_id))
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    monkeypatch.setattr(code_execution_tool.tempfile, "mkdtemp", Mock(side_effect=AssertionError("spawn")))
    captured: list[dict] = []
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", lambda _event, _task, **fields: captured.append(fields))
    with caplog.at_level(logging.DEBUG):
        result = json.loads(code_execution_tool.execute_code(source))
    with kb.connect() as conn:
        action, attention = kb.get_pending_action(conn, task_id), kb.get_current_attention(conn, task_id)
        events = kb.list_events(conn, task_id=task_id)
        assert action is not None and attention is not None and action.command_hash and action.fingerprint
        public = json.dumps({"result": result, "attention": repr(attention), "events": [(e.kind, e.payload) for e in events], "hook": captured, "logs": [r.getMessage() for r in caplog.records]}, sort_keys=True)
    for marker in (source, "RAW_SOURCE_SECRET_MARKER", action.command_hash, action.fingerprint, profile, str(workspace.resolve())):
        assert marker not in public


def test_exact_action_and_human_gate_authorizations_are_isolated(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M: an exact-action grant and a human gate cannot cross-authorize."""
    action_task = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    with kb.connect() as conn:
        action_run = kb.get_task(conn, action_task).current_run_id
        action = kb.record_pending_action_and_block(conn, task_id=action_task, run_id=action_run, command=command, summary="publish", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000)
        monkeypatch.delenv("HERMES_KANBAN_TASK")
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
        gate_task = kb.create_task(conn, title="separate human gate", assignee="backend-eng")
        claimed = kb.claim_task(conn, gate_task, claimer="gate-worker")
        assert claimed is not None
        assert kb.block_task(conn, gate_task, kind="needs_input", reason="human", expected_run_id=claimed.current_run_id, human_gate=True)
        gate_token = kb.issue_gate_token(conn, gate_task)
        gate_hash = conn.execute("SELECT gate_token_hash FROM tasks WHERE id=?", (gate_task,)).fetchone()[0]
        before = kb.get_pending_action_by_id(conn, action_task, action["action_id"])
        assert gate_token and before is not None
        assert kb.approve_pending_action(conn, action_task, action["action_id"])
        assert conn.execute("SELECT gate_token_hash FROM tasks WHERE id=?", (gate_task,)).fetchone()[0] == gate_hash
        assert kb.unblock_task(conn, action_task, token=gate_token) is False
        after = kb.get_pending_action_by_id(conn, action_task, action["action_id"])
        assert after is not None and (after.state, after.version) == ("approved", before.version + 1)
        assert kb.unblock_task(conn, gate_task, token=gate_token)
        assert kb.get_task(conn, gate_task).status == "ready"


def test_exact_action_grant_cannot_cross_authorize_identical_active_cards(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical payload/profile/workspace grants remain bound to their task/run."""
    command = "print('same exact script')"
    with kb.connect() as conn:
        task_a = kb.create_task(conn, title="A", assignee="backend-eng")
        task_b = kb.create_task(conn, title="B", assignee="backend-eng")
        claimed_a = kb.claim_task(conn, task_a, claimer="a-origin")
        claimed_b = kb.claim_task(conn, task_b, claimer="b-origin")
        assert claimed_a is not None and claimed_b is not None
        action_a = kb.record_pending_action_and_block(
            conn, task_id=task_a, run_id=claimed_a.current_run_id, command=command,
            summary="A", profile="backend-eng", workspace=str(isolated_board),
            expires_at=2_000_000_000, mutation_kind="execute-code-arbitrary",
        )
        # Only A owns an approval; B remains a regular active card with an
        # otherwise identical command/profile/workspace context.
        assert kb.approve_pending_action_and_unblock(conn, task_a, action_a["action_id"])
        resumed_a = kb.claim_task(conn, task_a, claimer="a-resumed")
        assert resumed_a is not None and resumed_a.current_run_id is not None
        assert claimed_b.current_run_id is not None
        a_before = kb.get_pending_action_by_id(conn, task_a, action_a["action_id"])
        assert a_before is not None and (a_before.state, a_before.consumed_at) == ("approved", None)

        assert not kb.consume_approved_action(
            conn, command=command, task_id=task_b, run_id=claimed_b.current_run_id,
            profile="backend-eng", workspace=str(isolated_board), mutation_kind="execute-code-arbitrary",
        )
        a_after_b_attempt = kb.get_pending_action_by_id(conn, task_a, action_a["action_id"])
        task_b_after = kb.get_task(conn, task_b)
        assert a_after_b_attempt is not None
        assert (a_after_b_attempt.state, a_after_b_attempt.consumed_at, a_after_b_attempt.version) == (
            "approved", None, a_before.version,
        )
        assert task_b_after is not None
        assert (task_b_after.status, task_b_after.current_run_id) == ("running", claimed_b.current_run_id)
        assert not [e for e in kb.list_events(conn, task_id=task_b) if e.kind == "terminal_approval_consumed"]

        assert kb.consume_approved_action(
            conn, command=command, task_id=task_a, run_id=resumed_a.current_run_id,
            profile="backend-eng", workspace=str(isolated_board), mutation_kind="execute-code-arbitrary",
        )
        a_after = kb.get_pending_action_by_id(conn, task_a, action_a["action_id"])
        assert a_after is not None
        assert a_after.state == "consumed" and a_after.consumed_at is not None
        assert a_after.version == a_before.version + 1
        a_consume_events = [e for e in kb.list_events(conn, task_id=task_a) if e.kind == "terminal_approval_consumed"]
        assert len(a_consume_events) == 1
        assert (a_consume_events[0].payload["action_id"], a_consume_events[0].run_id) == (
            action_a["action_id"], resumed_a.current_run_id,
        )
        assert not [e for e in kb.list_events(conn, task_id=task_b) if e.kind == "terminal_approval_consumed"]


def test_safe_readonly_heredoc_is_durable_noop_snapshot(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        run_id = kb.get_task(conn, task_id).current_run_id
        before = _atomic_snapshot(conn, task_id, run_id)
    record, queued = Mock(), Mock()
    monkeypatch.setattr(approval, "_record_kanban_pending_action", record)
    monkeypatch.setattr(approval, "submit_pending", queued)
    command = "sh <<'SAFE_READ'\ncat -- '/etc/hosts'\nSAFE_READ\n"

    result = approval.check_all_command_guards(command, "local")

    assert result["guard_outcome"] == "retry_with_safe_alternative"
    record.assert_not_called()
    queued.assert_not_called()
    with kb.connect() as conn:
        assert _atomic_snapshot(conn, task_id, run_id) == before


@pytest.mark.parametrize("force", [False, True])
def test_terminal_safe_readonly_heredoc_returns_redacted_structured_response_before_environment(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, force: bool,
) -> None:
    from tools import terminal_tool

    command = "sh <<'SAFE_READ'\ncat -- '/etc/hosts'\nSAFE_READ\n"
    monkeypatch.setattr(
        terminal_tool, "_create_environment",
        Mock(side_effect=AssertionError("environment creation must not run")),
    )

    result = json.loads(terminal_tool.terminal_tool(command, force=force))

    assert result == {
        "output": "",
        "exit_code": -1,
        "error": "Read-only audit recognized; use read_file.",
        "status": "safe_alternative_required",
        "outcome": "retry_with_safe_alternative",
        "guard_outcome": "retry_with_safe_alternative",
        "reason_code": "readonly-heredoc-audit",
        "safe_alternative": {
            "tool": "read_file",
            "kind": "literal-file-inspection",
            "execution": "not_run",
        },
    }
    public = json.dumps(result)
    assert command not in public and "/etc/hosts" not in public


def test_guard_outcome_matrix_only_approval_required_is_durable(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import approval

    task_id = _create_running_task(monkeypatch)
    bounded = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    _configure_terminal_pending_guard(monkeypatch, bounded)
    cases = {
        "allow": "echo harmless",
        "deny_hard": "rm -rf /",
        "retry_with_safe_alternative": "sh <<'SAFE_READ'\ncat -- '/etc/hosts'\nSAFE_READ\n",
        "approval_required": bounded,
    }
    outcomes = {name: approval.check_all_command_guards(command, "local") for name, command in cases.items()}

    assert {name: result["guard_outcome"] for name, result in outcomes.items()} == {
        "allow": "allow", "deny_hard": "deny_hard",
        "retry_with_safe_alternative": "retry_with_safe_alternative",
        "approval_required": "approval_required",
    }
    assert outcomes["deny_hard"].get("kanban_approval") is None
    assert outcomes["retry_with_safe_alternative"].get("kanban_approval") is None
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_pending_actions WHERE task_id=?", (task_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 1
        assert len([event for event in kb.list_events(conn, task_id=task_id) if event.kind in {"terminal_approval_pending", "blocked"}]) == 2
    retry = approval.check_all_command_guards(bounded, "local")
    assert retry["guard_outcome"] == "approval_required"
    assert retry["kanban_approval"]["reused"] is True


@pytest.mark.parametrize(
    ("command", "user_deny"),
    [
        ("rm -rf /", None),
        ("publish forbidden artifact", "publish forbidden *"),
        ("printf guessed-password | sudo -S id", None),
    ],
    ids=["hardline", "user-deny", "sudo-stdin"],
)
@pytest.mark.parametrize("force", [False, True], ids=["guarded", "forced"])
def test_public_terminal_hard_denies_are_kanban_durable_noops(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, command: str, user_deny: str | None,
    force: bool,
) -> None:
    """B: unconditional terminal denials never enter the Kanban approval lifecycle."""
    from tools import approval, terminal_tool

    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        run_id = task.current_run_id
        before = _atomic_snapshot(conn, task_id, run_id)
    if user_deny is not None:
        monkeypatch.setattr(approval, "_get_approval_config", lambda: {"deny": [user_deny]})
    record = Mock(wraps=approval._record_kanban_pending_action)
    queued = Mock(wraps=approval.submit_pending)
    lifecycle = Mock(wraps=kb._fire_kanban_lifecycle_hook)
    monkeypatch.setattr(approval, "_record_kanban_pending_action", record)
    monkeypatch.setattr(approval, "submit_pending", queued)
    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", lifecycle)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local", "cwd": str(isolated_board), "timeout": 30})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    create_environment = Mock(side_effect=AssertionError("environment spawn"))
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)

    result = json.loads(terminal_tool.terminal_tool(command, force=force))

    assert (result["status"], result["outcome"], result["guard_outcome"]) == (
        "blocked", "deny_hard", "deny_hard",
    )
    create_environment.assert_not_called()
    record.assert_not_called()
    queued.assert_not_called()
    lifecycle.assert_not_called()
    with kb.connect() as conn:
        assert _atomic_snapshot(conn, task_id, run_id) == before


def test_public_terminal_force_executes_bounded_approvable_command_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force remains an internal confirmation for commands outside the deny floor."""
    from tools import terminal_tool

    command = "git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic"
    fake_env = SimpleNamespace(
        cwd=str(isolated_board),
        execute=Mock(return_value={"output": "done", "returncode": 0}),
    )
    monkeypatch.setattr(
        terminal_tool, "_get_env_config",
        lambda: {"env_type": "local", "cwd": str(isolated_board), "timeout": 30},
    )
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **_kwargs: fake_env)

    forced = json.loads(terminal_tool.terminal_tool(command, force=True))

    assert forced["exit_code"] == 0
    fake_env.execute.assert_called_once()


@pytest.mark.parametrize(
    "variant",
    ["print('é')\r\n", "print('e\u0301')\n", "print('é')\n "],
    ids=["crlf", "nfd", "whitespace"],
)
def test_public_execute_code_variants_never_reuse_approved_exact_grant_and_later_consume_original_grant(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    """H: byte variants make their own durable request and cannot consume A."""
    from tools import code_execution_tool, terminal_tool

    code_a = "print('é')\n"
    assert variant.encode("utf-8") != code_a.encode("utf-8")
    monkeypatch.setattr(code_execution_tool, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    monkeypatch.setattr(terminal_tool, "_docker_has_host_access", lambda _cfg: False)

    task_id = _create_running_task(monkeypatch)
    pending_a = json.loads(code_execution_tool.execute_code(code_a))
    assert (pending_a["status"], pending_a["outcome"]) == ("pending_approval", "approval_required")
    action_a_id = pending_a["kanban_approval"]["action_id"]
    with kb.connect() as conn:
        assert kb.approve_pending_action_and_unblock(conn, task_id, action_a_id)
        resumed = kb.claim_task(conn, task_id, claimer="resumed-a")
        assert resumed is not None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(resumed.current_run_id))
        a_before = kb.get_pending_action_by_id(conn, task_id, action_a_id)
        assert a_before is not None and (a_before.state, a_before.consumed_at) == ("approved", None)

    no_spawn = Mock(side_effect=AssertionError("local spawn"))
    monkeypatch.setattr(code_execution_tool.tempfile, "mkdtemp", no_spawn)
    pending_b = json.loads(code_execution_tool.execute_code(variant))
    assert (pending_b["status"], pending_b["outcome"], pending_b["tool_calls_made"]) == ("pending_approval", "approval_required", 0)
    assert code_a not in json.dumps(pending_b, ensure_ascii=False)
    action_b_id = pending_b["kanban_approval"]["action_id"]
    assert action_b_id != action_a_id and no_spawn.call_count == 0
    with kb.connect() as conn:
        action_a = kb.get_pending_action_by_id(conn, task_id, action_a_id)
        action_b = kb.get_pending_action_by_id(conn, task_id, action_b_id)
        assert action_a is not None and action_b is not None
        assert (action_a.state, action_a.consumed_at, action_a.version) == ("approved", None, a_before.version)
        assert action_a.command_hash != action_b.command_hash
        assert (action_b.state, action_b.consumed_at) == ("pending", None)
        assert action_b.version >= 1

    with kb.connect() as conn:
        action_a_before_resolution = kb.get_pending_action_by_id(conn, task_id, action_a_id)
        action_b_before_resolution = kb.get_pending_action_by_id(conn, task_id, action_b_id)
        assert action_a_before_resolution is not None and action_b_before_resolution is not None
        assert kb.resolve_pending_action(conn, task_id, action_b_id)
        action_a_after_resolution = kb.get_pending_action_by_id(conn, task_id, action_a_id)
        action_b_after_resolution = kb.get_pending_action_by_id(conn, task_id, action_b_id)
        assert action_a_after_resolution is not None and action_b_after_resolution is not None
        assert (action_a_after_resolution.state, action_a_after_resolution.consumed_at, action_a_after_resolution.version) == (
            action_a_before_resolution.state, action_a_before_resolution.consumed_at, action_a_before_resolution.version,
        )
        assert action_b_after_resolution.state == "resolved" and action_b_after_resolution.consumed_at is None
        attention_a = kb.get_current_attention(conn, task_id)
        assert attention_a is not None and (attention_a.action_id, attention_a.state) == (action_a_id, "approved")
        # A second, byte-different request is not a technical failure of A.
        # It must not manufacture retry authority for A; only the dedicated
        # technical lifecycle may resume an approved exact action.
        assert conn.execute("SELECT retry_origin_run_id FROM task_pending_actions WHERE id=?", (action_a_id,)).fetchone()[0] is None


def test_resolve_pending_action_reports_not_found_conflict_and_gone(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id,
            command="resolve-cas-marker", summary="ignored", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.resolve_pending_action(conn, task_id, action.id, expected_version=action.version + 1, now=1).status == "conflict"
        assert kb.resolve_pending_action(conn, "foreign-task", action.id, expected_version=action.version, now=1).status == "not_found"
        resolved = kb.resolve_pending_action(conn, task_id, action.id, expected_version=action.version, now=1)
        assert resolved.status == "resolved" and resolved
        assert kb.resolve_pending_action(conn, task_id, action.id, expected_version=action.version, now=1).status == "gone"
        assert kb.get_current_attention(conn, task_id, now=1) is None


def test_approved_action_technical_failure_replaces_current_projection_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, resumed_run = _park_and_approve(conn, task_id, "technical-replacement-marker", isolated_board)
        before = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert before is not None and before.type == "exact_action" and before.state == "approved"
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=resumed_run,
            attention_type="capability", reason_code="missing_capability",
            cause_scope={"capability": "tool"}, now=1_900_000_001,
        )
        current = kb.get_current_attention(conn, task_id, now=1_900_000_001)
        internal = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert current is not None and current.type == "capability" and current.state == "pending"
        assert internal is not None and internal.state == "approved"
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 1
        assert kb.restore_approved_action_attention(conn, task_id, action.id, now=1_900_000_002)
        restored = kb.get_current_attention(conn, task_id, now=1_900_000_002)
        assert restored is not None and restored.type == "exact_action" and restored.state == "approved"


def test_technical_replacement_fault_rolls_back_full_snapshot(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, resumed_run = _park_and_approve(conn, task_id, "technical-rollback-marker", isolated_board)
        before = _atomic_snapshot(conn, task_id, resumed_run)
        monkeypatch.setattr(kb, "_technical_attention_after_replace_hook", lambda: (_ for _ in ()).throw(RuntimeError("fault")))
        with pytest.raises(RuntimeError, match="fault"):
            kb.block_approved_action_for_technical_failure(
                conn, task_id=task_id, action_id=action.id, expected_run_id=resumed_run,
                attention_type="transient", reason_code="external_transient",
                cause_scope={"subject": "external_service"}, now=1_900_000_001,
            )
        assert _atomic_snapshot(conn, task_id, resumed_run) == before


def test_restore_technical_attention_requires_unclaimed_blocked_retry_lifecycle(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, resumed_run = _park_and_approve(conn, task_id, "restore-lifecycle-marker", isolated_board)
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=resumed_run,
            attention_type="capability", reason_code="missing_capability", now=1_900_000_001,
        )
        for status in ("done", "archived", "ready", "todo", "running"):
            conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
            before = _atomic_snapshot(conn, task_id, resumed_run)
            assert not kb.restore_approved_action_attention(conn, task_id, action.id, now=1_900_000_002)
            assert _atomic_snapshot(conn, task_id, resumed_run) == before
        conn.execute("UPDATE tasks SET status='blocked', current_run_id=?, claim_lock='claimed' WHERE id=?", (resumed_run, task_id))
        before = _atomic_snapshot(conn, task_id, resumed_run)
        assert not kb.restore_approved_action_attention(conn, task_id, action.id, now=1_900_000_002)
        assert _atomic_snapshot(conn, task_id, resumed_run) == before
        conn.execute("DELETE FROM task_attentions WHERE task_id=?", (task_id,))
        before = _atomic_snapshot(conn, task_id, resumed_run)
        assert not kb.restore_approved_action_attention(conn, task_id, action.id, now=1_900_000_002)
        assert _atomic_snapshot(conn, task_id, resumed_run) == before


def test_technical_projection_is_deleted_on_expiry_resolve_and_cleanup_rollback(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, resumed_run = _park_and_approve(conn, task_id, "cleanup-marker", isolated_board)
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=resumed_run,
            attention_type="transient", reason_code="external_transient", now=1_900_000_001,
        )
        before = _atomic_snapshot(conn, task_id, resumed_run)
        monkeypatch.setattr(kb, "_attention_cleanup_hook", lambda: (_ for _ in ()).throw(RuntimeError("cleanup fault")))
        with pytest.raises(RuntimeError, match="cleanup fault"):
            kb.resolve_pending_action(conn, task_id, action.id, now=1_900_000_002)
        assert _atomic_snapshot(conn, task_id, resumed_run) == before
        monkeypatch.setattr(kb, "_attention_cleanup_hook", lambda: None)
        assert kb.resolve_pending_action(conn, task_id, action.id, now=1_900_000_002).status == "resolved"
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 0

    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, resumed_run = _park_and_approve(conn, task_id, "expiry-cleanup-marker", isolated_board)
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=resumed_run,
            attention_type="capability", reason_code="missing_capability", now=1_900_000_001,
        )
        assert kb.resolve_pending_action(conn, task_id, action.id, now=2_000_000_000).status == "gone"
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 0


def test_opaque_technical_retry_rejects_stale_origin_and_replay_without_drift(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, origin_run_id = _park_and_approve(conn, task_id, "opaque-retry-cas-marker", isolated_board)
        assert origin_run_id is not None
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=origin_run_id,
            attention_type="capability", reason_code="missing_capability", now=1_900_000_001,
        )
        technical = kb.get_current_attention(conn, task_id, now=1_900_000_001)
        assert technical is not None
        before = _atomic_snapshot(conn, task_id, origin_run_id)
        stale = kb.resume_approved_action_retry(
            conn, task_id=task_id, expected_attention_id=technical.id,
            expected_attention_version=technical.version + 1,
            expected_origin_run_id=origin_run_id, now=1_900_000_002,
        )
        wrong_origin = kb.resume_approved_action_retry(
            conn, task_id=task_id, expected_attention_id=technical.id,
            expected_attention_version=technical.version,
            expected_origin_run_id=origin_run_id + 1, now=1_900_000_002,
        )
        assert (stale.status, wrong_origin.status) == ("conflict", "conflict")
        assert _atomic_snapshot(conn, task_id, origin_run_id) == before
        assert kb.resume_approved_action_retry(
            conn, task_id=task_id, expected_attention_id=technical.id,
            expected_attention_version=technical.version,
            expected_origin_run_id=origin_run_id, now=1_900_000_002,
        ).status == "resumed"
        after_resume = _atomic_snapshot(conn, task_id, origin_run_id)
        replay = kb.resume_approved_action_retry(
            conn, task_id=task_id, expected_attention_id=technical.id,
            expected_attention_version=technical.version,
            expected_origin_run_id=origin_run_id, now=1_900_000_002,
        )
        assert replay.status == "gone"
        assert _atomic_snapshot(conn, task_id, origin_run_id) == after_resume


def test_opaque_technical_retry_restore_fault_rolls_back_snapshot(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, origin_run_id = _park_and_approve(conn, task_id, "opaque-retry-rollback-marker", isolated_board)
        assert origin_run_id is not None
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=origin_run_id,
            attention_type="transient", reason_code="external_transient", now=1_900_000_001,
        )
        technical = kb.get_current_attention(conn, task_id, now=1_900_000_001)
        assert technical is not None
        before = _atomic_snapshot(conn, task_id, origin_run_id)
        monkeypatch.setattr(
            kb, "_approved_action_retry_after_projection_restore_hook",
            lambda: (_ for _ in ()).throw(RuntimeError("retry restore fault")),
        )
        with pytest.raises(RuntimeError, match="retry restore fault"):
            kb.resume_approved_action_retry(
                conn, task_id=task_id, expected_attention_id=technical.id,
                expected_attention_version=technical.version,
                expected_origin_run_id=origin_run_id, now=1_900_000_002,
            )
        assert _atomic_snapshot(conn, task_id, origin_run_id) == before


def test_technical_projection_is_deleted_on_consume_done_and_archive(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = "terminal-cleanup-marker"
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        action, first_resumed_run = _park_and_approve(conn, task_id, command, isolated_board)
        assert first_resumed_run is not None
        approved = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert approved is not None and approved.state == "approved"
        approved_version = approved.version
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task_id, action_id=action.id, expected_run_id=first_resumed_run,
            attention_type="transient", reason_code="external_transient", now=1_900_000_001,
        )
        blocked_task = kb.get_task(conn, task_id)
        blocked_run = conn.execute(
            "SELECT status, outcome, ended_at, claim_lock FROM task_runs WHERE id=?",
            (first_resumed_run,),
        ).fetchone()
        technical_rows = conn.execute(
            "SELECT action_id, type FROM task_attentions WHERE task_id=?", (task_id,)
        ).fetchall()
        assert blocked_task is not None and (
            blocked_task.status, blocked_task.current_run_id, blocked_task.claim_lock,
        ) == ("blocked", None, None)
        assert blocked_run is not None and (
            blocked_run["status"], blocked_run["outcome"], blocked_run["ended_at"], blocked_run["claim_lock"],
        ) == ("blocked", "blocked", 1_900_000_001, None)
        assert [tuple(row) for row in technical_rows] == [(action.id, "transient")]
        technical = kb.get_current_attention(conn, task_id, now=1_900_000_001)
        assert technical is not None and (technical.action_id, technical.type, technical.state) == (
            action.id, "transient", "pending",
        )

        resumed = kb.resume_approved_action_retry(
            conn, task_id=task_id, expected_attention_id=technical.id,
            expected_attention_version=technical.version,
            expected_origin_run_id=first_resumed_run, now=1_900_000_002,
        )
        assert resumed.status == "resumed"
        restored = kb.get_current_attention(conn, task_id, now=1_900_000_002)
        restored_rows = conn.execute(
            "SELECT action_id, type FROM task_attentions WHERE task_id=?", (task_id,)
        ).fetchall()
        restored_action = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert restored is not None and (restored.id, restored.action_id, restored.type, restored.state) == (
            technical.id, action.id, "exact_action", "approved",
        )
        assert [tuple(row) for row in restored_rows] == [(action.id, "exact_action")]
        assert restored_action is not None and (
            restored_action.state, restored_action.version, restored_action.consumed_at,
        ) == ("approved", approved_version, None)

        ready = kb.get_task(conn, task_id)
        assert ready is not None and (
            ready.status, ready.current_run_id, ready.claim_lock, ready.claim_expires,
        ) == ("ready", None, None, None)
        second_resumed = kb.claim_task(conn, task_id, claimer="technical-retry")
        assert second_resumed is not None and second_resumed.current_run_id is not None
        second_resumed_run = second_resumed.current_run_id
        running_row = conn.execute(
            "SELECT status, outcome, ended_at, claim_lock FROM task_runs WHERE id=?",
            (second_resumed_run,),
        ).fetchone()
        assert running_row is not None and (
            running_row["status"], running_row["outcome"], running_row["ended_at"], running_row["claim_lock"],
        ) == ("running", None, None, "technical-retry")

        assert kb.consume_approved_action(
            conn, task_id=task_id, run_id=second_resumed_run, command=command,
            profile="backend-eng", workspace=str(isolated_board), now=1_900_000_003,
        )
        consumed = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert consumed is not None and (
            consumed.state, consumed.consumed_at, consumed.version,
        ) == ("consumed", 1_900_000_003, approved_version + 1)
        assert kb.get_current_attention(conn, task_id, now=1_900_000_003) is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,),
        ).fetchone()[0] == 0
        events = kb.list_events(conn, task_id=task_id)
        assert sum(event.kind == "terminal_approval_granted" for event in events) == 1
        assert sum(event.kind == "unblocked" for event in events) == 2
        assert sum(event.kind == "claimed" for event in events) == 3
        assert sum(event.kind == "blocked" for event in events) == 2
        retry_events = [event for event in events if event.kind == "approved_action_retry_resumed"]
        assert len(retry_events) == 1 and retry_events[0].payload is not None
        assert "action_id" not in retry_events[0].payload
        consume_events = [event for event in events if event.kind == "terminal_approval_consumed"]
        assert len(consume_events) == 1
        consume_event = consume_events[0]
        assert consume_event.payload is not None
        assert (consume_event.run_id, consume_event.payload["action_id"]) == (
            second_resumed_run, action.id,
        )
        assert command not in repr(events)
        assert action.fingerprint not in repr(events)

    for terminal in ("done", "archived"):
        task_id = _create_running_task(monkeypatch)
        with kb.connect() as conn:
            action, resumed_run = _park_and_approve(conn, task_id, f"{terminal}-cleanup-marker", isolated_board)
            assert kb.block_approved_action_for_technical_failure(
                conn, task_id=task_id, action_id=action.id, expected_run_id=resumed_run,
                attention_type="capability", reason_code="missing_capability", now=1_900_000_001,
            )
            assert (kb.complete_task(conn, task_id, summary="done") if terminal == "done" else kb.archive_task(conn, task_id))
            assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE task_id=?", (task_id,)).fetchone()[0] == 0
            current = kb.get_pending_action_by_id(conn, task_id, action.id)
            assert current is not None and current.state == "cancelled"


def test_parallel_versioned_resolves_have_exactly_one_winner(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id,
            command="parallel-resolve-cas-marker", summary="ignored", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000)
    barrier, results = threading.Barrier(2), []
    def resolve() -> None:
        with kb.connect() as other:
            barrier.wait(timeout=5)
            results.append(kb.resolve_pending_action(
                other, task_id, action.id, expected_version=action.version, now=1,
            ).status)
    workers = [threading.Thread(target=resolve) for _ in range(2)]
    [worker.start() for worker in workers]
    [worker.join(timeout=5) for worker in workers]
    assert not any(worker.is_alive() for worker in workers)
    assert sorted(results) == ["gone", "resolved"]
    with kb.connect() as conn:
        current = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert current is not None and (current.state, current.version) == ("resolved", action.version + 1)
        assert kb.get_current_attention(conn, task_id, now=1) is None



def test_versioned_approval_is_cas_and_keeps_waiting_projection(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command="git push origin HEAD",
            summary="push", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        assert kb.block_task(conn, task_id, kind="needs_input", expected_run_id=task.current_run_id)
        attention = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert attention is not None
        result = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, action.id, expected_version=attention.version, now=1_900_000_000,
        )
        assert result.status == "approved"
        assert result.attention_id == attention.id and result.attention_version == attention.version + 1
        waiting = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert waiting is not None and waiting.id == attention.id and waiting.state == "approved"
        assert not waiting.approvable and not waiting.requires_human_action
        current = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert current is not None
        assert kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, action.id, expected_version=current.version, now=1_900_000_000,
        ).status == "gone"


def test_parallel_versioned_approvals_return_approved_and_conflict_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale concurrent CAS request is conflict, not a terminal replay."""
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="parallel-versioned-approve", summary="ignored", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        assert kb.block_task(conn, task_id, kind="needs_input", expected_run_id=task.current_run_id)
        attention = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert attention is not None

    barrier, results = threading.Barrier(2), []

    def approve() -> None:
        with kb.connect() as other:
            barrier.wait(timeout=5)
            results.append(kb.approve_pending_action_and_unblock_versioned(
                other, task_id, action.id, expected_version=attention.version, now=1_900_000_000,
            ).status)

    workers = [threading.Thread(target=approve) for _ in range(2)]
    [worker.start() for worker in workers]
    [worker.join(timeout=5) for worker in workers]
    assert not any(worker.is_alive() for worker in workers)
    assert sorted(results) == ["approved", "conflict"]
    with kb.connect() as conn:
        current = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert current is not None and (current.state, current.version) == ("approved", action.version + 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='terminal_approval_granted'",
            (task_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE id=? AND status='ready'", (task_id,),
        ).fetchone()[0] == 1
        assert kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, action.id, expected_version=current.version, now=1_900_000_000,
        ).status == "gone"


def test_attention_transition_refuses_live_projection(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id, command="git push origin HEAD",
            summary="push", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_000,
        )
        assert kb.block_task(conn, task_id, kind="needs_input", expected_run_id=task.current_run_id)
        result = kb.transition_task_status_with_attention(
            conn=conn, task_id=task_id, status="ready", now=1_900_000_000,
        )
        assert result.status == "conflict" and result.attention_id is not None


def test_versioned_approval_human_gate_requires_token_and_consumes_only_on_success(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        origin_run_id = task.current_run_id
        action = kb.record_pending_action(conn, task_id=task_id, run_id=origin_run_id,
            command="git push origin gated", summary="push", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.block_task(conn, task_id, kind="needs_input", expected_run_id=task.current_run_id, human_gate=True)
        token = kb.issue_gate_token(conn, task_id, action="unblock")
        attention = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert token and attention
        before = _atomic_snapshot(conn, task_id, origin_run_id)
        denied = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, action.id, expected_version=attention.version, now=1_900_000_000)
        assert denied.status == "conflict"
        # Rejected gate attempts are auditable, but do not consume the token or
        # mutate the action/task lifecycle.
        after_denied = _atomic_snapshot(conn, task_id, origin_run_id)
        assert after_denied["task"] == before["task"]
        assert after_denied["actions"] == before["actions"]
        assert conn.execute("SELECT gate_token_hash FROM tasks WHERE id=?", (task_id,)).fetchone()[0] == kb.hash_gate_token(token)
        approved = kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, action.id, expected_version=attention.version, token=token, now=1_900_000_000)
        assert approved.status == "approved"
        assert conn.execute("SELECT gate_token_hash FROM tasks WHERE id=?", (task_id,)).fetchone()[0] is None


def test_versioned_approval_foreign_request_does_not_materialize_unrelated_expiry(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    monkeypatch.setattr(kb.time, "time", lambda: 100)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push origin expired", summary="push", profile="backend-eng",
            workspace=str(isolated_board), expires_at=101,
        )
        before = tuple(conn.execute("SELECT state, version FROM task_pending_actions WHERE id=?", (action.id,)).fetchone())
        result = kb.approve_pending_action_and_unblock_versioned(conn, "foreign-task", action.id, expected_version=action.version, now=102)
        assert result.status == "not_found"
        assert tuple(conn.execute("SELECT state, version FROM task_pending_actions WHERE id=?", (action.id,)).fetchone()) == before


@pytest.mark.parametrize(("attention_type", "reason_code", "scope", "requires_human_action"), [
    ("decision", "credential_choice", {"required_decision": "credential"}, True),
    ("protocol", "goal_closeout_missing", {"protocol": "goal_closeout"}, True),
    ("review", "review_required", {"subject": "review"}, True),
    ("loop_triage", "review_required", {"subject": "loop"}, False),
    ("capability", "missing_capability", {"capability": "tool"}, True),
    ("transient", "external_transient", {"subject": "external_service"}, False),
])
def test_block_task_all_actionless_typed_attentions_transition_by_id_and_version(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch, attention_type: str,
    reason_code: str, scope: dict[str, str], requires_human_action: bool,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="untrusted prose",
                             attention_type=attention_type, reason_code=reason_code,
                             cause_scope=scope, expected_run_id=task.current_run_id)
        attention = kb.get_current_attention(conn, task_id)
        assert attention is not None
        assert (attention.type, attention.version, attention.requires_human_action,
                attention.approvable, attention.action_id) == (
                    attention_type, 1, requires_human_action, False, None)
        assert not kb.unblock_task(conn, task_id)
        result = kb.transition_task_status_with_attention(
            conn, task_id=task_id, status="ready", expected_attention_id=attention.id,
            expected_attention_version=attention.version,
        )
        assert result.status == "transitioned"
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE id=?", (attention.id,)).fetchone()[0] == 0


def test_exact_approved_awaiting_worker_remains_visible_not_approvable(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        action = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id,
            command="approved-awaiting-worker", summary="ignored", profile="backend-eng",
            workspace=str(isolated_board), expires_at=2_000_000_000)
        assert kb.block_task(conn, task_id, kind="needs_input", expected_run_id=task.current_run_id)
        before = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert before is not None
        assert kb.approve_pending_action_and_unblock_versioned(
            conn, task_id, action.id, expected_version=before.version, now=1_900_000_000,
        ).status == "approved"
        waiting = kb.get_current_attention(conn, task_id, now=1_900_000_000)
        assert waiting is not None and waiting.id == before.id
        assert (waiting.type, waiting.state, waiting.approvable, waiting.requires_human_action) == (
            "exact_action", "approved", False, False)


def test_current_attentions_batch_is_read_only_deduped_and_bounded(isolated_board: Path) -> None:
    with kb.connect() as conn:
        ids = [f"batch-{n}" for n in range(901)]
        conn.executemany("INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, 'blocked', 1)",
                         [(task_id, task_id) for task_id in ids])
        conn.executemany("""INSERT INTO task_attentions
            (task_id, action_id, type, cause_fingerprint, summary, created_at, version, origin_run_id)
            VALUES (?, NULL, 'transient', ?, 'external_transient', 1, 1, NULL)""",
                         [(task_id, f"batch-{n}-cause") for n, task_id in enumerate(ids)])
        before = conn.total_changes
        selects: list[str] = []
        conn.set_trace_callback(lambda sql: selects.append(sql) if "FROM task_attentions x" in sql else None)
        assert kb.get_current_attentions(conn, []) == {}
        got = kb.get_current_attentions(conn, ids + ids[:40])
        conn.set_trace_callback(None)
        assert set(got) == set(ids)
        assert all(attention.action_id is None and attention.type == "transient" for attention in got.values())
        assert len(selects) == 2
        assert conn.total_changes == before
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (ids[0],))
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (ids[1],))
        visible = kb.get_current_attentions(conn, ids[:2])
        assert visible == {}


def test_typed_attention_transition_stale_version_and_two_connections_have_one_winner(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.current_run_id is not None
        assert kb.block_task(conn, task_id, kind="needs_input", attention_type="decision",
                             reason_code="credential_choice", cause_scope={"required_decision": "credential"},
                             expected_run_id=task.current_run_id)
        attention = kb.get_current_attention(conn, task_id)
        assert attention is not None
        stale = kb.transition_task_status_with_attention(conn, task_id=task_id, status="ready",
            expected_attention_id=attention.id, expected_attention_version=attention.version + 1)
        assert stale.status == "conflict"
    barrier, results = threading.Barrier(2), []
    def transition() -> None:
        with kb.connect() as other:
            barrier.wait(timeout=5)
            results.append(kb.transition_task_status_with_attention(
                other, task_id=task_id, status="ready", expected_attention_id=attention.id,
                expected_attention_version=attention.version,
            ).status)
    workers = [threading.Thread(target=transition) for _ in range(2)]
    [worker.start() for worker in workers]
    [worker.join(timeout=5) for worker in workers]
    assert not any(worker.is_alive() for worker in workers)
    assert sorted(results) == ["conflict", "transitioned"]
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE id=?", (attention.id,)).fetchone()[0] == 0

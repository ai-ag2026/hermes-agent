from __future__ import annotations

import json
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
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="different human decision",
            expected_run_id=resumed_run,
        )
        resolved = kb.get_pending_action_by_id(conn, task_id, action.id)
        assert resolved is not None
        assert resolved.cancelled_at is not None
        assert resolved.consumed_at is None
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


def test_terminal_same_identity_rebinds_stable_attention(
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
        assert kb.resolve_pending_action(conn, task_id, first.id, now=1_900_000_000)
        second = kb.record_pending_action(conn, task_id=task_id, run_id=task.current_run_id, command=command,
                                          summary="publish", profile="backend-eng", workspace=str(isolated_board), expires_at=2_000_000_100)
        current = kb.get_current_attention(conn, task_id)
        historic = kb.get_pending_action_by_id(conn, task_id, first.id)
        assert current is not None and historic is not None
        assert historic.state == "resolved"
        assert (current.id, current.action_id) == (old_attention.id, second.id)


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


def test_expired_pending_rerequest_rebinds_stable_attention(
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
        assert (old.state, second.id != first.id, current.id, current.action_id) == ("expired", True, attention.id, second.id)


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
    assert fields["assignee"] == "backend-eng"
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
                             expected_run_id=task.current_run_id)
        assert kb.unblock_task(conn, task_id, reason="credential choice supplied")
        assert kb.claim_task(conn, task_id, claimer="second-worker") is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="need credential choice")
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
                             expected_run_id=task.current_run_id)
        assert kb.unblock_task(conn, task_id, reason="credential choice supplied")
        assert kb.claim_task(conn, task_id, claimer="second-worker") is not None
        assert kb.block_task(conn, task_id, kind="needs_input", reason="need publication approval")
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


def test_execute_code_and_terminal_pending_paths_create_same_durable_approval_contract(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

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
    assert result["status"] == "error"

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

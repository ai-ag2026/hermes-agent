from __future__ import annotations

from pathlib import Path

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


def test_action_grant_is_exact_expiring_and_consume_once(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    command = "git push --force-with-lease=refs/heads/topic:abc123 fork HEAD:topic"

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
        assert kb.approve_pending_action(conn, task_id, action.id, now=1_900_000_000)

        context = dict(task_id=task_id, profile="backend-eng", workspace=str(isolated_board))
        assert not kb.consume_approved_action(
            conn, command=command + " --dry-run", now=1_900_000_001, **context,
        )
        assert not kb.consume_approved_action(
            conn, command=command, profile="reviewer", task_id=task_id,
            workspace=str(isolated_board), now=1_900_000_001,
        )
        assert kb.consume_approved_action(
            conn, command=command, now=1_900_000_001, **context,
        )
        assert not kb.consume_approved_action(
            conn, command=command, now=1_900_000_002, **context,
        )


def test_expired_grant_is_rejected(
    isolated_board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = _create_running_task(monkeypatch)
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        action = kb.record_pending_action(
            conn, task_id=task_id, run_id=task.current_run_id,
            command="git push --force-with-lease fork HEAD:topic", summary="git push",
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
            command="git push --force-with-lease fork HEAD:topic", summary="git push",
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
    command = "git push --force-with-lease=refs/heads/topic:abc123 fork HEAD:topic"
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
        assert task is not None
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="terminal approval required",
            expected_run_id=task.current_run_id,
        )
        assert kb.approve_pending_action_and_unblock(conn, task_id, action.id)

    second = approval.check_all_command_guards(command, "local")
    assert second["approved"] is True
    assert second["kanban_action_grant"] is True
    with kb.connect() as conn:
        assert kb.get_pending_action(conn, task_id) is None

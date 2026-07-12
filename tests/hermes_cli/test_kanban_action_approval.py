from __future__ import annotations

from pathlib import Path
import threading

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
        assert task is not None
        assert kb.block_task(
            conn, task_id, kind="needs_input", reason="terminal approval required",
            expected_run_id=task.current_run_id,
        )
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

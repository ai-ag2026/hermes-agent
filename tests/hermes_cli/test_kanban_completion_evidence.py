from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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


def _artifact_rows(conn, task_id: str):
    return conn.execute(
        "SELECT * FROM task_artifacts WHERE task_id=? ORDER BY id", (task_id,)
    ).fetchall()


def _event_payload(conn, task_id: str, kind: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    assert row is not None
    return json.loads(row["payload"] or "{}")


def test_missing_artifact_blocks_completion_with_typed_evidence(kanban_home: Path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="missing evidence", workspace_kind="dir", workspace_path=str(kanban_home))
        missing = kanban_home / "does-not-exist.txt"

        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="done",
                metadata={"artifacts": [str(missing)]},
            )

        assert exc.value.kind == "evidence_missing"
        assert kb.get_task(conn, task_id).status == "ready"
        payload = _event_payload(conn, task_id, "evidence_missing")
        assert payload["path"] == str(missing)
        assert not _artifact_rows(conn, task_id)


def test_unreadable_and_unsafe_artifacts_fail_closed(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    allowed = kanban_home / "workspace"
    allowed.mkdir()
    unreadable = allowed / "report.txt"
    unreadable.write_text("report", encoding="utf-8")

    real_access = os.access
    monkeypatch.setattr(
        kb.os,
        "access",
        lambda path, mode: False if Path(path) == unreadable else real_access(path, mode),
    )
    with kb.connect() as conn:
        unreadable_task = kb.create_task(
            conn, title="unreadable", workspace_kind="dir", workspace_path=str(allowed)
        )
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                unreadable_task,
                summary="done",
                metadata={"artifacts": [str(unreadable)]},
            )
        assert exc.value.kind == "evidence_missing"
        assert kb.get_task(conn, unreadable_task).status == "ready"

    monkeypatch.setattr(kb.os, "access", real_access)
    with kb.connect() as conn:
        unsafe_task = kb.create_task(
            conn, title="unsafe", workspace_kind="dir", workspace_path=str(allowed)
        )
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                unsafe_task,
                summary="done",
                metadata={"artifacts": ["/etc/hosts"]},
            )
        assert exc.value.kind == "artifact_not_durable"
        assert kb.get_task(conn, unsafe_task).status == "ready"
        assert _event_payload(conn, unsafe_task, "artifact_not_durable")["path"] == "/etc/hosts"


def test_artifact_copy_rejects_symlink_swap_at_open(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = kanban_home / "race-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("safe", encoding="utf-8")
    real_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == artifact.name and kwargs.get("dir_fd") is not None and not swapped:
            artifact.unlink()
            artifact.symlink_to("/etc/hosts")
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(kb.os, "open", swap_then_open)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="symlink race",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="must not copy swapped source",
                metadata={"artifacts": [str(artifact)]},
            )
        assert swapped is True
        assert exc.value.kind == "artifact_promotion_failed"
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert not _artifact_rows(conn, task_id)


def test_artifact_promotion_fsyncs_destination_directory(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = kanban_home / "fsync-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")
    real_fsync = os.fsync
    fsynced_directory = False

    def observe_fsync(fd: int) -> None:
        nonlocal fsynced_directory
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            fsynced_directory = True
        real_fsync(fd)

    monkeypatch.setattr(kb.os, "fsync", observe_fsync)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="directory fsync",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="durable",
            metadata={"artifacts": [str(artifact)]},
        )
    assert fsynced_directory is True


def test_scratch_artifact_is_promoted_before_cleanup_and_manifest_survives(
    kanban_home: Path,
):
    content = b"durable completion evidence\n"
    expected_hash = hashlib.sha256(content).hexdigest()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="promote scratch")
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task_id, workspace)
        source = workspace / "report.txt"
        source.write_bytes(content)

        assert kb.complete_task(
            conn,
            task_id,
            summary="artifact promoted",
            metadata={"artifacts": [str(source)]},
        )

        assert not workspace.exists()
        rows = _artifact_rows(conn, task_id)
        assert len(rows) == 1
        manifest = rows[0]
        durable = Path(manifest["durable_path"])
        assert durable.is_file()
        assert durable.read_bytes() == content
        assert manifest["original_path"] == str(source)
        assert manifest["sha256"] == expected_hash
        assert manifest["size"] == len(content)
        assert manifest["producer_run_id"] == 0
        assert manifest["retention_class"] == "task_completion"
        assert manifest["validated_at"] > 0
        assert manifest["content_type"].startswith("text/")

        payload = _event_payload(conn, task_id, "completed")
        assert payload["artifacts"] == [str(durable)]
        assert payload["artifact_manifest"][0]["sha256"] == expected_hash
        run = kb.latest_run(conn, task_id)
        assert run.metadata["artifacts"] == [str(durable)]
        assert run.metadata["artifact_manifest"][0]["original_path"] == str(source)


def test_promotion_is_collision_safe_and_completion_retry_does_not_duplicate(
    kanban_home: Path,
):
    workspace = kanban_home / "workspace"
    workspace.mkdir()
    left = workspace / "left" / "result.txt"
    right = workspace / "right" / "result.txt"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("left", encoding="utf-8")
    right.write_text("right", encoding="utf-8")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="collision", workspace_kind="dir", workspace_path=str(workspace)
        )
        metadata = {"artifacts": [str(left), str(right), str(left)]}
        assert kb.complete_task(conn, task_id, summary="done", metadata=metadata)
        assert kb.complete_task(conn, task_id, summary="again", metadata=metadata) is False

        rows = _artifact_rows(conn, task_id)
        assert len(rows) == 2
        durable_paths = {row["durable_path"] for row in rows}
        assert len(durable_paths) == 2
        assert {Path(path).read_text(encoding="utf-8") for path in durable_paths} == {"left", "right"}
        assert len(
            conn.execute(
                "SELECT id FROM task_events WHERE task_id=? AND kind='completed'", (task_id,)
            ).fetchall()
        ) == 1


def test_rejected_created_cards_do_not_leave_orphaned_durable_artifacts(
    kanban_home: Path,
):
    workspace = kanban_home / "hallucination-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="reject phantom", workspace_kind="dir", workspace_path=str(workspace)
        )
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn,
                task_id,
                summary="not done",
                metadata={"artifacts": [str(artifact)]},
                created_cards=["t_deadbeefcafe"],
            )

        assert kb.get_task(conn, task_id).status == "ready"
        assert not _artifact_rows(conn, task_id)
        durable_dir = kb.completion_artifacts_root() / task_id / "0"
        assert not durable_dir.exists() or not list(durable_dir.iterdir())


def test_manifest_transaction_failure_is_typed_and_removes_promoted_file(
    kanban_home: Path,
):
    workspace = kanban_home / "rollback-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="rollback", workspace_kind="dir", workspace_path=str(workspace)
        )
        conn.execute(
            """
            CREATE TRIGGER reject_artifact_manifest
            BEFORE INSERT ON task_artifacts
            BEGIN
                SELECT RAISE(ABORT, 'manifest rejected');
            END
            """
        )
        conn.commit()

        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="not done",
                metadata={"artifacts": [str(artifact)]},
            )

        assert exc.value.kind == "artifact_promotion_failed"
        assert kb.get_task(conn, task_id).status == "ready"
        assert not _artifact_rows(conn, task_id)
        payload = _event_payload(conn, task_id, "artifact_promotion_failed")
        assert payload["path"] == str(artifact)
        durable_dir = kb.completion_artifacts_root() / task_id / "0"
        assert not durable_dir.exists() or not list(durable_dir.iterdir())


def test_named_board_completion_uses_named_board_artifact_store(kanban_home: Path):
    board = "project-a"
    workspace = kb.workspaces_root(board=board) / "task-workspace"
    workspace.mkdir(parents=True)
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")

    with kb.connect(board=board) as conn:
        task_id = kb.create_task(
            conn, title="named board", workspace_kind="dir", workspace_path=str(workspace)
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="done",
            metadata={"artifacts": [str(artifact)]},
            board=board,
        )
        row = _artifact_rows(conn, task_id)[0]

    durable = Path(row["durable_path"])
    assert durable.is_relative_to(kb.completion_artifacts_root(board=board))
    assert not durable.is_relative_to(kb.completion_artifacts_root(board="default"))


def test_concurrent_completion_cas_loser_preserves_committed_artifact(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = kanban_home / "concurrent-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="concurrent", workspace_kind="dir", workspace_path=str(workspace)
        )

    original_txn = kb._completion_artifact_txn
    ready = threading.Barrier(2)

    @contextlib.contextmanager
    def ordered_txn(conn, created_paths):
        paths = list(created_paths)
        ready.wait(timeout=5)
        if paths:
            # Make the creator of the shared destination lose the done CAS.
            time.sleep(0.1)
        with original_txn(conn, paths) as state:
            yield state

    monkeypatch.setattr(kb, "_completion_artifact_txn", ordered_txn)
    outcomes: list[bool] = []
    errors: list[Exception] = []

    def complete() -> None:
        try:
            with kb.connect() as conn:
                outcomes.append(
                    kb.complete_task(
                        conn,
                        task_id,
                        summary="done",
                        metadata={"artifacts": [str(artifact)]},
                    )
                )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=complete) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert sorted(outcomes) == [False, True]
    with kb.connect() as conn:
        rows = _artifact_rows(conn, task_id)
        assert len(rows) == 1
        assert Path(rows[0]["durable_path"]).read_text(encoding="utf-8") == "proof"


@pytest.mark.parametrize("status", ["ready", "blocked", "running"])
def test_all_completion_entry_statuses_share_evidence_gate(kanban_home: Path, status: str):
    workspace = kanban_home / f"workspace-{status}"
    workspace.mkdir()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title=status, workspace_kind="dir", workspace_path=str(workspace)
        )
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()

        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="done",
                metadata={"artifacts": [str(workspace / "missing.txt")]},
            )
        assert exc.value.kind == "evidence_missing"
        assert kb.get_task(conn, task_id).status == status


def test_explicit_completion_contract_requires_tests_readback_and_artifacts(
    kanban_home: Path,
):
    workspace = kanban_home / "contract-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="contract",
            workspace_kind="dir",
            workspace_path=str(workspace),
            completion_contract={
                "tests_or_smokes": True,
                "readback": True,
                "artifacts": True,
            },
        )
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="not enough",
                metadata={"artifacts": [str(artifact)]},
            )
        assert exc.value.kind == "evidence_missing"
        assert set(exc.value.details["missing"]) == {"tests_or_smokes", "readback"}
        assert kb.get_task(conn, task_id).status == "ready"

        assert kb.complete_task(
            conn,
            task_id,
            summary="verified",
            metadata={
                "artifacts": [str(artifact)],
                "tests_or_smokes": ["scripts/run_tests.sh tests/hermes_cli/test_x.py -q"],
                "readback": "durable file and manifest read back",
            },
        )
        assert kb.get_task(conn, task_id).status == "done"


def test_legacy_minimal_completion_remains_compatible(kanban_home: Path):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy")
        assert kb.complete_task(conn, task_id, result="done")
        assert kb.get_task(conn, task_id).status == "done"


@pytest.mark.parametrize(
    "relative_path",
    [
        ".env.production",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".aws/credentials",
        ".kube/config",
        ".docker/config.json",
        ".config/gcloud/application_default_credentials.json",
        "pairing/device.json",
    ],
)
def test_secret_like_artifact_paths_fail_closed(kanban_home: Path, relative_path: str):
    workspace = kanban_home / "secret-workspace"
    artifact = workspace / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("secret", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="secret", workspace_kind="dir", workspace_path=str(workspace)
        )
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="nope",
                metadata={"artifacts": [str(artifact)]},
            )
        assert exc.value.kind == "artifact_not_durable"
        assert kb.get_task(conn, task_id).status == "ready"


def test_artifact_ancestor_symlink_swap_fails_closed(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = kanban_home / "ancestor-workspace"
    nested = workspace / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")
    outside = kanban_home / "outside"
    outside.mkdir()
    (outside / "proof.txt").write_text("outside", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="ancestor race", workspace_kind="dir", workspace_path=str(workspace)
        )

    real_open = os.open
    swapped = False

    def swap_ancestor(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "nested" and kwargs.get("dir_fd") is not None and not swapped:
            artifact.unlink()
            nested.rmdir()
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(kb.os, "open", swap_ancestor)
    with kb.connect() as conn:
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="nope",
                metadata={"artifacts": [str(artifact)]},
            )
        assert exc.value.kind == "artifact_promotion_failed"
        assert kb.get_task(conn, task_id).status == "ready"
        assert not _artifact_rows(conn, task_id)


def test_artifact_destination_ancestor_symlink_swap_fails_closed(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    workspace = kanban_home / "destination-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")
    destination_parent = kanban_home / "dest-parent"
    outside = kanban_home / "outside-destination"
    outside.mkdir()
    monkeypatch.setenv(
        "HERMES_KANBAN_ARTIFACTS_ROOT", str(destination_parent / "artifacts")
    )
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="destination race", workspace_kind="dir", workspace_path=str(workspace)
        )

    real_open = os.open
    swapped = False

    def swap_destination_ancestor(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == destination_parent.name and kwargs.get("dir_fd") is not None and not swapped:
            destination_parent.rmdir()
            destination_parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(kb.os, "open", swap_destination_ancestor)
    with kb.connect() as conn:
        with pytest.raises(kb.CompletionEvidenceError) as exc:
            kb.complete_task(
                conn,
                task_id,
                summary="nope",
                metadata={"artifacts": [str(artifact)]},
            )
        assert exc.value.kind == "artifact_promotion_failed"
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "ready"


def test_same_content_and_basename_preserve_both_manifests(kanban_home: Path):
    workspace = kanban_home / "duplicate-workspace"
    first = workspace / "first" / "proof.txt"
    second = workspace / "second" / "proof.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="duplicates", workspace_kind="dir", workspace_path=str(workspace)
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="done",
            metadata={"artifacts": [str(first), str(second)]},
        )
        rows = _artifact_rows(conn, task_id)
    assert len(rows) == 2
    assert len({row["durable_path"] for row in rows}) == 2
    assert {row["original_path"] for row in rows} == {str(first), str(second)}


def test_startup_scavenger_removes_only_stale_unreferenced_files(kanban_home: Path):
    workspace = kanban_home / "scavenge-workspace"
    workspace.mkdir()
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="scavenge", workspace_kind="dir", workspace_path=str(workspace)
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="done",
            metadata={"artifacts": [str(artifact)]},
        )
        durable = Path(_artifact_rows(conn, task_id)[0]["durable_path"])
        orphan = durable.parent / ".promoting-orphan"
        orphan.write_text("orphan", encoding="utf-8")
        old = time.time() - 7200
        os.utime(orphan, (old, old))
        assert kb._scavenge_completion_artifacts(
            conn, board=None, grace_seconds=3600, now=time.time()
        ) == 1
    assert durable.exists()
    assert not orphan.exists()


def test_strict_notifier_accepts_durable_completion_artifact(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
):
    from gateway.platforms.base import BasePlatformAdapter

    durable = kb.completion_artifacts_root() / "task" / "1" / "proof.txt"
    durable.parent.mkdir(parents=True)
    durable.write_text("proof", encoding="utf-8")
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_TRUST_RECENT", "0")
    assert BasePlatformAdapter.validate_media_delivery_path(str(durable)) == str(
        durable.resolve()
    )


def test_cli_complete_renders_typed_evidence_rejection(kanban_home: Path):
    from hermes_cli.kanban import run_slash

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="cli contract",
            completion_contract={"artifacts": True},
        )
    output = run_slash(f"complete {task_id}")
    assert "evidence_missing" in output
    assert "Traceback" not in output

from __future__ import annotations

import time
from pathlib import Path, PureWindowsPath

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    workspaces = tmp_path / "board-data" / "workspaces"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces))
    kb.init_db()
    return home


@pytest.mark.parametrize("field", ["title", "body"])
def test_create_rejects_foreign_workspace_reference_in_title_or_body(
    kanban_home: Path,
    field: str,
) -> None:
    with kb.connect_closing() as conn:
        foreign_id = kb.create_task(conn, title="producer")
        foreign_workspace = kb.workspaces_root() / foreign_id
        kwargs = {
            "title": f"Consume {foreign_workspace}" if field == "title" else "consumer",
            "body": (
                f"Read {foreign_workspace / 'result.json'}"
                if field == "body"
                else None
            ),
        }

        with pytest.raises(ValueError, match="scratch workspace") as exc_info:
            kb.create_task(conn, **kwargs)

    message = str(exc_info.value)
    assert foreign_id in message
    assert str(foreign_workspace) in message


def test_create_rejects_foreign_windows_workspace_reference(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows_root = PureWindowsPath(r"C:\kanban\workspaces")
    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: windows_root)

    with kb.connect_closing() as conn:
        foreign_id = kb.create_task(conn, title="producer")
        foreign_workspace = windows_root / foreign_id

        with pytest.raises(ValueError, match="scratch workspace") as exc_info:
            kb.create_task(
                conn,
                title="consumer",
                body=f"Read {foreign_workspace / 'result.json'}",
            )

    assert str(foreign_workspace) in str(exc_info.value)


def test_create_allows_foreign_workspace_reference_with_explicit_opt_out(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        foreign_id = kb.create_task(conn, title="parallel sibling")
        foreign_workspace = kb.workspaces_root() / foreign_id

        created_id = kb.create_task(
            conn,
            title="Intentional live sibling handoff",
            body=f"Read {foreign_workspace / 'result.json'} while sibling runs",
            allow_workspace_refs=True,
        )

        assert kb.get_task(conn, created_id) is not None


def test_create_allows_reference_to_its_own_workspace(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    own_id = "t_1234abcd"
    monkeypatch.setattr(kb, "_new_task_id", lambda: own_id)

    with kb.connect_closing() as conn:
        created_id = kb.create_task(
            conn,
            title="Self-contained task",
            body=f"Write output to {kb.workspaces_root() / own_id / 'result.json'}",
        )

    assert created_id == own_id


def test_rejection_lists_durable_artifact_paths_for_referenced_task(
    kanban_home: Path,
    tmp_path: Path,
) -> None:
    durable_path = tmp_path / "artifacts" / "producer" / "result.json"
    with kb.connect_closing() as conn:
        foreign_id = kb.create_task(conn, title="producer")
        conn.execute(
            """
            INSERT INTO task_artifacts (
                task_id, producer_run_id, original_path, durable_path, sha256,
                size, content_type, validated_at, retention_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                foreign_id,
                7,
                str(kb.workspaces_root() / foreign_id / "result.json"),
                str(durable_path),
                "0" * 64,
                12,
                "application/json",
                int(time.time()),
                "standard",
            ),
        )
        conn.commit()

        with pytest.raises(ValueError) as exc_info:
            kb.create_task(
                conn,
                title="consumer",
                body=f"Use {kb.workspaces_root() / foreign_id / 'result.json'}",
            )

    message = str(exc_info.value)
    assert "durable artifact" in message.lower()
    assert str(durable_path) in message

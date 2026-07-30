"""Tests for the Kanban dashboard plugin backend (plugins/kanban/dashboard/plugin_api.py).

The plugin mounts as /api/plugins/kanban/ inside the dashboard's FastAPI app,
but here we attach its router to a bare FastAPI instance so we can test the
REST surface without spinning up the whole dashboard.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_exact_terminal_action_requires_combined_approval_and_resume(client, tmp_path):
    task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "publish", "assignee": "backend-eng"},
    ).json()["task"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kb.connect() as conn:
        claimed = kb.claim_task(conn, task["id"], claimer="test")
        assert claimed is not None
        action = kb.record_pending_action(
            conn,
            task_id=task["id"],
            run_id=claimed.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic",
            summary="git push --force-with-lease to topic",
            profile="backend-eng",
            workspace=str(workspace),
            expires_at=int(time.time()) + 600,
        )
        assert kb.block_task(
            conn, task["id"], kind="needs_input",
            reason="terminal approval required",
            expected_run_id=claimed.current_run_id,
        )

    plain = client.patch(
        f"/api/plugins/kanban/tasks/{task['id']}", json={"status": "ready"},
    )
    assert plain.status_code == 409

    detail = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]
    attention = detail["attention"]
    assert attention["state"] == "pending"
    assert attention["operator_summary"] == kb.PENDING_ACTION_OPERATOR_SUMMARY

    approved = client.post(
        f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action",
        json={"attention_id": attention["id"], "attention_version": attention["version"]},
    )
    assert approved.status_code == 200, approved.text
    with kb.connect() as conn:
        resumed = kb.get_task(conn, task["id"])
        assert resumed is not None and resumed.status == "ready"
        assert resumed.current_run_id is None

    replay = client.post(
        f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action",
        json={"attention_id": attention["id"], "attention_version": attention["version"]},
    )
    assert replay.status_code == 410


def test_approve_terminal_action_race_conflicts_but_post_approval_replays_are_gone(
    client, tmp_path, monkeypatch,
):
    """A pre-CAS pending contender conflicts; an observed approved replay is gone."""
    task, _ = _pending_exact_action(client, tmp_path)
    other, _ = _pending_exact_action(client, tmp_path / "other")
    route = f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action"
    attention = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    payload = {"attention_id": attention["id"], "attention_version": attention["version"]}

    # Foreign opaque attention and an active stale version remain ordinary
    # not-found/conflict cases before the actual CAS race.
    assert client.post(
        f"/api/plugins/kanban/tasks/{other['id']}/approve-terminal-action", json=payload,
    ).status_code == 404
    assert client.post(route, json={**payload, "attention_version": attention["version"] + 1}).status_code == 409
    with kb.connect() as conn:
        unchanged = kb.get_action_by_attention_id(conn, task["id"], attention["id"])
        task_before_race = kb.get_task(conn, task["id"])
        assert unchanged is not None and (unchanged.state, unchanged.version) == ("pending", attention["version"])
        assert task_before_race is not None and task_before_race.status == "blocked"

    plugin = sys.modules["hermes_dashboard_plugin_kanban_test"]
    barrier = threading.Barrier(2)
    responses = []

    def approve() -> None:
        responses.append(client.post(route, json=payload))

    with monkeypatch.context() as patch:
        patch.setattr(
            plugin,
            "_approve_terminal_action_after_snapshot_hook",
            lambda: barrier.wait(timeout=5),
        )
        workers = [threading.Thread(target=approve) for _ in range(2)]
        [worker.start() for worker in workers]
        [worker.join(timeout=5) for worker in workers]
    assert not any(worker.is_alive() for worker in workers)
    assert sorted(response.status_code for response in responses) == [200, 409]

    with kb.connect() as conn:
        action = kb.get_action_by_attention_id(conn, task["id"], attention["id"])
        assert action is not None and (action.state, action.version) == ("approved", attention["version"] + 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='terminal_approval_granted'", (task["id"],),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'", (task["id"],),
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE id=? AND status='ready'", (task["id"],)).fetchone()[0] == 1

    # The original opaque version is now a terminal replay, while the current
    # approved version is also non-actionable. Neither response leaks action IDs.
    replay = client.post(route, json=payload)
    current = client.post(route, json={**payload, "attention_version": attention["version"] + 1})
    assert replay.status_code == current.status_code == 410
    assert "action_id" not in repr({"replay": replay.json(), "current": current.json()})


_ATTENTION_FIELDS = {
    "id", "type", "requires_human_action", "approvable", "mutation_kind",
    "state", "created_at", "expires_at", "version", "operator_summary",
}


def _pending_exact_action(client, tmp_path, *, board=None):
    query = f"?board={board}" if board else ""
    task = client.post(f"/api/plugins/kanban/tasks{query}", json={"title": "exact", "assignee": "worker"}).json()["task"]
    workspace = tmp_path / (board or "default") / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    with kb.connect(board=board) as conn:
        claimed = kb.claim_task(conn, task["id"], claimer="test")
        assert claimed is not None
        action = kb.record_pending_action(
            conn, task_id=task["id"], run_id=claimed.current_run_id,
            command="P5_RAW_COMMAND_MARKER", summary="P5_SUMMARY_MARKER",
            profile="P5_PROFILE_MARKER", workspace=str(workspace), expires_at=int(time.time()) + 600,
        )
        assert kb.block_task(conn, task["id"], kind="needs_input", reason="P5_BLOCKER_MARKER", expected_run_id=claimed.current_run_id)
    return task, action


def _board_task(payload, task_id):
    return next(t for col in payload["columns"] for t in col["tasks"] if t["id"] == task_id)




def test_named_board_identity_is_returned_by_board_detail_update_and_approve(client, tmp_path):
    board = "p5-named"
    kb.create_board(board)
    task, _ = _pending_exact_action(client, tmp_path, board=board)
    board_payload = client.get(f"/api/plugins/kanban/board?board={board}").json()
    detail = client.get(f"/api/plugins/kanban/tasks/{task['id']}?board={board}").json()
    updated = client.patch(f"/api/plugins/kanban/tasks/{task['id']}?board={board}", json={"title": "renamed"}).json()
    attention = detail["task"]["attention"]
    approved = client.post(
        f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action?board={board}",
        json={"attention_id": attention["id"], "attention_version": attention["version"]},
    )
    assert approved.status_code == 200, approved.text
    assert {board_payload["board"], detail["board"], updated["board"], approved.json()["board"]} == {board}
    assert client.get("/api/plugins/kanban/board").json()["board"] == "default"


def test_resume_approved_action_retry_is_opaque_and_versioned(client, tmp_path):
    task, action = _pending_exact_action(client, tmp_path)
    pending = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action", json={"attention_id": pending["id"], "attention_version": pending["version"]}).status_code == 200
    with kb.connect() as conn:
        resumed = kb.claim_task(conn, task["id"], claimer="retry-worker")
        assert resumed is not None and resumed.current_run_id is not None
        assert kb.block_approved_action_for_technical_failure(
            conn, task_id=task["id"], action_id=action.id, expected_run_id=resumed.current_run_id,
            attention_type="capability", reason_code="missing_capability", now=int(time.time()),
        )
        db_origin_run_id = resumed.current_run_id
    technical = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert technical["type"] == "capability"
    # 2026-07-15: take origin_run_id from the API, not from the DB. This route
    # requires it, and until now the schema never emitted it -- so this test
    # passed while no real client could ever call the endpoint, and the only
    # escape from a technical park sat unreachable behind a button that did not
    # exist. Reading it out of the DB here hid exactly that.
    origin_run_id = technical["origin_run_id"]
    assert origin_run_id == db_origin_run_id
    route = f"/api/plugins/kanban/tasks/{task['id']}/resume-approved-action-retry"
    base = {"attention_id": technical["id"], "attention_version": technical["version"], "origin_run_id": origin_run_id}
    assert client.post(route, json={**base, "origin_run_id": origin_run_id + 1}).status_code == 409
    assert client.post(route, json={**base, "attention_version": technical["version"] + 1}).status_code == 409
    for hostile in ({**base, "attention_id": True}, {**base, "action_id": action.id}, {**base, "origin_run_id": True}):
        assert client.post(route, json=hostile).status_code == 422
    response = client.post(route, json=base)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "ok": True, "board": "default", "task_id": task["id"], "attention_id": technical["id"],
        "attention_version": technical["version"], "task_status": "ready",
    }
    assert "action_id" not in repr(response.json())
    assert client.post(route, json=base).status_code == 410


def test_versioned_attention_routes_and_named_board_isolation(client, tmp_path):
    kb.create_board("other")
    task, _ = _pending_exact_action(client, tmp_path)
    other, _ = _pending_exact_action(client, tmp_path, board="other")
    attention = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/resolve-terminal-action?board=other", json={"attention_id": attention["id"], "attention_version": attention["version"]}).status_code == 404
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/resolve-terminal-action", json={"attention_id": attention["id"], "attention_version": attention["version"] + 1}).status_code == 409
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/resolve-terminal-action", json={"attention_id": attention["id"], "attention_version": attention["version"]}).status_code == 200
    assert "attention" not in client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]
    # Terminal/replay identity is task-bound opaque history and therefore
    # gone (not a new authorization conflict or an event reconstruction).
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/resolve-terminal-action", json={"attention_id": attention["id"], "attention_version": attention["version"]}).status_code == 410
    assert client.get(f"/api/plugins/kanban/tasks/{other['id']}?board=other").json()["task"]["attention"]["id"]
    assert client.post(f"/api/plugins/kanban/tasks/{other['id']}/resolve-terminal-action?board=other", json={"attention_id": True, "attention_version": 1}).status_code == 422
    assert client.post(f"/api/plugins/kanban/tasks/{other['id']}/resolve-terminal-action?board=other", json={"attention_id": 1, "attention_version": True}).status_code == 422


def test_approved_current_attention_is_non_actionable_and_terminal_absent(client, tmp_path):
    task, _ = _pending_exact_action(client, tmp_path)
    pending = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action", json={"attention_id": pending["id"], "attention_version": pending["version"]}).status_code == 200
    approved = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert approved["id"] == pending["id"] and approved["version"] == pending["version"] + 1
    assert approved["state"] == "approved-awaiting-worker"
    assert approved["requires_human_action"] is False and approved["approvable"] is False
    assert client.post(f"/api/plugins/kanban/tasks/{task['id']}/resolve-terminal-action", json={"attention_id": approved["id"], "attention_version": approved["version"]}).status_code == 200
    assert "attention" not in client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]


# ---------------------------------------------------------------------------
# GET /board on an empty DB
# ---------------------------------------------------------------------------


def test_board_empty(client):
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    # All canonical columns present (triage + the rest), each empty.
    names = [c["name"] for c in data["columns"]]
    assert set(names) == kb.VALID_STATUSES - {"archived"}
    for expected in ("triage", "todo", "scheduled", "ready", "running", "blocked", "done"):
        assert expected in names, f"missing column {expected}: {names}"
    assert all(len(c["tasks"]) == 0 for c in data["columns"])
    assert data["tenants"] == []
    assert data["assignees"] == []
    assert data["latest_event_id"] == 0


# ---------------------------------------------------------------------------
# POST /tasks then GET /board sees it
# ---------------------------------------------------------------------------


def test_create_task_appears_on_board(client):
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "Research LLM caching",
            "assignee": "researcher",
            "priority": 3,
            "tenant": "acme",
        },
    )
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    assert task["title"] == "Research LLM caching"
    assert task["assignee"] == "researcher"
    assert task["status"] == "ready"  # no parents -> immediately ready
    assert task["priority"] == 3
    assert task["tenant"] == "acme"
    task_id = task["id"]

    # Board now lists it under 'ready'.
    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    data = r.json()
    ready = next(c for c in data["columns"] if c["name"] == "ready")
    assert len(ready["tasks"]) == 1
    assert ready["tasks"][0]["id"] == task_id
    assert "acme" in data["tenants"]
    assert "researcher" in data["assignees"]


def test_patch_board_sets_project_directory(client, tmp_path):
    """Board-level default_workdir must be editable after creation."""
    kb.create_board("late-config")
    project_dir = tmp_path / "late-project"
    project_dir.mkdir()

    response = client.patch(
        "/api/plugins/kanban/boards/late-config",
        json={"default_workdir": str(project_dir)},
    )

    assert response.status_code == 200, response.text
    board = response.json()["board"]
    assert board["default_workdir"] == str(project_dir.resolve())
    # The recommendation flips from scratch to a persistent kind so the
    # create-task dialog's workspace default follows the board setting.
    assert board["default_workspace_kind"] == "dir"
    assert kb.read_board_metadata("late-config")["default_workdir"] == str(
        project_dir.resolve()
    )


def test_scheduled_tasks_have_their_own_column_not_todo(client):
    """Scheduled/time-delay tasks must not be silently bucketed into todo."""

    task = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "wait for indexed data", "assignee": "ops"},
    ).json()["task"]

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'scheduled' WHERE id = ?",
                (task["id"],),
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    columns = {c["name"]: c["tasks"] for c in r.json()["columns"]}
    assert any(t["id"] == task["id"] for t in columns["scheduled"])
    assert not any(t["id"] == task["id"] for t in columns["todo"])


def test_tenant_filter(client):
    client.post("/api/plugins/kanban/tasks", json={"title": "A", "tenant": "t1"})
    client.post("/api/plugins/kanban/tasks", json={"title": "B", "tenant": "t2"})

    r = client.get("/api/plugins/kanban/board?tenant=t1")
    counts = {c["name"]: len(c["tasks"]) for c in r.json()["columns"]}
    total = sum(counts.values())
    assert total == 1

    r = client.get("/api/plugins/kanban/board?tenant=t2")
    total = sum(len(c["tasks"]) for c in r.json()["columns"])
    assert total == 1


def test_dashboard_markdown_html_is_sanitized_before_render():
    """Markdown rendering must sanitize HTML before dangerouslySetInnerHTML."""

    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert "function sanitizeMarkdownHtml(html)" in js
    assert "MARKDOWN_ALLOWED_TAGS" in js
    assert "sanitizeMarkdownHtml(renderMarkdown(props.source || \"\"))" in js
    assert "dangerouslySetInnerHTML: { __html: renderMarkdown(props.source || \"\") }" not in js


# ---------------------------------------------------------------------------
# GET /tasks/:id returns body + comments + events + links
# ---------------------------------------------------------------------------


def test_task_detail_includes_links_and_events(client):
    parent = client.post(
        "/api/plugins/kanban/tasks", json={"title": "parent"},
    ).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "child", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"  # parent not done yet

    # Detail for the child shows the parent link.
    r = client.get(f"/api/plugins/kanban/tasks/{child['id']}")
    assert r.status_code == 200
    data = r.json()
    assert data["task"]["id"] == child["id"]
    assert parent["id"] in data["links"]["parents"]

    # Detail for the parent shows the child.
    r = client.get(f"/api/plugins/kanban/tasks/{parent['id']}")
    assert child["id"] in r.json()["links"]["children"]

    # Events exist from creation.
    assert len(data["events"]) >= 1


# ---------------------------------------------------------------------------
# PATCH /tasks/:id — status transitions
# ---------------------------------------------------------------------------


def test_patch_status_complete(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "done", "result": "shipped"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "done"

    # Board reflects the move.
    done = next(
        c for c in client.get("/api/plugins/kanban/board").json()["columns"]
        if c["name"] == "done"
    )
    assert any(x["id"] == t["id"] for x in done["tasks"])


def test_patch_done_uses_named_board_artifact_store(client):
    board = "project-a"
    kb.create_board(board)
    workspace = kb.workspaces_root(board=board) / "dashboard-workspace"
    workspace.mkdir(parents=True)
    artifact = workspace / "proof.txt"
    artifact.write_text("proof", encoding="utf-8")
    with kb.connect(board=board) as conn:
        task_id = kb.create_task(
            conn, title="named", workspace_kind="dir", workspace_path=str(workspace)
        )

    response = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}?board={board}",
        json={
            "status": "done",
            "summary": "done",
            "metadata": {"artifacts": [str(artifact)]},
        },
    )
    assert response.status_code == 200
    with kb.connect(board=board) as conn:
        row = conn.execute(
            "SELECT durable_path FROM task_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()
    assert row is not None
    assert Path(row[0]).is_relative_to(kb.completion_artifacts_root(board=board))


def test_patch_done_returns_typed_evidence_rejection(client):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="contract",
            completion_contract={"artifacts": True},
        )
    response = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "done", "summary": "missing evidence"},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["kind"] == "evidence_missing"
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"


def test_bulk_done_returns_typed_evidence_rejection(client):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="contract",
            completion_contract={"artifacts": True},
        )
    response = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [task_id], "status": "done", "summary": "missing evidence"},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["ok"] is False
    assert result["error_kind"] == "evidence_missing"


def test_patch_block_then_unblock(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "blocked", "block_reason": "need input"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "blocked"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "ready"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "ready"


def test_patch_schedule_then_unblock(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "scheduled", "block_reason": "run tomorrow"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "scheduled"

    columns = client.get("/api/plugins/kanban/board").json()["columns"]
    assert "scheduled" in [c["name"] for c in columns]
    scheduled = next(c for c in columns if c["name"] == "scheduled")
    assert any(x["id"] == t["id"] for x in scheduled["tasks"])

    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "ready"},
    )
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "ready"


def test_patch_drag_drop_move_todo_to_ready(client):
    """Direct status write: the drag-drop path for statuses without a
    dedicated verb (e.g. manually promoting todo -> ready).

    Promoting a child whose parent is not done is rejected (409).
    Promoting a child whose parent IS done is accepted (200)."""
    parent = client.post("/api/plugins/kanban/tasks", json={"title": "p"}).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "c", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"

    # Rejected: parent not done yet.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{child['id']}",
        json={"status": "ready"},
    )
    assert r.status_code == 409

    # The 409 detail must name the blocking parent so the dashboard can
    # render an actionable toast instead of a silent no-op (#26744).
    detail = r.json()["detail"]
    assert "Cannot move to 'ready'" in detail
    assert parent["id"] in detail
    assert "'p'" in detail
    assert "status=" in detail
    # Whatever non-``done`` status the parent currently has must show up
    # so the operator knows what to fix.
    assert f"status={parent['status']}" in detail
    assert parent["status"] != "done"

    # Complete the parent.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200

    # Now child auto-promoted by recompute_ready — already ready.
    child_after = client.get(f"/api/plugins/kanban/tasks/{child['id']}").json()["task"]
    assert child_after["status"] == "ready"


def test_reopening_parent_demotes_ready_child(client):
    """Reopening a completed parent must invalidate ready children immediately.

    The dispatcher re-checks parent completion on claim, but the dashboard
    should not keep showing a stale child as ready after an operator drags
    its parent back out of done for more work.
    """
    parent = client.post("/api/plugins/kanban/tasks", json={"title": "p"}).json()["task"]
    child = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "c", "parents": [parent["id"]]},
    ).json()["task"]
    assert child["status"] == "todo"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "done"},
    )
    assert r.status_code == 200

    child_after_done = client.get(
        f"/api/plugins/kanban/tasks/{child['id']}"
    ).json()["task"]
    assert child_after_done["status"] == "ready"

    r = client.patch(
        f"/api/plugins/kanban/tasks/{parent['id']}",
        json={"status": "todo"},
    )
    assert r.status_code == 200

    child_after_reopen = client.get(
        f"/api/plugins/kanban/tasks/{child['id']}"
    ).json()["task"]
    assert child_after_reopen["status"] == "todo"


# ---------------------------------------------------------------------------
# Human-Gate v1 repair (2026-07-11 adversarial re-review). The dashboard
# is not in the original design's file list, but the review named
# `_set_status_direct` explicitly as an exit out of `blocked` to check —
# it had NO source-status restriction at all, the most dangerous of the
# bypasses found. All human_gate=1 state below is set directly via
# kanban_db (no dashboard endpoint can set/clear the gate, by design).
# ---------------------------------------------------------------------------

def test_patch_status_refuses_gated_blocked_card_every_verb(client, kanban_home):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    with kb.connect() as conn:
        kb.block_task(
            conn, t["id"], reason="needs approval",
            kind="needs_input", human_gate=True,
        )

    # 'done' (complete_task).
    r = client.patch(f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "done"})
    assert r.status_code == 409
    assert "human-gated" in r.json()["detail"]

    # 'ready' (routes through unblock_task since current status is blocked).
    r = client.patch(f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "ready"})
    assert r.status_code == 409
    assert "human-gated" in r.json()["detail"]

    # 'archived' (archive_task).
    r = client.patch(f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "archived"})
    assert r.status_code == 409
    assert "human-gated" in r.json()["detail"]

    # 'todo' — the generic drag-drop path (_set_status_direct). This is
    # the actual reviewer-named finding: this function had no status
    # check of any kind before the repair.
    r = client.patch(f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "todo"})
    assert r.status_code == 409
    assert "human-gated" in r.json()["detail"]

    with kb.connect() as conn:
        task = kb.get_task(conn, t["id"])
        assert task.status == "blocked"
        assert task.human_gate is True


def test_bulk_update_gracefully_refuses_gated_blocked_card(client, kanban_home):
    """The bulk endpoint's existing per-id `except Exception` catch-all
    already turns a GateTokenError into a graceful per-id failure — this
    just proves it, so a future change can't silently regress it into an
    unhandled 500 that aborts the whole batch."""
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    with kb.connect() as conn:
        kb.block_task(
            conn, t["id"], reason="needs approval",
            kind="needs_input", human_gate=True,
        )
    r = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [t["id"]], "status": "done"},
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert results[0]["ok"] is False
    assert "human-gated" in results[0]["error"]
    with kb.connect() as conn:
        assert kb.get_task(conn, t["id"]).status == "blocked"


def test_reclaim_endpoint_refuses_gated_stall_blocked_card(client, kanban_home):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    with kb.connect() as conn:
        assert kb.claim_task(conn, t["id"]) is not None
        # Reproduce detect_resource_stalls's shape: blocked, claim_lock left set.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked' WHERE id = ? AND status='running'",
                (t["id"],),
            )
        assert kb.set_human_gate(conn, t["id"], on=True, actor="manfred") is True

    r = client.post(f"/api/plugins/kanban/tasks/{t['id']}/reclaim", json={})
    assert r.status_code == 409
    assert "human-gated" in r.json()["detail"]
    with kb.connect() as conn:
        assert kb.get_task(conn, t["id"]).status == "blocked"


def test_patch_status_ungated_blocked_card_unaffected(client, kanban_home):
    """Behavioural neutrality: a plain (never human_gate=1) blocked card
    still moves through the dashboard exactly as before this repair."""
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}",
        json={"status": "blocked", "block_reason": "need input"},
    )
    assert r.status_code == 200
    r = client.patch(f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "ready"})
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "ready"


# ---------------------------------------------------------------------------
# DELETE /tasks/:id
# ---------------------------------------------------------------------------

def test_delete_task(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "to-delete"}).json()["task"]
    # Hard-delete is retention cleanup, not a lifecycle shortcut.
    r = client.delete(
        f"/api/plugins/kanban/tasks/{t['id']}", params={"reason": "retention cleanup"}
    )
    assert r.status_code == 409
    r = client.patch(
        f"/api/plugins/kanban/tasks/{t['id']}", json={"status": "archived"}
    )
    assert r.status_code == 200
    r = client.delete(
        f"/api/plugins/kanban/tasks/{t['id']}", params={"reason": "retention cleanup"}
    )
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert r.json()["task_id"] == t["id"]

    # Gone from board
    board = client.get("/api/plugins/kanban/board").json()
    all_ids = [tt["id"] for col in board["columns"] for tt in col["tasks"]]
    assert t["id"] not in all_ids

    # Gone from detail
    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    assert r.status_code == 404


def test_delete_task_not_found(client):
    r = client.delete(
        "/api/plugins/kanban/tasks/t_nonexistent",
        params={"reason": "retention cleanup"},
    )
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Comments + Links
# ---------------------------------------------------------------------------


def test_add_comment(client):
    t = client.post("/api/plugins/kanban/tasks", json={"title": "x"}).json()["task"]
    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/comments",
        json={"body": "how's progress?", "author": "teknium"},
    )
    assert r.status_code == 200

    r = client.get(f"/api/plugins/kanban/tasks/{t['id']}")
    comments = r.json()["comments"]
    assert len(comments) == 1
    assert comments[0]["body"] == "how's progress?"
    assert comments[0]["author"] == "teknium"


# ---------------------------------------------------------------------------
# Dispatch nudge
# ---------------------------------------------------------------------------


def test_dispatch_dry_run(client):
    client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "work", "assignee": "researcher"},
    )
    r = client.post("/api/plugins/kanban/dispatch?dry_run=true&max=4")
    assert r.status_code == 200
    body = r.json()
    # DispatchResult is serialized as a dataclass dict.
    assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# Triage column (new v1 status)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Progress rollup (done children / total children)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Auto-init on first board read
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WebSocket auth (query-param token)
# ---------------------------------------------------------------------------


def test_ws_events_rejects_when_token_required(tmp_path, monkeypatch):
    """Loopback mode: a missing or wrong ?token= must be rejected with
    policy-violation; the correct token is accepted. The kanban WS now
    delegates to web_server._ws_auth_ok, so we stub that with the real
    loopback-token semantics (auth_required False → constant-time token
    compare)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    # Stub web_server with a loopback-mode _ws_auth_ok (auth_required False →
    # accept only the correct ?token=). Mirrors the real gate's loopback path.
    import hermes_cli
    import types

    def _fake_ws_auth_ok(ws):
        return ws.query_params.get("token", "") == "secret-xyz"

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=_fake_ws_auth_ok,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    c = TestClient(app)

    # No token → policy violation close.
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events"):
            pass
    assert exc.value.code == 1008

    # Wrong token → policy violation close.
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/kanban/events?token=nope"):
            pass
    assert exc.value.code == 1008

    # Correct token → accepted (connect then close cleanly from our side).
    with c.websocket_connect(
        "/api/plugins/kanban/events?token=secret-xyz"
    ) as ws:
        assert ws is not None  # handshake succeeded


    # The bug symptom was a traceback; we don't assert on stderr because
    # capturing asyncio's internal "exception was never retrieved" logging
    # is flaky. The assertion that matters is: no CancelledError escaped.


# ---------------------------------------------------------------------------
# Bulk actions
# ---------------------------------------------------------------------------


def test_bulk_status_ready(client):
    a = client.post("/api/plugins/kanban/tasks", json={"title": "a"}).json()["task"]
    b = client.post("/api/plugins/kanban/tasks", json={"title": "b"}).json()["task"]
    c2 = client.post("/api/plugins/kanban/tasks", json={"title": "c"}).json()["task"]
    # Parent-less tasks land in "ready" already; push them to blocked first.
    for tid in (a["id"], b["id"], c2["id"]):
        client.patch(f"/api/plugins/kanban/tasks/{tid}",
                     json={"status": "blocked", "block_reason": "wait"})

    r = client.post("/api/plugins/kanban/tasks/bulk",
                    json={"ids": [a["id"], b["id"], c2["id"]], "status": "ready"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert all(r["ok"] for r in results)
    # All three are now ready.
    board = client.get("/api/plugins/kanban/board").json()
    ready = next(col for col in board["columns"] if col["name"] == "ready")
    ids = {t["id"] for t in ready["tasks"]}
    assert {a["id"], b["id"], c2["id"]}.issubset(ids)


# ---------------------------------------------------------------------------
# /config endpoint
# ---------------------------------------------------------------------------


def test_config_reads_dashboard_kanban_section(tmp_path, monkeypatch, client):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "dashboard:\n"
        "  kanban:\n"
        "    default_tenant: acme\n"
        "    lane_by_profile: false\n"
        "    include_archived_by_default: true\n"
        "    render_markdown: false\n"
    )
    r = client.get("/api/plugins/kanban/config")
    assert r.status_code == 200
    data = r.json()
    assert data["default_tenant"] == "acme"
    assert data["lane_by_profile"] is False
    assert data["include_archived_by_default"] is True
    assert data["render_markdown"] is False


# ---------------------------------------------------------------------------
# Runs surfacing (vulcan-artivus RFC feedback)
# ---------------------------------------------------------------------------

def test_task_detail_includes_runs(client):
    """GET /tasks/:id carries a runs[] array with the attempt history."""
    r = client.post("/api/plugins/kanban/tasks",
                    json={"title": "port x", "assignee": "worker"}).json()
    tid = r["task"]["id"]

    # Drive status running to force a run creation: PATCH to running
    # doesn't call claim_task (the PATCH path uses _set_status_direct),
    # so use the bulk/claim indirection via the kernel.
    import hermes_cli.kanban_db as _kb
    conn = _kb.connect()
    try:
        _kb.claim_task(conn, tid)
        _kb.complete_task(
            conn, tid,
            result="done",
            summary="tested on rate limiter",
            metadata={"changed_files": ["limiter.py"]},
        )
    finally:
        conn.close()

    d = client.get(f"/api/plugins/kanban/tasks/{tid}").json()
    assert "runs" in d
    assert len(d["runs"]) == 1
    run = d["runs"][0]
    assert run["outcome"] == "completed"
    assert set(run) == {
        "id", "task_id", "step_key", "status", "max_runtime_seconds",
        "last_heartbeat_at", "started_at", "ended_at", "outcome",
    }
    assert "worker" not in repr(run)
    assert "tested on rate limiter" not in repr(run)
    assert "limiter.py" not in repr(run)
    assert run["ended_at"] is not None


def test_task_detail_runs_empty_before_claim(client):
    """A task that's never been claimed has an empty runs[] list, not
    a missing key."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "fresh"}).json()
    d = client.get(f"/api/plugins/kanban/tasks/{r['task']['id']}").json()
    assert d["runs"] == []


def test_patch_status_done_with_summary_and_metadata(client):
    """PATCH /tasks/:id with status=done + summary + metadata must
    reach complete_task, so the dashboard has CLI parity."""
    # Create + claim.
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={
            "status": "done",
            "summary": "shipped the thing",
            "metadata": {"changed_files": ["a.py", "b.py"], "tests_run": 7},
        },
    )
    assert r.status_code == 200, r.text

    # The run must have the summary + metadata attached.
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, tid)
        assert run.outcome == "completed"
        assert run.summary == "shipped the thing"
        assert run.metadata == {"changed_files": ["a.py", "b.py"], "tests_run": 7}
    finally:
        conn.close()


def test_patch_status_done_without_summary_still_works(client):
    """Back-compat: PATCH without the new fields still completes."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "y", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "done", "result": "legacy shape"},
    )
    assert r.status_code == 200, r.text
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, tid)
        assert run.outcome == "completed"
        assert run.summary == "legacy shape"  # falls back to result
    finally:
        conn.close()


def test_patch_status_archive_closes_running_run(client):
    """PATCH to archived while running must close the in-flight run."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "z", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
        open_run = kb.latest_run(conn, tid)
        assert open_run.ended_at is None
    finally:
        conn.close()
    r = client.patch(
        f"/api/plugins/kanban/tasks/{tid}",
        json={"status": "archived"},
    )
    assert r.status_code == 200, r.text
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        assert task.current_run_id is None
        assert kb.latest_run(conn, tid).outcome == "reclaimed"
    finally:
        conn.close()


def test_event_dict_includes_run_id(client):
    """GET /tasks/:id returns events with run_id populated."""
    r = client.post("/api/plugins/kanban/tasks", json={"title": "e", "assignee": "worker"})
    tid = r.json()["task"]["id"]
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb.claim_task(conn, tid)
        run_id = kb.latest_run(conn, tid).id
        kb.complete_task(conn, tid, summary="wss")
    finally:
        conn.close()

    r = client.get(f"/api/plugins/kanban/tasks/{tid}")
    assert r.status_code == 200
    events = r.json()["events"]
    # Every event in the response must have a run_id key (None or int).
    for e in events:
        assert "run_id" in e, f"missing run_id in event: {e}"
    # completed event must have the actual run_id.
    comp = [e for e in events if e["kind"] == "completed"]
    assert comp[0]["run_id"] == run_id


# ---------------------------------------------------------------------------
# Per-task force-loaded skills via REST
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dispatcher-presence warning in POST /tasks response
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _task_dict — outer try/except fallback when task_age raises
#
# Background: kanban_db.task_age was hardened in 061a1830 to return None for
# corrupt timestamp values via _safe_int. The companion fix added a belt-and-
# suspenders try/except in plugin_api._task_dict so that *any future* exception
# from task_age (not just ValueError on '%s') still yields a usable dict
# instead of 500'ing GET /board for the entire org.
#
# kanban_db._safe_int / task_age corruption paths are covered in
# tests/hermes_cli/test_kanban_db.py. The OUTER fallback here is not, which
# means a refactor that drops the try/except would not be caught by CI. The
# tests below pin that contract.
# ---------------------------------------------------------------------------


_FALLBACK_AGE = {
    "created_age_seconds": None,
    "started_age_seconds": None,
    "time_to_complete_seconds": None,
}


# ---------------------------------------------------------------------------
# Home-channel subscription endpoints (#19534 follow-up: GUI opt-in)
# ---------------------------------------------------------------------------
#
# Dashboard surface for per-task, per-platform notification toggles. The
# backend endpoints read the live GatewayConfig, so tests set env vars
# (BOT_TOKEN + HOME_CHANNEL) to simulate a user who has run /sethome on
# telegram and discord.


@pytest.fixture
def with_home_channels(monkeypatch):
    """Simulate a user with home channels set on telegram and discord."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc:fake")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "1234567")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "42")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_NAME", "Main TG")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "disc_fake")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL", "9999999")
    monkeypatch.setenv("DISCORD_HOME_CHANNEL_NAME", "Main Discord")
    # Slack has a token but NO home — should be excluded from the list.
    monkeypatch.setenv("SLACK_BOT_TOKEN", "slack_fake")


def test_home_channels_lists_only_platforms_with_home(client, with_home_channels):
    """GET /home-channels returns entries only for platforms where the
    user has set a home; untoggled-subscribed bool is false by default."""
    r = client.get("/api/plugins/kanban/home-channels")
    assert r.status_code == 200
    platforms = {h["platform"] for h in r.json()["home_channels"]}
    assert platforms == {"telegram", "discord"}, (
        f"slack has a token but no home — must not appear. got {platforms}"
    )
    for h in r.json()["home_channels"]:
        assert h["subscribed"] is False


# ---------------------------------------------------------------------------
# Recovery endpoints (reclaim + reassign) and warnings field
# ---------------------------------------------------------------------------


def test_reclaim_endpoint_releases_running_claim(client):
    """POST /tasks/<id>/reclaim drops the claim, returns ok, and emits
    a manual reclaimed event."""
    import secrets
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="running", assignee="x")
        lock = secrets.token_hex(8)
        future = int(time.time()) + 3600
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, future, 99999, t),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (t, lock, future, 99999, int(time.time())),
        )
        run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, t))
        conn.commit()
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reclaim",
        json={"reason": "browser recovery"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["task_id"] == t

    # Confirm the task is back to ready.
    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT status, claim_lock FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
    finally:
        conn2.close()


def test_reassign_endpoint_switches_profile(client):
    """POST /tasks/<id>/reassign changes the assignee field."""
    conn = kb.connect()
    try:
        t = kb.create_task(conn, title="task", assignee="orig")
    finally:
        conn.close()

    r = client.post(
        f"/api/plugins/kanban/tasks/{t}/reassign",
        json={"profile": "newbie", "reclaim_first": False},
    )
    assert r.status_code == 200, r.text
    assert r.json()["assignee"] == "newbie"

    conn2 = kb.connect()
    try:
        row = conn2.execute(
            "SELECT assignee FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["assignee"] == "newbie"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Diagnostics endpoint (/api/plugins/kanban/diagnostics)
# ---------------------------------------------------------------------------


def test_diagnostics_endpoint_surfaces_blocked_hallucination(client):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        real = kb.create_task(conn, title="real", assignee="x", created_by="alice")
        import pytest as _pytest
        with _pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent, summary="phantom",
                created_cards=[real, "t_ffff00001234"],
            )
    finally:
        conn.close()

    r = client.get("/api/plugins/kanban/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    row = data["diagnostics"][0]
    assert row["task_id"] == parent
    assert row["diagnostics"][0]["kind"] == "hallucinated_cards"
    assert row["diagnostics"][0]["severity"] == "error"
    assert "t_ffff00001234" in row["diagnostics"][0]["data"]["phantom_ids"]


# ---------------------------------------------------------------------------
# POST /tasks/:id/specify — triage specifier endpoint
# ---------------------------------------------------------------------------


def _patch_specifier_response(monkeypatch, *, content, model="test-model"):
    """Helper: install a fake auxiliary client so the specifier endpoint
    can run without hitting any real provider."""
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    # specify_task routes through call_llm now (#35566) — mock it directly.
    fake_call = MagicMock(return_value=resp)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call)
    return fake_call


def test_specify_happy_path(client, monkeypatch):
    import json as jsonlib

    # Create a triage task.
    t = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "one-liner", "triage": True},
    ).json()["task"]
    assert t["status"] == "triage"

    _patch_specifier_response(
        monkeypatch,
        content=jsonlib.dumps(
            {"title": "Polished", "body": "**Goal**\nDo the thing."}
        ),
    )

    r = client.post(
        f"/api/plugins/kanban/tasks/{t['id']}/specify",
        json={"author": "ui-tester"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["task_id"] == t["id"]
    assert body["new_title"] == "Polished"

    # Task should have moved off the triage column.
    detail = client.get(f"/api/plugins/kanban/tasks/{t['id']}").json()["task"]
    assert detail["status"] in {"todo", "ready"}
    assert detail["title"] == "Polished"
    assert "**Goal**" in (detail["body"] or "")


# ---------------------------------------------------------------------------
# Final result visibility for Done cards
# ---------------------------------------------------------------------------


def test_task_detail_exposes_result_and_latest_summary_separately(client):
    """The drawer receives both source fields without a duplicate alias."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "Task with explicit result"},
    )
    task_id = r.json()["task"]["id"]
    client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": "done", "result": "The final answer is 42.", "summary": "short handoff"},
    )
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200
    data = r.json()["task"]
    assert data["result"] == "The final answer is 42."
    assert data["latest_summary"] == "short handoff"
    assert "final_result" not in data


def test_task_detail_exposes_latest_summary_when_result_is_empty(client):
    """Summary-only completions remain available to the drawer fallback."""
    conn = kb.connect()
    task_id = kb.create_task(conn, title="Task with only run summary")
    kb.claim_task(conn, task_id)
    kb.complete_task(conn, task_id, summary="Report written to /output/report.md")
    conn.close()

    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200
    data = r.json()["task"]
    assert data["status"] == "done"
    assert not data["result"]
    assert data["latest_summary"] == "Report written to /output/report.md"


def test_task_detail_latest_summary_none_when_nothing_recorded(client):
    """When no run summary exists, the existing field remains None."""
    r = client.post(
        "/api/plugins/kanban/tasks",
        json={"title": "Task with no result at all"},
    )
    task_id = r.json()["task"]["id"]
    r = client.get(f"/api/plugins/kanban/tasks/{task_id}")
    assert r.status_code == 200
    assert r.json()["task"]["latest_summary"] is None


def test_board_tasks_include_latest_summary(client):
    """Board cards already expose the summary used by the drawer fallback."""
    conn = kb.connect()
    task_id = kb.create_task(conn, title="Board card with summary only")
    kb.claim_task(conn, task_id)
    kb.complete_task(conn, task_id, summary="Done: see attachment")
    conn.close()

    r = client.get("/api/plugins/kanban/board")
    assert r.status_code == 200
    done_col = next(c for c in r.json()["columns"] if c["name"] == "done")
    card = next((t for t in done_col["tasks"] if t["id"] == task_id), None)
    assert card is not None
    assert "Done: see attachment" in card["latest_summary"]


def test_dashboard_done_final_result_section_rendered_from_summary():
    """Frontend must render Final Result section from run summary when task.result is empty."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    assert "t.result || t.latest_summary" in dist
    assert "Final Result (run summary)" in dist
    assert "No final result was recorded" in dist
    assert "orchestrator" in dist or "parent task" in dist


def test_task_detail_includes_child_result_summaries(client):
    """Parent drawers should receive the child results they need to render."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="Research topic")
        child = kb.create_task(conn, title="Collect sources")
        kb.link_tasks(conn, parent, child)
        kb.complete_task(conn, parent, summary="Delegated research to child tasks.")
        kb.recompute_ready(conn)
        kb.complete_task(conn, child, summary="Collected five primary sources.")

    response = client.get(f"/api/plugins/kanban/tasks/{parent}")

    assert response.status_code == 200
    assert response.json()["child_results"] == [
        {
            "id": child,
            "title": "Collect sources",
            "status": "done",
            "latest_summary": "Collected five primary sources.",
            "result": None,
        }
    ]


def test_dashboard_final_result_uses_existing_fields_without_alias():
    """The drawer should not duplicate result/summary into another API field."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    api = (repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py").read_text()

    assert "var finalResult = t.result || t.latest_summary || null;" in dist
    assert "t.final_result" not in dist
    assert 'd["final_result"]' not in api


def test_dashboard_parent_notice_and_child_results_use_detail_links():
    """Parent detection must use links.children, which exists in task detail."""
    repo_root = Path(__file__).resolve().parents[2]
    dist = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    detail = dist[dist.index("function TaskDetail"):]

    assert "links.children.length > 0" in detail
    assert "t.link_counts" not in detail
    assert "Child Results" in detail
    assert "props.data.child_results" in detail


def test_current_attention_contract_and_redaction(client, tmp_path):
    task, action = _pending_exact_action(client, tmp_path)
    board_task = _board_task(client.get("/api/plugins/kanban/board").json(), task["id"])
    detail = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()
    assert board_task["attention"] == detail["task"]["attention"]
    assert set(detail["task"]["attention"]) == _ATTENTION_FIELDS
    public = repr({"board": board_task, "detail": detail})
    # Option 1 (2026-07-14 merge): the block reason now surfaces via latest_summary on
    # the operator dashboard (upstream Done-card direction) — dropped from the redaction
    # set. The raw exact-action command, the action summary, the profile, and the
    # command hashes/fingerprint still NEVER leak.
    for marker in ("P5_RAW_COMMAND_MARKER", "P5_SUMMARY_MARKER", "P5_PROFILE_MARKER", action.command_hash, action.fingerprint):
        assert marker not in public
    assert "pending_terminal_action" not in public


def test_task_payload_is_a_recursive_public_whitelist(client, tmp_path):
    task = client.post("/api/plugins/kanban/tasks", json={"title": "safe", "body": "public body", "assignee": "worker"}).json()["task"]
    markers = {
        "workspace_path": "P5_WORKSPACE_MARKER", "claim_lock": "P5_CLAIM_MARKER",
        "last_failure_error": "P5_FAILURE_MARKER", "result": "P5_RESULT_MARKER",
        "session_id": "P5_SESSION_MARKER", "run_summary": "P5_RUN_MARKER",
        "run_error": "P5_RUN_ERROR_MARKER", "run_metadata": "P5_RUN_METADATA_MARKER",
        "event_metadata": "P5_EVENT_METADATA_MARKER",
    }
    with kb.connect() as conn:
        claimed = kb.claim_task(conn, task["id"], claimer="redaction-worker")
        assert claimed is not None and claimed.current_run_id is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_path=?, claim_lock=?, last_failure_error=?, result=?, session_id=? WHERE id=?",
                (markers["workspace_path"], markers["claim_lock"], markers["last_failure_error"], markers["result"], markers["session_id"], task["id"]),
            )
            conn.execute(
                "UPDATE task_runs SET summary=?, metadata=?, error=? WHERE id=?",
                (markers["run_summary"], markers["run_metadata"], markers["run_error"], claimed.current_run_id),
            )
            conn.execute("UPDATE task_events SET payload=? WHERE task_id=?", (markers["event_metadata"], task["id"]))
    board = client.get("/api/plugins/kanban/board").json()
    detail = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()
    expected = {
        "id", "title", "body", "assignee", "status", "priority", "tenant", "created_at", "started_at", "completed_at",
        "max_runtime_seconds", "workflow_template_id", "current_step_key", "goal_mode", "goal_max_turns", "skills", "block_reason", "age", "latest_summary",
    }
    board_task = _board_task(board, task["id"])
    assert expected <= set(board_task)
    assert set(board_task) == expected | {"link_counts", "comment_count", "progress"}
    # Option 1 (2026-07-14 merge): the detail drawer additionally exposes `result`
    # (upstream Done-card direction). The board LIST stays result-free (our privacy).
    assert set(detail["task"]) == expected | {"result"}
    assert board_task["block_reason"] is None and detail["task"]["block_reason"] is None
    # `result` is now intentionally surfaced on the detail drawer, but must NOT leak
    # into the board list; every OTHER internal marker still never surfaces anywhere.
    assert markers["result"] not in repr(board)
    public = repr({"board": board, "detail": detail}) + repr({"board": board, "detail": detail})
    # Option 1 exposes `latest_summary` (the worker-authored run summary) on both the
    # board and the drawer — so `run_summary` is intentionally public now. `result` is
    # drawer-only. Every OTHER internal marker (command, hashes, profile, run_metadata,
    # run_error, session_id, …) must still never surface anywhere.
    for name, marker in markers.items():
        if name in ("result", "run_summary"):
            continue
        assert marker not in public

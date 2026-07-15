"""2026-07-15: the dashboard must be able to approve an exact action on a
HUMAN-GATED card.

Twin of the WebUI-bridge defect found the same day. Approving an exact action
also unblocks the card, so the Core enforces the human gate on it. This endpoint
passed no token, so every gated card refused approval -- and since /unblock
refuses while an exact action is pending, the card had no exit at all.

The endpoint's neighbours (gate_off, archive_running) already issue + redeem a
grant server-side on exactly this reasoning: a dashboard request sits behind the
web-server auth gate, so an authenticated human IS the operator act.

test_kanban_dashboard_plugin.py covers this endpoint well, but only ever on
UNGATED cards -- which is why the deadlock shipped.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb

from .test_kanban_dashboard_plugin import _load_plugin_router


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


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _card_with_pending_action(client, tmp_path, *, gated: bool):
    task = client.post(
        "/api/plugins/kanban/tasks", json={"title": "publish", "assignee": "backend-eng"},
    ).json()["task"]
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    with kb.connect() as conn:
        claimed = kb.claim_task(conn, task["id"], claimer="test")
        kb.record_pending_action(
            conn, task_id=task["id"], run_id=claimed.current_run_id,
            command="git push --force-with-lease=refs/heads/topic:abcdef1 fork HEAD:topic",
            summary="git push --force-with-lease to topic",
            profile="backend-eng", workspace=str(workspace),
            expires_at=int(time.time()) + 600,
        )
        assert kb.block_task(
            conn, task["id"], kind="needs_input", reason="terminal approval required",
            expected_run_id=claimed.current_run_id,
        )
        if gated:
            assert kb.set_human_gate(conn, task["id"], on=True, actor="test")
    detail = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]
    return task, detail["attention"]


def test_gated_card_approval_succeeds_via_dashboard(client, tmp_path):
    """The regression: this returned 409 before the fix, with no way out."""
    task, attention = _card_with_pending_action(client, tmp_path, gated=True)

    approved = client.post(
        f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action",
        json={"attention_id": attention["id"], "attention_version": attention["version"]},
    )

    assert approved.status_code == 200, approved.text
    with kb.connect() as conn:
        resumed = kb.get_task(conn, task["id"])
        assert resumed.status == "ready"
        assert bool(resumed.human_gate) is True, "the gate flag itself must persist"
        # The minted grant is one-shot and must not linger as a replayable token.
        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["gate_token_hash"] is None


def test_ungated_card_approval_still_succeeds(client, tmp_path):
    """Behavioural neutrality: ungated cards mint nothing and are unchanged."""
    task, attention = _card_with_pending_action(client, tmp_path, gated=False)

    approved = client.post(
        f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action",
        json={"attention_id": attention["id"], "attention_version": attention["version"]},
    )

    assert approved.status_code == 200, approved.text
    with kb.connect() as conn:
        assert kb.get_task(conn, task["id"]).status == "ready"
        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id=?", (task["id"],)
        ).fetchone()
        assert row["gate_token_hash"] is None, "no gate -> no token may ever be minted"


def test_gate_refusal_is_403_not_a_retryable_409(client, tmp_path, monkeypatch):
    """A gate refusal must not be dressed up as a retryable conflict.

    409 tells the operator "state moved, refresh and retry"; if the server
    genuinely cannot satisfy the gate, retrying is futile and must say so.
    """
    task, attention = _card_with_pending_action(client, tmp_path, gated=True)

    # Simulate a server that cannot mint a grant for this card.
    import hermes_cli.kanban_db as _kb
    monkeypatch.setattr(_kb, "issue_gate_token", lambda *a, **k: None)

    refused = client.post(
        f"/api/plugins/kanban/tasks/{task['id']}/approve-terminal-action",
        json={"attention_id": attention["id"], "attention_version": attention["version"]},
    )

    assert refused.status_code == 403, refused.text
    assert "human-gated" in refused.json()["detail"]
    with kb.connect() as conn:
        assert kb.get_task(conn, task["id"]).status == "blocked", "refusal must be inert"

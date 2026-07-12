"""Regression tests for the self-modification governance gate (bounded autonomy).

A ready card whose work would change how Hermes itself runs (config, worker
profiles, systemd, agent code, runtime plugins/skills, containment) must be
routed to blocked+human_gate by the dispatcher so a human approves before it
can run. Work on the operator's own projects/clients/docs is NOT gated. The
gate reuses Human-Gate v1 and approves once per card (a card already carrying
human_gate=1 has passed the gate).
"""
from __future__ import annotations

import sys
import tempfile
from types import SimpleNamespace

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_selfmod_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    # Pretend every assignee resolves to a real profile so the dispatcher
    # reaches the gate instead of bucketing the card as non-spawnable.
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True, raising=False)
    yield kanban_db, test_home


def _spawn(*args, **kwargs):
    return 12345


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,body,wp", [
    ("Fix worker profile config.yaml disabled_toolsets", "", None),
    ("Restart hermes-gateway.service after change", "systemctl --user restart", None),
    ("Repair kanban_db respawn guard", "edit hermes-agent dispatcher", None),
    ("Update tars-config-guard plugin", "", None),
    ("Set disabled_toolsets for workers", "docker_image swap", None),
    ("edit ~/.hermes/config.yaml", "", None),
    ("whatever", "", "/home/manfred/.hermes/workspace/repos/hermes-agent"),
])
def test_classifier_flags_runtime_changes(isolated_kanban_home, title, body, wp):
    kb, _ = isolated_kanban_home
    task = SimpleNamespace(title=title, body=body, workspace_path=wp, branch_name=None)
    assert kb.classify_self_modification(task) is not None


@pytest.mark.parametrize("title,body", [
    ("Amalun SharePic redesign for job posting", "build the job card in ~/amalun-wp"),
    ("noScribe transcription 2/5 integrate", "add whisper backend to qualtrans"),
    ("Write report on Thueringen dashboard", "draft the markdown"),
    ("Reachy v0.9 deploy 3/5 predeploy-review", "review the reachy app changes"),
    ("Refactor the pricing module", "clean up the client project's python code"),
])
def test_classifier_ignores_project_work(isolated_kanban_home, title, body):
    kb, _ = isolated_kanban_home
    task = SimpleNamespace(title=title, body=body, workspace_path=None, branch_name=None)
    assert kb.classify_self_modification(task) is None


# ---------------------------------------------------------------------------
# Dispatch integration
# ---------------------------------------------------------------------------

def test_self_modify_card_is_gated_not_spawned(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Fix worker profile config.yaml toolsets", assignee="backend-eng")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_spawn)
    assert [t for t, _r in res.self_modify_gated] == [tid]
    assert tid not in [s[0] for s in res.spawned]
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.human_gate
        token = kb.issue_gate_token(conn, tid)
        assert token
        row = conn.execute("SELECT gate_scope_hash, gate_scope_version, governance_target_ref, governance_mutation_class FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["gate_scope_hash"] and row["gate_scope_version"] == kb.GOVERNANCE_SCOPE_VERSION
        assert row["governance_target_ref"] == f"task/{tid}"
        assert row["governance_mutation_class"] == "live-config"
        # The blocked event carries the layman fields for the cockpit/Telegram.
        import json
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        payload = json.loads(ev[0])
        assert payload.get("human_summary")
        assert payload.get("human_action")
        assert "self-modification gate" in (payload.get("reason") or "")


def test_project_card_dispatches_normally(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Amalun SharePic redesign for job posting", assignee="backend-eng")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_spawn)
    assert not res.self_modify_gated
    assert tid in [s[0] for s in res.spawned]
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_approved_card_is_not_regated(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Edit ~/.hermes/config.yaml gateway settings", assignee="backend-eng")
    with kb.connect_closing() as conn:
        kb.dispatch_once(conn, spawn_fn=_spawn)  # gates it
    # Human approves: mint a token and unblock (the cockpit/Telegram gate path).
    with kb.connect_closing() as conn:
        token = kb.issue_gate_token(conn, tid, action="unblock")
        assert token
        assert kb.unblock_task(conn, tid, actor="operator", reason="approved", token=token)
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.human_gate
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_spawn)
    assert not res.self_modify_gated  # not re-gated
    assert tid in [s[0] for s in res.spawned]
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "running"


def test_gate_can_be_disabled_by_config(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr(kb, "_self_modify_gate_enabled", lambda: False)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Fix worker profile config.yaml toolsets", assignee="backend-eng")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_spawn)
    assert not res.self_modify_gated
    assert tid in [s[0] for s in res.spawned]


# ---------------------------------------------------------------------------
# CAS guard: dispatcher-initiated blocks must not de-claim a card that was
# claimed between the ready snapshot and the block (require_unclaimed=True).
# ---------------------------------------------------------------------------

def test_require_unclaimed_refuses_to_declaim_claimed_card(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Edit ~/.hermes/config.yaml", assignee="backend-eng")
        # Simulate the race: an external actor claims the ready card before
        # the dispatcher hook fires its block.
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert not kb.block_task(
            conn, tid, require_unclaimed=True,
            reason="self-modification gate: test", kind="needs_input",
            human_gate=True,
        )
        task = kb.get_task(conn, tid)
        # The running claim survives untouched — no de-claim, no double-spawn stage.
        assert task.status == "running"
        assert task.claim_lock


def test_require_unclaimed_blocks_unclaimed_ready_card(isolated_kanban_home):
    kb, _ = isolated_kanban_home
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Edit ~/.hermes/config.yaml", assignee="backend-eng")
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.block_task(
            conn, tid, require_unclaimed=True,
            reason="self-modification gate: test", kind="needs_input",
            human_gate=True,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.human_gate

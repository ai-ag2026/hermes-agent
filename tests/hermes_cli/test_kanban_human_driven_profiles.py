"""Hardening (2026-07-17): human-driven profiles never auto-spawn / route.

``work`` (and ``default`` where configured) are operator-driven, NOT autonomous
kanban workers. The dispatcher must never auto-spawn a card assigned to one, and
the decomposer must never route work to one. ``kanban.human_driven_profiles``
(config) makes the set operator-editable; an explicit empty list is a kill
switch. Note: ``default`` is NOT auto-excluded (list_profiles/profile_exists
report it as spawnable), so it must be listed in config to be protected.
See reference-profile-taxonomy.
"""
from __future__ import annotations

import sys
import tempfile

import pytest

_LC = "hermes_cli.config.load_config_readonly"  # what human_driven_profiles() reads


def _is_hermes_mod(name: str) -> bool:
    return name.startswith("hermes_cli") or name.startswith("hermes_state") or name == "hermes_constants"


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME + clean kanban DB.

    Snapshots and RESTORES the hermes_cli/* module state on teardown so this
    file's re-import + module-function monkeypatches (list_profiles,
    get_active_profile_name, ...) cannot leak into co-located tests on the same
    xdist worker — the plain del-sys.modules pattern (test_kanban_default_assignee)
    is contaminating once those functions are patched.
    """
    saved = {k: v for k, v in sys.modules.items() if _is_hermes_mod(k)}
    test_home = tempfile.mkdtemp(prefix="kanban_human_driven_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if _is_hermes_mod(mod):
            del sys.modules[mod]
    from hermes_cli import kanban_db
    try:
        yield kanban_db, test_home
    finally:
        for k in [k for k in sys.modules if _is_hermes_mod(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def _fake_spawn(*args, **kwargs):
    return 12345


# --- human_driven_profiles() config resolution ------------------------------

def test_default_is_work(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr(_LC, lambda: {})
    assert kb.human_driven_profiles() == frozenset({"work"})


def test_config_list_replaces_default(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": ["default", "work"]}})
    assert kb.human_driven_profiles() == frozenset({"default", "work"})


def test_empty_list_is_kill_switch(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": []}})
    assert kb.human_driven_profiles() == frozenset()


def test_malformed_config_fails_safe_to_default(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    # A bare string (not a list) is malformed → keep the fail-safe default.
    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": "work"}})
    assert kb.human_driven_profiles() == frozenset({"work"})


# --- dispatcher spawn guard (ready loop) ------------------------------------

def test_dispatch_skips_work_but_spawns_worker(isolated_kanban_home, monkeypatch):
    """A ready 'work' card is bucketed skipped_nonspawnable and never spawns,
    while a normal worker card on the same tick spawns. Both assignees resolve
    as real profiles so the existing profile_exists gate passes — the
    human-driven guard is what must stop 'work' (built-in default set)."""
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        work_id = kb.create_task(conn, title="human", assignee="work")
        eng_id = kb.create_task(conn, title="worker", assignee="backend-eng")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    spawned_ids = [s[0] for s in res.spawned]
    assert work_id in res.skipped_nonspawnable
    assert work_id not in spawned_ids
    assert eng_id in spawned_ids


def test_config_can_add_default(isolated_kanban_home, monkeypatch):
    """With config [default, work], a ready 'default' card is ALSO skipped —
    proving the config-driven protection for the private default profile that
    is NOT auto-excluded."""
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": ["default", "work"]}})
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        def_id = kb.create_task(conn, title="private", assignee="default")
        eng_id = kb.create_task(conn, title="worker", assignee="backend-eng")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    spawned_ids = [s[0] for s in res.spawned]
    assert def_id in res.skipped_nonspawnable
    assert def_id not in spawned_ids
    assert eng_id in spawned_ids


def test_kill_switch_lets_work_spawn(isolated_kanban_home, monkeypatch):
    """With the guard disabled (empty config list), a 'work' card spawns like
    any other — proving the guard is what gates it, not some other rule."""
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": []}})
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        work_id = kb.create_task(conn, title="human", assignee="work")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    spawned_ids = [s[0] for s in res.spawned]
    assert work_id in spawned_ids
    assert work_id not in res.skipped_nonspawnable


def test_human_driven_default_assignee_ignored(isolated_kanban_home, monkeypatch):
    """A human-driven default_assignee must NOT auto-assign unassigned ready
    tasks (which would strand them assigned-but-unspawned). The task stays
    skipped_unassigned and its row is untouched."""
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kb.connect_closing() as conn:
        # 'work' is human-driven by the built-in default set.
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False, default_assignee="work")
    assert task_id in res.skipped_unassigned
    assert not res.auto_assigned_default
    assert not res.spawned
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert row["assignee"] is None


# --- dispatcher spawn guard (review loop) -----------------------------------

def test_dispatch_skips_work_in_review(isolated_kanban_home, monkeypatch):
    """The review-loop guard mirrors the ready loop: a review card assigned to
    a human-driven profile is skipped, a worker review card spawns."""
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        work_id = kb.create_task(conn, title="human", assignee="work")
        eng_id = kb.create_task(conn, title="worker", assignee="backend-eng")
        conn.execute("UPDATE tasks SET status='review' WHERE id IN (?,?)", (work_id, eng_id))
        conn.commit()
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    spawned_ids = [s[0] for s in res.spawned]
    assert work_id in res.skipped_nonspawnable
    assert work_id not in spawned_ids
    assert eng_id in spawned_ids


# --- health telemetry -------------------------------------------------------

def test_has_spawnable_ready_excludes_work(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="human", assignee="work")
    with kb.connect_closing() as conn:
        assert kb.has_spawnable_ready(conn) is False
        kb.create_task(conn, title="worker", assignee="backend-eng")
    with kb.connect_closing() as conn:
        assert kb.has_spawnable_ready(conn) is True


def test_has_spawnable_review_excludes_work(isolated_kanban_home, monkeypatch):
    kb, _ = isolated_kanban_home
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        work_id = kb.create_task(conn, title="human", assignee="work")
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (work_id,))
        conn.commit()
    with kb.connect_closing() as conn:
        assert kb.has_spawnable_review(conn) is False
        eng_id = kb.create_task(conn, title="worker", assignee="backend-eng")
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (eng_id,))
        conn.commit()
    with kb.connect_closing() as conn:
        assert kb.has_spawnable_review(conn) is True


# --- decomposer roster exclusion --------------------------------------------

def test_roster_excludes_human_driven(isolated_kanban_home, monkeypatch):
    """_build_roster drops human-driven profiles (incl. 'default' when
    configured) from BOTH the roster the LLM sees and the valid set."""
    kb, _ = isolated_kanban_home
    from hermes_cli import kanban_decompose as kd

    class _P:
        def __init__(self, name, description):
            self.name = name
            self.description = description

    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [_P("default", ""), _P("work", "prof desc"), _P("backend-eng", "eng desc")],
    )
    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": ["default", "work"]}})
    roster, valid = kd._build_roster()
    names = {r["name"] for r in roster}
    assert "work" not in names and "work" not in valid
    assert "default" not in names and "default" not in valid
    assert "backend-eng" in names and "backend-eng" in valid


def test_resolve_default_assignee_never_human_driven(isolated_kanban_home, monkeypatch):
    """_resolve_default_assignee must never return a human-driven profile, even
    when default_assignee is unset and the active profile is 'default' (the
    dispatcher's own home). It falls to orchestrator_profile, then a worker."""
    kb, _ = isolated_kanban_home
    from hermes_cli import kanban_decompose as kd

    class _P:
        def __init__(self, name):
            self.name = name
            self.description = ""

    monkeypatch.setattr(_LC, lambda: {"kanban": {"human_driven_profiles": ["default", "work"]}})
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "hermes_cli.profiles.list_profiles",
        lambda: [_P("default"), _P("work"), _P("backend-eng")],
    )
    # orchestrator_profile (pm, non-hd) wins when no explicit default_assignee.
    assert kd._resolve_default_assignee({"kanban": {"orchestrator_profile": "pm"}}) == "pm"
    # No orchestrator, active='default' (human-driven) → first non-hd worker.
    assert kd._resolve_default_assignee({"kanban": {}}) == "backend-eng"
    # An explicitly human-driven default_assignee is ignored, not returned.
    assert kd._resolve_default_assignee({"kanban": {"default_assignee": "work"}}) == "backend-eng"

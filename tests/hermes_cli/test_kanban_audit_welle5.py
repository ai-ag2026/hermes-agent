"""Welle-5 audit rules + GC done-retention (Audit 2026-07-27).

The audit gained eyes for the silent-death states the incident exposed:
aging unclaimed review/triage cards, active subscriptions whose terminal
events nobody delivered, attention deliveries stuck mid-lease, and claim
residue. Plus: `kanban gc` now also sweeps scratch workspaces of DONE tasks
older than the retention window (3.6 GiB had piled up because only archived
tasks were covered and the weekly cron never ran).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_repair as kr


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _age_task(conn, tid, *, seconds):
    past = int(time.time()) - seconds
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (past, tid))
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ?", (past, tid)
        )


def _finding_kinds(conn):
    return {(f.kind, f.task_id) for f in kr.run_audit(conn).findings}


def test_audit_flags_stale_review_and_triage(kanban_home):
    with kb.connect() as conn:
        rev = kb.create_task(conn, title="dead review", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id = ?", (rev,))
        _age_task(conn, rev, seconds=2 * 3600)
        tri = kb.create_task(conn, title="old triage", assignee="worker", triage=True)
        _age_task(conn, tri, seconds=8 * 24 * 3600)
        fresh = kb.create_task(conn, title="fresh triage", assignee="worker", triage=True)

        kinds = _finding_kinds(conn)
        assert ("stale_review_task", rev) in kinds
        assert ("stale_triage_task", tri) in kinds
        assert ("stale_triage_task", fresh) not in kinds


def test_audit_flags_undelivered_subscription(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="finished", assignee="worker")
        kb.claim_task(conn, tid)
        kb.add_notify_sub(conn, task_id=tid, platform="webui", chat_id="sess1")
        assert kb.complete_task(conn, tid, summary="done") is True
        _age_task(conn, tid, seconds=3600)  # terminal events are old, cursor 0

        kinds = _finding_kinds(conn)
        assert ("undelivered_subscription", tid) in kinds

        # A delivered (advanced) cursor clears the finding.
        _, new_cursor, _ = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="webui", chat_id="sess1",
        )
        assert new_cursor > 0
        kinds = _finding_kinds(conn)
        assert ("undelivered_subscription", tid) not in kinds


def test_audit_flags_claim_residue(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="residue", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', claim_lock='host:123' WHERE id = ?",
                (tid,),
            )
        assert ("claim_residue", tid) in _finding_kinds(conn)


def test_audit_stale_promotable_skips_human_gated(kanban_home):
    """Parity with recompute_ready: a human-gated blocked card is NOT a
    stale-promotable finding — it is deliberately held."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.claim_task(conn, tid)
        kb.block_task(conn, tid, reason="hold", kind="needs_input", human_gate=True)
        kinds = _finding_kinds(conn)
        assert ("stale_promotable_task", tid) not in kinds


def test_gc_sweeps_old_done_scratch_workspaces(kanban_home, monkeypatch):
    import argparse
    from hermes_cli import kanban as kanban_cli
    root = kb.workspaces_root()
    with kb.connect() as conn:
        old_done = kb.create_task(conn, title="old done", assignee="w")
        recent_done = kb.create_task(conn, title="recent done", assignee="w")
        for tid in (old_done, recent_done):
            ws = root / tid
            ws.mkdir(parents=True)
            (ws / "junk.txt").write_text("scratch")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='done', workspace_kind='scratch', "
                    "workspace_path=?, completed_at=? WHERE id = ?",
                    (str(ws),
                     int(time.time()) - (30 * 86400 if tid == old_done else 3600),
                     tid),
                )

    rc = kanban_cli._cmd_gc(argparse.Namespace(
        event_retention_days=30, log_retention_days=30,
        workspace_retention_days=7,
    ))
    assert rc == 0
    assert not (root / old_done).exists()
    assert (root / recent_done).exists()


def test_scavenger_refuses_foreign_reference_set(kanban_home, monkeypatch):
    """Incident 2026-07-27: with HERMES_KANBAN_DB pointing at board X while
    the artifact root resolves to default, the scavenger walked the DEFAULT
    root with X's (empty) reference set and deleted every default artifact
    older than the grace window. The sweep must refuse when the connection's
    DB does not own the root it is about to sweep."""
    import os as _os

    # Default-board artifact: referenced row + on-disk file older than grace.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="with artifact", assignee="w")
        root = kb.completion_artifacts_root(None)
        dest = root / tid / "1" / "x"
        dest.mkdir(parents=True)
        path = dest / "evidence.md"
        path.write_text("precious")
        old = time.time() - 7200
        _os.utime(path, (old, old))
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_artifacts (task_id, producer_run_id, "
                "original_path, durable_path, sha256, size, content_type, "
                "validated_at, retention_class) VALUES (?, 0, ?, ?, 'a'*1, 8, "
                "'text/markdown', strftime('%s','now'), 'default')",
                (tid, str(path), str(path)),
            )

    # Foreign board DB + env override — the incident invocation shape.
    kb.init_db(board="otherboard")
    other_db = kb.kanban_db_path(board="otherboard")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(other_db))
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(other_db)
    conn.row_factory = _sqlite3.Row
    try:
        removed = kb._scavenge_completion_artifacts(conn, board=None)
    finally:
        conn.close()
    assert removed == 0
    assert path.exists(), "foreign-reference sweep must never delete"

    # Positive control: the OWNING connection still scavenges unreferenced
    # stale files (a second, unreferenced old file disappears; the
    # referenced one stays).
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    stray = dest / "stray.tmp"
    stray.write_text("junk")
    _os.utime(stray, (old, old))
    with kb.connect() as conn:
        removed = kb._scavenge_completion_artifacts(conn, board=None)
    assert removed >= 1
    assert not stray.exists()
    assert path.exists()


# ---------------------------------------------------------------------------
# Repair-Runde nach TARS-Review 2026-07-27
# ---------------------------------------------------------------------------

def test_gc_workspace_sweep_is_opt_in(kanban_home):
    """Default must NOT delete workspaces: the sweep is opt-in now."""
    import argparse
    from hermes_cli import kanban as kanban_cli
    root = kb.workspaces_root()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="old done", assignee="w")
        ws = root / tid
        ws.mkdir(parents=True)
        (ws / "junk.txt").write_text("scratch")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='scratch', "
                "workspace_path=?, completed_at=? WHERE id = ?",
                (str(ws), int(time.time()) - 30 * 86400, tid),
            )
    # Namespace without the flag at all → getattr default 0 → no sweep.
    rc = kanban_cli._cmd_gc(argparse.Namespace(
        event_retention_days=30, log_retention_days=30,
    ))
    assert rc == 0
    assert (root / tid).exists(), "default GC must not delete done workspaces"


def test_gc_protects_workspace_holding_last_artifact_copy(kanban_home):
    """A workspace holding the ONLY copy of a recorded artifact survives.

    This is the loss that actually happened on 2026-07-27: the scavenger had
    destroyed the durable copies, then a GC pass deleted the scratch dirs
    still holding the originals.
    """
    import argparse
    from hermes_cli import kanban as kanban_cli
    root = kb.workspaces_root()
    with kb.connect() as conn:
        keep = kb.create_task(conn, title="has last copy", assignee="w")
        drop = kb.create_task(conn, title="no artifacts", assignee="w")
        old = int(time.time()) - 30 * 86400
        for tid in (keep, drop):
            ws = root / tid
            ws.mkdir(parents=True)
            (ws / "evidence.md").write_text("precious")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='done', workspace_kind='scratch', "
                    "workspace_path=?, completed_at=? WHERE id = ?",
                    (str(ws), old, tid),
                )
        # Recorded artifact whose DURABLE copy is gone; original still lives
        # in the scratch workspace of `keep`.
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_artifacts (task_id, producer_run_id, "
                "original_path, durable_path, sha256, size, content_type, "
                "validated_at, retention_class) VALUES (?, 0, ?, ?, 'x', 8, "
                "'text/markdown', strftime('%s','now'), 'default')",
                (keep, str(root / keep / "evidence.md"),
                 str(kanban_home / "gone" / "durable.md")),
            )

    rc = kanban_cli._cmd_gc(argparse.Namespace(
        event_retention_days=30, log_retention_days=30,
        workspace_retention_days=7,
    ))
    assert rc == 0
    assert (root / keep).exists(), "last-copy workspace must be protected"
    assert not (root / drop).exists(), "unreferenced workspace still swept"


def test_gc_protects_cross_task_artifact_location(kanban_home):
    """Protection is global: task A's artifact may live in B's workspace."""
    import argparse
    from hermes_cli import kanban as kanban_cli
    root = kb.workspaces_root()
    with kb.connect() as conn:
        owner = kb.create_task(conn, title="owns artifact", assignee="w")
        host = kb.create_task(conn, title="hosts the file", assignee="w")
        old = int(time.time()) - 30 * 86400
        ws = root / host
        ws.mkdir(parents=True)
        (ws / "shared.md").write_text("cross-task evidence")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', workspace_kind='scratch', "
                "workspace_path=?, completed_at=? WHERE id = ?",
                (str(ws), old, host),
            )
            conn.execute(
                "INSERT INTO task_artifacts (task_id, producer_run_id, "
                "original_path, durable_path, sha256, size, content_type, "
                "validated_at, retention_class) VALUES (?, 0, ?, ?, 'y', 8, "
                "'text/markdown', strftime('%s','now'), 'default')",
                (owner, str(ws / "shared.md"), str(kanban_home / "gone.md")),
            )
    rc = kanban_cli._cmd_gc(argparse.Namespace(
        event_retention_days=30, log_retention_days=30,
        workspace_retention_days=7,
    ))
    assert rc == 0
    assert (root / host).exists()


def test_human_driven_profiles_reads_root_config(kanban_home, monkeypatch, tmp_path):
    """Board-global policy must not depend on the asking profile.

    A worker profile carries no kanban.human_driven_profiles, so the reviewer
    guard used to answer differently inside a worker than in the gateway that
    later has to dispatch the review (TARS review, P1).
    """
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  human_driven_profiles: [default, work]\n", encoding="utf-8"
    )
    kb._ROOT_KANBAN_CONFIG_CACHE = None
    assert "work" in kb.human_driven_profiles()

    # Now pretend we are a worker profile: HERMES_HOME points into profiles/,
    # whose config has no kanban section at all.
    profile_home = kanban_home / "profiles" / "backend-eng"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    kb._ROOT_KANBAN_CONFIG_CACHE = None
    assert "work" in kb.human_driven_profiles(), (
        "worker profile must see the same human-driven set as the gateway"
    )

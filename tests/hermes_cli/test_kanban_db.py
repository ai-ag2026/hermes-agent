"""Tests for the Kanban DB layer (hermes_cli.kanban_db)."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import types
import unittest.mock
from pathlib import Path

import pytest

import hermes_state
from hermes_cli import kanban_db as kb


def _kanban_multiprocess_writer(db_path: str, proc_idx: int, n: int, queue):
    """Small real-process writer used by the WAL concurrency regression test."""
    try:
        for i in range(n):
            with kb.connect_closing(db_path=Path(db_path)) as conn:
                kb.create_task(
                    conn,
                    title=f"stress p{proc_idx} #{i}",
                    body="multiprocess kanban sqlite stress",
                    assignee="worker",
                    created_by=f"proc-{proc_idx}",
                    workspace_kind="dir",
                    workspace_path=str(Path(db_path).parent),
                )
        queue.put(None)
    except Exception as exc:  # pragma: no cover - surfaced in parent assertion
        queue.put(f"{type(exc).__name__}: {exc}")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------







def test_cross_process_init_lock_uses_windows_byte_range_lock(tmp_path, monkeypatch):
    """Windows must use a real (non-blocking) process lock, not a no-op open.

    The init lock acquires with LK_NBLCK in a bounded retry loop (#36644) so a
    wedged holder can never block connect() forever; a clean acquire takes the
    lock once and releases it once.
    """
    calls: list[tuple[int, int, int]] = []
    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=3,
        LK_UNLCK=2,
        locking=lambda fd, mode, nbytes: calls.append((fd, mode, nbytes)),
    )
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    db_path = tmp_path / "kanban.db"
    with kb._cross_process_init_lock(db_path):
        # Acquired exactly once via the non-blocking byte-range lock.
        assert [call[1:] for call in calls] == [(fake_msvcrt.LK_NBLCK, 1)]

    # Released once on exit.
    assert [call[1:] for call in calls] == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]


def test_cross_process_init_lock_timeout_fails_closed(tmp_path, monkeypatch):
    """A lock timeout must not run potentially destructive migrations unlocked."""
    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=3,
        LK_UNLCK=2,
        locking=lambda *_args: (_ for _ in ()).throw(OSError("busy")),
    )
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(kb, "_INIT_LOCK_TIMEOUT_SECONDS", 0)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    with pytest.raises(TimeoutError, match="refusing to run schema migration"):
        with kb._cross_process_init_lock(tmp_path / "kanban.db"):
            pytest.fail("lock body must not run")


def test_connect_rejects_existing_zero_byte_db(tmp_path):
    """An existing empty file is truncation, not permission to recreate a board."""
    db_path = tmp_path / "kanban.db"
    db_path.touch()

    with pytest.raises(sqlite3.DatabaseError, match="existing zero-byte kanban DB"):
        kb.connect(db_path=db_path)

    assert db_path.stat().st_size == 0


def test_connect_rejects_tls_record_in_sqlite_header(tmp_path, monkeypatch):
    """Kanban should classify TLS-looking page-0 clobbers before WAL setup."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    corrupt = home / "kanban.db"
    corrupt.write_bytes(b"SQLit" + bytes.fromhex("17 03 03 00 13") + b"x" * 32)

    with pytest.raises(sqlite3.DatabaseError) as exc_info:
        kb.connect(board="default")

    msg = str(exc_info.value)
    assert "file is not a database" in msg
    assert "TLS record header detected at byte offset 5" in msg
    assert "53 51 4c 69 74 17 03 03 00 13" in msg


def test_connect_migrates_legacy_db_before_optional_column_indexes(tmp_path):
    """Legacy DBs missing additive indexed columns must migrate cleanly.

    SCHEMA_SQL runs in ``connect()`` before ``_migrate_add_optional_columns``.
    Indexes over additive columns therefore must be created after the
    migration adds those columns, or boards predating the column fail to
    open before migration can run.

    Covers all four indexes that sit on additive columns:
    - ``tasks.session_id``       -> ``idx_tasks_session_id``    (#28447)
    - ``tasks.tenant``           -> ``idx_tasks_tenant``        (#16081)
    - ``tasks.idempotency_key``  -> ``idx_tasks_idempotency``   (#17805)
    - ``task_events.run_id``     -> ``idx_events_run``          (#17805)
    """
    db_path = tmp_path / "legacy-kanban.db"
    conn = sqlite3.connect(str(db_path))
    # Pre-#16081 ``tasks`` shape: missing tenant, idempotency_key, session_id.
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
    """)
    # Pre-#17805 ``task_events`` shape: missing run_id. Required because
    # ``_migrate_add_optional_columns`` unconditionally runs PRAGMA on
    # ``task_events`` for run_id back-fill.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE task_pending_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            run_id INTEGER,
            command_hash TEXT NOT NULL,
            summary TEXT NOT NULL,
            profile TEXT NOT NULL,
            workspace TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            approved_at INTEGER,
            consumed_at INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old board task', 'ready', 1)"
    )
    conn.commit()
    conn.close()

    with kb.connect(db_path) as migrated:
        task_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")
        }
        event_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(task_events)")
        }
        action_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(task_pending_actions)")
        }
        indexes = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    # Additive columns added by migration:
    assert "session_id" in task_columns
    assert "tenant" in task_columns
    assert "idempotency_key" in task_columns
    assert "run_id" in event_columns
    assert {"fingerprint", "mutation_kind", "cancelled_at"} <= action_columns
    # And their indexes — the regression scope of this test:
    assert "idx_tasks_session_id" in indexes
    assert "idx_tasks_tenant" in indexes
    assert "idx_tasks_idempotency" in indexes
    assert "idx_events_run" in indexes


def test_connect_recreates_missing_pending_action_table_and_attention_projection(tmp_path):
    db_path = tmp_path / "missing-pending-actions.db"
    kb.init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DROP TABLE task_attentions")
        conn.execute("DROP TABLE task_pending_actions")
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        action_columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_pending_actions)")}
        attention_columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_attentions)")}
        indexes = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"state", "version", "updated_at", "resolved_at"} <= action_columns
        assert {"action_id", "type", "cause_fingerprint"} <= attention_columns
        assert {"uq_pending_action_active_identity", "uq_pending_action_active_origin"} <= indexes
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def _install_v1_pending_action_fixture(db_path: Path) -> dict[str, object]:
    """Build a real pre-lifecycle table, including V1 fingerprints."""
    kb.init_db(db_path)
    fixture: dict[str, object] = {"tasks": {}, "actions": {}}
    with kb.connect(db_path) as conn:
        board = kb._pending_action_board_identity(conn)
        fixture["board"] = board
        tasks = fixture["tasks"]
        actions = fixture["actions"]
        assert isinstance(tasks, dict) and isinstance(actions, dict)
        for name in (
            "pending", "approved", "duplicate", "invalid_no_run",
            "invalid_no_kind", "invalid_no_fingerprint",
        ):
            task_id = kb.create_task(conn, title=name, assignee="legacy")
            claimed = kb.claim_task(conn, task_id, claimer="legacy")
            assert claimed is not None and claimed.current_run_id is not None
            tasks[name] = {"run_id": claimed.current_run_id, "task_id": task_id}
        conn.execute("DROP TABLE task_attentions")
        conn.execute("DROP TABLE task_pending_actions")
        conn.execute("""CREATE TABLE task_pending_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, run_id INTEGER,
            command_hash TEXT NOT NULL, fingerprint TEXT, mutation_kind TEXT,
            summary TEXT NOT NULL, profile TEXT NOT NULL, workspace TEXT NOT NULL,
            created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, approved_at INTEGER,
            consumed_at INTEGER)""")
        def insert(name, *, task_name=None, approved=False, run_id="default", mutation_kind="terminal-command", fingerprint="valid"):
            task = tasks[task_name or name]
            assert isinstance(task, dict)
            origin, task_id = task["run_id"], task["task_id"]
            run_id = origin if run_id == "default" else run_id
            expires = 2_000_000_000
            profile = "legacy-profile"
            workspace = str((db_path.parent / "legacy-workspace").resolve())
            command = f"legacy-{name}-command"
            command_hash = kb._pending_action_hash(command)
            fp = fingerprint
            if fp == "valid":
                fp = kb._pending_action_fingerprint(board_identity=board, task_id=task_id, run_id=run_id,
                    command_hash=command_hash, mutation_kind=mutation_kind or "terminal-command",
                    profile=profile, workspace=workspace, expires_at=expires, version=1) if run_id is not None and mutation_kind is not None else None
            action_id = conn.execute("INSERT INTO task_pending_actions (task_id,run_id,command_hash,fingerprint,mutation_kind,summary,profile,workspace,created_at,expires_at,approved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, run_id, command_hash, fp, mutation_kind, "legacy", profile, workspace, 10, expires, 20 if approved else None)).lastrowid
            actions[name] = {"id": action_id, "task_id": task_id, "run_id": run_id, "command": command,
                             "command_hash": command_hash, "fingerprint_v1": fp, "mutation_kind": mutation_kind,
                             "profile": profile, "workspace": workspace, "expires_at": expires}
        insert("pending")
        insert("approved", approved=True)
        insert("duplicate")
        insert("duplicate_approved_low", task_name="duplicate", approved=True)
        insert("duplicate_approved_high", task_name="duplicate", approved=True)
        insert("invalid_no_run", run_id=None)
        insert("invalid_no_kind", mutation_kind=None)
        insert("invalid_no_fingerprint", fingerprint=None)
    return fixture


def test_v1_pending_action_migration_preserves_valid_history_and_is_idempotent(tmp_path):
    db_path = tmp_path / "v1-actions.db"
    fixture = _install_v1_pending_action_fixture(db_path)
    tasks = fixture["tasks"]
    actions = fixture["actions"]
    assert isinstance(tasks, dict) and isinstance(actions, dict)
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        rows = conn.execute("SELECT id,task_id,state,version,fingerprint,resolved_at FROM task_pending_actions ORDER BY id").fetchall()
        by_id = {row["id"]: row for row in rows}
        assert by_id[actions["pending"]["id"]]["state"] == "pending"
        assert by_id[actions["approved"]["id"]]["state"] == "approved"
        assert by_id[actions["duplicate"]["id"]]["state"] == "cancelled"
        assert by_id[actions["duplicate_approved_low"]["id"]]["state"] == "cancelled"
        assert by_id[actions["duplicate_approved_high"]["id"]]["state"] == "approved"
        assert actions["duplicate_approved_high"]["id"] > actions["duplicate_approved_low"]["id"]
        invalid_actions = [actions[name] for name in ("invalid_no_run", "invalid_no_kind", "invalid_no_fingerprint")]
        for action in invalid_actions:
            row = by_id[action["id"]]
            assert row["state"] == "resolved" and row["resolved_at"] is not None
            assert conn.execute("SELECT COUNT(*) FROM task_attentions WHERE action_id=?", (action["id"],)).fetchone()[0] == 0
            assert not kb.approve_pending_action(conn, action["task_id"], action["id"], now=100)
            if action["run_id"] is None:
                assert conn.execute("SELECT COUNT(*) FROM task_pending_actions WHERE id=? AND state='approved'", (action["id"],)).fetchone()[0] == 0
            else:
                conn.execute("UPDATE task_runs SET status='done', outcome='blocked', ended_at=30 WHERE id=?", (action["run_id"],))
                resumed = conn.execute("INSERT INTO task_runs (task_id,status,started_at,profile) VALUES (?, 'running', 40, ?)", (action["task_id"], action["profile"])).lastrowid
                conn.execute("UPDATE tasks SET status='running', current_run_id=? WHERE id=?", (resumed, action["task_id"]))
                assert not kb.consume_approved_action(conn, task_id=action["task_id"], run_id=resumed, command=action["command"], profile=action["profile"], workspace=action["workspace"], now=100)
        assert all(r["version"] == 1 for r in rows if r["state"] != "cancelled")
        assert [r["version"] for r in rows if r["state"] == "cancelled"] == [2, 2]
        assert all(len(r["fingerprint"] or "") == 64 for r in rows if r["state"] != "resolved")
        active_names = ("pending", "approved", "duplicate_approved_high")
        expected_attentions = []
        for name in active_names:
            action = actions[name]
            expected_v2 = kb._pending_action_fingerprint(board_identity=fixture["board"], task_id=action["task_id"], run_id=action["run_id"], command_hash=action["command_hash"], mutation_kind=action["mutation_kind"], profile=action["profile"], workspace=action["workspace"])
            assert by_id[action["id"]]["fingerprint"] == expected_v2
            expected_attentions.append((action["task_id"], action["id"], "exact_action", expected_v2))
        attentions = [tuple(row) for row in conn.execute("SELECT id,task_id,action_id,type,cause_fingerprint,summary,created_at FROM task_attentions ORDER BY id")]
        assert len(attentions) == len(expected_attentions)
        assert [(row[1], row[2], row[3], row[4]) for row in attentions] == sorted(expected_attentions)
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        snapshot = ([tuple(r) for r in rows], attentions)
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        again = ([tuple(r) for r in conn.execute("SELECT id,task_id,state,version,fingerprint,resolved_at FROM task_pending_actions ORDER BY id")], [tuple(r) for r in conn.execute("SELECT id,task_id,action_id,type,cause_fingerprint,summary,created_at FROM task_attentions ORDER BY id")])
        assert again == snapshot


def test_v1_approved_action_migrates_and_consumes_from_resumed_run(tmp_path):
    db_path = tmp_path / "v1-consume.db"
    fixture = _install_v1_pending_action_fixture(db_path)
    actions = fixture["actions"]
    assert isinstance(actions, dict)
    kb.init_db(db_path)
    approved = actions["approved"]
    with kb.connect(db_path) as conn:
        row = conn.execute("SELECT id,command_hash,fingerprint FROM task_pending_actions WHERE id=?", (approved["id"],)).fetchone()
        assert row is not None
        expected_v2 = kb._pending_action_fingerprint(board_identity=fixture["board"], task_id=approved["task_id"], run_id=approved["run_id"], command_hash=approved["command_hash"], mutation_kind=approved["mutation_kind"], profile=approved["profile"], workspace=approved["workspace"])
        assert row["command_hash"] == kb._pending_action_hash(approved["command"])
        assert row["fingerprint"] == expected_v2
        assert row["fingerprint"] != approved["fingerprint_v1"]
        conn.execute("UPDATE task_runs SET status='done', outcome='blocked', ended_at=30 WHERE id=?", (approved["run_id"],))
        resumed = conn.execute("INSERT INTO task_runs (task_id,status,started_at,profile) VALUES (?, 'running', 40, ?)", (approved["task_id"], approved["profile"])).lastrowid
        conn.execute("UPDATE tasks SET status='running', current_run_id=? WHERE id=?", (resumed, approved["task_id"]))
        conn.commit()
        assert kb.consume_approved_action(conn, task_id=approved["task_id"], run_id=resumed, command=approved["command"], profile=approved["profile"], workspace=approved["workspace"], now=100)
        assert kb.get_pending_action_by_id(conn, approved["task_id"], row["id"]).state == "consumed"
        events = conn.execute("SELECT kind FROM task_events WHERE task_id=? AND run_id=? AND kind='terminal_approval_consumed'", (approved["task_id"], resumed)).fetchall()
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Task creation + status inference
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Links + dependency resolution
# ---------------------------------------------------------------------------





def test_link_keeps_ready_child_when_parent_already_archived(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="archived parent")
        assert kb.complete_task(conn, parent)
        assert kb.archive_task(conn, parent)
        child = kb.create_task(conn, title="child")
        assert kb.get_task(conn, child).status == "ready"

        kb.link_tasks(conn, parent, child)

        assert kb.get_task(conn, child).status == "ready"


def test_link_rejects_self_loop(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        with pytest.raises(ValueError, match="itself"):
            kb.link_tasks(conn, a, a)


def test_link_detects_cycle(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b", parents=[a])
        c = kb.create_task(conn, title="c", parents=[b])
        with pytest.raises(ValueError, match="cycle"):
            kb.link_tasks(conn, c, a)
        with pytest.raises(ValueError, match="cycle"):
            kb.link_tasks(conn, b, a)


def test_recompute_ready_cascades_through_chain(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b", parents=[a])
        c = kb.create_task(conn, title="c", parents=[b])
        assert [kb.get_task(conn, x).status for x in (a, b, c)] == \
               ["ready", "todo", "todo"]
        kb.complete_task(conn, a)
        assert kb.get_task(conn, b).status == "ready"
        kb.complete_task(conn, b)
        assert kb.get_task(conn, c).status == "ready"


def test_recompute_ready_promotes_blocked_with_done_parents(kanban_home):
    """blocked tasks with all parents done should be promoted to ready,
    unless the circuit-breaker failure limit has been reached."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(
            conn, title="child", assignee="a", parents=[parent],
        )
        # Complete the parent
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="ok")
        # Manually block the child with zero failures (simulates a
        # dependency block, not a circuit-breaker block).
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=0, "
            "last_failure_error=NULL WHERE id=?",
            (child,),
        )
        conn.commit()
        assert kb.get_task(conn, child).status == "blocked"
        # recompute_ready should promote blocked → ready
        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None


def test_recompute_ready_fan_in_waits_for_all_parents(kanban_home):
    with kb.connect() as conn:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        c = kb.create_task(conn, title="c", parents=[a, b])
        kb.complete_task(conn, a)
        assert kb.get_task(conn, c).status == "todo"
        kb.complete_task(conn, b)
        assert kb.get_task(conn, c).status == "ready"


# ---------------------------------------------------------------------------
# Atomic claim (CAS)
# ---------------------------------------------------------------------------



def test_schedule_task_parks_time_delay_without_dispatching(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="delayed recheck", assignee="ops")
        assert kb.schedule_task(conn, t, reason="run next week") is True
        task = kb.get_task(conn, t)
        assert task.status == "scheduled"
        assert kb.claim_task(conn, t) is None

        events = kb.list_events(conn, t)
        assert any(e.kind == "scheduled" and e.payload == {"reason": "run next week"} for e in events)








def test_stale_claim_reclaim_event_records_diagnostic_payload(
    kanban_home, monkeypatch,
):
    """``reclaimed`` events should carry claim_expires, last_heartbeat_at,
    and worker_pid so operators can diagnose why a claim went stale
    (#23025: previous payload only had ``stale_lock`` which gives no
    timing context)."""
    import json
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 12345)
        old_expires = int(time.time()) - 3600
        hb_at = int(time.time()) - 1800
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ?",
            (old_expires, hb_at, t),
        )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None)
        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'reclaimed'",
            (t,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["claim_expires"] == old_expires
        assert payload["last_heartbeat_at"] == hb_at
        assert payload["worker_pid"] == 12345
        assert payload["host_local"] is True


def test_detect_crashed_workers_systemic_failure_fast_block(
    kanban_home, monkeypatch,
):
    """When many tasks crash with the same error, trip the breaker faster."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)

    with kb.connect() as conn:
        task_ids = []
        for i in range(4):
            tid = kb.create_task(conn, title=f"task-{i}", assignee="a")
            host = _kb._claimer_id().split(":", 1)[0]
            conn.execute(
                "UPDATE tasks SET status='running', worker_pid=?, "
                "claim_lock=? WHERE id=?",
                (90000 + i, f"{host}:w{i}", tid),
            )
            task_ids.append(tid)
        conn.commit()

        crashed = kb.detect_crashed_workers(conn)
        assert len(crashed) == 4

        for tid in task_ids:
            task = kb.get_task(conn, tid)
            # Automation-first (2026-07-16): systemic fast-block routes to triage.
            assert task.status == "triage", (
                f"task {tid} should be triaged (systemic), got {task.status}"
            )




# ---------------------------------------------------------------------------
# Rate-limit requeue: a worker that bails on a provider quota wall must be
# released back to ``ready`` WITHOUT counting a failure, so a long (e.g.
# 5-hour) quota window can't trip the circuit breaker and permanently block
# the card. The respawn guard then defers it on a cooldown until quota
# returns. Regression coverage for the kanban-rate-limit-failure report.
# ---------------------------------------------------------------------------


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8




def test_real_child_rate_limit_exit_reaps_into_transient_run(
    kanban_home, monkeypatch,
):
    """Exercise the real Popen -> waitpid -> reaper -> ledger contract.

    This deliberately does not call ``Popen.wait``/``poll`` because either would
    reap the child before ``reap_worker_zombies`` can record its raw wait status.
    ``waitid(..., WNOWAIT)`` blocks until exit while preserving the zombie for
    the dispatcher reaper.
    """
    if os.name == "nt" or not all(
        hasattr(os, attr) for attr in ("waitid", "P_PID", "WEXITED", "WNOWAIT")
    ):
        pytest.skip("requires POSIX waitid(..., WNOWAIT)")

    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    _kb._recent_worker_exits.clear()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="real rate-limit child", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:real-child")
        assert claimed is not None

        proc = subprocess.Popen(
            [sys.executable, "-c", f"raise SystemExit({_kb.KANBAN_RATE_LIMIT_EXIT_CODE})"]
        )
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (proc.pid, tid))
        conn.commit()

        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        assert proc.pid in _kb.reap_worker_zombies()

        crashed = kb.detect_crashed_workers(conn)
        assert tid not in crashed
        assert tid in getattr(_kb.detect_crashed_workers, "_last_rate_limited", [])

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        run = conn.execute(
            "SELECT status, outcome, error FROM task_runs WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run["status"] == "rate_limited"
        assert run["outcome"] == "rate_limited"
        assert "rate-limited" in (run["error"] or "")


def test_rate_limit_exit_requeues_without_counting_failure(
    kanban_home, monkeypatch,
):
    """A rate-limit sentinel exit releases the task to ``ready`` and leaves
    ``consecutive_failures`` untouched — the breaker must never trip on a
    transient throttle, even across many quota-wall hits."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl", assignee="a")

        # Simulate FAR more quota-wall hits than DEFAULT_FAILURE_LIMIT (2).
        # If any of these counted as a failure the task would be blocked.
        for i in range(6):
            pid = 70000 + i
            # Claim to open a real run (so detect_crashed_workers can close
            # it with a rate_limited outcome), then point the claim at this
            # host + a dead pid so the crash path acts on it.
            kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
            conn.execute(
                "UPDATE tasks SET worker_pid=?, consecutive_failures=? "
                "WHERE id=?",
                (pid, 0, tid),
            )
            conn.commit()
            _kb._record_worker_exit(
                pid, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE)
            )

            crashed = kb.detect_crashed_workers(conn)
            # Rate-limited requeues are NOT crashes.
            assert tid not in crashed
            rl = getattr(_kb.detect_crashed_workers, "_last_rate_limited", [])
            assert tid in rl

            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"hit {i}: should requeue ready, got {task.status}"
            )
            assert task.consecutive_failures == 0, (
                f"hit {i}: rate-limit must not count a failure, "
                f"got {task.consecutive_failures}"
            )

        # Last failure error stamped so the respawn guard recognizes the
        # quota wall.
        assert task.last_failure_error and "rate-limited" in task.last_failure_error

        # A ``rate_limited`` run outcome was recorded (not ``crashed``).
        outcomes = [
            r["outcome"] for r in conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=?", (tid,),
            ).fetchall()
        ]
        assert "rate_limited" in outcomes
        assert "crashed" not in outcomes


def test_real_crash_still_counts_and_trips_breaker(kanban_home, monkeypatch):
    """Sanity: a genuine non-zero crash (not the sentinel) still increments
    the failure counter and trips the breaker — the rate-limit carve-out is
    surgical, not a blanket "never count crashes"."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="crash", assignee="a")

        for i in range(2):  # DEFAULT_FAILURE_LIMIT == 2
            pid = 60000 + i
            conn.execute(
                "UPDATE tasks SET status='running', worker_pid=?, "
                "claim_lock=? WHERE id=?",
                (pid, f"{host}:w{i}", tid),
            )
            conn.commit()
            _kb._record_worker_exit(pid, _exited_status(1))  # generic failure
            kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        # Automation-first (2026-07-16): the trip routes a never-decomposed
        # card to triage instead of blocked.
        assert task.status == "triage", (
            f"genuine crashes should still trip the breaker, got {task.status}"
        )


def test_respawn_guard_defers_rate_limited_within_cooldown(
    kanban_home, monkeypatch,
):
    """Within the cooldown after a rate-limit requeue, the guard defers the
    respawn; after the cooldown it allows a probe — and crucially does NOT
    fall into ``blocker_auth`` (which would defer forever)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rl-guard", assignee="a")
        # Seed a rate_limited run that just ended + the stamped error.
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
        conn.commit()

        # Inside cooldown → defer with the rate-limit-specific reason.
        monkeypatch.setattr(_kb.time, "time", lambda: now + 100)
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # Past cooldown → allowed (None), NOT trapped by blocker_auth even
        # though last_failure_error contains "rate-limited".
        monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
        assert kb.check_respawn_guard(conn, tid) is None








# ---------------------------------------------------------------------------
# Complete / block / unblock / archive / assign
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Orphaned-run recovery (worker-containment 2026-07-16)
#
# Reproduces the wedge on card t_bb8742af: a worker's run is ended out from
# under the still-alive process (exact-action / self-modify gate blocks the
# card, operator approves + unblocks it back to 'ready'), leaving the card at
# status='ready' with current_run_id=NULL while the original worker process
# keeps working and finishes. Its terminal kanban_complete / kanban_request_review
# carry expected_run_id == the just-ended run, which no longer matches
# current_run_id (NULL) -> the finished work has no way back and the card
# deadlocks behind the active_pr respawn guard.
# ---------------------------------------------------------------------------

def _orphan_ready_after_ended_run(conn, *, assignee="backend-eng"):
    """Return (task_id, ended_run_id) for a card that is 'ready' with
    current_run_id=NULL and whose newest run was ended out from under a
    still-alive worker. Mirrors: claim -> gate-block(needs_input) -> operator
    approve+unblock -> ready.
    """
    t = kb.create_task(conn, title="orphan repro", assignee=assignee)
    claimed = kb.claim_task(conn, t, claimer="host:worker")
    assert claimed is not None and claimed.current_run_id is not None
    run_id = int(claimed.current_run_id)
    # Gate ends the run and parks the card for a human (needs_input).
    assert kb.block_task(
        conn, t, reason="terminal approval required",
        kind="needs_input", trusted_internal=True,
    )
    blocked = kb.get_task(conn, t)
    assert blocked.status == "blocked" and blocked.current_run_id is None
    # Operator approves and unblocks -> card is 'ready' again, still no run.
    assert kb.unblock_task(conn, t)
    ready = kb.get_task(conn, t)
    assert ready.status == "ready" and ready.current_run_id is None
    return t, run_id


def test_orphaned_run_completion_is_adopted(kanban_home):
    """A still-alive worker whose run was ended (card requeued to 'ready',
    current_run_id=NULL) must be able to land its finished work by passing its
    own (now-ended) run as expected_run_id. Before the fix this returns False
    and the card deadlocks."""
    with kb.connect() as conn:
        t, run_id = _orphan_ready_after_ended_run(conn)
        ok = kb.complete_task(
            conn, t, result="finished after gate", expected_run_id=run_id,
        )
        assert ok is True
        task = kb.get_task(conn, t)
    assert task.status == "done"
    assert task.result == "finished after gate"


def test_orphaned_run_review_request_is_adopted(kanban_home):
    """Same recovery for the review workflow: request_task_review from an
    orphaned 'ready' card owned by the just-ended run must succeed."""
    with kb.connect() as conn:
        t, run_id = _orphan_ready_after_ended_run(conn)
        ok = kb.request_task_review(
            conn, t, reviewer="reviewer", summary="please review",
            expected_run_id=run_id,
        )
        assert ok is True
        task = kb.get_task(conn, t)
    assert task.status == "review"
    assert task.assignee == "reviewer"


def test_orphaned_completion_refused_after_respawn(kanban_home):
    """SAFETY: once a NEW run has claimed the orphaned card, the stale worker's
    expected_run_id must NOT complete it (no double-deliver / no clobber of the
    run a newer worker now owns)."""
    with kb.connect() as conn:
        t, stale_run = _orphan_ready_after_ended_run(conn)
        reclaimed = kb.claim_task(conn, t, claimer="host:worker2")
        assert reclaimed is not None
        new_run = int(reclaimed.current_run_id)
        assert new_run != stale_run
        # Stale worker (old run) tries to land work -> refused.
        assert kb.complete_task(
            conn, t, result="stale", expected_run_id=stale_run,
        ) is False
        task = kb.get_task(conn, t)
    assert task.status == "running"
    assert int(task.current_run_id) == new_run


def test_orphaned_completion_refused_when_newer_orphan_run_exists(kanban_home):
    """SAFETY: only the *latest* ended run of an orphaned card is adoptable. An
    older run id must be refused even when current_run_id is NULL, so a worker
    from a superseded attempt cannot land against a fresher one."""
    with kb.connect() as conn:
        t, first_run = _orphan_ready_after_ended_run(conn)
        # A second attempt runs and is itself gate-blocked then unblocked,
        # producing a newer orphaned run. A different block kind avoids the
        # same-cause recurrence loop-breaker (BLOCK_RECURRENCE_LIMIT), so the
        # card lands back at 'ready' rather than 'triage'.
        second = kb.claim_task(conn, t, claimer="host:worker2")
        assert second is not None
        second_run = int(second.current_run_id)
        assert kb.block_task(
            conn, t, reason="gate again", kind="capability",
            trusted_internal=True,
        )
        assert kb.unblock_task(conn, t)
        # Old (superseded) run refused; latest orphan run adopted.
        assert kb.complete_task(
            conn, t, result="from first", expected_run_id=first_run,
        ) is False
        assert kb.complete_task(
            conn, t, result="from second", expected_run_id=second_run,
        ) is True
        task = kb.get_task(conn, t)
    assert task.status == "done"
    assert task.result == "from second"


def test_reclaimed_orphan_run_is_not_adoptable(kanban_home):
    """SAFETY / containment: a run ended by RECLAIM (operator abort or stale
    claim) leaves the card 'ready' + orphaned too, but the reclaimed worker
    must NOT be able to land its work — that is the run_identity containment
    the fix must preserve. Only outcome='blocked' orphans are adoptable."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="reclaimed", assignee="backend-eng")
        claimed = kb.claim_task(conn, t, claimer="host:worker")
        run_id = int(claimed.current_run_id)
        assert kb.reclaim_task(conn, t, reason="operator abort")
        reclaimed = kb.get_task(conn, t)
        assert reclaimed.status == "ready" and reclaimed.current_run_id is None
        # Reclaimed (aborted) worker tries to sneak its work back in -> refused.
        assert kb.complete_task(
            conn, t, result="aborted work", expected_run_id=run_id,
        ) is False
        task = kb.get_task(conn, t)
    assert task.status == "ready"
    assert task.result != "aborted work"


def test_orphaned_needs_input_block_not_bypassed(kanban_home):
    """SAFETY: adoption only applies to a 'ready' orphan. A card still parked at
    blocked+needs_input (human gate not yet cleared) must NOT be completable via
    a stale expected_run_id."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="still gated", assignee="backend-eng")
        claimed = kb.claim_task(conn, t, claimer="host:worker")
        run_id = int(claimed.current_run_id)
        assert kb.block_task(
            conn, t, reason="terminal approval required",
            kind="needs_input", trusted_internal=True,
        )
        blocked = kb.get_task(conn, t)
        assert blocked.status == "blocked" and blocked.current_run_id is None
        # No operator unblock yet -> gate still holds -> refuse.
        assert kb.complete_task(
            conn, t, result="sneaky", expected_run_id=run_id,
        ) is False
        task = kb.get_task(conn, t)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"


def test_block_then_unblock(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        assert kb.get_task(conn, t).status == "blocked"
        assert kb.unblock_task(conn, t)
        assert kb.get_task(conn, t).status == "ready"


def test_unblock_resets_failure_counters(kanban_home):
    """unblock_task must reset consecutive_failures and last_failure_error."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        # Simulate accumulated failures from the circuit breaker
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 5, "
            "last_failure_error = 'test error' WHERE id = ?",
            (t,),
        )
        conn.commit()
        assert kb.unblock_task(conn, t)
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None


def test_recompute_ready_skips_tasks_at_failure_limit(kanban_home):
    """recompute_ready must not auto-recover tasks whose consecutive_failures
    has reached the circuit-breaker limit (#35072).

    Without this guard, a task that repeatedly exhausts its iteration
    budget would cycle forever: block → auto-recover (counter reset)
    → respawn → budget exhausted → block → …
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(conn, title="child", assignee="a",
                               parents=[parent])
        # Complete the parent so the child's dependencies are satisfied.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="done")

        # Simulate the child having exhausted its budget twice,
        # hitting the default failure limit (2).
        kb.claim_task(conn, child)
        kb._record_task_failure(
            conn, child, error="budget exhausted 1",
            outcome="timed_out", release_claim=True, end_run=True,
            failure_limit=2,
        )
        kb._record_task_failure(
            conn, child, error="budget exhausted 2",
            outcome="timed_out", release_claim=True, end_run=True,
            failure_limit=2,
        )
        task = kb.get_task(conn, child)
        # Automation-first (2026-07-16): breaker trip routes to triage.
        assert task.status == "triage"
        assert task.consecutive_failures >= 2

        # recompute_ready must NOT promote this task — the circuit
        # breaker has tripped; automation-first (2026-07-16) parks it in
        # triage and recompute_ready must not promote it either way.
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "triage"

        # K-3 (2026-07-16): a tripped breaker now projects a typed operator
        # attention (so the cockpit shows a reason instead of a naked block).
        # Attention-bearing cards are resolved through the typed seam
        # (``transition_task_status_with_attention``); the legacy
        # ``unblock_task`` fail-closes on any live projection by design.
        att = kb.get_current_attention(conn, child)
        assert att is not None and att.requires_human_action
        assert kb.transition_task_status_with_attention(
            conn, task_id=child, status="ready",
            expected_attention_id=att.id,
            expected_attention_version=att.version,
        )
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        assert task.consecutive_failures == 0


def test_circuit_breaker_trip_leaves_operator_attention(kanban_home):
    """K-3 (2026-07-16) regression: the consecutive-failure circuit breaker
    must not leave a "naked block".

    A card blocked by ``_record_task_failure`` used to land in ``blocked``
    with NO ``block_kind``, NO ``block_reason_code``, NO ``blocked`` event,
    and NO ``task_attentions`` row — only a ``gave_up`` event. The cockpit
    reads ``get_current_attention`` as its operator surface, so such a card
    rendered "Kein Grund hinterlegt." (no reason on file). This reproduces
    the live symptom seen on card t_05ad5273 ("Repair WebUI PR #5836").

    After the fix the trip stamps ``block_kind='needs_input'`` +
    ``block_reason_code='gave_up'`` and projects a typed, human-actionable
    attention with a meaningful summary.
    """
    with kb.connect() as conn:
        t = kb.create_task(conn, title="repair card", assignee="a")
        kb.claim_task(conn, t)

        # Trip the breaker with a single failure at limit=1 (mirrors the
        # dispatcher spawn/crash path that blocked t_05ad5273).
        tripped = kb._record_task_failure(
            conn, t, error="pid 3592337 not alive",
            outcome="crashed", release_claim=True, end_run=True,
            failure_limit=1,
        )
        assert tripped is True

        task = kb.get_task(conn, t)
        # Automation-first (2026-07-16): a FIRST trip on a never-decomposed
        # card routes to triage so the decompose automat tries before a human
        # is paged. The durable block fields are stamped either way.
        assert task.status == "triage"
        assert task.block_kind == "needs_input"
        reason_code = conn.execute(
            "SELECT block_reason_code FROM tasks WHERE id=?", (t,),
        ).fetchone()[0]
        assert reason_code == "gave_up"

        # The operator surface the cockpit actually reads must be populated
        # and flagged as needing a human decision, with a non-empty summary.
        att = kb.get_current_attention(conn, t)
        assert att is not None, "gave_up trip must leave an operator attention"
        assert att.type == "decision"
        assert att.requires_human_action is True
        assert att.summary and "gave up" in att.summary.lower()

        # The lifecycle ``gave_up`` event is still emitted (and now
        # self-describes the projection it created).
        events = kb.list_events(conn, t)
        gave_up = [e for e in events if e.kind == "gave_up"]
        assert gave_up, f"expected gave_up event, got {[e.kind for e in events]}"
        assert gave_up[-1].payload.get("reason_code") == "gave_up"


def test_circuit_breaker_trip_after_decompose_blocks_for_human(kanban_home):
    """Cascade guard for automation-first routing (2026-07-16): a card the
    triage automat already decomposed once must NOT re-enter automation on the
    next breaker trip — it blocks for a human, exactly the pre-change
    behavior."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="already decomposed", assignee="a")
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'decomposed', ?)",
            (t, int(time.time())),
        )
        kb.claim_task(conn, t)
        tripped = kb._record_task_failure(
            conn, t, error="pid gone", outcome="crashed",
            release_claim=True, end_run=True, failure_limit=1,
        )
        assert tripped is True
        task = kb.get_task(conn, t)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


# ---------------------------------------------------------------------------
# Circuit-breaker trip on a still-OPEN run closes it as adoptable 'blocked'
# (2026-07-17, external-parker run-identity fix)
#
# _record_task_failure has two calling modes: release_claim=False/end_run=
# False (crash/timeout/reclaim paths) where the CALLER already pre-closed
# the run with its own terminal outcome before calling in -- those stay
# untouched and non-adoptable, by design. release_claim=True/end_run=True
# (spawn-failure, and agent/turn_finalizer.py's goal-loop iteration-budget-
# exhaustion self-report) hands _record_task_failure a genuinely OPEN run.
# When that mode trips the breaker, closing the run as outcome='gave_up'
# made it permanently non-adoptable (_adoptable_orphan_run only treats
# outcome='blocked' as adoptable) even though nothing crashed/timed out —
# the worker may still be alive and returning with finished work.
# ---------------------------------------------------------------------------

def test_circuit_breaker_trip_closes_open_run_as_adoptable_blocked(kanban_home):
    """A still-open run (release_claim=True/end_run=True — e.g. the goal-loop
    budget-exhaustion self-report) that trips the breaker must close with
    outcome='blocked' so a worker returning after the card is requeued to
    'ready' can still land its finished work via orphan adoption."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="goal card", assignee="a", max_retries=1)
        claimed = kb.claim_task(conn, t)
        run_id = int(claimed.current_run_id)

        tripped = kb._record_task_failure(
            conn, t, error="Iteration budget exhausted (90/90)",
            outcome="timed_out", release_claim=True, end_run=True,
        )
        assert tripped is True

        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        assert run["ended_at"] is not None
        assert run["outcome"] == "blocked"
        assert run["status"] == "blocked"

        task = kb.get_task(conn, t)
        assert task.status in ("blocked", "triage")
        assert task.current_run_id is None

        # Operator/automation resolves the attention and requeues the card.
        att = kb.get_current_attention(conn, t)
        assert att is not None
        assert kb.transition_task_status_with_attention(
            conn, task_id=t, status="ready",
            expected_attention_id=att.id,
            expected_attention_version=att.version,
        )

        # A worker that is still alive past its own budget-exhaustion report
        # can now land its finished work citing the closed-but-open-at-trip-
        # time run as expected_run_id.
        assert kb.complete_task(
            conn, t, result="delivered after budget trip",
            expected_run_id=run_id,
        ) is True
        assert kb.get_task(conn, t).status == "done"


def test_circuit_breaker_preclosed_run_outcome_unaffected(kanban_home, monkeypatch):
    """SAFETY: crash/timeout/reclaim pre-close their own run with their own
    terminal outcome BEFORE calling _record_task_failure (release_claim=
    False, end_run=False) -- the trip must never touch that outcome. Those
    stay NOT adoptable (see test_reclaimed_orphan_run_is_not_adoptable) —
    this fix only concerns the release_claim=True/end_run=True (still-open-
    run) path."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="crashy", assignee="a", max_retries=1)
        host = _kb._claimer_id().split(":", 1)[0]
        claimed = kb.claim_task(conn, t, claimer=f"{host}:w")
        run_id = int(claimed.current_run_id)
        conn.execute(
            "UPDATE tasks SET worker_pid=? WHERE id=?", (777777, t),
        )
        conn.commit()

        crashed = _kb.detect_crashed_workers(conn)
        assert t in crashed

        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        # The crash path already closed this run with outcome='crashed'
        # before _record_task_failure's trip ever ran; it must stay that
        # way, not be relabeled 'blocked'.
        assert run["outcome"] == "crashed"
        assert run["ended_at"] is not None

        task = kb.get_task(conn, t)
        assert task.status in ("blocked", "triage")

        # And it must stay refused for adoption (mirrors
        # test_reclaimed_orphan_run_is_not_adoptable's invariant).
        conn.execute(
            "UPDATE tasks SET status='ready', block_kind=NULL, "
            "current_run_id=NULL WHERE id=?", (t,),
        )
        conn.commit()
        assert kb.complete_task(
            conn, t, result="sneaking in a crash", expected_run_id=run_id,
        ) is False


def test_block_task_prefer_triage_first_occurrence(kanban_home):
    """prefer_triage routes a TECHNICAL first-occurrence block straight to
    triage (automation first); a card with a prior decompose falls back to
    blocked (cascade guard)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="guard escalation", assignee="a")
        assert kb.block_task(
            conn, t, reason="guard defer", kind="needs_input",
            prefer_triage=True,
        )
        assert kb.get_task(conn, t).status == "triage"

        t2 = kb.create_task(conn, title="guard escalation decomposed", assignee="a")
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'decomposed', ?)",
            (t2, int(time.time())),
        )
        assert kb.block_task(
            conn, t2, reason="guard defer", kind="needs_input",
            prefer_triage=True,
        )
        assert kb.get_task(conn, t2).status == "blocked"


def test_recompute_ready_recovers_below_limit(kanban_home):
    """recompute_ready auto-recovers blocked tasks that haven't hit the
    failure limit yet — the counter is preserved across recovery."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="task", assignee="a")
        kb.claim_task(conn, t)
        # One failure, below the default limit of 2.
        kb._record_task_failure(
            conn, t, error="budget exhausted 1",
            outcome="timed_out", release_claim=True, end_run=True,
            failure_limit=2,
        )
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == 1

        # Simulate being blocked by something else (not circuit breaker).
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (t,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        # Counter must be preserved, not reset.
        assert task.consecutive_failures == 1




def test_recompute_ready_honours_dispatcher_failure_limit(kanban_home):
    """The guard's effective limit must follow the same resolution order
    as the circuit breaker (#35072): per-task max_retries → dispatcher
    failure_limit → DEFAULT_FAILURE_LIMIT.

    Without threading the dispatcher's ``kanban.failure_limit`` through,
    the guard falls back to DEFAULT_FAILURE_LIMIT and disagrees with the
    breaker — sticking a task prematurely (config limit > default) or
    letting a tripped task escape (config limit < default).
    """
    with kb.connect() as conn:
        # Config allows MORE retries than the default. A task blocked
        # with failures below the configured limit must still recover.
        t = kb.create_task(conn, title="lenient", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=? "
            "WHERE id=?",
            (kb.DEFAULT_FAILURE_LIMIT, t),
        )
        conn.commit()
        # Default-limit call would stick it (failures >= default).
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, t).status == "blocked"
        # Dispatcher configured a higher limit → recover, preserve counter.
        promoted = kb.recompute_ready(
            conn, failure_limit=kb.DEFAULT_FAILURE_LIMIT + 2
        )
        assert promoted == 1
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == kb.DEFAULT_FAILURE_LIMIT

        # Config allows FEWER retries than the default. A task at the
        # stricter limit must stay blocked even though it's below default.
        t2 = kb.create_task(conn, title="strict", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1 "
            "WHERE id=?",
            (t2,),
        )
        conn.commit()
        # Default-limit (2) would recover it (1 < 2).
        # Stricter config limit (1) must keep it blocked (1 >= 1).
        assert kb.recompute_ready(conn, failure_limit=1) == 0
        assert kb.get_task(conn, t2).status == "blocked"




# ---------------------------------------------------------------------------
# Parent-completion invariant at the claim gate (RCA t_a6acd07d)
# ---------------------------------------------------------------------------














def test_delete_archived_task_removes_related_rows(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent], assignee="worker")
        kb.add_comment(conn, tid, "user", "cleanup me")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="done")
        assert kb.archive_task(conn, tid)
        conn.execute(
            "INSERT INTO kanban_notify_subs(task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id) "
            "VALUES (?, 'telegram', '123', '', 'u', 0, 0)",
            (tid,),
        )
        conn.commit()

        assert kb.delete_archived_task(conn, tid) is True
        assert kb.get_task(conn, tid) is None
        assert conn.execute("SELECT COUNT(*) FROM task_links WHERE child_id = ? OR parent_id = ?", (tid, tid)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (tid,)).fetchone()[0] == 0


def test_delete_task_removes_task_and_cascades(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="to-delete", assignee="alice")
        kb.add_comment(conn, t, "user", "comment")
        kb.add_comment(conn, t, "user", "another")
        assert kb.archive_task(conn, t)
        assert kb.delete_task(conn, t, operator_reason="retention expired")
        assert kb.get_task(conn, t) is None
        assert len(kb.list_comments(conn, t)) == 0
        assert len(kb.list_events(conn, t)) == 0
        assert len(kb.list_runs(conn, t)) == 0


def test_delete_task_returns_false_for_missing_task(kanban_home):
    with kb.connect() as conn:
        assert not kb.delete_task(conn, "t_nonexistent")


def test_delete_task_cascades_links(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="parent")
        c = kb.create_task(conn, title="child", parents=[p])
        child = kb.get_task(conn, c)
        assert child is not None and child.status == "todo"
        kb.archive_task(conn, p)
        kb.delete_task(conn, p, operator_reason="retention expired")
        assert kb.get_task(conn, p) is None
        child_after = kb.get_task(conn, c)
        assert child_after is not None and child_after.status == "ready"


# ---------------------------------------------------------------------------
# Comments / events / worker context
# ---------------------------------------------------------------------------





def test_events_capture_lifecycle(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        kb.complete_task(conn, t, result="ok")
        events = kb.list_events(conn, t)
    kinds = [e.kind for e in events]
    assert "created" in kinds
    assert "claimed" in kinds
    assert "completed" in kinds


def test_record_goal_progress_event_persists_and_round_trips(kanban_home):
    """The S4 goal-loop budget/progress sink: a caller wiring
    ``goals.run_kanban_goal_loop``'s ``emit_progress`` callback to this
    function must get a retrievable ``goal_progress`` event with the
    payload intact."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        run = kb.latest_run(conn, t)
        kb.record_goal_progress_event(
            conn,
            t,
            {
                "task_id": t,
                "phase": "closeout",
                "turns_used": 2,
                "max_turns": 3,
                "verdict_history": ["continue", "continue"],
            },
            run_id=run.id if run else None,
        )
        events = [e for e in kb.list_events(conn, t) if e.kind == "goal_progress"]
    assert len(events) == 1
    assert events[0].payload["phase"] == "closeout"
    assert events[0].payload["turns_used"] == 2
    if run is not None:
        assert events[0].run_id == run.id


def test_record_goal_progress_event_truncates_oversized_payload(kanban_home):
    """An oversized payload must never be silently dropped — it's replaced
    with a short, clearly-marked summary instead of growing the events
    table unboundedly."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        huge_payload = {
            "task_id": t,
            "phase": "continue",
            "turns_used": 1,
            "max_turns": 10,
            "junk": "x" * 10000,
        }
        kb.record_goal_progress_event(conn, t, huge_payload)
        events = [e for e in kb.list_events(conn, t) if e.kind == "goal_progress"]
    assert len(events) == 1
    assert events[0].payload.get("truncated") is True
    assert "junk" not in events[0].payload


def test_worker_context_includes_parent_results_and_comments(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="p")
        kb.complete_task(conn, p, result="PARENT_RESULT_MARKER")
        c = kb.create_task(conn, title="child", parents=[p])
        kb.add_comment(conn, c, "user", "CLARIFICATION_MARKER")
        ctx = kb.build_worker_context(conn, c)
    assert "PARENT_RESULT_MARKER" in ctx
    assert "CLARIFICATION_MARKER" in ctx
    assert c in ctx
    assert "child" in ctx


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Respawn guard (check_respawn_guard + dispatch_once integration)
# ---------------------------------------------------------------------------







def test_respawn_guard_blocker_auth_on_authentication_error(kanban_home):
    """Full word 'Authentication' triggers blocker_auth (regex covers auth\\w*)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="authn-task", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("Authentication failed: invalid credentials", t),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "blocker_auth"


def test_respawn_guard_blocker_auth_on_authorization_error(kanban_home):
    """Full word 'authorization' triggers blocker_auth (regex covers auth\\w*)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="authz-task", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("authorization denied for scope repo", t),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "blocker_auth"


def test_respawn_guard_recent_success(kanban_home):
    """A completed run within the guard window triggers recent_success."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="already-done", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 120, now - 60),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "recent_success"


def test_respawn_guard_recent_success_bypassed_by_requeue(kanban_home):
    """An explicit re-queue after a recent success (operator done->ready,
    promote, unblock, reclaim) is a deliberate re-run and must bypass the
    recent_success guard — otherwise a manual done->ready just sits there
    until the window elapses."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="rerun-me", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 120, now - 60),
        )
        # Baseline: a recent completion defers the respawn.
        assert kb.check_respawn_guard(conn, t) == "recent_success"
        # Operator drags done -> ready: a 'status' event after completion.
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'status', ?)",
            (t, now - 10),
        )
        assert kb.check_respawn_guard(conn, t) is None


def test_respawn_guard_stale_success_not_guarded(kanban_home):
    """A completed run outside the guard window does not block re-spawn."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="old-done", assignee="alice")
        old_end = int(time.time()) - kb._RESPAWN_GUARD_SUCCESS_WINDOW - 60
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, old_end - 300, old_end),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_active_pr_in_comment(kanban_home):
    """A GitHub PR URL in a recent comment triggers active_pr."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(
            conn, t, "worker",
            "PR created: https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_active_pr_bypassed_by_requeue(kanban_home):
    """An explicit re-queue after the newest PR-URL comment is a deliberate
    re-run and must bypass the active_pr guard — mirrors the recent_success
    bypass. Without it, a card whose job is repairing an existing PR (its
    comments necessarily cite the PR URL) is deferred for the whole 24h
    window and no WebUI unblock/ready can free it (2026-07-16, t_5a43abe1)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="repair-pr", assignee="alice")
        kb.add_comment(
            conn, t, "worker",
            "Repairing https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        # Baseline: the PR URL in a fresh comment defers the respawn.
        assert kb.check_respawn_guard(conn, t) == "active_pr"
        # Operator re-queues AFTER that comment: bypass.
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'status', ?)",
            (t, int(time.time()) + 1),
        )
        assert kb.check_respawn_guard(conn, t) is None
        # A newer PR-URL comment re-arms the guard.
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'worker', "
            "'Updated https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, int(time.time()) + 2),
        )
        assert kb.check_respawn_guard(conn, t) == "active_pr"


def test_respawn_guard_old_pr_comment_not_guarded(kanban_home):
    """A GitHub PR URL in a comment older than the PR window does not block."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="old-pr", assignee="alice")
        old_ts = int(time.time()) - kb._RESPAWN_GUARD_PR_WINDOW - 60
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'worker', "
            "'PR: https://github.com/totemx-AI/subsidysmart/pull/10', ?)",
            (t, old_ts),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_dispatch_respawn_guard_defers_auth_error_without_auto_block(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once defers (does NOT auto-block) a ready task whose last
    error is a blocker_auth.

    The old behaviour auto-blocked on first occurrence, which was too
    aggressive: a transient 429 rate-limit (which typically clears in
    seconds to minutes) would end up requiring manual unblock. The new
    behaviour defers the spawn this tick; the task stays in ``ready``
    and gets another chance next tick. If the auth error genuinely
    persists, the existing ``consecutive_failures`` circuit breaker
    will auto-block via the normal failure-limit path.
    """
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="quota-storm", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("rate limit exceeded: 429 Too Many Requests", t),
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    # Critical: task is NOT auto-blocked on first occurrence.
    assert t not in res.auto_blocked, (
        f"blocker_auth should defer, not auto-block on first occurrence; "
        f"got auto_blocked={res.auto_blocked!r}"
    )
    # It IS recorded as respawn_guarded with the reason.
    assert (t, "blocker_auth") in res.respawn_guarded, (
        f"expected (task_id, 'blocker_auth') in respawn_guarded; "
        f"got {res.respawn_guarded!r}"
    )
    # And it's NOT spawned this tick.
    assert t not in spawned_ids
    # Status stays ``ready`` so a future tick (or operator action) can
    # retry without manual unblock.
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"


def test_dispatch_respawn_guard_skips_recent_success(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once skips (but does not block) a task with a recent completed run."""
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="recent-winner", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 300, now - 60),
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert (t, "recent_success") in res.respawn_guarded
    assert t not in spawned_ids
    assert t not in res.auto_blocked
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"  # not blocked, just skipped


def test_dispatch_respawn_guard_skips_active_pr(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once skips (but does not block) a task with an active PR comment."""
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(
            conn, t, "worker",
            "Opened https://github.com/totemx-AI/subsidysmart/pull/99",
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert (t, "active_pr") in res.respawn_guarded
    assert t not in spawned_ids
    assert t not in res.auto_blocked
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"


def test_dispatch_respawn_guard_dry_run_no_auto_block(
    kanban_home, all_assignees_spawnable
):
    """In dry_run mode, blocker_auth tasks are recorded in respawn_guarded (not auto-blocked)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="dry-quota", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("quota exceeded", t),
        )
        res = kb.dispatch_once(conn, dry_run=True)

    assert (t, "blocker_auth") in res.respawn_guarded
    assert t not in res.auto_blocked
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"  # dry_run: no writes


def test_dispatch_respawn_guard_allows_clean_task(
    kanban_home, all_assignees_spawnable
):
    """A task with no guard triggers is spawned normally."""
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="clean-task", assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert t in spawned_ids
    assert not res.respawn_guarded
    assert t not in res.auto_blocked


def test_dispatch_respawn_guard_emits_event_for_skipped_task(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once emits a respawn_guarded task_event so operators can diagnose stuck-ready tasks."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="event-check", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 300, now - 60),
        )
        kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        events = kb.list_events(conn, t)

    kinds = [e.kind for e in events]
    assert "respawn_guarded" in kinds
    guarded_evt = next(e for e in events if e.kind == "respawn_guarded")
    # Event.payload is already parsed as a dict by list_events.
    assert isinstance(guarded_evt.payload, dict)
    assert guarded_evt.payload.get("reason") == "recent_success"


def _insert_guard_event(conn, task_id, created_at, kind="respawn_guarded"):
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, ?, '{\"reason\": \"active_pr\"}', ?)",
        (task_id, kind, created_at),
    )


def test_dispatch_active_pr_guard_escalates_to_block_after_window(
    kanban_home, all_assignees_spawnable
):
    """An active_pr guard that has continuously deferred a ready task for
    longer than _RESPAWN_GUARD_PR_ESCALATE_SECONDS escalates to an explicit
    needs_input block instead of livelocking in ready forever."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="gate-livelock", assignee="alice")
        kb.add_comment(
            conn, t, "reviewer",
            "verdict ACCEPT — https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        now = int(time.time())
        _insert_guard_event(
            conn, t, now - kb._RESPAWN_GUARD_PR_ESCALATE_SECONDS - 120
        )
        res = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        task = kb.get_task(conn, t)

    assert (t, "active_pr") in res.respawn_guarded
    assert t in res.auto_blocked
    # Automation-first (2026-07-16): guard escalation routes to triage.
    assert task.status == "triage"
    assert task.block_kind == "needs_input"


def test_dispatch_active_pr_guard_defers_before_escalate_window(
    kanban_home, all_assignees_spawnable
):
    """Within the escalate window the guard defers only — task stays ready."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="gate-fresh", assignee="alice")
        kb.add_comment(
            conn, t, "reviewer",
            "PR: https://github.com/totemx-AI/subsidysmart/pull/43",
        )
        res = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        task = kb.get_task(conn, t)

    assert (t, "active_pr") in res.respawn_guarded
    assert t not in res.auto_blocked
    assert task.status == "ready"


def test_dispatch_active_pr_guard_escalate_clock_resets_on_promote(
    kanban_home, all_assignees_spawnable
):
    """A promote/spawn/claim/unblock event resets the escalation clock: old
    guard events before the marker do not count toward the window."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="gate-reset", assignee="alice")
        kb.add_comment(
            conn, t, "reviewer",
            "PR: https://github.com/totemx-AI/subsidysmart/pull/44",
        )
        now = int(time.time())
        _insert_guard_event(
            conn, t, now - kb._RESPAWN_GUARD_PR_ESCALATE_SECONDS - 600
        )
        _insert_guard_event(conn, t, now - 100, kind="promoted")
        res = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        task = kb.get_task(conn, t)

    assert (t, "active_pr") in res.respawn_guarded
    assert t not in res.auto_blocked
    assert task.status == "ready"


# ---------------------------------------------------------------------------
# Respawn-guard event storm cap (2026-07-17)
#
# A persistent guard (active_pr) re-fires every dispatcher tick for as long
# as it holds, which used to append a fresh 'respawn_guarded' task_event
# EVERY tick -- 605 events in one incident window, 406 on a single card
# (t_614c91e9). The guard itself must keep deferring the spawn every tick
# (unchanged); only the event-log spam is capped.
# ---------------------------------------------------------------------------

def test_dispatch_respawn_guard_storm_is_capped(kanban_home, all_assignees_spawnable):
    """Repeated ticks of the same guard append at most _RESPAWN_GUARD_EVENT_CAP
    respawn_guarded events plus one storm-capped marker -- never one per tick."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="storm", assignee="alice")
        kb.add_comment(
            conn, t, "reviewer",
            "verdict PENDING — https://github.com/totemx-AI/subsidysmart/pull/77",
        )
        for _ in range(20):
            res = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
            # The guard must keep firing (deferring the spawn) every single
            # tick -- capping the EVENT LOG must never weaken the guard.
            assert (t, "active_pr") in res.respawn_guarded
            assert kb.get_task(conn, t).status == "ready"

        events = kb.list_events(conn, t)

    guarded = [e for e in events if e.kind == "respawn_guarded"]
    capped = [e for e in events if e.kind == "respawn_guard_storm_capped"]
    assert len(guarded) == kb._RESPAWN_GUARD_EVENT_CAP, (
        f"expected exactly {kb._RESPAWN_GUARD_EVENT_CAP} respawn_guarded "
        f"events across 20 ticks, got {len(guarded)}"
    )
    assert len(capped) == 1
    assert capped[0].payload.get("reason") == "active_pr"


def test_dispatch_respawn_guard_storm_cap_resets_on_progress(
    kanban_home, all_assignees_spawnable
):
    """A reset event (promote/spawn/claim/unblock) clears the storm-cap
    window exactly like it clears the active_pr escalation window -- a
    fresh guard streak after progress is not silently swallowed forever."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="storm-reset", assignee="alice")
        kb.add_comment(
            conn, t, "reviewer",
            "verdict PENDING — https://github.com/totemx-AI/subsidysmart/pull/78",
        )
        now = int(time.time())
        for i in range(kb._RESPAWN_GUARD_EVENT_CAP):
            _insert_guard_event(conn, t, now - 1000 + i)
        _insert_guard_event(conn, t, now - 500, kind="promoted")
        conn.commit()

        res = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        events = kb.list_events(conn, t)

    assert (t, "active_pr") in res.respawn_guarded
    guarded_after_reset = [
        e for e in events
        if e.kind == "respawn_guarded" and e.created_at > now - 500
    ]
    assert len(guarded_after_reset) == 1, (
        "a reset event must clear the storm-cap window, not carry the "
        "pre-reset count forward"
    )


def test_dispatch_active_pr_guard_escalates_after_storm_cap(
    kanban_home, all_assignees_spawnable
):
    """SAFETY: the active_pr auto-block escalation depends only on the
    EARLIEST respawn_guarded event since the last reset (MIN(created_at)).
    Capping later duplicates must never remove that earliest event or
    otherwise weaken the 30-minute escalation."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="storm-then-escalate", assignee="alice")
        kb.add_comment(
            conn, t, "reviewer",
            "verdict ACCEPT — https://github.com/totemx-AI/subsidysmart/pull/79",
        )
        now = int(time.time())
        # Simulate a storm that already hit the cap, starting well before
        # the escalation window.
        start = now - kb._RESPAWN_GUARD_PR_ESCALATE_SECONDS - 600
        for i in range(kb._RESPAWN_GUARD_EVENT_CAP):
            _insert_guard_event(conn, t, start + i * 5)
        _insert_guard_event(conn, t, start + 60, kind="respawn_guard_storm_capped")
        conn.commit()

        res = kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        task = kb.get_task(conn, t)

    assert (t, "active_pr") in res.respawn_guarded
    assert t in res.auto_blocked
    assert task.status == "triage"
    assert task.block_kind == "needs_input"


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------







def test_worktree_no_path_anchors_on_board_default_workdir(kanban_home, tmp_path):
    """A worktree task created with no explicit path inherits the board's
    default_workdir as its anchor and materializes a per-task linked worktree
    at ``<repo>/.worktrees/<id>`` — NOT the dispatcher's CWD, and NOT the
    shared default_workdir verbatim (which would collapse every task into one
    directory)."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("wt-default-board", default_workdir=str(repo))
    with kb.connect(board="wt-default-board") as conn:
        t = kb.create_task(
            conn, title="ship", workspace_kind="worktree", board="wt-default-board"
        )
        task = kb.get_task(conn, t)
        assert task is not None
        ws = kb.resolve_workspace(task, board="wt-default-board")

    expected = repo / ".worktrees" / t
    assert ws == expected
    assert ws.exists()
    assert ws != repo  # not the shared default verbatim


def test_worktree_no_path_no_board_default_raises(kanban_home, tmp_path, monkeypatch):
    """With neither an explicit workspace_path nor a board default_workdir,
    resolution fails loudly pointing at default_workdir / worktree:<path> —
    rather than silently materializing under the dispatcher's CWD (the old
    behavior that scattered worktrees under whatever dir launched the
    gateway)."""
    # Park the dispatcher CWD inside a real git repo so the OLD cwd-anchored
    # code would have "succeeded" — proving the new code does NOT use cwd.
    decoy_repo = tmp_path / "decoy"
    _init_git_repo(decoy_repo)
    monkeypatch.chdir(decoy_repo)
    with kb.connect() as conn:
        # Legacy-Zeile simulieren: create_task lehnt pfadlose worktree-Karten
        # ohne Board-Default inzwischen an der Quelle ab (K-4) — der
        # Dispatch-Zeit-Check hier bleibt der Backstop für Bestandszeilen.
        t = kb.create_task(
            conn, title="ship", workspace_kind="worktree",
            workspace_path=str(tmp_path / "legacy" / "path"),
        )
        conn.execute(
            "UPDATE tasks SET workspace_path = NULL WHERE id = ?", (t,)
        )
        conn.commit()
        task = kb.get_task(conn, t)
        assert task is not None
        with pytest.raises(ValueError, match="default_workdir"):
            kb.resolve_workspace(task)


def test_create_worktree_task_without_path_rejected_at_source(kanban_home):
    """K-4 (Vollaudit 2026-07-16): pathless worktree card on a board without
    default_workdir is refused at CREATE time — the error goes to the
    creating agent instead of burning spawn_failed/gave_up cycles later."""
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="default_workdir"):
            kb.create_task(conn, title="ship", workspace_kind="worktree")


def test_create_worktree_task_without_path_ok_with_board_default(
    kanban_home, tmp_path,
):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("wt-src-board", default_workdir=str(repo))
    with kb.connect(board="wt-src-board") as conn:
        t = kb.create_task(
            conn, title="ship", workspace_kind="worktree", board="wt-src-board",
        )
        assert kb.get_task(conn, t) is not None


def test_worktree_workspace_explicit_target_materializes_linked_worktree(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "custom-task"
    branch = "wt/custom-task"
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=branch,
        )
        task = kb.get_task(conn, t)
        assert task is not None
        ws = kb.resolve_workspace(task)

    assert ws == target
    assert ws.exists()
    repo_common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ws_common = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert ws_common == repo_common
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"worktree {target}" in listed
    assert f"branch refs/heads/{branch}" in listed


# ---------------------------------------------------------------------------
# Scratch cleanup containment (#28818)
# ---------------------------------------------------------------------------










def test_complete_task_persists_scratch_artifacts_before_cleanup(kanban_home):
    """Completion artifacts from scratch workspaces survive workspace cleanup."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="render chart")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, t, ws)
        artifact = ws / "chart.png"
        artifact.write_bytes(b"png-bytes")

        assert kb.complete_task(
            conn,
            t,
            result="ok",
            metadata={"artifacts": [str(artifact)]},
        )

        completed = [e for e in kb.list_events(conn, t) if e.kind == "completed"][-1]
        persisted = Path(completed.payload["artifacts"][0])
        run = kb.latest_run(conn, t)

    assert not ws.exists(), "scratch workspace should still be cleaned up"
    assert persisted.exists(), "artifact copy should survive scratch cleanup"
    # fork(tars) 31.07.2026: upstream persistiert nach attachments/<task>/ und
    # prüfte `persisted.parent == task_attachments_dir(t)`. Unser Fork legt
    # Completion-Artefakte bewusst im run-versionierten Artefakt-Store ab
    # (completion_artifacts_root()/<task>/<run>/<hash>, manifest-geführt —
    # die Deliverables-Pipeline). Upstreams eigentliche Zusicherung — die
    # Kopie überlebt das Scratch-Cleanup und die Events/Runs zeigen auf die
    # Kopie, nicht aufs Original — bleibt vollständig geprüft.
    assert kb.completion_artifacts_root() in persisted.parents
    assert persisted.read_bytes() == b"png-bytes"
    assert str(persisted) != str(artifact)
    assert run is not None
    assert run.metadata["artifacts"] == [str(persisted)]




# ---------------------------------------------------------------------------
# Deferred scratch cleanup for parent/child handoff (#33774)
# ---------------------------------------------------------------------------

def test_cleanup_workspace_refuses_path_outside_scratch_root(kanban_home, tmp_path):
    """A scratch task with a user path outside the workspaces root must NOT be deleted (#28818).

    Reproduces the data-loss vector where a board's ``default_workdir`` is set
    to a real source directory; tasks created without an explicit
    ``workspace_kind`` inherit ``scratch`` semantics, and the old cleanup path
    would ``shutil.rmtree`` the user's source tree on task completion.
    """
    real_source = tmp_path / "real-source"
    real_source.mkdir()
    (real_source / ".git").mkdir()
    (real_source / "README.md").write_text("important", encoding="utf-8")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="ship")
        # Simulate the bad state directly: workspace_kind='scratch' (default)
        # but workspace_path pointing at the user's real source tree, which is
        # exactly what board.default_workdir produces when the task is created
        # without an explicit workspace_kind.
        conn.execute(
            "UPDATE tasks SET workspace_kind=?, workspace_path=? WHERE id=?",
            ("scratch", str(real_source), t),
        )
        conn.commit()
        kb.complete_task(conn, t, result="ok")

    assert real_source.exists(), "User source tree must not be deleted by scratch cleanup"
    assert (real_source / ".git").exists()
    assert (real_source / "README.md").read_text(encoding="utf-8") == "important"


def test_cleanup_workspace_honors_workspaces_root_env_override(tmp_path, monkeypatch):
    """``HERMES_KANBAN_WORKSPACES_ROOT`` extends the managed-scratch set.

    Worker subprocesses run with this env var injected by the dispatcher. The
    cleanup containment check must treat paths under it as managed even when
    they sit outside the active kanban home.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workspaces_override = tmp_path / "ext-workspaces"
    workspaces_override.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(workspaces_override))
    kb.init_db()

    with kb.connect() as conn:
        t = kb.create_task(conn, title="ext")
        scratch_dir = workspaces_override / t
        scratch_dir.mkdir()
        conn.execute(
            "UPDATE tasks SET workspace_kind=?, workspace_path=? WHERE id=?",
            ("scratch", str(scratch_dir), t),
        )
        conn.commit()
        kb.complete_task(conn, t, result="ok")

    assert not scratch_dir.exists(), "Override-root scratch dir should be cleaned up"


# ---------------------------------------------------------------------------
# Deferred scratch cleanup for parent/child handoff (#33774)
# ---------------------------------------------------------------------------




def test_dir_child_completion_unblocks_deferred_scratch_parent(kanban_home, tmp_path):
    """A non-scratch ('dir') child completing must still sweep its scratch parent.

    Regression for the gap where ``_cleanup_workspace`` returned early for a
    non-scratch task and never ran the parent sweep — leaking the parent's
    deferred scratch dir forever.
    """
    child_dir = tmp_path / "persistent-child"
    child_dir.mkdir()
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="scratch parent")
        child = kb.create_task(
            conn, title="dir child", workspace_kind="dir",
            workspace_path=str(child_dir),
        )
        kb.link_tasks(conn, parent, child)
        p_task = kb.get_task(conn, parent)
        parent_ws = kb.resolve_workspace(p_task)
        kb.set_workspace_path(conn, parent, parent_ws)

        kb.complete_task(conn, parent, result="handoff")
        assert parent_ws.exists(), "deferred while dir child active"

        kb.complete_task(conn, child, result="built")

    assert not parent_ws.exists(), (
        "A 'dir' child completing must trigger the parent scratch sweep"
    )
    assert child_dir.exists(), "Non-scratch 'dir' child workspace is never deleted"




def test_is_managed_scratch_path_rejects_kanban_metadata_subtrees(kanban_home):
    """Hermes' own DB/metadata/log subtrees under ``<kanban_home>/kanban`` are NOT managed.

    Regression guard for the Copilot finding on #28819: a scratch task whose
    ``workspace_path`` was mis-set to the kanban home, the logs dir, or a
    board's metadata dir (i.e. the board root itself, not its ``workspaces/``
    child) must be refused. Without this, the containment check would happily
    ``shutil.rmtree`` Hermes' DB/metadata/logs on task completion.
    """
    kanban_root = kanban_home / "kanban"
    kanban_root.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(kanban_root)

    logs_dir = kanban_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(logs_dir)

    board_root = kanban_root / "boards" / "my-board"
    board_root.mkdir(parents=True, exist_ok=True)
    # The board root itself is NOT a managed scratch dir — only the
    # ``workspaces/`` child (and its descendants) are.
    assert not kb._is_managed_scratch_path(board_root)

    # Sibling subtrees of ``workspaces/`` under a board (e.g. its kanban.db
    # or board.json living next to ``workspaces/``) are also not managed.
    board_logs = board_root / "logs"
    board_logs.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(board_logs)

    # Now create the board's workspaces dir and a task scratch dir under it —
    # the latter is the only thing the guard should allow.
    board_workspaces = board_root / "workspaces"
    board_workspaces.mkdir(parents=True, exist_ok=True)
    # The workspaces root itself is also NOT managed — deleting it would
    # wipe every task's scratch dir at once.
    assert not kb._is_managed_scratch_path(board_workspaces)
    task_dir = board_workspaces / "task-42"
    task_dir.mkdir(parents=True, exist_ok=True)
    assert kb._is_managed_scratch_path(task_dir)


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Originating session id (ACP propagation)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Shared-board path resolution (issue #19348)
#
# The kanban board is a cross-profile coordination primitive: a worker
# spawned with `hermes -p <profile>` must read/write the same kanban.db
# as the dispatcher that claimed the task. These tests exercise the
# path-resolution layer directly and would have caught the regression
# where `kanban_db_path()` resolved to the active profile's HERMES_HOME.
# ---------------------------------------------------------------------------

class TestSharedBoardPaths:
    """`kanban_home`/`kanban_db_path`/`workspaces_root`/`worker_log_path`
    must anchor at the **shared root**, not the active profile's HERMES_HOME."""

    def _set_home(self, monkeypatch, tmp_path, hermes_home):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)


    def test_profile_worker_resolves_to_shared_root(
        self, tmp_path, monkeypatch
    ):
        # Reproduces the bug: dispatcher uses ~/.hermes/kanban.db,
        # worker spawned with -p <profile> previously resolved to
        # ~/.hermes/profiles/<profile>/kanban.db. After the fix both
        # converge on ~/.hermes/kanban.db.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile_home)

        # All four resolvers must anchor at the shared root, not the
        # profile-local HERMES_HOME.
        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_0d214f19")
            == default_home / "kanban" / "logs" / "t_0d214f19.log"
        )

        # Sanity: the profile-local path that used to be returned is
        # explicitly NOT what we resolve to anymore.
        assert kb.kanban_db_path() != profile_home / "kanban.db"






    def test_dispatcher_and_worker_share_a_real_database(
        self, tmp_path, monkeypatch
    ):
        # Belt-and-suspenders: round-trip a task across the two
        # HERMES_HOME perspectives via a real SQLite file. Without the
        # fix the worker would open a different file and see no rows.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)

        # Dispatcher creates the board and a task.
        self._set_home(monkeypatch, tmp_path, default_home)
        kb.init_db()
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="cross-profile")

        # Worker switches to the profile HERMES_HOME and reads.
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        with kb.connect() as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "cross-profile"




    def test_dispatcher_spawn_injects_kanban_paths_without_stale_session(
        self, tmp_path, monkeypatch
    ):
        # The dispatcher must pin board paths while stripping any unrelated
        # HERMES_SESSION_* identity inherited from the long-lived gateway.
        # The one exception is HERMES_SESSION_SOURCE, which the dispatcher
        # re-sets to its own `kanban` tag AFTER the strip — a value it owns,
        # never one inherited from whatever the gateway last routed.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        shared_gh_config = tmp_path / "broker" / "gh"
        shared_gh_config.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, default_home)
        monkeypatch.delenv("GH_CONFIG_DIR", raising=False)

        from gateway import session_context as sc

        # A dispatcher can launch before the gateway binds its first session.
        monkeypatch.setattr(sc, "_session_context_engaged", False)
        sc.reset_session_vars()
        for key in sc._VAR_MAP:
            monkeypatch.setenv(key, "stale-routing-value")

        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs.get("env", {})
                self.pid = 4242

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        task = kb.Task(
            id="t_dispatch_env",
            title="x",
            body=None,
            assignee="coder",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "ws"),
            claim_lock=None,
            claim_expires=None,
            tenant=None,
            branch_name="wt/t_dispatch_env",
        )
        kb._default_spawn(task, str(tmp_path / "ws"))

        env = captured["env"]
        assert env["HERMES_KANBAN_DB"] == str(default_home / "kanban.db")
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
            default_home / "kanban" / "workspaces"
        )
        assert env["HERMES_KANBAN_TASK"] == "t_dispatch_env"
        assert env["HERMES_KANBAN_BRANCH"] == "wt/t_dispatch_env"
        for key in sc._VAR_MAP:
            if key == "HERMES_SESSION_SOURCE":
                # Re-set by the dispatcher, so what matters is that it carries
                # the worker's own tag rather than the inherited routing value.
                assert env[key] == "kanban"
                continue
            assert key not in env
        # Operator ~/.config/gh is not discovered or injected implicitly.
        assert "GH_CONFIG_DIR" not in env

        # A caller-provided least-privilege broker/delegation config remains
        # authoritative; the dispatcher neither replaces nor copies it.
        monkeypatch.setenv("GH_CONFIG_DIR", str(shared_gh_config))
        kb._default_spawn(task, str(tmp_path / "ws"))
        assert captured["env"]["GH_CONFIG_DIR"] == str(shared_gh_config)


# ---------------------------------------------------------------------------
# latest_summary / latest_summaries — surface task_runs.summary handoffs
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# NFS / network-filesystem fallback (see hermes_state.apply_wal_with_fallback)
# ---------------------------------------------------------------------------

def test_connect_falls_back_to_delete_on_locking_protocol(tmp_path, monkeypatch, caplog):
    """kanban_db.connect() must handle ``locking protocol`` on NFS/SMB.

    Without this fallback, the gateway's kanban dispatcher crashes every
    60s and the kanban migration (``consecutive_failures`` ADD COLUMN) is
    retried forever — which is what the real-world user report shows
    (see hermes-agent issue #22032).

    NOTE: We do NOT use the ``kanban_home`` fixture here because that
    fixture pre-initializes the DB via ``kb.init_db()`` — putting the
    file in WAL on disk. The Bug D safety guard now refuses to downgrade
    to DELETE when the on-disk header is already WAL, so testing the
    NFS-fallback path requires a truly-fresh DB file (NFS scenario in
    production: first connection of the first process ever to touch the
    file, where downgrading is safe because nobody else has WAL state
    yet).
    """
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # These tests exercise the WAL-attempt path; assume a fixed SQLite so the
    # WAL-reset vulnerability gate doesn't short-circuit before the pragma.
    import hermes_state as _hermes_state
    monkeypatch.setattr(
        _hermes_state, "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )
    _hermes_state._wal_fallback_warned_paths.clear()

    # Clear module cache so a fresh connect() is attempted
    kb._INITIALIZED_PATHS.clear()
    hermes_state._wal_fallback_warned_paths.clear()

    real_connect = _sqlite3.connect

    class _WalBlockingConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                raise _sqlite3.OperationalError("locking protocol")
            return super().execute(sql, *args, **kwargs)

    def wal_blocking_connect(*args, **kwargs):
        # connect_tracked passes a tracking-augmented factory; drop it and
        # substitute the double, which connect_tracked re-applies to the
        # returned instance.
        kwargs.pop("factory", None)
        return real_connect(
            *args, factory=_WalBlockingConnection, **kwargs
        )

    with _patch("hermes_cli.kanban_db.sqlite3.connect", side_effect=wal_blocking_connect):
        with caplog.at_level("ERROR", logger="hermes_state"):
            conn = kb.connect()

    # One fallback error, naming kanban.db
    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "kanban.db" in r.getMessage()
    ]
    assert len(errors) >= 1, (
        f"Expected a kanban.db ERROR, got: {[r.getMessage() for r in caplog.records]}"
    )

    # DB still usable end-to-end — create + list a task
    t = kb.create_task(conn, title="post-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()


def test_connect_works_when_wal_is_silently_refused(tmp_path, monkeypatch, caplog):
    """kanban_db.connect() must stay usable when WAL silently no-ops to DELETE."""
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    hermes_state._wal_fallback_warned_paths.clear()
    # Assume a fixed SQLite so the WAL-reset gate doesn't short-circuit.
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )

    real_connect = _sqlite3.connect

    class _WalSilentNoOpConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                return super().execute("PRAGMA journal_mode=delete", *args, **kwargs)
            return super().execute(sql, *args, **kwargs)

    def wal_silent_noop_connect(*args, **kwargs):
        kwargs.pop("factory", None)
        return real_connect(
            *args, factory=_WalSilentNoOpConnection, **kwargs
        )

    with _patch(
        "hermes_cli.kanban_db.sqlite3.connect",
        side_effect=wal_silent_noop_connect,
    ):
        with caplog.at_level("ERROR", logger="hermes_state"):
            conn = kb.connect()

    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    t = kb.create_task(conn, title="post-silent-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()

    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "kanban.db" in r.getMessage()
    ]
    assert len(errors) >= 1, (
        f"Expected a kanban.db ERROR, got: {[r.getMessage() for r in caplog.records]}"
    )


def test_unlink_tasks_triggers_recompute_ready(kanban_home):
    """Regression test for issue #22459.

    Removing a dependency via unlink_tasks must immediately promote the child
    to ready when all remaining parents are done — same contract as
    complete_task and unblock_task.

    Before the fix, child stayed 'todo' indefinitely after unlink; only the
    next dispatcher tick or a manual 'hermes kanban recompute' would promote it.
    """
    with kb.connect() as conn:
        # A is done.
        a = kb.create_task(conn, title="parent-done")
        kb.complete_task(conn, a)

        # C is running (not done) — blocks child B.
        c = kb.create_task(conn, title="parent-running")
        kb.claim_task(conn, c, claimer="worker:1")

        # B depends on both A (done) and C (running) → stays todo.
        b = kb.create_task(conn, title="child", parents=[a, c])
        assert kb.get_task(conn, b).status == "todo"

        # Remove the blocking dependency C → B.
        removed = kb.unlink_tasks(conn, c, b)
        assert removed is True

        # B's only remaining parent is A (done) → must be ready immediately.
        assert kb.get_task(conn, b).status == "ready", (
            "child should promote to ready immediately after unlink_tasks "
            "removes its last blocking dependency"
        )



# ---------------------------------------------------------------------------
# _add_column_if_missing / _migrate_add_optional_columns idempotency (#21708)
# ---------------------------------------------------------------------------

def test_add_column_if_missing_is_idempotent_on_race(kanban_home):
    """``_add_column_if_missing`` must swallow 'duplicate column name' errors.

    Regression for #21708: the kanban dispatcher opens the DB twice per tick
    (once via _tick_once_for_board, once via init_db's discard-and-reconnect
    path).  A second concurrent connection runs _migrate_add_optional_columns
    before the first one commits, so ALTER TABLE raises OperationalError with
    'duplicate column name: consecutive_failures'.  Without the idempotency
    guard that crashes the dispatcher on the first tick after every restart.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
    )

    # First call adds the column — returns True.
    added = kb._add_column_if_missing(conn, "tasks", "extra_col", "extra_col TEXT")
    assert added is True
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "extra_col" in cols

    # Second call on same connection — column already exists — must return
    # False without raising, simulating the race the dispatcher hits.
    added_again = kb._add_column_if_missing(
        conn, "tasks", "extra_col", "extra_col TEXT"
    )
    assert added_again is False

    conn.close()


def test_migrate_add_optional_columns_tolerates_concurrent_migration(kanban_home):
    """Full _migrate_add_optional_columns must not raise when columns already
    exist (issue #21708 race window — two connections migrate concurrently)."""
    import sqlite3

    # Schema already in fully-migrated state (all optional columns present).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            branch_name TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            workflow_template_id TEXT,
            current_step_key TEXT,
            skills TEXT,
            max_retries INTEGER,
            session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL DEFAULT '',
            run_id     INTEGER,
            kind       TEXT NOT NULL DEFAULT '',
            payload    TEXT,
            created_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Running migration on an already-migrated schema must not raise.
    kb._migrate_add_optional_columns(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Dispatcher spawn invocation — _resolve_hermes_argv()
#
# Workers spawned by the dispatcher must use a `hermes` invocation that does
# not depend on PATH being set up correctly. cron jobs, systemd User= services,
# launchd jobs, and other detached processes routinely run with a stripped
# $PATH that doesn't include the venv's bin/, so a bare `["hermes", ...]`
# spawn fails with FileNotFoundError and the task gets stuck. The resolver
# prefers the PATH shim (familiar `ps` output) but falls back to the module
# form so the spawn keeps working when PATH is missing the shim.
# ---------------------------------------------------------------------------


def test_resolve_hermes_argv_prefers_path_shim(monkeypatch):
    """When `hermes` is on PATH, use the shim — preserves familiar ps output."""
    import shutil
    import hermes_cli.kanban_db as kb

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/hermes")
    argv = kb._resolve_hermes_argv()
    assert argv == ["/usr/local/bin/hermes"]


def test_resolve_hermes_argv_absolutizes_relative_exe_shim(monkeypatch, tmp_path):
    """A relative executable override must not remain workspace-cwd-dependent."""
    import hermes_cli.kanban_db as kb

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_BIN", ".\\hermes.exe")
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)

    assert kb._resolve_hermes_argv() == [os.path.abspath(".\\hermes.exe")]


def test_resolve_hermes_argv_avoids_implicit_windows_batch_shim(monkeypatch, tmp_path):
    """Implicit .cmd/.bat shims use the module fallback, not batch argv[0]."""
    import sys
    import hermes_cli.kanban_db as kb

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "hermes.CMD").write_text("@echo off\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("PATHEXT", ".CMD")
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)

    assert kb._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_honors_hermes_bin_path_override(monkeypatch, tmp_path):
    """An explicit path-like HERMES_BIN lets service managers pin the executable."""
    import shutil
    import hermes_cli.kanban_db as kb

    shim = tmp_path / "bin" / "hermes"
    shim.parent.mkdir()
    shim.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_BIN", str(shim))
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert kb._resolve_hermes_argv() == [str(shim)]


def test_resolve_hermes_argv_hermes_bin_bare_name_uses_path(monkeypatch, tmp_path):
    """Bare HERMES_BIN values keep PATH semantics instead of cwd shadowing."""
    import stat
    import hermes_cli.kanban_db as kb

    cwd_hermes = tmp_path / "hermes"
    cwd_hermes.write_text("wrong\n", encoding="utf-8")
    cwd_hermes.chmod(cwd_hermes.stat().st_mode | stat.S_IXUSR)
    path_hermes = tmp_path / "bin" / "hermes"
    path_hermes.parent.mkdir()
    path_hermes.write_text("right\n", encoding="utf-8")
    path_hermes.chmod(path_hermes.stat().st_mode | stat.S_IXUSR)
    real_access = os.access
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: (
            True if os.path.abspath(path) == str(path_hermes) and mode == os.X_OK else real_access(path, mode)
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(path_hermes.parent))
    monkeypatch.setenv("HERMES_BIN", "hermes")

    assert kb._resolve_hermes_argv() == [str(path_hermes)]


def test_resolve_hermes_argv_hermes_bin_bare_name_ignores_cwd(monkeypatch, tmp_path):
    """Bare HERMES_BIN does not accept current-directory shadow executables."""
    import sys
    import hermes_cli.kanban_db as kb

    (tmp_path / "hermes.exe").write_text("wrong\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_BIN", "hermes")
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)

    assert kb._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_hermes_bin_bare_cmd_uses_module_fallback(monkeypatch, tmp_path):
    """A PATH-resolved HERMES_BIN batch shim is not used as worker argv[0]."""
    import sys
    import hermes_cli.kanban_db as kb

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "hermes.CMD").write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("PATHEXT", ".CMD")
    monkeypatch.setenv("HERMES_BIN", "hermes")
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)

    assert kb._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_hermes_bin_unresolved_bare_name_falls_back(monkeypatch):
    """Unresolved HERMES_BIN command names do not delegate cwd search to Popen."""
    import sys
    import hermes_cli.kanban_db as kb

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HERMES_BIN", "hermes")

    assert kb._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_falls_back_to_module_form_when_no_path_shim(monkeypatch):
    """When the shim is not on PATH, fall back to `python -m hermes_cli.main`.

    Pins the correct module name (NOT `hermes` — there is no top-level
    `hermes` package). Regression for #23198: the original PR shipped
    `python -m hermes` which fails with `No module named hermes` on every
    invocation.
    """
    import shutil
    import sys
    import hermes_cli.kanban_db as kb

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    argv = kb._resolve_hermes_argv()
    assert argv == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_module_actually_runs():
    """The fallback module name must be importable + runnable.

    A unit test that pins the literal string is necessary but not
    sufficient — if `hermes_cli.main` ever loses `if __name__ == "__main__"`
    handling or its argparse setup, `python -m hermes_cli.main --version`
    would fail and so would every dispatcher spawn that hits the fallback.
    Run it as a real subprocess to catch that regression.
    """
    import subprocess
    import hermes_cli.kanban_db as kb
    import shutil
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_BIN", None)
        with mock.patch.object(shutil, "which", return_value=None):
            argv = kb._resolve_hermes_argv()
    r = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (
        f"`{' '.join(argv)} --version` failed (rc={r.returncode}); "
        f"stderr={r.stderr[:200]!r}"
    )
    assert "Hermes Agent" in r.stdout, f"unexpected output: {r.stdout[:200]!r}"


# ---------------------------------------------------------------------------
# task_age — guard against corrupt timestamp values
#
# The Task dataclass declares ``created_at: int`` but rows come from sqlite
# without coercion at the boundary. A row that ever held a non-int (e.g. an
# unsubstituted ``'%s'`` from a logged format string, ``None``, an arbitrary
# string, or a float-as-string) used to crash ``task_age`` with ``ValueError``
# and turn ``GET /api/plugins/kanban/board`` into a 500 because the dashboard
# calls ``task_age`` unguarded for every task in the response.
#
# After the fix, ``_safe_int`` returns ``None`` on bad input and ``task_age``
# degrades gracefully (per-field ``None`` rather than a hard crash).
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> "kb.Task":
    """Minimal Task with all required fields filled in. Override anything."""
    defaults = dict(
        id="t_age",
        title="x",
        body=None,
        assignee=None,
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    defaults.update(overrides)
    return kb.Task(**defaults)












# ---------------------------------------------------------------------------
# Board-level default_workdir
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# dispatch_once — max_in_progress
# ---------------------------------------------------------------------------


# Review column dispatch
# ---------------------------------------------------------------------------


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    """Test helper: set a task's status directly."""
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))








def test_dispatch_review_dry_run(kanban_home, all_assignees_spawnable):
    """dispatch_once dry-run sees review tasks and reports them as spawned."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="alice")
        _set_task_status(conn, t, "review")
        res = kb.dispatch_once(conn, dry_run=True)
    assert len(res.spawned) == 1
    assert res.spawned[0][0] == t
    # Dry run must NOT mutate status.
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "review"


def test_dispatch_review_spawns_with_correct_skills(
    kanban_home, all_assignees_spawnable,
):
    """Review tasks get sdlc-review skill set before spawning."""
    spawned_tasks = []

    def capture_spawn(task, workspace, board=None):
        spawned_tasks.append(task)
        return 42  # fake PID

    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="alice")
        _set_task_status(conn, t, "review")
        res = kb.dispatch_once(conn, spawn_fn=capture_spawn)
    assert len(res.spawned) == 1
    assert len(spawned_tasks) == 1
    assert spawned_tasks[0].skills == ["sdlc-review"]


def _mk_profile_skill(home, assignee: str, skill: str) -> None:
    d = home / "profiles" / assignee / "skills" / "devops" / skill
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {skill}\n---\nbody\n", encoding="utf-8")


def test_dispatch_review_merges_card_skills_with_forced_skill(
    kanban_home, all_assignees_spawnable,
):
    """Review spawn keeps the card's own skills alongside sdlc-review."""
    _mk_profile_skill(kanban_home, "alice", "sdlc-review")
    _mk_profile_skill(kanban_home, "alice", "test-driven-development")
    spawned_tasks = []

    def capture_spawn(task, workspace, board=None):
        spawned_tasks.append(task)
        return 42

    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="review me", assignee="alice",
            skills=["test-driven-development"],
        )
        _set_task_status(conn, t, "review")
        kb.dispatch_once(conn, spawn_fn=capture_spawn)
    assert spawned_tasks[0].skills == ["sdlc-review", "test-driven-development"]


def test_dispatch_review_degrades_when_forced_skill_unavailable(
    kanban_home, all_assignees_spawnable,
):
    """Profile without sdlc-review: spawn drops it, keeps the card's skills
    and leaves a durable degradation comment — instead of passing an
    all-missing ``--skills`` list that hard-crashes the worker at startup
    (CLI raises on all-missing) and burns the failure budget (K-3)."""
    _mk_profile_skill(kanban_home, "alice", "test-driven-development")
    spawned_tasks = []

    def capture_spawn(task, workspace, board=None):
        spawned_tasks.append(task)
        return 42

    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="review me", assignee="alice",
            skills=["test-driven-development"],
        )
        _set_task_status(conn, t, "review")
        kb.dispatch_once(conn, spawn_fn=capture_spawn)
        comments = kb.list_comments(conn, t)
    assert spawned_tasks[0].skills == ["test-driven-development"]
    assert any("skill-degradation" in c.body for c in comments)


def test_dispatch_review_spawns_bare_when_no_skills_available(
    kanban_home, all_assignees_spawnable,
):
    """Even with nothing loadable the review spawn must not crash-loop:
    spawn with no forced skills plus a degradation comment."""
    (kanban_home / "profiles" / "alice" / "skills").mkdir(parents=True)
    spawned_tasks = []

    def capture_spawn(task, workspace, board=None):
        spawned_tasks.append(task)
        return 42

    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="alice")
        _set_task_status(conn, t, "review")
        kb.dispatch_once(conn, spawn_fn=capture_spawn)
        comments = kb.list_comments(conn, t)
    assert spawned_tasks[0].skills == []
    assert any("skill-degradation" in c.body for c in comments)


def test_review_skill_degradation_comment_is_deduplicated(
    kanban_home, all_assignees_spawnable,
):
    """Repeated degraded spawns of the SAME card must not spam identical
    skill-degradation comments (self-audit finding 16.07.: a crash-looping
    review card would otherwise add a comment per spawn)."""
    _mk_profile_skill(kanban_home, "alice", "test-driven-development")
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="review me", assignee="alice",
            skills=["test-driven-development"],
        )
        task = kb.get_task(conn, t)
        first = kb._resolve_review_skills(conn, task)
        second = kb._resolve_review_skills(conn, task)
        comments = [
            c for c in kb.list_comments(conn, t)
            if "skill-degradation" in c.body
        ]
    assert first == second == ["test-driven-development"]
    assert len(comments) == 1


def test_dispatch_review_skips_unassigned(kanban_home):
    """Unassigned review tasks go to skipped_unassigned, not spawned."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review floater")
        _set_task_status(conn, t, "review")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_unassigned
    assert not res.spawned


def test_dispatch_review_counts_toward_max_spawn(
    kanban_home, all_assignees_spawnable,
):
    """Review spawns count against max_spawn alongside ready tasks."""
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        # Create 2 ready tasks + 1 review task, max_spawn=2
        t1 = kb.create_task(conn, title="ready 1", assignee="alice")
        t2 = kb.create_task(conn, title="ready 2", assignee="bob")
        t3 = kb.create_task(conn, title="review", assignee="alice")
        _set_task_status(conn, t3, "review")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)
    # Only 2 should spawn (ready tasks get priority in the loop)
    assert len(res.spawned) == 2
    assert len(spawns) == 2


def test_dispatch_review_spawns_when_ready_empty(
    kanban_home, all_assignees_spawnable,
):
    """When only review tasks exist, they still get dispatched."""
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="alice")
        _set_task_status(conn, t, "review")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    assert len(res.spawned) == 1
    assert spawns[0] == t


def test_has_spawnable_review_true(kanban_home):
    """has_spawnable_review returns True when review tasks exist with real profiles."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review me", assignee="default")
        _set_task_status(conn, t, "review")
        # default profile should exist in the test env
        assert kb.has_spawnable_review(conn) is True


def test_has_spawnable_review_false_on_empty(kanban_home):
    """has_spawnable_review returns False when no review tasks exist."""
    with kb.connect() as conn:
        assert kb.has_spawnable_review(conn) is False


def test_has_spawnable_review_false_when_only_terminal_lanes(
    kanban_home, monkeypatch,
):
    """has_spawnable_review returns False when review tasks are terminal lanes."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review", assignee="orion-cc")
        _set_task_status(conn, t, "review")
        assert kb.has_spawnable_review(conn) is False


def test_dispatch_review_skips_nonspawnable(kanban_home, monkeypatch):
    """Review tasks with non-existent profiles go to skipped_nonspawnable."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="review", assignee="orion-cc")
        _set_task_status(conn, t, "review")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_nonspawnable
    assert not res.spawned


def test_review_status_in_valid_statuses():
    """'review' is a valid task status."""
    assert "review" in kb.VALID_STATUSES


def test_dispatch_review_does_not_claim_ready_tasks(
    kanban_home, all_assignees_spawnable,
):
    """Review dispatch uses claim_review_task, which only claims review tasks."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="ready task", assignee="alice")
        # claim_review_task should NOT claim a ready task
        claimed = kb.claim_review_task(conn, t)
    assert claimed is None


# ---------------------------------------------------------------------------
# Stale detection — detect_stale_running
# ---------------------------------------------------------------------------

def test_detect_stale_returns_running_task_with_no_heartbeat(kanban_home, monkeypatch):
    """A task running > timeout with zero heartbeats gets reclaimed as stale."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="stale-no-hb", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        # Rewind started_at so the task appears to have been running for 5 hours.
        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )
        # No heartbeat set — last_heartbeat_at stays NULL.

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        killed = []
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: killed.append(s),
        )
        assert t in stale, "Task with no heartbeat for >4h should be reclaimed"
        task = kb.get_task(conn, t)
        assert task.status == "ready"
def test_detect_stale_returns_task_with_stale_heartbeat(kanban_home, monkeypatch):
    """A task running > timeout with a heartbeat older than the SAME timeout
    window gets reclaimed.

    Fund 1/7 repair: staleness now uses one unified window
    (stale_timeout_seconds) for liveness, not the old separate hardcoded
    _STALE_HEARTBEAT_GAP_SECONDS (1h) gate that used to fire independently
    of the configured timeout. A heartbeat merely older than 1h but still
    within the 4h timeout window is NOT stale (see
    test_detect_stale_skips_task_with_recent_heartbeat's sibling below) --
    only a heartbeat older than the timeout itself is.
    """
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="stale-hb", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        heartbeat_4_5h_ago = int(time.time()) - int(4.5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = ? "
                "WHERE id = ?",
                (five_hours_ago, heartbeat_4_5h_ago, t),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert t in stale, (
            "Task with heartbeat older than the 4h timeout window should be stale"
        )
        assert kb.get_task(conn, t).status == "ready"


def test_detect_stale_skips_heartbeat_older_than_gap_but_within_timeout(kanban_home, monkeypatch):
    """A heartbeat 2h old is NOT stale when stale_timeout_seconds is 4h --
    proves the old hardcoded 1h _STALE_HEARTBEAT_GAP_SECONDS gate is gone
    and staleness uses ONE window (stale_timeout_seconds) throughout."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="hb-2h-old", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        heartbeat_2h_ago = int(time.time()) - (2 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = ? "
                "WHERE id = ?",
                (five_hours_ago, heartbeat_2h_ago, t),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: True)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert stale == [], (
            "heartbeat 2h old is within the 4h timeout window -- not stale"
        )
        assert kb.get_task(conn, t).status == "running"


def test_detect_stale_skips_task_with_recent_heartbeat(kanban_home, monkeypatch):
    """A task running > timeout but with a recent heartbeat is NOT reclaimed.

    Fund 2 repair: this must prove the real "recent heartbeat -> skip" path
    -- i.e. the liveness check short-circuits BEFORE any termination attempt
    is made -- not just that the end result happens to be status='running'.
    Before the fix, a recent last_heartbeat_at alone did not stop
    detect_stale_running from treating the task as stale (staleness was
    gated on last_semantic_progress_at only), so this test only passed
    because it fell into the "worker survived termination -> defer" path,
    which calls _terminate_reclaimed_worker and (with _pid_alive patched to
    True and a no-op signal_fn) burns ~5s retrying SIGTERM before giving up.
    """
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="alive-hb", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        heartbeat_now = int(time.time())  # heartbeat just happened
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = ? "
                "WHERE id = ?",
                (five_hours_ago, heartbeat_now, t),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: True)
        terminate_calls = []
        real_terminate = _kb._terminate_reclaimed_worker
        monkeypatch.setattr(
            _kb, "_terminate_reclaimed_worker",
            lambda *a, **k: terminate_calls.append((a, k)) or real_terminate(*a, **k),
        )

        started = time.monotonic()
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        elapsed = time.monotonic() - started

        assert stale == [], "Task with recent heartbeat should not be reclaimed"
        assert kb.get_task(conn, t).status == "running"
        assert terminate_calls == [], (
            "recent last_heartbeat_at must short-circuit the liveness check "
            "before any termination is attempted"
        )
        assert elapsed < 1.0, (
            f"took {elapsed:.2f}s -- recent-heartbeat skip must be immediate, "
            "not fall through to the 5s SIGTERM-retry/defer path"
        )


def test_detect_stale_skips_task_with_only_auto_activity(kanban_home, monkeypatch):
    """A task running > timeout with ONLY automatic activity heartbeats
    (last_heartbeat_at/last_activity_at fresh, last_semantic_progress_at
    NULL) is NOT reclaimed. This is the exact scenario that made d7d5b3dae's
    semantic-progress-only staleness gate dangerous: almost no worker calls
    the explicit kanban_heartbeat tool, so last_semantic_progress_at stays
    NULL for a healthy multi-hour worker and it would be killed the instant
    dispatch_stale_timeout_seconds elapsed. Fund 1 repair."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="auto-activity-only", assignee="worker")
        kb.claim_task(conn, t)
        run_id = kb.latest_run(conn, t).id

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (five_hours_ago, run_id),
            )

        # Simulate the runtime's auto-heartbeat: semantic=False, sent on
        # essentially every tool call. last_semantic_progress_at stays NULL.
        assert kb.heartbeat_worker(conn, t, expected_run_id=run_id, semantic=False)
        row = conn.execute(
            "SELECT last_heartbeat_at, last_activity_at, last_semantic_progress_at "
            "FROM tasks WHERE id=?", (t,),
        ).fetchone()
        assert row["last_heartbeat_at"] is not None
        assert row["last_activity_at"] is not None
        assert row["last_semantic_progress_at"] is None

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: True)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert stale == [], (
            "worker with only auto-activity heartbeats (no semantic progress) "
            "must NOT be reclaimed -- it's still alive"
        )
        assert kb.get_task(conn, t).status == "running"


def test_detect_stale_reclaims_task_with_no_activity_at_all(kanban_home, monkeypatch):
    """Gegenprobe: a task with NO fresh activity of any kind (no heartbeat,
    no auto-activity, no semantic progress) is still reclaimed as stale.
    Fund 1 repair -- confirms the liveness relaxation didn't also disable
    the original no-heartbeat-ever reclaim path."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="truly-idle", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )
        # No heartbeat, no activity, no semantic progress set at all.

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert stale == [t], "Task with zero liveness signals for >4h should be reclaimed"
        assert kb.get_task(conn, t).status == "ready"


def test_detect_stale_skips_recently_started_task(kanban_home, monkeypatch):
    """A task started < timeout ago is NOT reclaimed even with no heartbeat."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="fresh", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        # Started only 1 hour ago — well within the 4h threshold.
        one_hour_ago = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (one_hour_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (one_hour_ago, t),
            )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: True)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert stale == [], "Task started <4h ago should not be reclaimed"
        assert kb.get_task(conn, t).status == "running"


def test_detect_stale_skips_when_timeout_zero(kanban_home, monkeypatch):
    """stale_timeout_seconds=0 disables stale detection entirely."""

    with kb.connect() as conn:
        t = kb.create_task(conn, title="disabled", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )

        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=0, signal_fn=lambda p, s: None,
        )
        assert stale == [], "timeout=0 should disable stale detection"
        assert kb.get_task(conn, t).status == "running"


def test_detect_stale_skips_blocked_tasks(kanban_home, monkeypatch):
    """Blocked tasks are NOT reclaimed by stale detection."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="blocked-task", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )
        # Block the task explicitly.
        kb.block_task(conn, t, reason="human requested block")

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert stale == [], "Blocked task should not be reclaimed by stale detection"
        assert kb.get_task(conn, t).status == "blocked"


def test_detect_stale_does_not_tick_failure_counter(kanban_home, monkeypatch):
    """Stale reclaim must NOT tick consecutive_failures.

    Stale detection is dispatcher-side absence-of-heartbeat detection,
    not a worker failure. Counting it as a failure would let two
    legitimately-long-running tasks (>4h without explicit heartbeat) trip
    the circuit breaker and auto-block at the default failure_limit=2,
    even though no worker actually failed. The 'stale' event in
    task_events is the right audit surface; the consecutive_failures
    counter is reserved for spawn_failed / timed_out / crashed.
    """
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="stale-no-counter-tick", assignee="worker")
        kb.claim_task(conn, t)
        kb._set_worker_pid(conn, t, os.getpid())

        five_hours_ago = int(time.time()) - (5 * 3600)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (five_hours_ago, t)
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (five_hours_ago, t),
            )
            # Counter starts at 0; assert that's our baseline.
            row = conn.execute(
                "SELECT consecutive_failures FROM tasks WHERE id = ?", (t,)
            ).fetchone()
            assert row["consecutive_failures"] in (0, None)

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        stale = kb.detect_stale_running(
            conn, stale_timeout_seconds=14400, signal_fn=lambda p, s: None,
        )
        assert t in stale, "Task should be reclaimed by stale detection"

        # Critical assertion: the failure counter MUST NOT have ticked.
        # Stale reclaim resets to ready for re-dispatch without penalty.
        row = conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (t,)
        ).fetchone()
        assert row["consecutive_failures"] in (0, None), (
            f"Stale reclaim ticked consecutive_failures to "
            f"{row['consecutive_failures']!r}; should remain 0/NULL."
        )

        # And the audit trail still records the stale event so operators
        # can see what happened.
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t,),
        ).fetchall()
        kinds = [e["kind"] for e in events]
        assert "stale" in kinds, (
            f"Expected 'stale' event in task_events; got {kinds!r}"
        )


# ---------------------------------------------------------------------------
# Corruption guard (issue #30687)
# ---------------------------------------------------------------------------

@pytest.mark.requires_wal  # upstream-Marker (Merge 30.07.): übersprungen,
# wo Hermes wegen des SQLite-WAL-Reset-Bugs auf journal_mode=DELETE ausweicht.
def test_file_length_invariant_is_skipped_in_wal_mode(tmp_path, monkeypatch):
    """Main-file length checks are invalid while committed frames live in WAL."""
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)

    with kb.connect(db_path=db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        calls = []

        def fake_getsize(_path):
            calls.append(_path)
            return 0

        monkeypatch.setattr(kb.os.path, "getsize", fake_getsize)
        kb._check_file_length_invariant(conn)

    assert calls == []


def test_file_length_invariant_still_protects_rollback_journal(tmp_path, monkeypatch):
    """The torn-extend guard still applies when the main DB is authoritative."""
    db_path = tmp_path / "delete-mode.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO t(value) VALUES ('x')")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        monkeypatch.setattr(kb.os.path, "getsize", lambda _path: 0)
        with pytest.raises(sqlite3.DatabaseError, match="torn-extend detected"):
            kb._check_file_length_invariant(conn)
    finally:
        conn.close()


def test_file_length_invariant_uses_read_snapshot_in_rollback_mode(tmp_path, monkeypatch):
    """Rollback-mode checks hold a transaction across logical/physical reads."""
    db_path = tmp_path / "delete-mode-snapshot.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY)")
        observed = []
        real_getsize = os.path.getsize

        def checking_getsize(path):
            observed.append(conn.in_transaction)
            return real_getsize(path)

        monkeypatch.setattr(kb.os.path, "getsize", checking_getsize)
        kb._check_file_length_invariant(conn)
        assert observed == [True]
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_wal_multiprocess_writes_preserve_parallelism_without_false_corruption(tmp_path):
    """Real worker processes should not need a global write sidecar lock.

    This is intentionally a small smoke, not a benchmark.  The historical
    regression was that hot WAL writers could trip the post-commit main-file
    length invariant even when the final DB was healthy.
    """
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    workers = 6
    per_worker = 40
    procs = [
        ctx.Process(
            target=_kanban_multiprocess_writer,
            args=(str(db_path), idx, per_worker, queue),
        )
        for idx in range(workers)
    ]
    for proc in procs:
        proc.start()

    errors = [queue.get(timeout=30) for _ in procs]
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0

    errors = [err for err in errors if err is not None]
    assert errors == []

    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == workers * per_worker

    assert not db_path.with_name(db_path.name + ".lock").exists()


def _write_corrupt_db(path: Path) -> bytes:
    """Write a kanban DB with a VALID SQLite header but malformed page content.

    This is the corruption shape the integrity guard specifically targets
    (e.g. issue #29507 follow-up reports where the file's first 16 bytes
    pass the header byte check but ``PRAGMA integrity_check`` then fails
    because the internal pages are damaged). It's what main's header-only
    validator was letting through, and what this PR adds the full guard
    for.
    """
    # 100-byte SQLite header (magic + minimal valid-looking fields) so the
    # cheap header check passes, then deliberate garbage so sqlite refuses
    # to read the file past the header.
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    payload = b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64
    blob = header + payload
    path.write_bytes(blob)
    return blob




def test_repeated_corrupt_open_reuses_single_backup(tmp_path):
    """Repeated quarantines of the same corrupt bytes must not amplify disk usage.

    Regression for the gateway dispatcher's 5-min retry loop on shared kanban
    DBs across multi-profile fleets: each retry on an unchanged corrupt file
    used to create a fresh ``.corrupt.<timestamp>.bak`` until disk filled. The
    content-addressed backup name is deterministic in the DB's sha256, so
    N retries of the same bytes share one backup.
    """
    db_path = tmp_path / "kanban.db"
    original = _write_corrupt_db(db_path)

    backups: set[Path] = set()
    for _ in range(10):
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        assert excinfo.value.backup_path is not None
        backups.add(excinfo.value.backup_path)

    assert len(backups) == 1, f"expected 1 deterministic backup, got {len(backups)}"
    (backup,) = backups
    assert backup.exists()
    assert backup.read_bytes() == original

    # Mutate the corrupt bytes — fingerprint changes, separate backup preserved.
    with db_path.open("r+b") as f:
        f.seek(4096)
        f.write(b"\xAB" * 64)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo2:
        kb.connect(db_path=db_path)
    second_backup = excinfo2.value.backup_path
    assert second_backup is not None
    assert second_backup != backup
    assert second_backup.exists()


def test_corrupt_backup_rejects_source_that_changes_during_copy(tmp_path, monkeypatch):
    """Never preserve a mixed-generation main/WAL forensic set."""
    db_path = tmp_path / "kanban.db"
    _write_corrupt_db(db_path)
    real_copy = shutil.copyfileobj
    mutated = False

    def copy_then_mutate(reader, writer, *args, **kwargs):
        nonlocal mutated
        result = real_copy(reader, writer, *args, **kwargs)
        if not mutated:
            with db_path.open("ab") as source:
                source.write(b"changed-during-copy")
            mutated = True
        return result

    monkeypatch.setattr(kb.shutil, "copyfileobj", copy_then_mutate)

    assert kb._backup_corrupt_db(db_path) is None
    assert list(tmp_path.glob("kanban.db.corrupt.*")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corrupt_backup_preserves_existing_sidecar_evidence(tmp_path):
    """Same main hash with a different WAL must not overwrite prior evidence."""
    db_path = tmp_path / "kanban.db"
    _write_corrupt_db(db_path)
    wal_path = Path(str(db_path) + "-wal")
    wal_path.write_bytes(b"first-generation")

    backup = kb._backup_corrupt_db(db_path)
    assert backup is not None
    backup_wal = Path(str(backup) + "-wal")
    assert backup_wal.read_bytes() == b"first-generation"

    wal_path.write_bytes(b"second-generation")
    assert kb._backup_corrupt_db(db_path) is None
    assert backup_wal.read_bytes() == b"first-generation"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_locked_healthy_db_does_not_classify_as_corrupt(tmp_path, monkeypatch):
    """A transient lock during the probe must not produce a .corrupt backup
    and must not be reported as :class:`KanbanDbCorruptError`. Raw sqlite
    ``OperationalError`` (lock/busy) is acceptable and expected."""
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    real_connect = sqlite3.connect

    def flaky_connect(*args, **kwargs):
        # First call is the integrity probe — simulate a lock.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb.sqlite3, "connect", flaky_connect)

    with pytest.raises(sqlite3.OperationalError):
        kb.connect(db_path=db_path)

    # No .corrupt backup may be produced for a healthy-but-locked DB.
    backups = list(tmp_path.glob("*.corrupt.*"))
    assert backups == [], f"unexpected corrupt backups: {backups}"

    # And once the lock clears, normal access still works.
    monkeypatch.setattr(kb.sqlite3, "connect", real_connect)
    with kb.connect(db_path=db_path) as conn:
        kb.create_task(conn, title="still here")
        titles = [t.title for t in kb.list_tasks(conn)]
    assert "still here" in titles




# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------

def test_maybe_emit_scratch_tip_fires_once_per_install(kanban_home, caplog):
    """First scratch workspace materialization warns + emits an event.

    Subsequent scratch workspaces on the SAME install stay silent — the
    sentinel file under kanban_home() flips after the first emit.
    """
    import logging

    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="first scratch")
        t2 = kb.create_task(conn, title="second scratch")

    # Sentinel must not exist yet on a fresh install.
    assert not kb._scratch_tip_shown()

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect() as conn:
            kb._maybe_emit_scratch_tip(conn, t1, "scratch")

    # Sentinel is now set.
    assert kb._scratch_tip_shown()
    assert kb._scratch_tip_sentinel_path().exists()

    # Warning was logged exactly once.
    tip_records = [
        r for r in caplog.records
        if "scratch workspaces are ephemeral" in r.getMessage()
    ]
    assert len(tip_records) == 1, (
        f"Expected exactly one tip warning, got {len(tip_records)}: "
        f"{[r.getMessage() for r in tip_records]!r}"
    )

    # An event row was appended on the first task.
    with kb.connect() as conn:
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t1,),
        ).fetchall()
    kinds = [e["kind"] for e in events]
    assert "tip_scratch_workspace" in kinds, (
        f"Expected tip_scratch_workspace event on first scratch task; "
        f"got {kinds!r}"
    )

    # Second scratch materialization on the same install stays silent.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect() as conn:
            kb._maybe_emit_scratch_tip(conn, t2, "scratch")
    tip_records2 = [
        r for r in caplog.records
        if "scratch workspaces are ephemeral" in r.getMessage()
    ]
    assert tip_records2 == [], (
        f"Tip should not re-fire after sentinel is set; got "
        f"{[r.getMessage() for r in tip_records2]!r}"
    )
    with kb.connect() as conn:
        events2 = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t2,),
        ).fetchall()
    assert "tip_scratch_workspace" not in [e["kind"] for e in events2], (
        "Tip event should not be appended for subsequent scratch tasks."
    )




# ---------------------------------------------------------------------------
# Connection pragmas (secure_delete, cell_size_check, synchronous=FULL)
# ---------------------------------------------------------------------------


def test_connect_sets_secure_delete_on(tmp_path):
    """secure_delete=ON must be active on every new connection."""
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path=db_path) as conn:
        row = conn.execute("PRAGMA secure_delete").fetchone()
    assert row[0] == 1, f"expected secure_delete=1, got {row[0]}"





# write_txn — rollback handler must not mask the original exception
# ---------------------------------------------------------------------------


def test_write_txn_preserves_original_exception_when_rollback_fails(kanban_home):
    """When a write inside write_txn raises an OperationalError that SQLite
    has already auto-rolled-back (e.g. ``disk I/O error``,
    ``database is locked``, ``database disk image is malformed``), the
    explicit ROLLBACK in ``write_txn.__exit__`` itself raises
    ``cannot rollback - no transaction is active``. The original cause
    must NOT be masked by the secondary rollback failure — operators rely
    on the original cause to diagnose the underlying issue.
    """

    class FailingConnWrapper:
        """Delegate to a real connection, simulating an EIO during an INSERT
        that SQLite has already auto-rolled-back."""

        def __init__(self, real):
            self._real = real
            self._fail_armed = True

        def execute(self, sql, *args, **kwargs):
            if (
                self._fail_armed
                and sql.lstrip().upper().startswith("INSERT")
                and "task_events" in sql.lower()
            ):
                self._fail_armed = False  # one-shot
                # Simulate SQLite auto-rolling back the transaction by
                # issuing a real ROLLBACK now. After this, BEGIN IMMEDIATE
                # is no longer active and an explicit ROLLBACK would error.
                try:
                    self._real.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with kb.connect() as conn:
        wrapper = FailingConnWrapper(conn)
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with kb.write_txn(wrapper):
                kb._append_event(wrapper, "t_bogus", "promoted", None)

    msg = str(excinfo.value)
    assert "disk I/O error" in msg, (
        f"write_txn masked the original exception with rollback failure; "
        f"got {msg!r} (expected to contain 'disk I/O error')"
    )
    assert "cannot rollback" not in msg, (
        f"write_txn surfaced the rollback failure instead of the original "
        f"OperationalError; got {msg!r}"
    )
def test_write_txn_healthy_commit_no_exception(tmp_path):
    """Normal commit does not trigger the torn-extend check."""
    from hermes_cli.kanban_db import connect, write_txn
    db = tmp_path / "test.db"
    conn = connect(db_path=db)
    # Should not raise
    with write_txn(conn) as c:
        c.execute(
            "INSERT INTO tasks (id, title, assignee, status, priority, created_at) "
            "VALUES ('t_test01', 'test task', 'tester', 'todo', 0, 1234567890)"
        )
    row = conn.execute("SELECT title FROM tasks WHERE id='t_test01'").fetchone()
    assert row["title"] == "test task"
    conn.close()


def test_write_txn_raises_on_truncated_file(tmp_path):
    """A mocked smaller file size triggers the torn-extend check.

    The check now reads the header side via ``PRAGMA page_count`` over the
    existing connection instead of ``open()``-ing the database file (an
    open/close would cancel this process's POSIX locks). The on-disk side is
    still ``stat()``, so that is what this test fakes. The invariant only
    applies under a rollback journal — in WAL a committed page may still be
    in the -wal file, so the main file legitimately lags.
    """
    from hermes_cli.kanban_db import connect, write_txn
    db = tmp_path / "test.db"
    conn = connect(db_path=db)
    conn.execute("PRAGMA journal_mode=DELETE")
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    original_getsize = os.path.getsize

    def fake_getsize(path):
        # Return a size that implies at least 1 fewer page than header claims
        real_size = original_getsize(path)
        return max(0, real_size - page_size)

    # fork(tars): unser write_txn verpackt eine NACH dem COMMIT gerissene
    # Integritätsprüfung in PostCommitIntegrityError (RuntimeError) — die
    # Mutation ist bereits dauerhaft und darf nicht blind wiederholt werden.
    # Upstreams Test kennt diesen Wrapper nicht und erwartet nur den nackten
    # sqlite3.DatabaseError; beide Typen sind hier zulässig, die Ursache steht
    # in der verketteten Ausnahme.
    from hermes_cli.kanban_db import PostCommitIntegrityError

    with pytest.raises((sqlite3.DatabaseError, PostCommitIntegrityError)) as excinfo:
        with unittest.mock.patch(
            "hermes_cli.sqlite_safe_read.os.path.getsize", side_effect=fake_getsize
        ):
            with write_txn(conn) as c:
                c.execute(
                    "INSERT INTO tasks (id, title, assignee, status, priority, created_at) "
                    "VALUES ('t_test02', 'test task 2', 'tester', 'todo', 0, 1234567890)"
                )
    chain = str(excinfo.value) + str(excinfo.value.__cause__ or "")
    assert "torn-extend" in chain or "page count mismatch" in chain
    conn.close()


def test_write_txn_post_commit_check_fires_every_call(tmp_path):
    """The invariant check runs on every write_txn call."""
    from hermes_cli.kanban_db import connect, write_txn
    import hermes_cli.kanban_db as kanban_db_module
    db = tmp_path / "test.db"
    conn = connect(db_path=db)
    call_count = 0
    real_check = kanban_db_module._check_file_length_invariant

    def counting_check(c):
        nonlocal call_count
        call_count += 1
        real_check(c)

    with unittest.mock.patch.object(kanban_db_module, "_check_file_length_invariant", counting_check):
        for i in range(3):
            with write_txn(conn) as c:
                c.execute(
                    f"INSERT INTO tasks (id, title, assignee, status, priority, created_at) "
                    f"VALUES ('t_fire{i:02d}', 'task {i}', 'tester', 'todo', 0, 1234567890)"
                )
    assert call_count == 3
    conn.close()


def test_write_txn_post_commit_failure_is_explicit_and_not_rolled_back(tmp_path, monkeypatch):
    """Callers must know that a post-COMMIT guard failure is not retry-safe."""
    db = tmp_path / "post-commit.db"
    conn = kb.connect(db_path=db)

    def fail_guard(_conn):
        raise sqlite3.DatabaseError("synthetic invariant failure")

    monkeypatch.setattr(kb, "_check_file_length_invariant", fail_guard)
    with pytest.raises(kb.PostCommitIntegrityError, match="must not be retried"):
        with kb.write_txn(conn) as txn:
            txn.execute(
                "INSERT INTO tasks (id, title, status, priority, created_at) "
                "VALUES ('t_committed', 'durable', 'todo', 0, 1)"
            )

    assert conn.execute(
        "SELECT title FROM tasks WHERE id='t_committed'"
    ).fetchone()[0] == "durable"
    conn.close()


def test_connect_sets_default_wal_autocheckpoint_1000(tmp_path):
    """connect() keeps SQLite's default 1000-page WAL auto-checkpoint."""
    from hermes_cli.kanban_db import connect
    db = tmp_path / "test.db"
    conn = connect(db_path=db)
    val = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    assert val == 1000
    conn.close()


def test_fast_path_keeps_default_wal_autocheckpoint_1000(tmp_path):
    """A cached-path reconnect must not revert first-open settings to 100."""
    db = tmp_path / "test-fast-path.db"
    first = kb.connect(db_path=db)
    first.close()
    assert str(db.resolve()) in kb._INITIALIZED_PATHS
    second = kb.connect(db_path=db)
    try:
        assert second.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 1000
    finally:
        second.close()


def test_write_txn_check_reads_correct_header_fields(tmp_path):
    """A genuinely truncated DB is never reported as passing the invariant.

    The check no longer opens the database file to read header bytes (that
    open/close would cancel this process's POSIX advisory locks — the
    corruption route in sqlite.org/howtocorrupt.html §2.2). It asks SQLite for
    ``page_count`` instead. On a truncated file SQLite refuses that pragma, so
    the helper reports "not healthy" rather than a page-count mismatch; either
    way the file must never come back clean.
    """
    import struct
    from hermes_cli.kanban_db import connect
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    db = tmp_path / "synthetic.db"
    conn = connect(db_path=db)
    conn.execute("PRAGMA journal_mode=DELETE")
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()

    with open(db, "rb") as f:
        data = bytearray(f.read())
    real_page_count = struct.unpack(">I", data[28:32])[0]
    if real_page_count < 2:
        pytest.skip("DB too small for synthetic truncation test")
    truncated = bytes(data[: (real_page_count - 1) * page_size])
    with open(db, "wb") as f:
        f.write(truncated)

    raw_conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        assert file_length_matches_header(raw_conn) is not True
    finally:
        raw_conn.close()


# ---------------------------------------------------------------------------
# reap_worker_zombies() tests
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# connect_closing(): context manager that actually closes the FD
# Regression coverage for #33159 (kanban.db FD leak — gateway crashes after
# ~4 days). sqlite3.Connection's built-in __exit__ commits/rollbacks but
# does NOT close, so `with kb.connect() as conn:` leaks the FD in
# long-lived processes (gateway run_slash, dashboard decompose handler).
# `connect_closing()` is the leak-safe replacement.
# ---------------------------------------------------------------------------




def test_bare_connect_does_not_close_on_context_exit(tmp_path):
    """Document the leak that connect_closing exists to prevent.

    sqlite3.Connection's __exit__ commits/rollbacks but doesn't close.
    This is the upstream behaviour we cannot change; the regression
    guard is to make sure connect_closing() does the right thing.
    """
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path=db_path) as conn:
        pass
    # Still usable after with-block exit (the leak).
    conn.execute("SELECT 1").fetchone()
    conn.close()  # explicit close to avoid leaking THIS test


# ---------------------------------------------------------------------------
# Quality-class routing hardening (audit follow-up 2026-07-09)
# ---------------------------------------------------------------------------

class TestSetTaskClassNormalisation:
    """set_task_class centralises the 'clear' semantics so a plugin calling
    the setter directly can't leak a literal sentinel string into task_class."""

    def test_real_class_is_stored(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x")
            assert kb.set_task_class(conn, tid, "hard") is True
            assert kb.get_task(conn, tid).task_class == "hard"

    @pytest.mark.parametrize("sentinel", ["none", "None", "NULL", "-", "", "  "])
    def test_sentinels_clear_the_class(self, kanban_home, sentinel):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x")
            kb.set_task_class(conn, tid, "hard")
            assert kb.set_task_class(conn, tid, sentinel) is True
            # No literal 'none'/'null'/etc. leaks through — the class is cleared.
            assert kb.get_task(conn, tid).task_class is None

    def test_whitespace_is_stripped(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x")
            kb.set_task_class(conn, tid, "  hard  ")
            assert kb.get_task(conn, tid).task_class == "hard"


class TestProtectedBranchGuard:
    """_is_protected_branch keeps the worktree cleanup from ``branch -D``'ing a
    shared/long-lived branch named via a custom tasks.branch_name."""

    def _init_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        for cmd in (["git", "init", "-b", "main"], ["git", "commit", "--allow-empty", "-m", "init"]):
            subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)
        return repo

    def test_default_and_conventional_branches_are_protected(self, tmp_path):
        repo = self._init_repo(tmp_path)
        for b in ("main", "master", "develop", "trunk", "", "   "):
            assert kb._is_protected_branch(repo, b) is True

    def test_task_scoped_branch_is_not_protected(self, tmp_path):
        repo = self._init_repo(tmp_path)
        assert kb._is_protected_branch(repo, "wt/t_abc123") is False
        assert kb._is_protected_branch(repo, "feature/some-work") is False

    def test_current_head_branch_is_protected(self, tmp_path):
        # A branch checked out in the main worktree must not be deleted even if
        # it isn't the conventional default name.
        repo = self._init_repo(tmp_path)
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "checkout", "-b", "integration"], cwd=repo, env=env,
                       check=True, capture_output=True)
        assert kb._is_protected_branch(repo, "integration") is True


# ---------------------------------------------------------------------------
# First-class review handshake / deterministic promotion (S2)
# ---------------------------------------------------------------------------


def _request_review(conn, tid, *, reviewer="reviewer"):
    task = kb.get_task(conn, tid)
    assert task is not None
    return kb.request_task_review(
        conn,
        tid,
        reviewer=reviewer,
        summary="implementation ready",
        metadata={"tests": ["pytest -q"]},
        expected_run_id=task.current_run_id,
    )


def test_review_request_is_dispatchable_and_idempotent(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None

        assert _request_review(conn, tid) is True
        assert _request_review(conn, tid) is True

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.current_run_id is None
        requested = [e for e in kb.list_events(conn, tid) if e.kind == "review_requested"]
        assert len(requested) == 1

        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert review.status == "running"
        assert review.assignee == "reviewer"
        assert review.current_run_id is not None


def test_pending_review_cannot_bypass_accept_with_complete(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid)
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.complete_task(
            conn,
            tid,
            summary="bypass",
            expected_run_id=review.current_run_id,
        ) is False
        assert kb.get_task(conn, tid).status == "running"


# ---------------------------------------------------------------------------
# Stale review handshake recovery (2026-07-17, card t_5a43abe1)
#
# Repro: an implementation run hands off to review (`review_requested`); the
# reviewer claims it (`claim_review_task`) but its process CRASHES before
# calling `decide_task_review`. `_pending_review_request` used to keep
# returning the original `review_requested` payload forever (the crashed
# claim never produced a `review_decided`), permanently wedging
# `complete_task` and making `request_task_review` either refuse outright or
# no-op without dispatching a fresh reviewer. This produced hours of
# `block_loop_detected` noise (18:41-22:59 on the incident date) before an
# operator manually intervened.
# ---------------------------------------------------------------------------

def _crash_reviewer_claim(conn, tid, review):
    """Kill the reviewer's claimed run via the real crash-detection path.

    Mirrors ``test_detect_crashed_workers_*``: stamp a worker_pid, force
    ``_pid_alive`` False, and run the real ``detect_crashed_workers`` so the
    run ends exactly the way a genuine crash would (outcome='crashed',
    task reset to 'ready', current_run_id cleared).
    """
    import hermes_cli.kanban_db as _kb

    host = _kb._claimer_id().split(":", 1)[0]
    conn.execute(
        "UPDATE tasks SET worker_pid=?, claim_lock=? WHERE id=?",
        (555555, f"{host}:reviewer-proc", tid),
    )
    conn.execute(
        "UPDATE task_runs SET claim_lock=? WHERE id=?",
        (f"{host}:reviewer-proc", review.current_run_id),
    )
    conn.commit()
    crashed = _kb.detect_crashed_workers(conn)
    assert tid in crashed, "test setup: expected the review claim to crash"


def test_stale_review_handshake_does_not_wedge_complete(kanban_home, monkeypatch):
    """Repro (a): a crashed, undecided review handshake must not block
    complete_task forever — nobody is ever coming back to decide it."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid) is True
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        _crash_reviewer_claim(conn, tid, review)

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None

        # The crashed claim carries no decision -- the handshake must now
        # read as resolved/stale, not eternally pending.
        assert _kb._pending_review_request(conn, tid) is None

        assert kb.complete_task(
            conn, tid, result="landed despite dead reviewer",
        ) is True
        assert kb.get_task(conn, tid).status == "done"


def test_healthy_review_claim_still_pending_after_crash_helper_added(kanban_home):
    """Regression guard: a reviewer claim that is still genuinely RUNNING
    (never crashed) must keep blocking complete_task -- the review-bypass
    protection (2026-07-13) must not regress."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid) is True
        review = kb.claim_review_task(conn, tid)
        assert review is not None

        assert kb._pending_review_request(conn, tid) is not None
        assert kb.complete_task(
            conn, tid, summary="bypass", expected_run_id=review.current_run_id,
        ) is False
        assert kb.get_task(conn, tid).status == "running"


def test_unclaimed_review_request_still_pending(kanban_home):
    """Regression guard: a review_requested handshake that nobody has
    claimed yet (no reviewer run at all) is genuinely pending, not stale --
    must keep blocking complete_task and stay idempotent on re-request."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid) is True

        assert kb._pending_review_request(conn, tid) is not None
        assert kb.complete_task(conn, tid, summary="bypass") is False
        # Idempotent re-request: still exactly one review_requested event.
        assert _request_review(conn, tid) is True
        requested = [e for e in kb.list_events(conn, tid) if e.kind == "review_requested"]
        assert len(requested) == 1


def test_stale_review_handshake_allows_fresh_review_request(kanban_home, monkeypatch):
    """Repro (b): after a crashed/undecided review claim, request_task_review
    (citing the dead reviewer's run) must be able to open a FRESH handshake
    -- flipping the card back to 'review' so the review-lane dispatcher picks
    up a new reviewer -- instead of refusing or silently no-op'ing."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid) is True
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        stale_run_id = int(review.current_run_id)
        _crash_reviewer_claim(conn, tid, review)

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None

        ok = kb.request_task_review(
            conn, tid, reviewer="reviewer", summary="retry after crash",
            expected_run_id=stale_run_id,
        )
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        requested = [e for e in kb.list_events(conn, tid) if e.kind == "review_requested"]
        assert len(requested) == 2

        # And the fresh handshake is claimable exactly like a normal one.
        fresh = kb.claim_review_task(conn, tid)
        assert fresh is not None
        assert fresh.status == "running"


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_assignee"),
    [
        ("ACCEPT", "done", "backend-eng"),
        ("NEEDS_REPAIR", "ready", "backend-eng"),
        ("BLOCK", "blocked", "backend-eng"),
    ],
)
def test_review_decisions_are_atomic(
    kanban_home, decision, expected_status, expected_assignee,
):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid)
        review = kb.claim_review_task(conn, tid)
        assert review is not None

        assert kb.decide_task_review(
            conn,
            tid,
            decision=decision,
            summary=f"review says {decision}",
            metadata={"reviewer": "independent"},
            expected_run_id=review.current_run_id,
        ) is True

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == expected_status
        assert task.assignee == expected_assignee
        assert task.current_run_id is None
        latest = kb.latest_run(conn, tid)
        assert latest is not None
        assert latest.outcome == decision.lower()
        events = kb.list_events(conn, tid)
        # BLOCK additionally appends a 'blocked' event so _has_sticky_block
        # recognizes the review gate (S4d) — hence three accepted tail kinds.
        assert events[-1].kind in {"review_decided", "completed", "blocked"}


def test_stale_review_decision_is_rejected(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid)
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.decide_task_review(
            conn,
            tid,
            decision="ACCEPT",
            summary="stale",
            expected_run_id=int(review.current_run_id) + 1,
        ) is False
        assert kb.get_task(conn, tid).status == "running"


def test_review_accept_enforces_and_merges_completion_contract(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="contract", assignee="backend-eng",
            completion_contract={"tests_or_smokes": True, "readback": True},
        )
        impl = kb.claim_task(conn, tid)
        assert kb.request_task_review(
            conn, tid, reviewer="reviewer", summary="ready",
            metadata={"tests_or_smokes": ["pytest"]},
            expected_run_id=impl.current_run_id,
        )
        review = kb.claim_review_task(conn, tid)
        with pytest.raises(kb.CompletionEvidenceError):
            kb.decide_task_review(
                conn, tid, decision="ACCEPT", metadata={},
                expected_run_id=review.current_run_id,
            )
        assert kb.get_task(conn, tid).status == "running"
        assert kb.decide_task_review(
            conn, tid, decision="ACCEPT", metadata={"readback": ["file:1"]},
            expected_run_id=review.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "done"
        closed = kb.latest_run(conn, tid)
        assert closed.metadata["tests_or_smokes"] == ["pytest"]
        assert closed.metadata["readback"] == ["file:1"]


def test_dependency_block_hook_observes_committed_state(kanban_home, monkeypatch):
    seen = []

    def hook(_name, task_id, **_payload):
        with kb.connect() as observer:
            seen.append(kb.get_task(observer, task_id).status)

    monkeypatch.setattr(kb, "_fire_kanban_lifecycle_hook", hook)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="waiting", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.block_task(
            conn,
            tid,
            reason="parent outstanding",
            kind="dependency",
            expected_run_id=claimed.current_run_id,
        ) is True
    assert seen[-1] == "todo"


def test_link_to_done_parent_promotes_child_immediately(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        assert kb.complete_task(conn, parent, summary="done")
        kb.link_tasks(conn, parent, child)
        assert kb.get_task(conn, child).status == "ready"


def test_create_with_done_parent_is_immediately_ready(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        assert kb.complete_task(conn, parent, summary="done")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.get_task(conn, child).status == "ready"


def test_recompute_does_not_promote_sticky_block(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="human decision", assignee="worker")
        assert kb.block_task(conn, tid, reason="choose", kind="needs_input")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"


def test_archive_complete_race_leaves_one_consistent_terminal_state(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="race", assignee="worker")

    def complete():
        with kb.connect() as conn:
            return kb.complete_task(conn, tid, summary="done")

    def archive():
        with kb.connect() as conn:
            return kb.archive_task(conn, tid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(complete), pool.submit(archive)]
        won = [future.result() for future in futures]

    # done -> archived is a legitimate explicit transition, so both operations
    # may serialize successfully. The invariant is one final terminal state and
    # no orphaned active run/claim regardless of ordering.
    assert any(bool(result) for result in won)
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task.status in {"done", "archived"}
        assert task.current_run_id is None
        assert task.claim_lock is None


# ---------------------------------------------------------------------------
# S4d: needs_input human-gate enforcement + unblock attribution +
# forced-promote/claim consistency (Audit 2026-07-10)
# ---------------------------------------------------------------------------

def test_review_block_is_sticky_against_recompute_ready(kanban_home):
    """A review BLOCK must emit a 'blocked' event so _has_sticky_block holds
    and recompute_ready does NOT silently reopen the needs_input gate."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid)
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.decide_task_review(
            conn, tid, decision="BLOCK", summary="needs human approval",
            expected_run_id=review.current_run_id,
        ) is True
        assert kb._has_sticky_block(conn, tid) is True
        kb.recompute_ready(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"


def test_review_block_increments_block_recurrences(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement", assignee="backend-eng")
        kb.claim_task(conn, tid)
        assert _request_review(conn, tid)
        review = kb.claim_review_task(conn, tid)
        assert kb.decide_task_review(
            conn, tid, decision="BLOCK", summary="round 1",
            expected_run_id=review.current_run_id,
        )
        row = conn.execute(
            "SELECT block_recurrences FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert int(row["block_recurrences"]) >= 1


def test_unblock_event_carries_actor_and_reason(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gate", assignee="worker")
        kb.block_task(conn, tid, reason="approval required", kind="needs_input")
        assert kb.unblock_task(
            conn, tid, actor="manfred", reason="approved via chat"
        ) is True
        events = kb.list_events(conn, tid)
        unblocked = [e for e in events if e.kind == "unblocked"][-1]
        assert unblocked.payload["actor"] == "manfred"
        assert unblocked.payload["reason"] == "approved via chat"


def test_claim_honors_forced_manual_promote(kanban_home):
    """claim_task must not silently revert an operator's promote --force:
    the forced promoted_manual event overrides the parent gate once."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="review subject", assignee="worker")
        child = kb.create_task(
            conn, title="quality gate", assignee="reviewer", parents=(parent,)
        )
        kb.block_task(conn, parent, reason="review-required", kind="needs_input")
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (child,)
        )
        with kb.write_txn(conn):
            kb._append_event(
                conn, child, "promoted_manual",
                {"actor": "operator", "forced": True},
            )
        claimed = kb.claim_task(conn, child)
        assert claimed is not None, "forced promote must survive claim"
        assert kb.get_task(conn, child).status == "running"


def test_claim_still_demotes_without_forced_promote(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=(parent,)
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (child,)
        )
        assert kb.claim_task(conn, child) is None
        assert kb.get_task(conn, child).status == "todo"


def test_cli_unblock_refused_in_worker_session(kanban_home, monkeypatch, capsys):
    import argparse
    from hermes_cli import kanban as kanban_cli
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_someworker")
    rc = kanban_cli._cmd_unblock(argparse.Namespace(task_ids=["t_x"], reason=None))
    assert rc == 1
    assert "refused" in capsys.readouterr().err


def test_cli_unblock_needs_input_requires_reason(kanban_home, monkeypatch, capsys):
    import argparse
    from hermes_cli import kanban as kanban_cli
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gate", assignee="worker")
        kb.block_task(conn, tid, reason="approval required", kind="needs_input")
    rc = kanban_cli._cmd_unblock(argparse.Namespace(task_ids=[tid], reason=None))
    assert rc == 1
    assert "requires" in capsys.readouterr().err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
        rc2 = kanban_cli._cmd_unblock(
            argparse.Namespace(task_ids=[tid], reason="approved by operator")
        )
        assert rc2 == 0


# ---------------------------------------------------------------------------
# Human-Gate v1 (human-gate-design.md, 2026-07-11)
# ---------------------------------------------------------------------------

def test_human_gate_token_roundtrip(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="needs approval", kind="needs_input", human_gate=True)
        assert kb.get_task(conn, tid).human_gate is True
        token = kb.issue_gate_token(conn, tid)
        assert token
        assert kb.unblock_task(conn, tid, actor="manfred", reason="approved", token=token) is True
        assert kb.get_task(conn, tid).status in ("ready", "todo")
        events = kb.list_events(conn, tid)
        unblocked = [e for e in events if e.kind == "unblocked"][-1]
        assert unblocked.payload["human_gate"] is True


def test_human_gate_token_race_has_one_winner(kanban_home):
    """Two real connections redeeming one token produce one transition."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="needs approval", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
    barrier = threading.Barrier(2)
    outcomes: list[bool] = []
    failures: list[BaseException] = []

    def redeem() -> None:
        try:
            with kb.connect() as other:
                barrier.wait(timeout=5)
                outcomes.append(kb.unblock_task(other, tid, token=token))
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=redeem) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not failures
    assert outcomes.count(True) == 1 and outcomes.count(False) == 1
    with kb.connect() as conn:
        row = conn.execute("SELECT status, gate_token_hash FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] in ("ready", "todo") and row["gate_token_hash"] is None
        assert sum(event.kind == "unblocked" for event in kb.list_events(conn, tid)) == 1


def test_human_gate_grant_binds_board_task_and_action(kanban_home):
    with kb.scoped_current_board("default"), kb.connect() as conn:
        first = kb.create_task(conn, title="first", assignee="worker")
        second = kb.create_task(conn, title="second", assignee="worker")
        for tid in (first, second):
            kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, first, action="unblock", board="default")
        assert token
        with pytest.raises(kb.GateTokenError, match="different action"):
            kb.complete_task(conn, first, result="x", token=token)
        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, second, token=token)
        with kb.scoped_current_board("other"):
            with pytest.raises(kb.GateTokenError, match="different board"):
                kb.unblock_task(conn, first, token=token)
        with kb.scoped_current_board("default"):
            assert kb.unblock_task(conn, first, token=token) is True


def test_governance_gate_token_binds_current_canonical_scope(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="gated", body="preserve **exact** bytes", assignee="worker",
            workspace_kind="worktree", workspace_path="/repo/.worktrees/gated",
            branch_name="governance/gated",
        )
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        row = conn.execute(
            "SELECT gate_scope_hash, gate_scope_version FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert token and row["gate_scope_version"] == kb.GOVERNANCE_SCOPE_VERSION
        assert row["gate_scope_hash"] == kb.governance_scope_hash(conn, tid)
        assert "gate_scope_hash" not in vars(kb.get_task(conn, tid))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("body", "materially changed"),
        ("workspace_path", "/different/repo"),
        ("branch_name", "governance/other"),
        ("governance_mutation_class", "runtime-restart"),
    ],
)
def test_governance_gate_rejects_token_after_material_scope_change(kanban_home, column, value):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="gated", body="original", assignee="worker",
            workspace_kind="worktree", workspace_path="/repo/.worktrees/gated",
            branch_name="governance/gated",
        )
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        before = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["gate_token_hash"]
        before_events = [e.kind for e in kb.list_events(conn, tid)]
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, tid))
        with pytest.raises(kb.GateTokenError, match="governance scope changed"):
            kb.unblock_task(conn, tid, token=token)
        row = conn.execute(
            "SELECT status, gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["status"] == "blocked" and row["gate_token_hash"] == before
        assert [e.kind for e in kb.list_events(conn, tid)] == before_events


def test_governance_gate_token_survives_formatting_only_body_normalization(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", body="line one\r\nline two\r\n\r\n", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        conn.execute("UPDATE tasks SET body = ? WHERE id = ?", ("line one\nline two\n", tid))
        assert kb.unblock_task(conn, tid, token=token) is True


def test_governance_gate_scope_change_requires_fresh_token(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", body="old", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        stale = kb.issue_gate_token(conn, tid)
        conn.execute("UPDATE tasks SET body = 'new' WHERE id = ?", (tid,))
        with pytest.raises(kb.GateTokenError, match="governance scope changed"):
            kb.unblock_task(conn, tid, token=stale)
        fresh = kb.issue_gate_token(conn, tid)
        assert fresh and kb.unblock_task(conn, tid, token=fresh) is True


def test_legacy_human_gate_token_without_scope_binding_fails_closed(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        conn.execute(
            "UPDATE tasks SET gate_scope_hash = NULL, gate_scope_version = NULL WHERE id = ?", (tid,)
        )
        with pytest.raises(kb.GateTokenError, match="governance scope changed"):
            kb.unblock_task(conn, tid, token=token)
        assert kb.get_task(conn, tid).status == "blocked"



def _governance_snapshot(conn, task_id):
    """Stable durable state used by P4b no-op/rollback contracts."""
    task = tuple(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    events = [tuple(r) for r in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,)
    ).fetchall()]
    runs = [tuple(r) for r in conn.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)
    ).fetchall()]
    return task, events, runs


@pytest.mark.parametrize("column,value", [
    ("title", "renamed intent"),
    ("workspace_kind", "dir"),
    ("project_id", "different-project"),
])
def test_governance_scope_matrix_rejects_material_card_changes(kanban_home, column, value):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", body="body", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        before = _governance_snapshot(conn, tid)
        # These are durable card fields with no public editor API; this is an
        # explicit storage-layer characterization of the scope contract.
        conn.execute(f"UPDATE tasks SET {column}=? WHERE id=?", (value, tid))
        with pytest.raises(kb.GateTokenError, match="governance scope changed"):
            kb.unblock_task(conn, tid, token=token)
        after = _governance_snapshot(conn, tid)
        assert after[0][0] == before[0][0]  # task identity is stable
        assert after[1:] == before[1:]      # no event/run/hook-visible effect
        assert conn.execute("SELECT gate_token_hash FROM tasks WHERE id=?", (tid,)).fetchone()[0]


def test_governance_internal_writer_is_closed_and_scope_bound(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(
            conn, tid, reason="x", kind="needs_input", human_gate=True,
            _governance_target_ref="refs/heads/main",
            _governance_mutation_class="agent-runtime",
        )
        token = kb.issue_gate_token(conn, tid)
        row = conn.execute("SELECT governance_target_ref, governance_mutation_class FROM tasks WHERE id=?", (tid,)).fetchone()
        assert tuple(row) == ("refs/heads/main", "agent-runtime")
        with pytest.raises(ValueError):
            kb._set_internal_governance_declarations(
                conn, tid, target_ref="bad ref with spaces", mutation_class="agent-runtime"
            )
        kb._set_internal_governance_declarations(
            conn, tid, target_ref="refs/heads/repaired", mutation_class="agent-runtime"
        )
        with pytest.raises(kb.GateTokenError, match="governance scope changed"):
            kb.unblock_task(conn, tid, token=token)


def test_governance_scope_negative_lifecycle_controls_do_not_invalidate(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", body="body", assignee="worker", priority=1)
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        conn.execute(
            "UPDATE tasks SET priority=9, assignee='other', skills='[\\\"x\\\"]', "
            "task_class='hard', max_runtime_seconds=12 WHERE id=?", (tid,)
        )
        assert kb.unblock_task(conn, tid, token=token) is True


def test_governance_scope_fault_after_consume_rolls_back_everything(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        before = _governance_snapshot(conn, tid)
        original = kb._append_event
        def boom(*args, **kwargs):
            if len(args) > 2 and args[2] == "unblocked":
                raise RuntimeError("injected")
            return original(*args, **kwargs)
        monkeypatch.setattr(kb, "_append_event", boom)
        with pytest.raises(RuntimeError, match="injected"):
            kb.unblock_task(conn, tid, token=token)
        assert _governance_snapshot(conn, tid) == before


def test_governance_scope_parallel_consume_has_exactly_one_winner(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
    barrier = threading.Barrier(2)
    outcomes = []
    def consume():
        with kb.connect() as other:
            barrier.wait(timeout=5)
            outcomes.append(kb.unblock_task(other, tid, token=token))
    threads = [threading.Thread(target=consume) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=10)
    assert outcomes.count(True) == 1 and outcomes.count(False) == 1



def test_governance_scope_change_vs_consume_is_serialized_fail_closed(kanban_home):
    """Real independent writers: drift wins => old grant never authorizes."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", body="before", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
    entered = threading.Event()
    release = threading.Event()
    outcome = []
    def drift():
        with kb.connect() as writer, kb.write_txn(writer):
            writer.execute("UPDATE tasks SET body='after' WHERE id=?", (tid,))
            entered.set()
            assert release.wait(5)
    def consume():
        with kb.connect() as reader:
            try:
                outcome.append(kb.unblock_task(reader, tid, token=token))
            except kb.GateTokenError as exc:
                outcome.append(type(exc).__name__)
    writer = threading.Thread(target=drift)
    writer.start(); assert entered.wait(5)
    consumer = threading.Thread(target=consume); consumer.start()
    release.set(); writer.join(timeout=10); consumer.join(timeout=10)
    assert outcome == ["GateTokenError"]
    with kb.connect() as conn:
        row = conn.execute("SELECT status, gate_token_hash FROM tasks WHERE id=?", (tid,)).fetchone()
        assert row["status"] == "blocked" and row["gate_token_hash"]


def test_governance_scope_public_and_durable_surfaces_redact_internal_values(kanban_home, caplog):
    secret_ref = "refs/heads/private-governance-anchor"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True,
                      _governance_target_ref=secret_ref,
                      _governance_mutation_class="agent-runtime")
        token = kb.issue_gate_token(conn, tid)
        task = kb.get_task(conn, tid)
        assert secret_ref not in repr(vars(task))
        events_before = json.dumps([e.payload for e in kb.list_events(conn, tid)])
        assert secret_ref not in events_before and token not in events_before
        kb._set_internal_governance_declarations(
            conn, tid, target_ref="refs/heads/changed", mutation_class="agent-runtime"
        )
        with pytest.raises(kb.GateTokenError, match="governance scope changed"):
            kb.unblock_task(conn, tid, token=token)
        assert secret_ref not in caplog.text and token not in caplog.text


def test_human_gate_failed_attempt_lockout_persists_across_connections(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_human_gate_config", lambda: {
        "token_ttl_seconds": 600, "max_failed_attempts": 2,
        "failure_window_seconds": 60, "lockout_seconds": 60,
    })
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        for _ in range(2):
            with pytest.raises(kb.GateTokenError):
                kb.unblock_task(conn, tid, token="wrong")
    with kb.connect() as fresh:
        row = fresh.execute(
            "SELECT gate_failed_attempts, gate_locked_until FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["gate_failed_attempts"] == 2
        assert row["gate_locked_until"] > int(time.time())
        with pytest.raises(kb.GateTokenError, match="temporarily locked"):
            kb.unblock_task(fresh, tid, token=token)


def test_human_gate_rotation_resets_failed_attempts(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, token="wrong")
        token = kb.issue_gate_token(conn, tid)
        row = conn.execute(
            "SELECT gate_failed_attempts, gate_locked_until FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["gate_failed_attempts"] == 0
        assert row["gate_locked_until"] is None
        assert kb.unblock_task(conn, tid, token=token) is True


def test_human_gate_refuses_when_no_token_issued(kanban_home):
    """A human_gate=1 card with a NULL hash (never issued, or ntfy failed)
    must refuse ANY token, not just a wrong one — that's the fail-closed
    state, and the only rescue path is `kanban gate <id> off`."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, actor="a", reason="x", token="whatever")
        assert kb.get_task(conn, tid).status == "blocked"


def test_human_gate_refuses_missing_or_wrong_token(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        kb.issue_gate_token(conn, tid)

        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, actor="a", reason="x")  # no token at all
        assert kb.get_task(conn, tid).status == "blocked"

        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, actor="a", reason="x", token="not-the-token")
        assert kb.get_task(conn, tid).status == "blocked"


def test_human_gate_token_is_single_use(kanban_home):
    """A token that was already consumed must not work again, even for a
    fresh re-block of the SAME card (gate persists across re-block, but
    the old hash was cleared on the successful unblock)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        assert kb.unblock_task(conn, tid, actor="a", reason="ok", token=token) is True

        # Get the card back into blocked without issuing a new token.
        if kb.get_task(conn, tid).status == "todo":
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        assert kb.claim_task(conn, tid) is not None
        # kind=None (not "needs_input" again) so this re-block doesn't also
        # trip the unrelated S4 unblock-loop breaker (BLOCK_RECURRENCE_LIMIT)
        # and get routed to triage instead of blocked.
        kb.block_task(conn, tid, reason="still needs input")
        assert kb.get_task(conn, tid).human_gate is True, "gate must persist across re-block"

        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, actor="a", reason="replay", token=token)
        assert kb.get_task(conn, tid).status == "blocked"


def test_human_gate_reblock_rotates_token(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        old_token = kb.issue_gate_token(conn, tid)
        assert kb.unblock_task(conn, tid, actor="a", reason="ok", token=old_token) is True

        if kb.get_task(conn, tid).status == "todo":
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        assert kb.claim_task(conn, tid) is not None
        # kind=None here too, for the same reason as the single-use test.
        kb.block_task(conn, tid, reason="again")
        new_token = kb.issue_gate_token(conn, tid)
        assert new_token != old_token

        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, actor="a", reason="replay old", token=old_token)
        assert kb.unblock_task(conn, tid, actor="a", reason="ok again", token=new_token) is True


def test_human_gate_token_never_persisted_in_plaintext(kanban_home):
    """Dump every text-bearing row the token could have leaked into
    (tasks columns, event payloads, comments) and confirm the plaintext
    never appears — only its sha256 hash does, and only until consumed."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid)
        assert token
        assert kb.hash_gate_token(token) != token
        assert kb.unblock_task(conn, tid, actor="a", reason="ok", token=token) is True

        task_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        for key in task_row.keys():
            val = task_row[key]
            if isinstance(val, str):
                assert token not in val, f"plaintext token leaked into tasks.{key}"
        assert task_row["gate_token_hash"] is None, "token must be cleared after use"

        for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ?", (tid,)
        ).fetchall():
            if row["payload"]:
                assert token not in row["payload"], "plaintext token leaked into task_events.payload"

        for row in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = ?", (tid,)
        ).fetchall():
            assert token not in (row["body"] or ""), "plaintext token leaked into task_comments.body"


def test_human_gate_ntfy_failure_is_fail_closed(kanban_home, monkeypatch):
    """A mocked HTTP failure during the ntfy push must leave the card
    hard-blocked: the token hash stays persisted (nobody holds the
    plaintext), so unblock_task keeps refusing until `gate off`.

    The ntfy push is legacy-opt-in since gate_notify_ntfy defaulted to off
    (the durable ops channel replaced it); this test pins the LEGACY path,
    so it enables the flag explicitly."""
    monkeypatch.setattr(kb, "gate_notify_ntfy_enabled", lambda: True)
    monkeypatch.setenv("NTFY_TOPIC", "test-topic")

    def _raise(*_a, **_kw):
        raise OSError("network unreachable")

    monkeypatch.setattr(kb.urllib.request, "urlopen", _raise)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)

        delivered = kb.issue_and_notify_gate_token(conn, tid)
        assert delivered is False

        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["gate_token_hash"] is not None  # issued, but never delivered

        with pytest.raises(kb.GateTokenError):
            kb.unblock_task(conn, tid, actor="a", reason="x", token="anything")
        assert kb.get_task(conn, tid).status == "blocked"

        # Step④ (2026-07-14): H1 removed — `gate off` is again the FREE rescue for a
        # coarse human-gate hold (this card has no pending exact-action). ntfy being
        # down no longer strands it: the operator lifts the soft hold directly (it is
        # budget-bounded + logged + reversible). The exact-action approval — the real
        # grant-enforced authority gate — is a separate mechanism, unaffected.
        assert kb.set_human_gate(conn, tid, on=False, actor="manfred") is True
        assert kb.get_task(conn, tid).human_gate is False


def test_human_gate_token_issuance_succeeds_without_ntfy_by_default(
    kanban_home, monkeypatch,
):
    """Current contract: with gate_notify_ntfy OFF (the default) issuance
    succeeds WITHOUT any delivery dependency — the durable ops channel and
    the cockpit are the notify/release paths, ntfy is out of the loop and
    must not even be attempted."""
    monkeypatch.delenv("NTFY_TOPIC", raising=False)

    def _must_not_send(*_a, **_kw):
        raise AssertionError("ntfy push attempted although gate_notify_ntfy is off")

    monkeypatch.setattr(kb, "send_gate_token_ntfy", _must_not_send)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        assert kb.issue_and_notify_gate_token(conn, tid) is True
        assert kb.get_task(conn, tid).status == "blocked"
        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["gate_token_hash"] is not None


def test_human_gate_missing_ntfy_config_is_fail_closed(kanban_home, monkeypatch):
    """Legacy path (gate_notify_ntfy explicitly on): no configured topic
    means no delivery — fail-closed."""
    monkeypatch.setattr(kb, "gate_notify_ntfy_enabled", lambda: True)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        assert kb.issue_and_notify_gate_token(conn, tid) is False
        assert kb.get_task(conn, tid).status == "blocked"


def test_ungated_block_unaffected_by_human_gate(kanban_home):
    """Behavioural neutrality: a card never marked human_gate=1 keeps
    working exactly like before — no token required."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
        kb.block_task(conn, tid, reason="waiting", kind="needs_input")
        assert kb.get_task(conn, tid).human_gate is False
        assert kb.unblock_task(conn, tid, actor="a", reason="ok") is True


def test_cmd_gate_on_off_toggles_flag_and_audits(kanban_home, monkeypatch):
    import argparse
    from hermes_cli import kanban as kanban_cli
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")

    rc = kanban_cli._cmd_gate(argparse.Namespace(task_id=tid, state="on"))
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).human_gate is True
        events = kb.list_events(conn, tid)
        on_events = [e for e in events if e.kind == "gate_set" and e.payload.get("on") is True]
        assert on_events

    rc2 = kanban_cli._cmd_gate(argparse.Namespace(task_id=tid, state="off"))
    assert rc2 == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).human_gate is False
        events = kb.list_events(conn, tid)
        off_events = [e for e in events if e.kind == "gate_set" and e.payload.get("on") is False]
        assert off_events


def test_cmd_gate_refused_in_worker_session(kanban_home, monkeypatch, capsys):
    """A worker must not be able to strip its own hard gate by shelling
    out to `kanban gate <id> off` (same pattern as the S4d unblock
    refusal — Audit 2026-07-10)."""
    import argparse
    from hermes_cli import kanban as kanban_cli
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_someworker")
    rc = kanban_cli._cmd_gate(argparse.Namespace(task_id=tid, state="off"))
    assert rc == 1
    assert "refused" in capsys.readouterr().err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).human_gate is False  # never touched


@pytest.mark.parametrize("action", ["complete", "promote"])
def test_cmd_gate_token_issues_delivers_and_redeems_action_grant(
    kanban_home, monkeypatch, action
):
    import argparse
    from hermes_cli import kanban as kanban_cli

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    delivered = {}

    def _capture(task_id, token, *, board=None, action="unblock"):
        delivered.update(task_id=task_id, token=token, board=board, action=action)
        return True

    # Legacy ntfy delivery path — opt-in since gate_notify_ntfy defaulted off.
    monkeypatch.setattr(kb, "gate_notify_ntfy_enabled", lambda: True)
    monkeypatch.setattr(kb, "send_gate_token_ntfy", _capture)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="approval", kind="needs_input", human_gate=True)

    rc = kanban_cli._cmd_gate_token(argparse.Namespace(task_id=tid, action=action))
    assert rc == 0
    assert delivered["task_id"] == tid
    assert delivered["action"] == action

    with kb.connect() as conn:
        if action == "complete":
            assert kb.complete_task(
                conn,
                tid,
                result="approved",
                summary="approved",
                token=delivered["token"],
            ) is True
            assert kb.get_task(conn, tid).status == "done"
        else:
            ok, err = kb.promote_task(
                conn,
                tid,
                actor="operator",
                reason="approved",
                token=delivered["token"],
            )
            assert (ok, err) == (True, None)
            assert kb.get_task(conn, tid).status == "ready"
        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["gate_token_hash"] is None


def test_cmd_gate_token_refused_in_worker_session(kanban_home, monkeypatch, capsys):
    import argparse
    from hermes_cli import kanban as kanban_cli

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="approval", kind="needs_input", human_gate=True)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    rc = kanban_cli._cmd_gate_token(argparse.Namespace(task_id=tid, action="complete"))
    assert rc == 1
    assert "refused" in capsys.readouterr().err


def test_gate_token_parser_requires_and_captures_action():
    import argparse
    from hermes_cli import kanban as kanban_cli

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="top")
    kanban_cli.build_parser(subparsers)
    args = parser.parse_args(
        ["kanban", "gate-token", "t_example", "--action", "complete"]
    )
    assert args.kanban_action == "gate-token"
    assert args.task_id == "t_example"
    assert args.action == "complete"


def test_cli_block_human_gate_flag_and_unblock_token(kanban_home, monkeypatch, capsys):
    import argparse
    from hermes_cli import kanban as kanban_cli
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")

    rc = kanban_cli._cmd_block(argparse.Namespace(
        task_id=tid, reason=["approval", "needed"], ids=None,
        kind="needs_input", human_gate=True,
    ))
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).human_gate is True
        token = kb.issue_gate_token(conn, tid)

    # Without --token: refused, bulk loop continues (single id here, but
    # exercises the try/except path instead of an uncaught raise).
    rc_fail = kanban_cli._cmd_unblock(
        argparse.Namespace(task_ids=[tid], reason="ok", token=None)
    )
    assert rc_fail == 1
    assert "human-gated" in capsys.readouterr().err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"

    rc_ok = kanban_cli._cmd_unblock(
        argparse.Namespace(task_ids=[tid], reason="approved", token=token)
    )
    assert rc_ok == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status in ("ready", "todo")


def test_human_gate_migration_idempotent_on_legacy_db(tmp_path, monkeypatch):
    """A DB created before Human-Gate v1 shipped (missing all three
    columns) must migrate cleanly on open, and a second open must be a
    no-op rather than erroring on 'duplicate column'."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="legacy_gate")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    # A deliberately minimal pre-Human-Gate `tasks` table — everything from
    # `tenant` onward (including human_gate/gate_token_*) is missing, the
    # same shape a DB predating dozens of additive migrations would have.
    # `SCHEMA_SQL`'s `CREATE TABLE IF NOT EXISTS tasks` is a no-op against
    # this table on the next `kb.connect()`, so only `_migrate_add_optional_
    # columns` can add the missing columns — exercising the real migration
    # path instead of `ALTER TABLE ... DROP COLUMN`, which is unreliable
    # against a heavily commented CREATE TABLE like the current SCHEMA_SQL.
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE tasks (
            id             TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            body           TEXT,
            assignee       TEXT,
            status         TEXT NOT NULL,
            priority       INTEGER DEFAULT 0,
            created_by     TEXT,
            created_at     INTEGER NOT NULL,
            started_at     INTEGER,
            completed_at   INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            branch_name    TEXT,
            claim_lock     TEXT,
            claim_expires  INTEGER
        );
        """
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('t-legacy', 'T', 'blocked', 1000)"
    )
    conn.commit()
    conn.close()

    with kb.connect(db_path) as c1:
        cols = {r["name"] for r in c1.execute("PRAGMA table_info(tasks)")}
        assert {"human_gate", "gate_token_hash", "gate_token_issued_at", "gate_scope_hash", "gate_scope_version", "governance_target_ref", "governance_mutation_class"} <= cols
        assert c1.execute("SELECT title FROM tasks WHERE id = 't-legacy'").fetchone()["title"] == "T"
        assert c1.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        legacy_task = kb.get_task(c1, "t-legacy")
        assert legacy_task.human_gate is False  # safe default for a pre-existing row

    # Re-running the migration (fresh connect) must not error.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as c2:
        cols2 = {r["name"] for r in c2.execute("PRAGMA table_info(tasks)")}
        assert {"human_gate", "gate_token_hash", "gate_token_issued_at"} <= cols2


def test_p4b_legacy_human_gate_fixture_migrates_fail_closed_then_reissues_once(
    tmp_path, monkeypatch, caplog
):
    """Real pre-P4b Human-Gate schema: token metadata exists, scope columns do not."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = home / "legacy-p4b.db"
    task_id, token = "t_legacy_gate", "legacy-plaintext-token"
    private_body, private_workspace = "raw legacy body", "/private/workspace"
    issued_at = int(time.time())

    # This is the P4a tasks shape with exactly P4b's four columns absent;
    # the historical token binding, run and event are created before any P4b
    # initializer touches the file (no new-helper fixture normalization).
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
            status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT,
            created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT,
            branch_name TEXT, project_id TEXT, claim_lock TEXT, claim_expires INTEGER,
            tenant TEXT, result TEXT, idempotency_key TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0, worker_pid INTEGER,
            last_failure_error TEXT, max_runtime_seconds INTEGER, last_heartbeat_at INTEGER,
            current_run_id INTEGER, workflow_template_id TEXT, current_step_key TEXT,
            skills TEXT, model_override TEXT, task_class TEXT, effort TEXT,
            max_retries INTEGER, goal_mode INTEGER NOT NULL DEFAULT 0,
            goal_max_turns INTEGER, session_id TEXT, block_kind TEXT,
            block_recurrences INTEGER NOT NULL DEFAULT 0,
            block_cause_fingerprint TEXT, block_reason_code TEXT,
            block_cause_version INTEGER, completion_contract TEXT,
            human_gate INTEGER NOT NULL DEFAULT 0, gate_token_hash TEXT,
            gate_token_issued_at INTEGER, gate_token_board TEXT,
            gate_token_task_id TEXT, gate_token_action TEXT,
            gate_failed_attempts INTEGER NOT NULL DEFAULT 0,
            gate_failure_window_started_at INTEGER, gate_locked_until INTEGER
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            profile TEXT, step_key TEXT, status TEXT NOT NULL, claim_lock TEXT,
            claim_expires INTEGER, worker_pid INTEGER, max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER, last_activity_at INTEGER,
            last_semantic_progress_at INTEGER, worker_start_ticks INTEGER,
            d_state_since INTEGER, resource_sample TEXT,
            termination_pending_since INTEGER, started_at INTEGER NOT NULL,
            ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT, error TEXT
        );
    """)
    legacy.execute(
        """INSERT INTO tasks (id, title, body, assignee, status, created_at,
            workspace_kind, workspace_path, branch_name, claim_lock, claim_expires,
            current_run_id, human_gate, gate_token_hash, gate_token_issued_at,
            gate_token_board, gate_token_task_id, gate_token_action)
           VALUES (?, 'legacy gated', ?, 'legacy', 'blocked', 101, 'dir', ?,
                   'legacy/main', NULL, NULL, 73, 1, ?, ?,
                   'default', ?, 'unblock')""",
        (task_id, private_body, private_workspace, kb.hash_gate_token(token), issued_at, task_id),
    )
    legacy.execute("INSERT INTO task_runs (id, task_id, status, started_at, summary) VALUES (73, ?, 'blocked', 102, 'historical run')", (task_id,))
    legacy.execute("INSERT INTO task_events (id, task_id, run_id, kind, payload, created_at) VALUES (41, ?, 73, 'blocked', '{\"historical\":true}', 103)", (task_id,))
    legacy.execute("INSERT INTO tasks (id, title, status, created_at, workspace_kind, human_gate) VALUES ('t_legacy_plain', 'legacy plain', 'blocked', 104, 'scratch', 0)")
    legacy.commit()
    legacy.close()

    kb.init_db(db_path)  # public/real migration path
    with kb.connect(db_path) as conn:
        assert {"gate_scope_hash", "gate_scope_version", "governance_target_ref", "governance_mutation_class"} <= {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        historical = (
            tuple(conn.execute("SELECT id, title, body, workspace_path, claim_lock, current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()),
            tuple(conn.execute("SELECT id, task_id, run_id, kind, payload, created_at FROM task_events WHERE id=41").fetchone()),
            tuple(conn.execute("SELECT id, task_id, status, started_at, summary FROM task_runs WHERE id=73").fetchone()),
        )
        assert historical == (
            (task_id, "legacy gated", private_body, private_workspace, None, 73),
            (41, task_id, 73, "blocked", '{"historical":true}', 103),
            (73, task_id, "blocked", 102, "historical run"),
        )
        before_reject = _governance_snapshot(conn, task_id)
        with caplog.at_level("WARNING", logger="hermes_cli.kanban_db"):
            with pytest.raises(kb.GateTokenError) as error:
                kb.unblock_task(conn, task_id, token=token)
        assert "governance scope changed" in str(error.value)
        assert _governance_snapshot(conn, task_id) == before_reject
        mismatch_surfaces = "\n".join((
            str(error.value), caplog.text,
            json.dumps([e.payload for e in kb.list_events(conn, task_id)]),
        ))
        # Task body/workspace are established public card fields; the P4b
        # boundary is that the mismatch itself and public Task projection do
        # not introduce token/hash/scope-declaration data.
        for secret in (token, kb.hash_gate_token(token), private_body, private_workspace):
            assert secret not in mismatch_surfaces
        public_task = repr(vars(kb.get_task(conn, task_id)))
        for marker in ("gate_token_hash", "gate_scope_hash", "governance_target_ref", "governance_mutation_class"):
            assert marker not in public_task
        # New controlled issuance binds current scope, redeems exactly once.
        fresh = kb.issue_gate_token(conn, task_id)
        assert fresh and kb.unblock_task(conn, task_id, token=fresh) is True
        assert kb.unblock_task(conn, task_id, token=fresh) is False
        assert kb.unblock_task(conn, "t_legacy_plain", actor="operator", reason="unchanged") is True
        # The consumed grant's cleared binding is itself durable state; second
        # init must not resurrect a legacy token or rewrite historical rows.
        retained_scope = tuple(conn.execute(
            "SELECT gate_scope_hash, gate_scope_version FROM tasks WHERE id=?", (task_id,)
        ).fetchone())

    kb.init_db(db_path)  # second public init must be idempotent
    with kb.connect(db_path) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert tuple(conn.execute("SELECT id, title, body, workspace_path, claim_lock FROM tasks WHERE id=?", (task_id,)).fetchone()) == historical[0][:5]
        assert tuple(conn.execute("SELECT gate_scope_hash, gate_scope_version FROM tasks WHERE id=?", (task_id,)).fetchone()) == retained_scope
        assert tuple(conn.execute("SELECT id, task_id, run_id, kind, payload, created_at FROM task_events WHERE id=41").fetchone()) == historical[1]
        assert tuple(conn.execute("SELECT id, task_id, started_at, summary FROM task_runs WHERE id=73").fetchone()) == (historical[2][0], historical[2][1], historical[2][3], historical[2][4])


# ---------------------------------------------------------------------------
# Human-Gate v1 repair (2026-07-11 adversarial re-review REJECT: two
# reproduced bypasses in complete_task/reclaim_task, one structurally
# identical in promote_task — all closed via the shared
# _assert_human_gate_open() guard). See human-gate-design.md.
# ---------------------------------------------------------------------------

def test_repair_complete_task_refuses_gated_blocked_card(kanban_home):
    """Reviewer Finding #1 repro: complete_task/_complete_task_locked
    accepted blocked -> done with NO human_gate check at all — reachable
    via Dashboard PATCH status=done, the bulk-update endpoint, and a bare
    `hermes kanban complete` run outside a worker context."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="needs approval", kind="needs_input", human_gate=True)

        # Exact repro: complete_task with no expected_run_id (the bare-CLI
        # / dashboard shape), no token at all.
        with pytest.raises(kb.GateTokenError):
            kb.complete_task(conn, tid, result="done anyway")
        assert kb.get_task(conn, tid).status == "blocked"
        no_hash = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert no_hash["gate_token_hash"] is None  # nothing issued yet

        # Wrong token: also refused, and a bad guess must not mutate the
        # (now-issued) hash — a legitimate holder's token must still work.
        token = kb.issue_gate_token(conn, tid, action="complete")
        with pytest.raises(kb.GateTokenError):
            kb.complete_task(conn, tid, result="done anyway", token="wrong-guess")
        unchanged = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert unchanged["gate_token_hash"] == kb.hash_gate_token(token)
        assert kb.get_task(conn, tid).status == "blocked"

        # Positive path: the correct token completes it and consumes it.
        assert kb.complete_task(conn, tid, result="done for real", token=token) is True
        assert kb.get_task(conn, tid).status == "done"


def test_repair_complete_task_with_expected_run_id_also_gated(kanban_home):
    """The second (expected_run_id-pinned) UPDATE branch in
    _complete_task_locked must be gated too, not just the bare one.

    ``block_task`` always closes the run (nulling ``current_run_id``), so
    that path can't reach here with a matching ``expected_run_id`` once
    blocked. A real S4c resource-stall block (detect_resource_stalls's
    own UPDATE) deliberately leaves ``current_run_id`` set on the
    now-blocked task — reproduce that shape instead.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        run_id = claimed.current_run_id
        assert run_id is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked', block_kind='capability' "
                "WHERE id = ? AND status = 'running' AND current_run_id = ?",
                (tid, run_id),
            )
        assert kb.set_human_gate(conn, tid, on=True, actor="manfred") is True
        with pytest.raises(kb.GateTokenError):
            kb.complete_task(conn, tid, result="x", expected_run_id=run_id)
        assert kb.get_task(conn, tid).status == "blocked"


def test_repair_reclaim_task_refuses_stall_blocked_gated_card(kanban_home):
    """Reviewer Finding #2 repro: detect_resource_stalls's own UPDATE
    (kanban_db.py, S4c) sets status='blocked' WITHOUT clearing
    claim_lock/worker_pid. reclaim_task's early-return only checked
    `status != 'running'`, so such a card slipped past it (claim_lock is
    not None) straight back to 'ready' via the plain UPDATE, with no
    gate check at all. Refuse-only: no token parameter exists for
    reclaim; the only way past this is `kanban gate <id> off` first."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        # Reproduce detect_resource_stalls's exact update shape: status
        # flips to blocked, claim_lock/worker_pid are deliberately left
        # untouched (see kanban_db.py's detect_resource_stalls comment).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked', block_kind='capability' "
                "WHERE id = ? AND status = 'running'",
                (tid,),
            )
        row = conn.execute(
            "SELECT claim_lock FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["claim_lock"] is not None  # precondition for the exploit

        assert kb.set_human_gate(conn, tid, on=True, actor="manfred") is True

        with pytest.raises(kb.GateTokenError):
            kb.reclaim_task(conn, tid, reason="operator abort")
        assert kb.get_task(conn, tid).status == "blocked"
        still_locked = conn.execute(
            "SELECT claim_lock FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert still_locked["claim_lock"] is not None  # untouched by the refused attempt

        # Rescue path: gate off, then reclaim succeeds normally. H1 (2026-07-13):
        # gate off now itself requires a fresh gate_off grant; issue one directly
        # (the delivery surface is not under test here) and pass it.
        gate_off_token = kb.issue_gate_token(conn, tid, action="gate_off")
        assert gate_off_token is not None
        assert kb.set_human_gate(conn, tid, on=False, actor="manfred", token=gate_off_token) is True
        assert kb.reclaim_task(conn, tid, reason="operator abort") is True
        assert kb.get_task(conn, tid).status == "ready"


def test_gate_off_free_step4(kanban_home):
    """Step④ (2026-07-14): H1 removed — disabling a LIVE (blocked+gated) gate is now
    FREE (no grant). The human_gate is a soft, reversible hold; lifting it is bounded
    by the Step② budget + logged. It cannot bypass the exact-action approval (that is
    enforced by the pending_action/attention state machine, verified separately)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        assert kb.get_task(conn, tid).human_gate is True
        # No token needed anymore.
        assert kb.set_human_gate(conn, tid, on=False, actor="op") is True
        assert kb.get_task(conn, tid).human_gate is False


def test_gate_off_bulk_burst_hits_budget(kanban_home, monkeypatch):
    """Step④ + Step②: a gate-off is free, but an ad-hoc BURST still trips the
    mutation budget (blast-radius bound) — the structural replacement for H1."""
    monkeypatch.setenv("HERMES_KANBAN_MUTATION_RATE_LIMIT", "3")
    with kb.connect() as conn:
        ids = []
        for i in range(6):
            t = kb.create_task(conn, title=f"g{i}", assignee="worker")
            kb.block_task(conn, t, reason="x", kind="needs_input", human_gate=True)
            ids.append(t)
        done = 0
        tripped = False
        for t in ids:
            try:
                kb.set_human_gate(conn, t, on=False, actor="op")
                done += 1
            except kb.MutationBudgetError:
                tripped = True
                break
        assert done == 3 and tripped is True


def test_gate_off_free_when_not_live_gate_h1(kanban_home):
    """H1: a human_gate flag on a non-blocked card is inert; gate off stays free
    (no grant), matching _assert_human_gate_open's blocked/scheduled scope."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
        assert kb.set_human_gate(conn, tid, on=True, actor="op") is True  # gated but todo
        assert kb.set_human_gate(conn, tid, on=False, actor="op") is True
        assert kb.get_task(conn, tid).human_gate is False


def test_archive_running_free_step4(kanban_home):
    """Step④ (2026-07-14): H2 removed — archiving a RUNNING card no longer needs an
    archive_running grant. Step③ verified the reclaim is non-destructive (session/
    workspace/history preserved) and unarchive_task restores + resumes it, so nothing
    is irreversibly destroyed. Blast radius is bounded by the Step② budget."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="running", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        assert kb.get_task(conn, tid).status == "running"
        # No token needed anymore.
        assert kb.archive_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "archived"
        # The reclaimed run is preserved (recoverable), not destroyed.
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        assert run["outcome"] == "reclaimed"


def test_archive_nonrunning_is_free_h2(kanban_home):
    """H2: archiving a non-running, unbound card stays free (no grant required)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="todo", assignee="worker")
        assert kb.archive_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "archived"


def test_repair_promote_task_refuses_gated_blocked_card(kanban_home):
    """Reviewer Finding #3 (structurally identical): promote_task's
    UPDATE `WHERE status IN ('todo', 'blocked')` reached ready from
    blocked with no gate check."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="needs approval", kind="needs_input", human_gate=True)

        ok, err = kb.promote_task(conn, tid, actor="a", force=True)
        assert ok is False
        assert err is not None and "human-gated" in err
        assert kb.get_task(conn, tid).status == "blocked"
        no_hash = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert no_hash["gate_token_hash"] is None

        token = kb.issue_gate_token(conn, tid, action="promote")
        bad_ok, bad_err = kb.promote_task(conn, tid, actor="a", force=True, token="wrong")
        assert bad_ok is False
        assert bad_err is not None and "human-gated" in bad_err
        unchanged = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert unchanged["gate_token_hash"] == kb.hash_gate_token(token)

        ok2, err2 = kb.promote_task(conn, tid, actor="a", force=True, token=token)
        assert ok2 is True and err2 is None
        assert kb.get_task(conn, tid).status == "ready"


def test_repair_promote_task_dry_run_ignores_gate(kanban_home):
    """dry_run is a read-only preview of the parent-dependency gate only
    — it must never consume a token, and (documented scope boundary)
    doesn't report the human gate as a blocker either."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid, action="promote")
        ok, err = kb.promote_task(conn, tid, actor="a", force=True, dry_run=True, token=token)
        assert ok is True and err is None
        # Token must survive a dry_run untouched (never spent by a preview).
        row = conn.execute(
            "SELECT gate_token_hash FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["gate_token_hash"] == kb.hash_gate_token(token)
        assert kb.get_task(conn, tid).status == "blocked"


def test_repair_schedule_task_refuses_gated_blocked_card(kanban_home):
    """Self-audit finding: schedule_task's UPDATE `WHERE status IN
    ('todo','ready','running','blocked')` let a gated card be parked in
    'scheduled' with no gate check, "laundering" it out of the blocked
    column with no authorization at all. Refuse-only (no token param)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        with pytest.raises(kb.GateTokenError):
            kb.schedule_task(conn, tid, reason="park it")
        assert kb.get_task(conn, tid).status == "blocked"


def test_repair_archive_task_refuses_gated_blocked_card(kanban_home):
    """Self-audit finding: archive_task's UPDATE `WHERE status !=
    'archived'` disposed of a gated blocked card with no gate check —
    no human decision ever required. Refuse-only (no token param)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        with pytest.raises(kb.GateTokenError):
            kb.archive_task(conn, tid)
        assert kb.get_task(conn, tid).status == "blocked"


def test_repair_recompute_ready_skips_gated_card_even_when_not_sticky(kanban_home):
    """Self-audit finding: recompute_ready's blocked -> ready auto-promotion
    already skipped "sticky" blocks (an explicit kanban_block event), but
    a card blocked via direct DB manipulation (no 'blocked' event — e.g.
    the circuit breaker path, or any future writer) and THEN gated via
    `kanban gate <id> on` was not sticky and would have been silently
    auto-promoted once its parent finished, with no gate check."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker", parents=(parent,))
        kb.claim_task(conn, parent)
        assert kb.complete_task(conn, parent, result="done") is True

        kb.claim_task(conn, child)
        # Direct manipulation, deliberately bypassing block_task so NO
        # 'blocked' event is emitted — the pre-existing non-sticky case.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (child,))
        assert kb._has_sticky_block(conn, child) is False  # sanity: genuinely not sticky

        assert kb.set_human_gate(conn, child, on=True, actor="manfred") is True
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "blocked"


def test_repair_cli_complete_and_promote_token_flags(kanban_home):
    import argparse
    from hermes_cli import kanban as kanban_cli

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker")
        kb.block_task(conn, tid, reason="x", kind="needs_input", human_gate=True)
        token = kb.issue_gate_token(conn, tid, action="complete")

    rc_fail = kanban_cli._cmd_complete(argparse.Namespace(
        task_ids=[tid], result="x", summary=None, metadata=None, token=None,
    ))
    assert rc_fail == 1
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"

    rc_ok = kanban_cli._cmd_complete(argparse.Namespace(
        task_ids=[tid], result="x", summary=None, metadata=None, token=token,
    ))
    assert rc_ok == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"

    tid2 = None
    with kb.connect() as conn:
        tid2 = kb.create_task(conn, title="gated2", assignee="worker")
        kb.block_task(conn, tid2, reason="x", kind="needs_input", human_gate=True)
        token2 = kb.issue_gate_token(conn, tid2, action="promote")

    rc_fail2 = kanban_cli._cmd_promote(argparse.Namespace(
        task_id=tid2, reason=[], ids=None, force=True, dry_run=False,
        json=False, token=None,
    ))
    assert rc_fail2 == 1
    rc_ok2 = kanban_cli._cmd_promote(argparse.Namespace(
        task_id=tid2, reason=[], ids=None, force=True, dry_run=False,
        json=False, token=token2,
    ))
    assert rc_ok2 == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid2).status == "ready"


# ---------------------------------------------------------------------------
# Audit 2026-07-11 H1/H2: wedged-worker respawn brake + per-run crash grace
# ---------------------------------------------------------------------------

def _wedge_and_reclaim(conn, monkeypatch, t):
    """Make task t look wedged (pid alive, heartbeat stale >1h, TTL expired)
    and run release_stale_claims with a successful termination stub."""
    import hermes_cli.kanban_db as _kb
    kb._set_worker_pid(conn, t, 12345)
    conn.execute(
        "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
        (int(time.time()) - 60, int(time.time()) - 7200, t),
    )
    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        _kb, "_terminate_reclaimed_worker",
        lambda *a, **k: {
            "termination_attempted": True,
            "host_local": True,
            "terminated": True,
        },
    )
    return kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None)


def test_wedged_reclaim_counts_as_failure(kanban_home, monkeypatch):
    """H1: a wedged-but-alive worker (pid up, heartbeat stale >1h) that gets
    reclaimed must increment the failure counter — before the fix the same
    structural wedge respawned forever and the circuit breaker never tripped."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="wedges forever", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        reclaimed = _wedge_and_reclaim(conn, monkeypatch, t)
        assert reclaimed == 1
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["consecutive_failures"] == 1
        assert row["status"] == "ready"  # first wedge: retry allowed


def test_wedged_reclaim_trips_circuit_breaker(kanban_home, monkeypatch):
    """H1: repeated wedge-reclaims must eventually auto-block (gave_up),
    not respawn unboundedly."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="wedges forever", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        for _ in range(kb.DEFAULT_FAILURE_LIMIT):
            claimed = kb.claim_task(conn, t, claimer=f"{host}:worker")
            assert claimed is not None
            assert _wedge_and_reclaim(conn, monkeypatch, t) == 1
        task = kb.get_task(conn, t)
        # Automation-first (2026-07-16): first trip routes to triage.
        assert task.status == "triage"
        kinds = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (t,),
            ).fetchall()
        ]
        assert "gave_up" in kinds


def test_ttl_reclaim_of_dead_pid_does_not_count_failure(kanban_home, monkeypatch):
    """Scope guard for H1: an ordinary TTL reclaim of a DEAD worker keeps its
    existing semantics (no failure accounting here — the crash path owns that)."""
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="died quietly", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 12345)
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 60, t),
        )
        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (t,),
        ).fetchone()
        assert row["status"] == "ready"
        assert (row["consecutive_failures"] or 0) == 0


def test_crash_grace_measured_from_active_run(kanban_home, monkeypatch):
    """H2: the launch-window crash grace must be measured from the ACTIVE
    run, not tasks.started_at (frozen at the first-ever claim) — otherwise
    the grace is a no-op from the first respawn on, exactly the case it
    exists for."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "3600")
    with kb.connect() as conn:
        t = kb.create_task(conn, title="respawned", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        # First claim happened long ago (tasks.started_at is frozen there) …
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 7200, t),
        )
        # … the ACTIVE run however started just now (fresh respawn).
        kb._set_worker_pid(conn, t, 999999)
        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        crashed = kb.detect_crashed_workers(conn)
        # Fresh respawn is inside the grace window → NOT reclaimed as crashed.
        assert crashed == []
        assert kb.get_task(conn, t).status == "running"


# ---------------------------------------------------------------------------
# C4 (Audit 2026-07-11): worker spawn env must not leak notification secrets
# ---------------------------------------------------------------------------

def test_spawn_strips_notification_secrets(kanban_home, monkeypatch):
    """NTFY_* serves the Human-Gate/attention channel; a worker must not
    hold it. GITEA_TOKEN stays (workers push branches/PRs — accepted)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("NTFY_TOKEN", "sekrit")
    monkeypatch.setenv("NTFY_ADMIN_TOKEN", "sekrit2")
    monkeypatch.setenv("GITEA_TOKEN", "needed-for-pushes")
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            self.pid = 4242

    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="coder")
        task = kb.get_task(conn, t)
    _kb._default_spawn(task, str(kanban_home))

    env = captured["env"]
    assert "NTFY_TOKEN" not in env
    assert "NTFY_ADMIN_TOKEN" not in env
    assert env.get("GITEA_TOKEN") == "needed-for-pushes"


def test_spawn_env_strip_extendable_via_config(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("MY_EXTRA_SECRET", "x")
    import hermes_cli.config as _cfg
    monkeypatch.setattr(
        _cfg, "load_config",
        lambda *a, **k: {"kanban": {"worker_env_strip": ["MY_EXTRA_SECRET"]}},
    )
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            self.pid = 4242

    monkeypatch.setattr("subprocess.Popen", _FakePopen)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="coder")
        task = kb.get_task(conn, t)
    _kb._default_spawn(task, str(kanban_home))
    assert "MY_EXTRA_SECRET" not in captured["env"]


def test_complete_task_rejects_missing_declared_scratch_artifact(kanban_home):
    """A declared scratch deliverable must not disappear behind a false Done."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="missing report")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, t, ws)
        missing = ws / "report.md"

        # Choice A (2026-07-14 merge): our completion-artifact EVIDENCE validation runs
        # before the write txn and fails closed on a non-existent deliverable, pre-empting
        # upstream's in-txn ArtifactPreservationError. Same safety outcome (no false Done,
        # scratch kept for retry), our exception type.
        with pytest.raises(kb.CompletionEvidenceError, match="does not exist"):
            kb.complete_task(
                conn,
                t,
                result="report complete",
                metadata={"artifacts": [str(missing)]},
            )

        assert kb.get_task(conn, t).status == "ready"
        assert kb.list_attachments(conn, t) == []
    assert ws.exists(), "failed completion must keep scratch available for retry"


def test_complete_task_rejects_out_of_root_artifact(
    kanban_home,
    tmp_path,
):
    """Choice A (2026-07-14 merge): our completion-artifact EVIDENCE validation is kept,
    so a declared artifact OUTSIDE the allowed roots (the managed workspace / board
    artifact roots) fails closed — even if the file really exists. This deliberately
    diverges from upstream's opaque model (which let external paths ride into the payload
    unchanged): under our posture, workers must place deliverables inside their workspace.
    The task stays in-flight and the external file is left untouched."""
    external = tmp_path / "report.md"
    external.write_text("keep me here", encoding="utf-8")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="external report")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, t, ws)

        with pytest.raises(
            kb.CompletionEvidenceError, match="outside allowed roots"
        ):
            kb.complete_task(
                conn,
                t,
                result="ok",
                metadata={"artifacts": [str(external)]},
            )

        assert kb.get_task(conn, t).status == "ready"
        assert kb.list_attachments(conn, t) == []

    assert external.exists(), "a rejected completion must not touch the external file"
    assert ws.exists(), "failed completion keeps the scratch workspace for retry"

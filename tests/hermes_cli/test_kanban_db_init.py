from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _make_legacy_db(path: Path) -> None:
    """Write a kanban DB with the pre-AUTOINCREMENT (TEXT PK) schema for the
    four tables #35096 affects, keeping every other table current so the
    additive-column migration runs cleanly on top.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript(
        """
        DROP TABLE task_events;
        DROP TABLE task_comments;
        DROP TABLE task_runs;
        DROP TABLE kanban_notify_subs;
        CREATE TABLE task_comments (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            author TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL);
        CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL);
        CREATE TABLE task_runs (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL);
        CREATE TABLE kanban_notify_subs (task_id TEXT NOT NULL, platform TEXT NOT NULL,
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,
            created_at INTEGER NOT NULL, last_event_id TEXT,
            PRIMARY KEY (task_id, platform, chat_id, thread_id));
        """
    )
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('task-1', 'T', 'done', 1000)")
    conn.execute("INSERT INTO task_comments VALUES ('c-1', 'task-1', 'agent', 'hi', 1500)")
    conn.execute("INSERT INTO task_events VALUES ('e-1', 'task-1', 'completed', NULL, 2000)")
    conn.execute("INSERT INTO task_events VALUES ('e-2', 'task-1', 'blocked', NULL, 2100)")
    conn.execute("INSERT INTO task_runs VALUES ('r-1', 'task-1', 'default', 'done', 1000)")
    conn.execute(
        "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, created_at, last_event_id) "
        "VALUES ('task-1', 'telegram', '123', 1000, 'e-1')"
    )
    conn.commit()
    conn.close()


def _setup_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="legacy")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return db_path


def _table_struct(conn: sqlite3.Connection, table: str):
    cols = [
        (r["name"], (r["type"] or "").upper(), r["notnull"], r["pk"])
        for r in conn.execute(f"PRAGMA table_info({table})")
    ]
    idx = sorted(
        r["name"]
        for r in conn.execute(f"PRAGMA index_list({table})")
        if not r["name"].startswith("sqlite_")
    )
    return cols, idx




def test_legacy_text_pk_tables_rebuilt_to_integer_autoincrement(tmp_path, monkeypatch):
    """A pre-AUTOINCREMENT DB is migrated in place: id columns become INTEGER
    PKs, ``last_event_id`` becomes INTEGER, data is preserved, and indexes
    are recreated (DROP TABLE would otherwise take them down)."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path) as conn:
        for table in ("task_events", "task_comments", "task_runs"):
            id_col = {r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")}["id"]
            assert id_col["type"].upper() == "INTEGER" and id_col["pk"] == 1

        lei = {r["name"]: r for r in conn.execute("PRAGMA table_info(kanban_notify_subs)")}
        assert lei["last_event_id"]["type"].upper() == "INTEGER"
        assert "delivery_metadata" in lei

        # Data preserved across the rebuild.
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2
        assert conn.execute("SELECT body FROM task_comments").fetchone()["body"] == "hi"
        assert len(conn.execute("SELECT * FROM task_runs").fetchall()) == 1
        # Non-numeric legacy cursor ("e-1") casts to 0.
        assert conn.execute("SELECT last_event_id FROM kanban_notify_subs").fetchone()["last_event_id"] == 0

        # Indexes restored, including idx_events_run (added by the additive pass).
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        for name in ("idx_events_task", "idx_events_run", "idx_comments_task",
                     "idx_runs_task", "idx_runs_status", "idx_notify_task"):
            assert name in indexes

        # AUTOINCREMENT actually works after the rebuild.
        conn.execute("INSERT INTO task_events (task_id, kind, created_at) VALUES ('task-1', 'completed', 3000)")
        new_id = conn.execute("SELECT id FROM task_events ORDER BY id DESC LIMIT 1").fetchone()["id"]
        assert isinstance(new_id, int) and new_id >= 1




def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Re-opening an already-migrated DB is a no-op and leaves data intact."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path):
        pass
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        id_col = {r["name"]: r for r in conn.execute("PRAGMA table_info(task_events)")}["id"]
        assert id_col["type"].upper() == "INTEGER"
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2


def test_unseen_events_for_sub_survives_migrated_db(tmp_path, monkeypatch):
    """The crash that motivated #35096 — ``int(None)`` on a NULL cursor — is
    gone after migration; the notifier query returns an integer cursor."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path) as conn:
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id="task-1", platform="telegram", chat_id="123"
        )
        assert isinstance(cursor, int)
        assert isinstance(events, list)


def _make_pre_typed_attention_db(path: Path, *, duplicate_projection: bool = False, unbound_action: bool = False) -> None:
    """Create the immediately preceding attention schema via sqlite DDL."""
    conn = sqlite3.connect(path)
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript("""
        DROP TABLE task_attentions;
        DROP TABLE task_pending_actions;
        CREATE TABLE task_pending_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, run_id INTEGER,
            command_hash TEXT NOT NULL, fingerprint TEXT NOT NULL, mutation_kind TEXT NOT NULL,
            profile TEXT NOT NULL, workspace TEXT NOT NULL, summary TEXT NOT NULL,
            state TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
            expires_at INTEGER NOT NULL, approved_at INTEGER, approved_by TEXT,
            consumed_at INTEGER, version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE task_attentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            action_id INTEGER NOT NULL UNIQUE, type TEXT NOT NULL,
            cause_fingerprint TEXT NOT NULL, summary TEXT NOT NULL, created_at INTEGER NOT NULL,
            UNIQUE(task_id, cause_fingerprint)
        );
    """)
    conn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('legacy-attention', 'legacy', 'blocked', 1)")
    conn.execute("""INSERT INTO task_pending_actions
        (id, task_id, command_hash, fingerprint, mutation_kind, profile, workspace, summary, state, created_at, updated_at, expires_at)
        VALUES (41, 'legacy-attention', 'h', 'f', 'test', 'default', '/tmp', 'operator', 'pending', 1, 1, 2000000000)""")
    if not unbound_action:
        conn.execute("INSERT INTO task_attentions VALUES (777, 'legacy-attention', 41, 'exact_action', 'legacy-exact', 'operator', 1)")
    if duplicate_projection:
        conn.executescript("DROP TABLE task_attentions; CREATE TABLE task_attentions (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, action_id INTEGER NOT NULL, type TEXT NOT NULL, cause_fingerprint TEXT NOT NULL, summary TEXT NOT NULL, created_at INTEGER NOT NULL);")
        conn.executemany("INSERT INTO task_attentions VALUES (?, 'legacy-attention', 41, 'exact_action', ?, 'operator', 1)", [(777, 'legacy-exact'), (778, 'legacy-duplicate')])
    conn.commit()
    conn.close()


def _attention_snapshot(path: Path) -> dict[str, object]:
    """Capture affected rows and schema before a fail-closed migration attempt."""
    conn = sqlite3.connect(path)
    try:
        tables = ("task_attentions", "task_pending_actions")
        return {
            "rows": {
                table: conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                for table in tables
            },
            "table_info": {
                table: conn.execute(f"PRAGMA table_info({table})").fetchall()
                for table in tables
            },
            "index_list": {
                table: conn.execute(f"PRAGMA index_list({table})").fetchall()
                for table in tables
            },
            "sqlite_master": conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE tbl_name IN (?, ?) OR name='task_attentions_rebuild' ORDER BY type, name",
                tables,
            ).fetchall(),
        }
    finally:
        conn.close()


def test_pre_typed_attention_migration_preserves_ids_backfills_and_reinits(tmp_path, monkeypatch):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_pre_typed_attention_db(db_path)
    with kb.connect(db_path) as conn:
        projection = conn.execute("SELECT id, action_id, version, origin_run_id FROM task_attentions").fetchone()
        action = conn.execute("SELECT attention_id FROM task_pending_actions WHERE id=41").fetchone()
        columns = {row["name"]: row for row in conn.execute("PRAGMA table_info(task_attentions)")}
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(task_attentions)")}
        assert tuple(projection) == (777, 41, 1, None)
        assert action["attention_id"] == 777
        assert columns["action_id"]["notnull"] == 0
        assert {"uq_task_attentions_action_live", "uq_task_attentions_task_cause", "idx_task_attentions_task_created"} <= indexes
        assert "uq_pending_actions_attention_id" in {row["name"] for row in conn.execute("PRAGMA index_list(task_pending_actions)")}
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    kb.init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM task_attentions").fetchone()[0] == 777
        assert conn.execute("SELECT attention_id FROM task_pending_actions WHERE id=41").fetchone()[0] == 777
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_pre_typed_attention_migration_rejects_ambiguous_drift_without_partial_rebuild(tmp_path, monkeypatch):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_pre_typed_attention_db(db_path, duplicate_projection=True)
    before = _attention_snapshot(db_path)
    with pytest.raises(RuntimeError, match="duplicate non-null action_id|ambiguous exact attention"):
        kb.connect(db_path)
    assert _attention_snapshot(db_path) == before


def test_pre_typed_attention_legacy_action_without_unique_projection_stays_unbound(tmp_path, monkeypatch):
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_pre_typed_attention_db(db_path, unbound_action=True)
    with kb.connect(db_path) as conn:
        assert conn.execute("SELECT attention_id FROM task_pending_actions WHERE id=41").fetchone()[0] is None
        assert conn.execute("SELECT COUNT(*) FROM task_attentions").fetchone()[0] == 0

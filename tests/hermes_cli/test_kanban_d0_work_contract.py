"""D0: additive work-contract columns and board identity (ADR-1).

Two things are under test, and they fail in different ways:

* The **columns** are expand-only. The risk is not that they fail to appear
  but that they quietly change existing rows -- a back-filled default would
  make a legacy card indistinguishable from one that genuinely declared its
  terms.
* The **board uuid** is an identity. The risk is that it changes. Every
  ``work_uid`` minted against a board dies the moment its uuid is rewritten,
  silently and all at once, so the guard has to be mechanical rather than a
  convention.

ADR-1 asks for UPDATE, restore, clone and double-mount to be covered
separately, because they look alike and mean different things.
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from hermes_cli import kanban_db


D0_COLUMNS = [
    "work_kind", "standing_state", "owner_core_id", "origin_kind",
    "origin_key", "origin_conversation_ref", "payload_digest",
    "acceptance_required", "review_policy_version", "authority_epoch",
    "writer_mode", "commitment", "commitment_due_at", "definition_of_done",
    "route_tenant_snapshot",
]


def _columns(conn: sqlite3.Connection, table: str = "tasks") -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture
def board(tmp_path):
    """A freshly initialised board database."""
    db = tmp_path / "kanban.db"
    conn = kanban_db.connect(db)
    yield conn
    conn.close()


# ------------------------------------------------------------ the columns


def test_all_work_contract_columns_exist(board):
    cols = _columns(board)
    missing = [c for c in D0_COLUMNS if c not in cols]
    assert not missing, f"D0 columns missing: {missing}"


def test_migration_is_idempotent(board):
    """``init_db`` runs on every open; a second pass must be a no-op."""
    before = _columns(board)
    kanban_db._migrate_add_optional_columns(board)
    kanban_db._migrate_add_optional_columns(board)
    assert _columns(board) == before


def test_legacy_rows_get_null_not_a_guessed_default(tmp_path):
    """The point of NULL.

    A legacy card predates the contract. Back-filling ``acceptance_required=0``
    would assert that it needs no review; back-filling 1 would assert it does.
    Both are claims the migration is not entitled to make.
    """
    db = tmp_path / "legacy.db"
    conn = kanban_db.connect(db)

    # A row created before the contract existed: strip the D0 values.
    task_id = kanban_db.create_task(conn, title="legacy card", body="from before")
    conn.execute(
        "UPDATE tasks SET " + ", ".join(f"{c}=NULL" for c in D0_COLUMNS)
        + " WHERE id=?", (task_id,))
    conn.commit()

    kanban_db._migrate_add_optional_columns(conn)

    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    for col in D0_COLUMNS:
        assert row[col] is None, f"{col} was back-filled with {row[col]!r}"
    assert row["title"] == "legacy card"
    conn.close()


def test_existing_rows_survive_the_migration(board):
    ids = [kanban_db.create_task(board, title=f"t{i}") for i in range(5)]
    board.commit()
    before = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    kanban_db._migrate_add_optional_columns(board)

    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
    for task_id in ids:
        assert board.execute(
            "SELECT id FROM tasks WHERE id=?", (task_id,)).fetchone() is not None


# ---------------------------------------------------- origin_key uniqueness


def test_origin_key_is_unique_per_origin_kind(board):
    """ADR-4: the dedup key replaces the racy check-then-insert."""
    a = kanban_db.create_task(board, title="first")
    b = kanban_db.create_task(board, title="second")
    board.execute(
        "UPDATE tasks SET origin_kind='cron', origin_key='job-42' WHERE id=?", (a,))
    board.commit()

    with pytest.raises(sqlite3.IntegrityError):
        board.execute(
            "UPDATE tasks SET origin_kind='cron', origin_key='job-42' WHERE id=?",
            (b,))
        board.commit()


def test_same_key_under_a_different_origin_kind_is_allowed(board):
    """The key is only meaningful within its producer's namespace."""
    a = kanban_db.create_task(board, title="from cron")
    b = kanban_db.create_task(board, title="from webhook")
    board.execute(
        "UPDATE tasks SET origin_kind='cron', origin_key='shared' WHERE id=?", (a,))
    board.execute(
        "UPDATE tasks SET origin_kind='webhook', origin_key='shared' WHERE id=?", (b,))
    board.commit()  # must not raise


def test_many_legacy_null_keys_do_not_collide(board):
    """Why the index has to be partial.

    A plain UNIQUE would treat every legacy row as a duplicate of every other
    legacy row and make the migration unrunnable on a real board.
    """
    for i in range(20):
        kanban_db.create_task(board, title=f"legacy {i}")
    board.commit()
    n = board.execute(
        "SELECT COUNT(*) FROM tasks WHERE origin_key IS NULL").fetchone()[0]
    assert n == 20


# ------------------------------------------------------- board identity


def test_board_uuid_is_created_once_and_is_stable(board):
    first = kanban_db.get_board_uuid(board)
    assert first
    kanban_db._migrate_board_identity(board)
    kanban_db._migrate_add_optional_columns(board)
    assert kanban_db.get_board_uuid(board) == first, \
        "re-running the migration minted a new identity"


def test_board_uuid_update_is_mechanically_refused(board):
    """Negative case 1: a direct UPDATE.

    This is the one that would go unnoticed -- it succeeds silently on a
    plain table and invalidates every existing work_uid.
    """
    original = kanban_db.get_board_uuid(board)
    with pytest.raises(sqlite3.IntegrityError):
        board.execute(
            "UPDATE board_meta SET value='hijacked' WHERE key='board_uuid'")
    assert kanban_db.get_board_uuid(board) == original


def test_board_uuid_delete_is_refused(board):
    """Deleting and re-inserting would be an UPDATE with extra steps."""
    original = kanban_db.get_board_uuid(board)
    with pytest.raises(sqlite3.IntegrityError):
        board.execute("DELETE FROM board_meta WHERE key='board_uuid'")
    assert kanban_db.get_board_uuid(board) == original


def test_other_board_meta_keys_stay_writable(board):
    """The guard is scoped to the identity, not to the whole table."""
    board.execute(
        "INSERT INTO board_meta (key, value) VALUES ('display_name', 'Ops')")
    board.execute(
        "UPDATE board_meta SET value='Operations' WHERE key='display_name'")
    assert board.execute(
        "SELECT value FROM board_meta WHERE key='display_name'"
    ).fetchone()[0] == "Operations"


def test_restore_of_the_same_board_keeps_its_uuid(tmp_path):
    """Negative case 2: restore is not re-identification.

    A file-level restore of the same logical board must keep the uuid, or
    every work_uid reference breaks after disaster recovery -- turning a
    successful restore into silent data corruption at the reference layer.
    """
    src = tmp_path / "board.db"
    conn = kanban_db.connect(src)
    original = kanban_db.get_board_uuid(conn)
    conn.close()

    backup = tmp_path / "backup.db"
    shutil.copy2(src, backup)
    src.unlink()
    shutil.copy2(backup, src)

    conn = kanban_db.connect(src)  # reopening runs the migration again
    assert kanban_db.get_board_uuid(conn) == original
    conn.close()


def test_two_independently_created_boards_get_different_uuids(tmp_path):
    """Negative case 3: a clone that is meant to be a second board.

    Creating a board from scratch must never reuse another board's identity;
    otherwise the resolver sees one uuid on two paths.
    """
    uuids = set()
    for name in ("a", "b", "c"):
        conn = kanban_db.connect(tmp_path / f"{name}.db")
        uuids.add(kanban_db.get_board_uuid(conn))
        conn.close()
    assert len(uuids) == 3


def test_file_copy_duplicates_the_uuid_which_is_why_reidentify_exists(tmp_path):
    """Negative case 4: the double-mount hazard, stated honestly.

    Copying a board file copies its identity -- that is *correct* for restore
    and *wrong* for a clone meant to be used alongside the original. The
    migration cannot tell the two apart from inside the file, which is exactly
    why ADR-1 puts re-identification in an explicit tool step before first
    mount, and why the resolver must fail closed on a duplicate rather than
    pick one.

    This test pins the hazard so the resolver work in D1 cannot forget it.
    """
    src = tmp_path / "orig.db"
    conn = kanban_db.connect(src)
    original = kanban_db.get_board_uuid(conn)
    conn.close()

    clone = tmp_path / "clone.db"
    shutil.copy2(src, clone)
    conn = kanban_db.connect(clone)
    assert kanban_db.get_board_uuid(conn) == original, \
        "a file copy shares the identity -- reidentify is mandatory before mount"
    conn.close()


def test_work_uid_is_namespaced_by_board(board):
    uid = kanban_db.build_work_uid(kanban_db.get_board_uuid(board), "t_abcd1234")
    assert uid.startswith("kanban:")
    assert uid.endswith(":t_abcd1234")
    assert kanban_db.get_board_uuid(board) in uid


def test_identical_task_ids_on_two_boards_yield_distinct_work_uids(tmp_path):
    """The concrete reason work_uid is namespaced.

    Task ids are ``"t_" + token_hex(4)`` generated per board with a single
    in-database retry. Two boards colliding is not hypothetical, and nothing
    in the schema detects it.
    """
    uids = []
    for name in ("a", "b"):
        conn = kanban_db.connect(tmp_path / f"{name}.db")
        uids.append(kanban_db.build_work_uid(
            kanban_db.get_board_uuid(conn), "t_deadbeef"))
        conn.close()
    assert uids[0] != uids[1]

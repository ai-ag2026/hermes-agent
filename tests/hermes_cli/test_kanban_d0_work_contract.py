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


# ----------------------------------------------- D0 DoD: origin_key race


def test_two_creators_with_the_same_origin_key_yield_exactly_one_item(tmp_path):
    """The DoD race test: two creators, one key, one item.

    This is the scenario the old ``idempotency_key`` check-then-insert loses.
    Two threads read "no such key", both decide to insert, and two cards for
    one request appear. The unique index moves the decision into SQLite, where
    it is atomic: the loser gets IntegrityError instead of a duplicate.
    """
    import threading

    db = tmp_path / "race.db"
    kanban_db.connect(db).close()  # initialise once, outside the race

    created, conflicts, errors = [], [], []
    barrier = threading.Barrier(8)

    def creator(n: int) -> None:
        conn = kanban_db.connect(db)
        try:
            task_id = kanban_db.create_task(conn, title=f"creator {n}")
            barrier.wait(timeout=10)  # maximise overlap on the write
            try:
                conn.execute(
                    "UPDATE tasks SET origin_kind='cron', origin_key='the-one-key' "
                    "WHERE id=?", (task_id,))
                conn.commit()
                created.append(task_id)
            except sqlite3.IntegrityError:
                conn.rollback()
                conflicts.append(task_id)
        except Exception as exc:  # pragma: no cover - surfaces real breakage
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=creator, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"unexpected errors: {errors}"
    assert len(created) == 1, f"expected exactly one winner, got {len(created)}"
    assert len(conflicts) == 7

    conn = kanban_db.connect(db)
    n = conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE origin_key='the-one-key'").fetchone()[0]
    conn.close()
    assert n == 1, "the unique index did not hold under concurrency"


# ------------------------------------ D0 DoD: acceptance_required bypass


def test_acceptance_required_blocks_direct_completion(board):
    """The negative test the DoD asks for.

    A card marked ``acceptance_required`` must not reach ``done`` through the
    generic completion path -- which is where the dashboard PATCH, bulk update
    and bare CLI completion all converge.
    """
    task_id = kanban_db.create_task(board, title="needs acceptance")
    board.execute("UPDATE tasks SET acceptance_required=1 WHERE id=?", (task_id,))
    board.commit()

    assert kanban_db.complete_task(board, task_id, result="done by worker") is False
    assert kanban_db.get_task(board, task_id).status != "done"

    kinds = [r[0] for r in board.execute(
        "SELECT kind FROM task_events WHERE task_id=?", (task_id,))]
    assert "completion_blocked_acceptance_required" in kinds, \
        "the refusal must be auditable, not silent"


def test_acceptance_required_absent_or_zero_is_a_no_op(board):
    """The guard must not change behaviour for the existing corpus."""
    for value in (None, 0):
        task_id = kanban_db.create_task(board, title=f"legacy {value}")
        board.execute(
            "UPDATE tasks SET acceptance_required=?, status='ready' WHERE id=?",
            (value, task_id))
        board.commit()
        assert kanban_db.complete_task(board, task_id, result="ok") is True


def test_no_decision_yet_counts_as_not_accepted(board):
    """The default direction matters more than the mechanism.

    "No decision on record" must read as *not accepted*. Reading it as *not
    refused* would make the flag look like a safeguard while letting
    everything through.
    """
    task_id = kanban_db.create_task(board, title="awaiting review")
    board.execute("UPDATE tasks SET acceptance_required=1 WHERE id=?", (task_id,))
    board.commit()
    assert kanban_db._acceptance_satisfied(board, task_id) is False


def test_a_reject_decision_does_not_satisfy_acceptance(board):
    task_id = kanban_db.create_task(board, title="rejected")
    board.execute("UPDATE tasks SET acceptance_required=1 WHERE id=?", (task_id,))
    board.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?,?,?,strftime('%s','now'))",
        (task_id, "review_decided", '{"decision": "REJECT"}'))
    board.commit()
    assert kanban_db._acceptance_satisfied(board, task_id) is False


def test_an_accept_decision_satisfies_acceptance(board):
    task_id = kanban_db.create_task(board, title="accepted")
    board.execute("UPDATE tasks SET acceptance_required=1 WHERE id=?", (task_id,))
    board.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?,?,?,strftime('%s','now'))",
        (task_id, "review_decided", '{"decision": "ACCEPT"}'))
    board.commit()
    assert kanban_db._acceptance_satisfied(board, task_id) is True


def test_a_corrupt_decision_payload_is_not_an_acceptance(board):
    """Failing open on a malformed payload would be a universal bypass."""
    task_id = kanban_db.create_task(board, title="corrupt")
    board.execute("UPDATE tasks SET acceptance_required=1 WHERE id=?", (task_id,))
    board.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?,?,?,strftime('%s','now'))",
        (task_id, "review_decided", "{not json"))
    board.commit()
    assert kanban_db._acceptance_satisfied(board, task_id) is False


# ------------------------------- R5: the D0 -> D1 intermediate upgrade path


def test_an_unscoped_origin_index_is_replaced_not_kept(tmp_path):
    """The upgrade bug an independent review reproduced.

    An intermediate revision created ``idx_tasks_origin_key`` with an
    UNSCOPED definition. ``CREATE UNIQUE INDEX IF NOT EXISTS`` is a no-op
    against an existing name, so a board opened once by that revision kept the
    wrong constraint -- and ``quick_check=ok`` plus an intact task count
    concealed it completely. A drop targeting a *different* name never fired.

    Fresh boards hid this: they have no prior index, so the correct one is
    created and everything looks right. Only a board that passed through the
    intermediate revision shows it.

    Names are not definitions. The migration now reads the stored SQL.
    """
    db = tmp_path / "upgraded.db"
    conn = kanban_db.connect(db)

    # Recreate exactly what the intermediate revision left behind.
    conn.execute("DROP INDEX IF EXISTS idx_tasks_origin_key")
    conn.execute(
        "CREATE UNIQUE INDEX idx_tasks_origin_key "
        "ON tasks(origin_kind, origin_key) "
        "WHERE origin_key IS NOT NULL AND origin_kind IS NOT NULL")
    conn.commit()
    stale = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='idx_tasks_origin_key'"
    ).fetchone()[0]
    assert "owner_core_id" not in stale
    conn.close()

    # ``connect()`` caches initialised paths per process, so reopening in the
    # same process would skip the migration entirely -- which is why the
    # manual two-process reproduction of this bug behaved differently from a
    # single-process test. ``init_db()`` is the documented way to force the
    # migration pass, and it is what a restarted gateway effectively does.
    kanban_db.init_db(db)
    conn = kanban_db.connect(db)
    repaired = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='idx_tasks_origin_key'"
    ).fetchone()[0]
    assert "owner_core_id" in repaired, \
        "the unscoped index survived the upgrade"

    # And the repaired constraint actually behaves as scoped.
    a = kanban_db.create_task(conn, title="core a")
    b = kanban_db.create_task(conn, title="core b")
    conn.execute("UPDATE tasks SET owner_core_id='core-a', origin_kind='cron', "
                 "origin_key='shared' WHERE id=?", (a,))
    conn.execute("UPDATE tasks SET owner_core_id='core-b', origin_kind='cron', "
                 "origin_key='shared' WHERE id=?", (b,))
    conn.commit()  # must not raise
    conn.close()


def test_the_repair_is_idempotent(tmp_path):
    """Reopening must not churn an already-correct index."""
    db = tmp_path / "stable.db"
    conn = kanban_db.connect(db)
    first = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='idx_tasks_origin_key'"
    ).fetchone()[0]
    conn.close()

    for _ in range(3):
        kanban_db.init_db(db)
        conn = kanban_db.connect(db)
        again = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='idx_tasks_origin_key'"
        ).fetchone()[0]
        conn.close()
        assert again == first

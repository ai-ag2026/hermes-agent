"""Board resolution: the duplicate case is the point.

Resolving a unique uuid is arithmetic. What this module has to get right is
what it does when the answer is *not* unique -- because that is the state a
copied-and-mounted board file produces, and the tempting behaviours (pick the
first, pick the newest, pick the shortest path) all route work to the wrong
board while looking like they worked.
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from hermes_cli import board_resolver, kanban_db


def _make_board(path):
    conn = kanban_db.connect(path)
    board_uuid = kanban_db.get_board_uuid(conn)
    conn.close()
    return board_uuid


# ---------------------------------------------------------------- parsing


def test_parses_a_well_formed_work_uid():
    board, task = board_resolver.parse_work_uid(
        "kanban:123e4567-e89b-12d3-a456-426614174000:t_deadbeef")
    assert board == "123e4567-e89b-12d3-a456-426614174000"
    assert task == "t_deadbeef"


@pytest.mark.parametrize("bad", [
    "",
    "t_deadbeef",                       # bare task id
    "kanban::t_deadbeef",               # empty board
    "kanban:not-a-uuid:t_deadbeef",
    "kanban:123e4567-e89b-12d3-a456-426614174000:deadbeef",  # no t_ prefix
    "other:123e4567-e89b-12d3-a456-426614174000:t_deadbeef",
])
def test_rejects_malformed_work_uids(bad):
    """A permissive parser would mint references that look resolvable."""
    with pytest.raises(board_resolver.MalformedWorkUid):
        board_resolver.parse_work_uid(bad)


# --------------------------------------------------------------- resolving


def test_resolves_a_uuid_to_its_board(tmp_path):
    db = tmp_path / "boards" / "ops" / "kanban.db"
    db.parent.mkdir(parents=True)
    board_uuid = _make_board(db)

    resolved = board_resolver.resolve(
        f"kanban:{board_uuid}:t_abcd1234", roots=[tmp_path])
    assert resolved.db_path == db
    assert resolved.task_id == "t_abcd1234"


def test_unknown_uuid_raises(tmp_path):
    db = tmp_path / "kanban.db"
    _make_board(db)
    with pytest.raises(board_resolver.UnknownBoard):
        board_resolver.resolve(
            "kanban:123e4567-e89b-12d3-a456-426614174000:t_abcd1234",
            roots=[tmp_path])


def test_several_boards_resolve_independently(tmp_path):
    uuids = {}
    for name in ("ops", "dev", "client"):
        db = tmp_path / name / "kanban.db"
        db.parent.mkdir(parents=True)
        uuids[name] = _make_board(db)

    for name, board_uuid in uuids.items():
        resolved = board_resolver.resolve(
            f"kanban:{board_uuid}:t_00000001", roots=[tmp_path])
        assert resolved.db_path.parent.name == name


# ------------------------------------------------- the duplicate: fail closed


def test_duplicate_uuid_refuses_writes(tmp_path):
    """A copied board mounted alongside the original.

    Picking either path is a coin flip that routes work to the wrong board.
    Picking by mtime or path order is the same coin flip with a
    plausible-sounding rule attached.
    """
    original = tmp_path / "a" / "kanban.db"
    original.parent.mkdir(parents=True)
    board_uuid = _make_board(original)

    clone = tmp_path / "b" / "kanban.db"
    clone.parent.mkdir(parents=True)
    shutil.copy2(original, clone)

    with pytest.raises(board_resolver.AmbiguousBoard) as exc:
        board_resolver.resolve(
            f"kanban:{board_uuid}:t_abcd1234", roots=[tmp_path], for_write=True)
    assert len(exc.value.paths) == 2


def test_duplicate_uuid_refuses_reads_too(tmp_path):
    """Reads are not the safe exception.

    A read that silently picks one of two boards answers a question the caller
    did not ask, and does so without any signal that it guessed.
    """
    original = tmp_path / "a" / "kanban.db"
    original.parent.mkdir(parents=True)
    board_uuid = _make_board(original)
    clone = tmp_path / "b" / "kanban.db"
    clone.parent.mkdir(parents=True)
    shutil.copy2(original, clone)

    with pytest.raises(board_resolver.AmbiguousBoard):
        board_resolver.resolve(
            f"kanban:{board_uuid}:t_abcd1234", roots=[tmp_path], for_write=False)


def test_the_error_names_every_conflicting_path(tmp_path):
    """The operator has to be able to act on it without hunting."""
    original = tmp_path / "a" / "kanban.db"
    original.parent.mkdir(parents=True)
    board_uuid = _make_board(original)
    for name in ("b", "c"):
        clone = tmp_path / name / "kanban.db"
        clone.parent.mkdir(parents=True)
        shutil.copy2(original, clone)

    with pytest.raises(board_resolver.AmbiguousBoard) as exc:
        board_resolver.resolve(f"kanban:{board_uuid}:t_abcd1234", roots=[tmp_path])
    message = str(exc.value)
    assert message.count("kanban.db") == 3
    assert board_uuid in message


def test_removing_the_clone_restores_resolution(tmp_path):
    """The refusal is a state, not a permanent poisoning."""
    original = tmp_path / "a" / "kanban.db"
    original.parent.mkdir(parents=True)
    board_uuid = _make_board(original)
    clone = tmp_path / "b" / "kanban.db"
    clone.parent.mkdir(parents=True)
    shutil.copy2(original, clone)

    with pytest.raises(board_resolver.AmbiguousBoard):
        board_resolver.resolve(f"kanban:{board_uuid}:t_1", roots=[tmp_path])

    clone.unlink()
    assert board_resolver.resolve(
        f"kanban:{board_uuid}:t_1", roots=[tmp_path]).db_path == original


# --------------------------------------------------------------- robustness


def test_scan_ignores_unreadable_and_pre_d0_files(tmp_path):
    """One bad file must not abort the scan.

    A board that predates D0 has no ``board_meta``; a truncated or unrelated
    ``kanban.db`` is not a board at all. Both are skipped rather than raising,
    because a single unreadable file otherwise makes every reference in the
    whole tree unresolvable.
    """
    good = tmp_path / "good" / "kanban.db"
    good.parent.mkdir(parents=True)
    board_uuid = _make_board(good)

    junk = tmp_path / "junk" / "kanban.db"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"this is not a database")

    legacy = tmp_path / "legacy" / "kanban.db"
    legacy.parent.mkdir(parents=True)
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE tasks (id TEXT)")
    conn.commit()
    conn.close()

    index = board_resolver.scan_boards([tmp_path])
    assert list(index) == [board_uuid]
    assert board_resolver.resolve(
        f"kanban:{board_uuid}:t_1", roots=[tmp_path]).db_path == good


def test_scan_does_not_mutate_the_boards_it_reads(tmp_path):
    """Scanning must not initialise or migrate.

    A resolver that opens boards read-write would migrate every board in the
    tree as a side effect of resolving one reference.
    """
    db = tmp_path / "kanban.db"
    _make_board(db)
    before = db.stat().st_mtime_ns, db.stat().st_size

    board_resolver.scan_boards([tmp_path])
    board_resolver.scan_boards([tmp_path])

    assert (db.stat().st_mtime_ns, db.stat().st_size) == before


def test_work_uid_for_mints_a_resolvable_reference(tmp_path):
    """Round trip: mint on an open board, resolve it back to the same file."""
    db = tmp_path / "board" / "kanban.db"
    db.parent.mkdir(parents=True)
    conn = kanban_db.connect(db)
    task_id = kanban_db.create_task(conn, title="round trip")
    conn.commit()
    work_uid = board_resolver.work_uid_for(conn, task_id)
    conn.close()

    assert work_uid
    resolved = board_resolver.resolve(work_uid, roots=[tmp_path])
    assert resolved.db_path == db
    assert resolved.task_id == task_id

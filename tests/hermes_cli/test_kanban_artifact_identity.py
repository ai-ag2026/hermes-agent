"""A0 (2026-07-28): the source identity is pinned at PROMOTION time.

Why this exists. The last-copy guard has to answer "is this directory holding
the only remaining copy of a recorded artifact?" — and until now it answered it
by re-resolving ``original_path`` at GC time. That identifies whatever the
string points at THEN. A symlink ancestor rebent after promotion makes it a
different object, so the guard protects the wrong inode, or none.

TARS' design review of 2026-07-28 called this the first of two P1 gaps: the
identity must be captured while we still hold the proof that the path IS the
file we copied. Everything here is about that capture. The descriptor-based
sweep that will USE it is a later step (A2); the workspace sweep stays opt-in
and off in the meantime.
"""

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _artifact_rows(conn):
    return [
        dict(row) for row in conn.execute(
            "SELECT original_path, original_realpath, original_dev, original_ino, "
            "durable_path FROM task_artifacts"
        )
    ]


def test_columns_exist_and_migrate(board):
    cols = {row["name"] for row in board.execute("PRAGMA table_info(task_artifacts)")}
    assert {"original_dev", "original_ino", "original_realpath"} <= cols


def test_promotion_pins_dev_and_ino(board, tmp_path):
    """The recorded identity must be the identity of the file we copied."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    artifact = workspace / "result.md"
    artifact.write_text("Ergebnis\n", encoding="utf-8")
    expected = os.stat(artifact)

    task_id = kb.create_task(
        board, title="mit Artefakt", assignee="worker1",
        workspace_kind="dir", workspace_path=str(workspace),
    )
    kb.complete_task(
        board, task_id, summary="fertig",
        metadata={"artifacts": [str(artifact)]},
    )
    conn = board

    rows = _artifact_rows(conn)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["original_dev"] == expected.st_dev
    assert row["original_ino"] == expected.st_ino
    assert row["original_realpath"] == os.path.realpath(artifact)
    assert row["original_path"] == str(artifact), "the raw producer path is kept too"


def test_pinned_identity_survives_a_rebent_symlink_ancestor(board, tmp_path):
    """The exact scenario the string-based lookup gets wrong.

    Promotion happens through ``link/result.md``; afterwards ``link`` is rebent
    to a different directory holding a DIFFERENT file under the same name.
    Re-resolving the stored path now finds the impostor — the pinned identity
    still names the original.
    """
    real = tmp_path / "real"
    real.mkdir()
    (real / "result.md").write_text("echt\n", encoding="utf-8")
    impostor = tmp_path / "impostor"
    impostor.mkdir()
    (impostor / "result.md").write_text("fremd\n", encoding="utf-8")

    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    promoted_as = link / "result.md"
    original_identity = os.stat(real / "result.md")

    task_id = kb.create_task(
        board, title="Symlink-Fall", assignee="worker1",
        workspace_kind="dir", workspace_path=str(link),
    )
    kb.complete_task(
        board, task_id, summary="fertig",
        metadata={"artifacts": [str(promoted_as)]},
    )

    link.unlink()
    link.symlink_to(impostor, target_is_directory=True)

    row = _artifact_rows(board)[0]
    now_resolves_to = os.stat(promoted_as)
    assert (now_resolves_to.st_dev, now_resolves_to.st_ino) != (
        original_identity.st_dev, original_identity.st_ino
    ), "precondition: the stored path now points somewhere else"
    assert row["original_dev"] == original_identity.st_dev
    assert row["original_ino"] == original_identity.st_ino, (
        "the pinned identity must still name the file that was promoted"
    )


def test_legacy_rows_stay_null_and_are_therefore_not_provable(board):
    """Rows written before A0 must be recognisable as unprovable, not guessed."""
    task_id = kb.create_task(board, title="Altbestand", assignee="worker1")
    board.execute(
        "INSERT INTO task_artifacts (task_id, producer_run_id, original_path, "
        "durable_path, sha256, size, content_type, validated_at, retention_class) "
        "VALUES (?, 0, '/gone/old.md', '/durable/old.md', 'abc', 1, 'text/plain', "
        "0, 'task_completion')",
        (task_id,),
    )
    board.commit()
    row = _artifact_rows(board)[0]
    assert row["original_dev"] is None and row["original_ino"] is None, (
        "a legacy row must not pretend to carry an identity"
    )


def test_unpinnable_source_fails_closed(board, tmp_path, monkeypatch):
    """If the identity cannot be captured, the promotion must not succeed.

    An artifact whose identity is unknown is exactly the one a later guard
    would have to guess about — and guessing is what destroyed 44 files.
    """
    workspace = tmp_path / "ws2"
    workspace.mkdir()
    artifact = workspace / "result.md"
    artifact.write_text("Ergebnis\n", encoding="utf-8")

    real_open = os.open

    def refusing_open(path, flags, *args, **kwargs):
        if str(path).endswith("result.md") and flags & os.O_RDONLY == os.O_RDONLY:
            raise OSError(5, "Input/output error")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refusing_open)

    task_id = kb.create_task(
        board, title="unpinbar", assignee="worker1",
        workspace_kind="dir", workspace_path=str(workspace),
    )
    with pytest.raises(Exception):
        kb.complete_task(
            board, task_id, summary="fertig",
            metadata={"artifacts": [str(artifact)]},
        )
    assert _artifact_rows(board) == [], "no manifest row without a pinned identity"

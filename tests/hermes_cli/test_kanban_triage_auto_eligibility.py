"""Loop-geparkte und human-gated Triage-Karten sind für Automation tabu.

Vorfall 2026-08-02 (t_33a7ec10): Der Loop-Breaker parkte die Karte terminal in
triage, der auto-decompose-Sweep nahm sie über ``list_triage_ids`` wieder auf,
re-spezifizierte und promotete sie — 17 Rekurrenzen in einer Nacht. Beide
Sweep-Kopien (``kanban_decompose``/``kanban_specify``) filtern jetzt über das
gemeinsame Prädikat ``kanban_db.triage_auto_eligible``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose, kanban_specify


@pytest.fixture
def isolated_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    db_path = home / "kanban.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_PROFILE", "backend-eng")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(db_path)
    return workspace


def _triage_task(conn, *, human_gate: int = 0, recurrences: int = 0) -> str:
    task_id = kb.create_task(conn, title="triage card", assignee="backend-eng")
    conn.execute(
        "UPDATE tasks SET status='triage', human_gate=?, block_recurrences=? WHERE id=?",
        (human_gate, recurrences, task_id),
    )
    conn.commit()
    return task_id


def test_fresh_triage_card_is_swept(isolated_board):
    with kb.connect() as conn:
        tid = _triage_task(conn)
    assert tid in kanban_decompose.list_triage_ids()
    assert tid in kanban_specify.list_triage_ids()


def test_loop_parked_card_is_not_swept(isolated_board):
    with kb.connect() as conn:
        tid = _triage_task(conn, recurrences=kb.BLOCK_RECURRENCE_LIMIT)
    assert tid not in kanban_decompose.list_triage_ids()
    assert tid not in kanban_specify.list_triage_ids()


def test_human_gated_card_is_not_swept(isolated_board):
    with kb.connect() as conn:
        tid = _triage_task(conn, human_gate=1)
    assert tid not in kanban_decompose.list_triage_ids()
    assert tid not in kanban_specify.list_triage_ids()


def test_recurrences_below_limit_stay_eligible(isolated_board):
    with kb.connect() as conn:
        tid = _triage_task(conn, recurrences=kb.BLOCK_RECURRENCE_LIMIT - 1)
    assert tid in kanban_decompose.list_triage_ids()
    assert tid in kanban_specify.list_triage_ids()

"""Tests für das Decompose-Versuchslimit (K-10, Vollaudit 2026-07-16).

``kanban.decompose_max_attempts`` stand seit Wochen in der Live-Config,
hatte aber keinen Consumer — der Auto-Decompose-Tick retryte gescheiterte
Karten unbegrenzt jeden Tick (Aux-LLM-Spend, Log-Rauschen, keine
Eskalation). Jetzt: fehlgeschlagene Versuche werden als durable
``decompose_attempt_failed``-Events gezählt; ab dem Limit überspringt der
Tick die Karte und hinterlässt EINMALIG ``decompose_gave_up`` — den
Human-Ping übernimmt der Attention-Cron (Grace-Fenster, 2026-07-16).
Default bleibt 0 = unbegrenzt (verhaltensneutral ohne Config-Key).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.kanban_watchers import (
    _count_failed_decompose_attempts,
    _record_decompose_gave_up_once,
    _record_failed_decompose_attempt,
    _resolve_decompose_max_attempts,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    yield home


def test_resolver_reads_config_value():
    assert _resolve_decompose_max_attempts(
        lambda: {"kanban": {"decompose_max_attempts": 2}}
    ) == 2


def test_resolver_defaults_to_unlimited():
    assert _resolve_decompose_max_attempts(lambda: {}) == 0
    assert _resolve_decompose_max_attempts(lambda: {"kanban": {}}) == 0


def test_resolver_fails_safe():
    def boom():
        raise RuntimeError("config unreadable")

    assert _resolve_decompose_max_attempts(boom) == 0
    assert _resolve_decompose_max_attempts(
        lambda: {"kanban": {"decompose_max_attempts": "kaputt"}}
    ) == 0
    assert _resolve_decompose_max_attempts(
        lambda: {"kanban": {"decompose_max_attempts": -3}}
    ) == 0


def test_failed_attempts_are_counted_durably(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="stubborn card", triage=True)
    assert _count_failed_decompose_attempts(kb, tid) == 0
    assert _record_failed_decompose_attempt(kb, tid, "no aux client") == 1
    assert _record_failed_decompose_attempt(kb, tid, "no aux client") == 2
    assert _count_failed_decompose_attempts(kb, tid) == 2


def test_gave_up_event_is_recorded_exactly_once(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="stubborn card", triage=True)
    _record_failed_decompose_attempt(kb, tid, "x")
    _record_failed_decompose_attempt(kb, tid, "x")
    assert _record_decompose_gave_up_once(kb, tid, attempts=2, limit=2) is True
    assert _record_decompose_gave_up_once(kb, tid, attempts=2, limit=2) is False
    with kb.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind = 'decompose_gave_up'",
            (tid,),
        ).fetchone()[0]
    assert n == 1

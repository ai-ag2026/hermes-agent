"""Tests for the AUTO-decompose operator-card exemption (Kanban-Krise
2026-07-10, geparkter Repair-Punkt 8).

Incident: the gateway's auto-decompose tick fanned out an operator-created
(``created_by="claude-code"``) notice card into a duplicate task chain after
a needs_input-recurrence escalation routed it to triage. Operator-created
cards must stay in triage for a human/operator decision; the unattended
dispatcher-tick loop (``_auto_decompose_tick`` -> ``list_triage_ids`` ->
``decompose_task(author="auto-decomposer")``) is the only thing gated here.
Explicit decomposition (``hermes kanban decompose <id>``, or
``decompose_task(..., author=<anything else>)`` called directly) never goes
through ``_filter_operator_owned_triage_ids`` and is unaffected -- see
hermes_cli/kanban.py's ``_cmd_decompose``, which calls
``kanban_decompose.list_triage_ids`` directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.kanban_watchers import (
    _filter_operator_owned_triage_ids,
    _operator_owned_skip_logged,
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    _operator_owned_skip_logged.clear()
    yield home
    _operator_owned_skip_logged.clear()


def test_operator_card_filtered_normal_card_kept(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        op_id = kb.create_task(
            conn, title="operator notice", triage=True, created_by="claude-code",
        )
        normal_id = kb.create_task(
            conn, title="normal triage", triage=True, created_by="auto-decomposer",
        )

    kept = _filter_operator_owned_triage_ids(
        [op_id, normal_id],
        board_slug="default",
        operator_authors=frozenset({"claude-code", "manfred"}),
        kb_module=kb,
    )
    assert kept == [normal_id]


def test_no_created_by_is_not_filtered(kanban_home):
    """A task created with created_by=None (legacy / no-author path) must
    not be swept up by the filter -- only an explicit operator match."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no-author task", triage=True)

    kept = _filter_operator_owned_triage_ids(
        [tid],
        board_slug="default",
        operator_authors=frozenset({"claude-code", "manfred"}),
        kb_module=kb,
    )
    assert kept == [tid]


def test_empty_operator_authors_keeps_everything(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="operator notice", triage=True, created_by="claude-code",
        )

    kept = _filter_operator_owned_triage_ids(
        [tid], board_slug="default", operator_authors=frozenset(), kb_module=kb,
    )
    assert kept == [tid]


def test_empty_triage_ids_short_circuits(kanban_home):
    from hermes_cli import kanban_db as kb

    kept = _filter_operator_owned_triage_ids(
        [], board_slug="default", operator_authors=frozenset({"claude-code"}), kb_module=kb,
    )
    assert kept == []


def test_lookup_failure_fails_open(kanban_home):
    """A DB hiccup during the operator-author lookup must not wedge
    auto-decompose for the whole board -- return the ids unfiltered."""
    broken_kb = MagicMock()
    broken_kb.connect.side_effect = RuntimeError("db unavailable")

    kept = _filter_operator_owned_triage_ids(
        ["t_whatever"],
        board_slug="default",
        operator_authors=frozenset({"claude-code"}),
        kb_module=broken_kb,
    )
    assert kept == ["t_whatever"]


def test_skip_logged_once_not_every_tick(kanban_home, caplog):
    """Repeated ticks over the same untouched operator card must log the
    skip exactly once, not on every dispatcher tick."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        op_id = kb.create_task(
            conn, title="operator notice", triage=True, created_by="manfred",
        )

    caplog.set_level(logging.INFO, logger="gateway.run")
    for _ in range(3):
        kept = _filter_operator_owned_triage_ids(
            [op_id],
            board_slug="default",
            operator_authors=frozenset({"manfred"}),
            kb_module=kb,
        )
        assert kept == []

    skip_records = [
        r for r in caplog.records
        if "skipped" in r.message and op_id in r.message
    ]
    assert len(skip_records) == 1, (
        f"expected exactly one skip log for {op_id}, got {len(skip_records)}"
    )


def test_explicit_decompose_path_never_calls_the_filter():
    """Documents the seam choice: hermes_cli/kanban.py's `_cmd_decompose`
    (the explicit `hermes kanban decompose --all` CLI path) calls
    `kanban_decompose.list_triage_ids` directly and never routes through
    `_filter_operator_owned_triage_ids` -- the filter lives only in the
    gateway's unattended `_auto_decompose_tick` loop."""
    import inspect

    from hermes_cli import kanban as kanban_cli

    source = inspect.getsource(kanban_cli._cmd_decompose)
    assert "_filter_operator_owned_triage_ids" not in source
    assert "operator_authors" not in source

"""Tests for pm-supervisor config resolution and candidate selection
(Autonomie-Umbau Baustein B, 2026-07-17).

``_resolve_pm_supervisor_settings`` mirrors ``_resolve_auto_decompose_settings``
(live re-read every tick, fail-safe on error) but defaults to OFF -- this is
a NEW autonomy step, not pre-existing default-on behaviour, so an operator
must opt in.

``_list_pm_supervisor_candidate_ids`` is the fail-safe candidate query: cards
in ``triage``/``blocked`` with ``block_kind='needs_input'`` (covers plain
needs_input blocks, the gave_up spawn-retry breaker, and the generic
unblock-loop breaker's triage route) OR a ``triage`` card carrying a
``decompose_gave_up`` event (which never touches ``block_kind`` at all --
see the function's own docstring for why that split is necessary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.kanban_watchers import (
    _DECOMPOSE_GAVE_UP_EVENT,
    _list_pm_supervisor_candidate_ids,
    _resolve_pm_supervisor_settings,
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


# ---------------------------------------------------------------------------
# _resolve_pm_supervisor_settings
# ---------------------------------------------------------------------------


def test_disabled_by_default_when_key_absent():
    enabled, per_tick, max_attempts = _resolve_pm_supervisor_settings(lambda: {"kanban": {}})
    assert enabled is False
    assert per_tick == 3
    assert max_attempts == 2


def test_enabled_when_flag_true():
    enabled, _, _ = _resolve_pm_supervisor_settings(
        lambda: {"kanban": {"pm_supervisor_enabled": True}}
    )
    assert enabled is True


def test_per_tick_and_max_attempts_respected_and_clamped():
    enabled, per_tick, max_attempts = _resolve_pm_supervisor_settings(
        lambda: {
            "kanban": {
                "pm_supervisor_enabled": True,
                "pm_supervisor_per_tick": 5,
                "pm_supervisor_max_attempts": 4,
            }
        }
    )
    assert (enabled, per_tick, max_attempts) == (True, 5, 4)

    # 0 is treated as "unset" by the `or default` fallback -> default,
    # same convention as auto_decompose_per_tick.
    _, _, max_attempts_zero = _resolve_pm_supervisor_settings(
        lambda: {"kanban": {"pm_supervisor_max_attempts": 0}}
    )
    assert max_attempts_zero == 2

    # A genuine negative value clamps up to 1.
    _, per_tick_neg, max_attempts_neg = _resolve_pm_supervisor_settings(
        lambda: {
            "kanban": {
                "pm_supervisor_per_tick": -5,
                "pm_supervisor_max_attempts": -3,
            }
        }
    )
    assert per_tick_neg == 1
    assert max_attempts_neg == 1


def test_malformed_values_fall_back_to_defaults():
    _, per_tick, max_attempts = _resolve_pm_supervisor_settings(
        lambda: {
            "kanban": {
                "pm_supervisor_per_tick": "lots",
                "pm_supervisor_max_attempts": "many",
            }
        }
    )
    assert per_tick == 3
    assert max_attempts == 2


def test_config_read_error_fails_safe_disabled():
    def _boom():
        raise RuntimeError("config read failed")

    enabled, per_tick, max_attempts = _resolve_pm_supervisor_settings(_boom)
    assert enabled is False
    assert per_tick == 3
    assert max_attempts == 2


def test_non_dict_config_fails_safe_disabled():
    enabled, _, _ = _resolve_pm_supervisor_settings(lambda: None)
    assert enabled is False


def test_live_toggle_takes_effect_between_calls():
    state = {"kanban": {"pm_supervisor_enabled": False}}
    assert _resolve_pm_supervisor_settings(lambda: state)[0] is False
    state["kanban"]["pm_supervisor_enabled"] = True
    assert _resolve_pm_supervisor_settings(lambda: state)[0] is True


# ---------------------------------------------------------------------------
# _list_pm_supervisor_candidate_ids
# ---------------------------------------------------------------------------


def test_needs_input_blocked_card_is_a_candidate(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="blocked card")
        kb.block_task(conn, tid, kind="needs_input", reason="need input")

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == [tid]


def test_plain_ready_card_is_not_a_candidate(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        kb.create_task(conn, title="normal ready card")

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == []


def test_plain_triage_card_without_gave_up_marker_is_not_a_candidate(kanban_home):
    """A fresh, never-decomposed triage card is NOT pm-supervisor's job --
    that's the auto-decomposer's lane. Only a triage card that already
    carries a decompose_gave_up marker (or a needs_input block_kind, via
    the generic loop-breaker's triage route) qualifies."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        kb.create_task(conn, title="fresh triage card", triage=True)

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == []


def test_decompose_gave_up_triage_card_is_a_candidate(kanban_home):
    """decompose_gave_up is recorded ONLY as a task_events row -- it never
    touches tasks.block_kind/block_reason_code -- so this needs the
    dedicated EXISTS(task_events) branch, not the block_kind column."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="gave up decomposing", triage=True)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, _DECOMPOSE_GAVE_UP_EVENT, {"attempts": 2, "limit": 2})

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == [tid]


def test_human_gate_card_is_excluded(kanban_home):
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="hard gated")
        kb.block_task(conn, tid, kind="needs_input", reason="x", human_gate=True)

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == []


def test_pending_exact_action_card_is_excluded(kanban_home):
    """A card waiting on a Tier-3 exact-action approval is untouchable --
    the operator's pending decision, not a worker stall."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="pending exact action")
        kb.block_task(conn, tid, kind="needs_input", reason="x")
        conn.execute(
            "INSERT INTO task_pending_actions "
            "(task_id, run_id, command_hash, fingerprint, mutation_kind, summary, "
            " profile, workspace, created_at, state, expires_at, version, updated_at) "
            "VALUES (?, NULL, 'h', 'f', 'shell', 's', 'p', 'w', ?, 'pending', ?, 1, ?)",
            (tid, 0, 9999999999, 0),
        )
        conn.commit()

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == []


def test_capability_block_kind_is_not_a_candidate(kanban_home):
    """Only block_kind='needs_input' qualifies -- a 'capability' block is a
    different (tool-gate) shape the pm-supervisor is not scoped to touch."""
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="capability blocked")
        kb.block_task(conn, tid, kind="capability", reason="needs a tool grant")

    assert _list_pm_supervisor_candidate_ids(kb, board_slug="default") == []


def test_lookup_failure_fails_open_to_empty_list(kanban_home):
    from unittest.mock import MagicMock

    broken_kb = MagicMock()
    broken_kb.connect.side_effect = RuntimeError("db unavailable")

    assert _list_pm_supervisor_candidate_ids(broken_kb, board_slug="default") == []

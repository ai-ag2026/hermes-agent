"""Review-lane hardening (Audit 2026-07-27, P1 findings).

Three invariants:

1. ``request_task_review`` rejects reviewers the dispatcher can never spawn
   (unknown profile, human-driven profile) — AT REQUEST TIME, loudly. Before
   this, a typo'd reviewer opened a handshake that the dispatcher skipped
   silently every tick while ``complete_task`` refused forever: the card was
   dead with no event, no attention, no audit finding. Validation is gated on
   a profile registry existing on disk, so fresh installs and test homes that
   drive the lane manually keep the legacy behaviour.

2. The worker↔reviewer NEEDS_REPAIR loop is bounded
   (``kanban.review_repair_limit``, default ``BLOCK_RECURRENCE_LIMIT``): at
   the limit the card routes to ``triage`` instead of round N+1 — the same
   escalation target the unblock-recurrence breaker uses. It was the only
   unbounded loop in the system, at a full model run per round.

3. ``acceptance_required`` is reachable through ``create_task`` directly (not
   only the internal work_contract promotion path), so ordinary cards can opt
   into an enforced review instead of an advisory one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (no profile registry)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _profiles_root() -> Path:
    from hermes_cli.profiles import _get_profiles_root
    return _get_profiles_root()


def _running_task(conn, *, assignee="backend-eng"):
    tid = kb.create_task(conn, title="implement", assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    return tid


def _request(conn, tid, reviewer):
    task = kb.get_task(conn, tid)
    return kb.request_task_review(
        conn, tid, reviewer=reviewer, summary="handoff",
        expected_run_id=task.current_run_id,
    )


# ---------------------------------------------------------------------------
# 1. Reviewer validation
# ---------------------------------------------------------------------------

def test_request_review_rejects_unknown_reviewer_when_registry_exists(kanban_home):
    root = _profiles_root()
    (root / "reviewer-bot").mkdir(parents=True)
    with kb.connect() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError, match="does not exist"):
            _request(conn, tid, "no-such-profile")
        # The refusal left no handshake behind: the card is untouched and a
        # correctly addressed request still works.
        assert kb.get_task(conn, tid).status == "running"
        assert _request(conn, tid, "reviewer-bot") is True
        assert kb.get_task(conn, tid).status == "review"


def test_request_review_rejects_human_driven_reviewer(kanban_home):
    root = _profiles_root()
    (root / "work").mkdir(parents=True)
    with kb.connect() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError, match="human-driven"):
            _request(conn, tid, "work")
        assert kb.get_task(conn, tid).status == "running"


def test_request_review_without_registry_keeps_legacy_behaviour(kanban_home):
    # No profiles root on disk (fresh install / manual-lane test home):
    # arbitrary reviewer names stay accepted.
    assert not _profiles_root().is_dir()
    with kb.connect() as conn:
        tid = _running_task(conn)
        assert _request(conn, tid, "made-up-reviewer") is True
        assert kb.get_task(conn, tid).status == "review"


# ---------------------------------------------------------------------------
# 2. NEEDS_REPAIR breaker
# ---------------------------------------------------------------------------

def _one_repair_round(conn, tid):
    """running → review → (reviewer claims) → NEEDS_REPAIR decision."""
    assert _request(conn, tid, "reviewer") is True
    review = kb.claim_review_task(conn, tid)
    assert review is not None
    assert kb.decide_task_review(
        conn, tid, decision="NEEDS_REPAIR", summary="fix it",
        expected_run_id=review.current_run_id,
    ) is True


def test_needs_repair_breaks_to_triage_at_limit(kanban_home):
    with kb.connect() as conn:
        tid = _running_task(conn)

        # Round 1 (limit is BLOCK_RECURRENCE_LIMIT = 2): back to ready.
        _one_repair_round(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        events = [e for e in kb.list_events(conn, tid) if e.kind == "review_decided"]
        assert events[-1].payload["repair_rounds"] == 1
        assert "routed_to" not in events[-1].payload

        # Round 2: breaker trips, card routes to triage instead of ready.
        assert kb.claim_task(conn, tid) is not None
        _one_repair_round(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        events = [e for e in kb.list_events(conn, tid) if e.kind == "review_decided"]
        assert events[-1].payload["repair_rounds"] == 2
        assert events[-1].payload["routed_to"] == "triage"
        assert events[-1].payload["limit"] == kb.BLOCK_RECURRENCE_LIMIT


def test_needs_repair_breaker_disabled_via_config(kanban_home, monkeypatch):
    import hermes_cli.config as _cfg
    monkeypatch.setattr(
        _cfg, "load_config_readonly",
        lambda: {"kanban": {"review_repair_limit": 0}},
    )
    with kb.connect() as conn:
        tid = _running_task(conn)
        for expected_round in (1, 2, 3):
            _one_repair_round(conn, tid)
            task = kb.get_task(conn, tid)
            assert task.status == "ready", f"round {expected_round}"
            assert kb.claim_task(conn, tid) is not None


# ---------------------------------------------------------------------------
# 3. acceptance_required via create_task
# ---------------------------------------------------------------------------

def test_create_task_acceptance_required_param_enforces_review(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="must be reviewed", assignee="backend-eng",
            acceptance_required=True,
        )
        row = conn.execute(
            "SELECT acceptance_required FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["acceptance_required"] == 1

        # Worker cannot self-complete: the D0 guard refuses and records why.
        assert kb.claim_task(conn, tid) is not None
        assert kb.complete_task(conn, tid, summary="done by my own say-so") is False
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "completion_blocked_acceptance_required" in kinds

        # The typed lane satisfies it: request → claim → ACCEPT → done.
        assert _request(conn, tid, "reviewer") is True
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.decide_task_review(
            conn, tid, decision="ACCEPT", summary="verified",
            expected_run_id=review.current_run_id,
        ) is True
        assert kb.get_task(conn, tid).status == "done"


def test_create_task_default_leaves_acceptance_unset(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary", assignee="backend-eng")
        row = conn.execute(
            "SELECT acceptance_required FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["acceptance_required"] is None

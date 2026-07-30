"""Tests for kanban goal_mode — per-card Ralph-style goal loop.

Covers three layers:

1. DB: goal_mode / goal_max_turns persist through create_task + from_row,
   and a legacy DB (without the columns) migrates cleanly.
2. Spawn: _default_spawn sets the HERMES_KANBAN_GOAL_MODE env vars only
   when the card opts in.
3. Loop: goals.run_kanban_goal_loop continuation / completion / budget
   behaviour, driven entirely through injected callbacks (no live model).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import goals


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------





def test_legacy_db_migrates_goal_columns(tmp_path, monkeypatch):
    """A tasks table created without goal columns must gain them on init."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal legacy schema: tasks table missing goal_mode / goal_max_turns.
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.commit()
    legacy.close()

    # init_db runs the additive migration.
    kb.init_db()
    with kb.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "goal_mode" in cols
        assert "goal_max_turns" in cols
        task = kb.get_task(conn, "legacy1")
    # Existing row keeps the safe default.
    assert task.goal_mode is False
    assert task.goal_max_turns is None


# ---------------------------------------------------------------------------
# Spawn env
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Goal loop logic (callback-injected, no live model)
# ---------------------------------------------------------------------------

def _patch_judge(monkeypatch, verdicts):
    """Make judge_goal return a scripted sequence of verdicts."""
    seq = list(verdicts)

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        v = seq.pop(0) if seq else "done"
        # 5-tuple contract: verdict, reason, parse failure, wait, transport failure.
        return v, f"scripted:{v}", False, None, False

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)


def test_loop_stops_when_worker_already_completed(monkeypatch):
    # Worker called kanban_complete on its first turn — no judging needed.
    _patch_judge(monkeypatch, ["continue"])  # should never be consulted
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "done",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="done already",
    )
    assert res["outcome"] == "completed_by_worker"
    assert turns == []  # no extra turns




def test_loop_blocks_on_budget_exhaustion(monkeypatch):
    # With a small budget (max_turns=3) and a work_budget of 2, turn 3 is the
    # reserved closeout turn. The worker never finalizes, so this now lands
    # on the typed terminal_step_not_taken outcome rather than a generic
    # "blocked_budget" — see test_reserved_closeout_* below for the more
    # targeted DoD coverage of that budget split.
    _patch_judge(monkeypatch, ["continue"] * 10)
    blocked = {}

    def _block(reason):
        blocked["reason"] = reason

    res = goals.run_kanban_goal_loop(
        task_id="t3",
        goal_text="endless task",
        run_turn=lambda p: "still going",
        task_status_fn=lambda: "running",
        block_fn=_block,
        max_turns=3,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_terminal_step_not_taken"
    assert res["turns_used"] == 3
    assert "closeout" in blocked["reason"].lower()


def test_loop_finalize_nudge_when_judge_done_but_open(monkeypatch):
    # Judge says done, but worker never terminated → one finalize nudge,
    # then worker completes.
    _patch_judge(monkeypatch, ["done", "done"])
    statuses = iter(["running", "done"])
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t4",
        goal_text="task",
        run_turn=lambda p: turns.append(p) or "ok",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="looks done",
    )
    assert res["outcome"] == "completed_by_worker"
    assert len(turns) == 1
    assert "still open" in turns[0]


def test_loop_blocks_when_judge_done_but_never_finalizes(monkeypatch):
    # Judge keeps saying done, worker never calls kanban_complete → block
    # after the single finalize nudge, typed as terminal_step_not_taken (the
    # lifecycle call was never made even though the judge saw "done" twice).
    _patch_judge(monkeypatch, ["done", "done"])
    blocked = {}

    res = goals.run_kanban_goal_loop(
        task_id="t5",
        goal_text="task",
        run_turn=lambda p: "still not finalizing",
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=10,
        first_response="looks done",
    )
    assert res["outcome"] == "blocked_terminal_step_not_taken"
    assert "finalize" in blocked["reason"].lower()


def test_loop_stops_if_task_reclaimed(monkeypatch):
    _patch_judge(monkeypatch, ["continue"])
    res = goals.run_kanban_goal_loop(
        task_id="t6",
        goal_text="task",
        run_turn=lambda p: pytest.fail("should not run a turn"),
        task_status_fn=lambda: "archived",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="x",
    )
    assert res["outcome"] == "stopped"


# ---------------------------------------------------------------------------
# S4: reserved closeout turn (DoD 1 + 2)
# ---------------------------------------------------------------------------

def test_reserved_closeout_turn_caps_ordinary_work(monkeypatch):
    """max_turns=3 proves at most two work turns plus one reserved closeout
    turn: turn 1 (first_response) + turn 2 (ordinary continuation) are work,
    turn 3 must be the lifecycle-only closeout prompt — never a second
    ordinary continuation."""
    _patch_judge(monkeypatch, ["continue"] * 10)
    prompts = []

    def _run_turn(p):
        prompts.append(p)
        return f"work #{len(prompts)}"  # always a fresh response — no no_progress interference

    blocked = {}
    res = goals.run_kanban_goal_loop(
        task_id="t7",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=3,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_terminal_step_not_taken"
    assert res["turns_used"] == 3
    assert len(prompts) == 2
    # The first fed prompt is ordinary continuation, the second (final) one
    # is the lifecycle-only closeout prompt — never a repeat of ordinary
    # continuation wording.
    assert "Take the next concrete step" in prompts[0]
    assert "reserved" in prompts[1].lower() and "closeout" in prompts[1].lower()
    assert "Take the next concrete step" not in prompts[1]


def test_reserved_closeout_turn_success_ends_normally(monkeypatch):
    """A successful kanban_complete call made during the reserved closeout
    turn ends the loop normally — no more prompts are fed afterward."""
    _patch_judge(monkeypatch, ["continue"] * 10)
    prompts = []
    statuses = iter(["running", "running", "done"])

    def _run_turn(p):
        prompts.append(p)
        return f"work #{len(prompts)}"

    res = goals.run_kanban_goal_loop(
        task_id="t8",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=3,
        first_response="turn1",
    )
    assert res["outcome"] == "completed_by_worker"
    assert res["turns_used"] == 3
    assert len(prompts) == 2
    assert "closeout" in prompts[1].lower()


def test_reserved_closeout_never_receives_ordinary_continuation(monkeypatch):
    """Even with a larger reserved corridor, once the loop enters closeout
    phase it is never handed an ordinary continuation prompt again."""
    _patch_judge(monkeypatch, ["continue"] * 10)
    prompts = []

    def _run_turn(p):
        prompts.append(p)
        return f"work #{len(prompts)}"

    blocked = {}
    res = goals.run_kanban_goal_loop(
        task_id="t8b",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=5,
        reserved_closeout_turns=2,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_terminal_step_not_taken"
    # work_budget = 5 - 2 = 3; turn 2 ordinary continuation (turn 1 =
    # first_response fills the rest of work_budget), turns 3 and 4 spend
    # the two reserved closeout-corridor turns before turn 5 is refused.
    assert len(prompts) == 4
    assert "closeout" in prompts[2].lower()
    assert "closeout" in prompts[3].lower()
    for p in prompts[2:]:
        assert "Take the next concrete step" not in p


def test_no_room_for_closeout_is_blocked_budget_not_terminal(monkeypatch):
    """max_turns=1 leaves no room to even attempt the reserved closeout
    turn — that's a configuration edge, not a worker failure, so it stays
    the generic blocked_budget outcome."""
    _patch_judge(monkeypatch, ["continue"])
    res = goals.run_kanban_goal_loop(
        task_id="t8c",
        goal_text="task",
        run_turn=lambda p: pytest.fail("no budget left for any turn"),
        task_status_fn=lambda: "running",
        block_fn=lambda r: None,
        max_turns=1,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_budget"
    assert res["turns_used"] == 1


# ---------------------------------------------------------------------------
# S4: WAIT barrier (DoD 3)
# ---------------------------------------------------------------------------

def test_wait_parks_without_spending_a_turn_then_resumes(monkeypatch):
    """A valid WAIT (pid) directive must not increment turns_used, and the
    loop must resume once the pid dies."""

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        if not hasattr(_fake_judge, "n"):
            _fake_judge.n = 0
        _fake_judge.n += 1
        if _fake_judge.n == 1:
            return "wait", "waiting on background build", False, {"pid": 4242}, False
        return "continue", "resumed", False, None, False

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)

    alive = {"value": True}
    monkeypatch.setattr(goals, "_pid_alive", lambda pid: alive["value"])

    sleeps = []

    def _sleep(seconds):
        sleeps.append(seconds)
        alive["value"] = False  # barrier clears on the first poll

    clock = {"t": 0.0}

    def _monotonic():
        return clock["t"]

    events = []
    prompts = []
    statuses = iter(["running", "running", "done"])

    res = goals.run_kanban_goal_loop(
        task_id="t9",
        goal_text="task",
        run_turn=lambda p: prompts.append(p) or "ok",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="turn1",
        sleep_fn=_sleep,
        monotonic_fn=_monotonic,
        emit_progress=events.append,
    )
    assert res["outcome"] == "completed_by_worker"
    # Only one ordinary continuation turn actually ran — the WAIT round
    # burned no turn at all.
    assert res["turns_used"] == 2
    assert len(sleeps) == 1
    phases = [e["phase"] for e in events]
    assert "wait" in phases
    assert "wait_cleared" in phases


def test_wait_barrier_gives_up_after_max_wait_seconds(monkeypatch):
    """A pid that never dies must not hang the loop forever — it resumes
    once the bounded max_wait_seconds ceiling is hit."""

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        if not hasattr(_fake_judge, "n"):
            _fake_judge.n = 0
        _fake_judge.n += 1
        if _fake_judge.n == 1:
            return "wait", "waiting forever", False, {"pid": 999}, False
        return "continue", "gave up waiting", False, None, False

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)
    monkeypatch.setattr(goals, "_pid_alive", lambda pid: True)  # never dies

    clock = {"t": 0.0}

    def _monotonic():
        return clock["t"]

    def _sleep(seconds):
        clock["t"] += seconds

    res = goals.run_kanban_goal_loop(
        task_id="t10",
        goal_text="task",
        run_turn=lambda p: "ok",
        task_status_fn=lambda: "done" if getattr(_fake_judge, "n", 0) > 1 else "running",
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="turn1",
        sleep_fn=_sleep,
        monotonic_fn=_monotonic,
        max_wait_seconds=30.0,
        wait_poll_seconds=10.0,
    )
    assert res["outcome"] == "completed_by_worker"
    # Bounded: at most ceiling/poll_interval + 1 polls happened before giving up.
    assert clock["t"] <= 40.0


def test_invalid_wait_directive_degrades_to_continue_without_hanging(monkeypatch):
    """A WAIT verdict with no usable pid/seconds must not park forever — it
    degrades to a normal continue turn instead."""
    calls = {"sleep": 0}

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        if not hasattr(_fake_judge, "n"):
            _fake_judge.n = 0
        _fake_judge.n += 1
        if _fake_judge.n == 1:
            return "wait", "no target given", False, {}, False
        return "continue", "ok", False, None, False

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)

    statuses = iter(["running", "done"])
    prompts = []
    res = goals.run_kanban_goal_loop(
        task_id="t11",
        goal_text="task",
        run_turn=lambda p: prompts.append(p) or "ok",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="turn1",
        sleep_fn=lambda s: calls.__setitem__("sleep", calls["sleep"] + 1),
    )
    assert res["outcome"] == "completed_by_worker"
    assert calls["sleep"] == 0  # never actually parked
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# S4: no_progress classification (DoD 4)
# ---------------------------------------------------------------------------

def test_repeated_identical_response_is_typed_no_progress(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 20)
    blocked = {}
    res = goals.run_kanban_goal_loop(
        task_id="t12",
        goal_text="task",
        run_turn=lambda p: "same prose every time",
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=20,
        no_progress_limit=2,
        first_response="same prose every time",
    )
    assert res["outcome"] == "blocked_no_progress"
    # Caught well before the numeric budget (20) or reserved closeout (19).
    assert res["turns_used"] < 19
    assert "no detectable" in blocked["reason"].lower()


def test_real_delta_resets_no_progress_counter(monkeypatch):
    """A genuine workspace/test-manifest delta reported via progress_fn must
    reset the no-progress counter even if the response text repeats."""
    _patch_judge(monkeypatch, ["continue"] * 20)
    deltas = iter([
        {"workspace_fingerprint": "a"},
        {"workspace_fingerprint": "a"},  # repeat -> would trip at limit=2
        {"workspace_fingerprint": "b"},  # real delta -> resets counter
        {"workspace_fingerprint": "b"},
    ])
    statuses = iter(["running", "running", "running", "running", "done"])
    res = goals.run_kanban_goal_loop(
        task_id="t13",
        goal_text="task",
        run_turn=lambda p: "same prose every time",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail(f"should not block: {r}"),
        max_turns=20,
        no_progress_limit=2,
        first_response="same prose every time",
        progress_fn=lambda: next(deltas, {"workspace_fingerprint": "b"}),
    )
    assert res["outcome"] == "completed_by_worker"


# ---------------------------------------------------------------------------
# S4: terminal_step_not_taken before hitting a large numeric budget (DoD 5)
# ---------------------------------------------------------------------------

def test_terminal_step_not_taken_before_large_budget_exhausted(monkeypatch):
    """Mirrors the real incident (a card dying 90/90 with usable artifacts
    but no reserved closeout corridor): with a large max_turns, varying
    responses (so no_progress never trips) still end in a typed
    terminal_step_not_taken once the reserved closeout turn is spent, not a
    bare numeric exhaustion."""
    _patch_judge(monkeypatch, ["continue"] * 200)
    prompts = []

    def _run_turn(p):
        prompts.append(p)
        return f"turn #{len(prompts)} distinct output"

    blocked = {}
    res = goals.run_kanban_goal_loop(
        task_id="t14",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=90,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_terminal_step_not_taken"
    assert res["turns_used"] == 90
    assert "closeout" in prompts[-1].lower()


# ---------------------------------------------------------------------------
# S4: classified run_turn failure retries (DoD 6)
# ---------------------------------------------------------------------------

def test_transient_failure_is_retried_and_recovers(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 10)
    attempts = {"n": 0}
    sleeps = []

    def _run_turn(p):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise goals.KanbanTransientTurnError("429 rate limited")
        return "recovered"

    statuses = iter(["running", "done"])
    res = goals.run_kanban_goal_loop(
        task_id="t15",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="turn1",
        sleep_fn=sleeps.append,
    )
    assert res["outcome"] == "completed_by_worker"
    assert attempts["n"] == 2
    assert len(sleeps) == 1  # one bounded backoff before recovering


def test_deterministic_failure_gets_exactly_one_retry(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 10)
    attempts = {"n": 0}

    def _run_turn(p):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("unexpected tool schema mismatch")
        return "recovered after one retry"

    statuses = iter(["running", "done"])
    res = goals.run_kanban_goal_loop(
        task_id="t16",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="turn1",
    )
    assert res["outcome"] == "completed_by_worker"
    assert attempts["n"] == 2


def test_repeated_identical_failure_fingerprint_escalates_to_triage(monkeypatch):
    """A second occurrence of the SAME failure must not be blind-retried
    again — it escalates straight to triage."""
    _patch_judge(monkeypatch, ["continue"] * 10)
    blocked = {}

    def _run_turn(p):
        raise ValueError("same deterministic bug every time")

    res = goals.run_kanban_goal_loop(
        task_id="t17",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=10,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_triage"
    assert "same" in blocked["reason"].lower()


def test_changed_failure_after_retry_keeps_going(monkeypatch):
    """A different failure on the retry (changed strategy / a new transient
    blip) is not the "identical second fingerprint" case — it may keep
    going through its own classification instead of being force-escalated."""
    _patch_judge(monkeypatch, ["continue"] * 10)
    attempts = {"n": 0}

    def _run_turn(p):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("first distinct failure")
        if attempts["n"] == 2:
            raise ValueError("a completely different failure message")
        return "finally recovered"

    statuses = iter(["running", "done"])
    res = goals.run_kanban_goal_loop(
        task_id="t18",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail(f"should not block: {r}"),
        max_turns=10,
        first_response="turn1",
    )
    assert res["outcome"] == "completed_by_worker"
    assert attempts["n"] == 3


def test_transient_retries_exhausted_with_evidence_routes_to_resume_closeout(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 10)
    blocked = {}
    n = {"i": 0}

    def _run_turn(p):
        n["i"] += 1
        raise goals.KanbanTransientTurnError(f"timeout attempt {n['i']}")

    res = goals.run_kanban_goal_loop(
        task_id="t19",
        goal_text="task",
        run_turn=_run_turn,
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=10,
        max_transient_retries=2,
        first_response="already produced something",
        sleep_fn=lambda s: None,
    )
    assert res["outcome"] == "blocked_resume_closeout"


# ---------------------------------------------------------------------------
# S4: bounded, secret-free progress/budget telemetry (DoD 7)
# ---------------------------------------------------------------------------

def test_emit_progress_events_are_bounded_and_secret_free(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 10)
    events = []
    secret = "sk-super-secret-token-should-never-appear-1234567890"

    res = goals.run_kanban_goal_loop(
        task_id="t20",
        goal_text="task",
        run_turn=lambda p: f"response containing {secret}",
        task_status_fn=lambda: "running",
        block_fn=lambda r: None,
        max_turns=4,
        first_response=f"response containing {secret}",
        emit_progress=events.append,
    )
    assert events  # at least one event was emitted
    for ev in events:
        blob = json.dumps(ev)
        assert secret not in blob
        assert len(blob) < 4000  # bounded payload, no unbounded response echo
        assert "turns_used" in ev and "max_turns" in ev
        assert "verdict_history" in ev
        assert len(ev["verdict_history"]) <= 5


# ---------------------------------------------------------------------------
# S4 review: "progress_fn anschließen" — cli._kanban_progress_snapshot feeds
# goals.run_kanban_goal_loop's no_progress fingerprint with task_event_count
# / workspace_fingerprint / test_manifest_fingerprint. Before this, the
# no_progress detector only ever saw response-hash + judge verdict.
# ---------------------------------------------------------------------------

def test_progress_snapshot_task_event_count_increases_with_new_event(kanban_home):
    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    before = _kanban_progress_snapshot(tid, None)["task_event_count"]
    assert before is not None

    with kb.connect() as conn:
        kb.add_comment(conn, tid, "worker", "did some work")

    after = _kanban_progress_snapshot(tid, None)["task_event_count"]
    assert after == before + 1


def test_progress_snapshot_missing_workspace_gives_none_keys_no_crash(kanban_home):
    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    # workspace=None mirrors a missing HERMES_KANBAN_WORKSPACE env var.
    snap = _kanban_progress_snapshot(tid, None)
    assert snap["workspace_fingerprint"] is None
    assert snap["test_manifest_fingerprint"] is None
    # task_event_count is resolved independently of the workspace.
    assert snap["task_event_count"] is not None


def test_progress_snapshot_nonexistent_workspace_path_no_crash(kanban_home, tmp_path):
    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    missing = str(tmp_path / "does-not-exist")
    snap = _kanban_progress_snapshot(tid, missing)
    assert snap["workspace_fingerprint"] is None
    assert snap["test_manifest_fingerprint"] is None


def test_progress_snapshot_git_fingerprint_changes_with_commit(kanban_home, tmp_path):
    import subprocess

    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    ws = tmp_path / "workspace"
    ws.mkdir()

    def _git(*args):
        subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True)

    _git("init", "-q")
    _git("config", "user.email", "test@example.com")
    _git("config", "user.name", "Test")
    (ws / "a.txt").write_text("one")
    _git("add", "a.txt")
    _git("commit", "-q", "-m", "first")

    snap1 = _kanban_progress_snapshot(tid, str(ws))
    assert snap1["workspace_fingerprint"] is not None

    (ws / "a.txt").write_text("two")
    _git("add", "a.txt")
    _git("commit", "-q", "-m", "second")

    snap2 = _kanban_progress_snapshot(tid, str(ws))
    assert snap2["workspace_fingerprint"] is not None
    assert snap2["workspace_fingerprint"] != snap1["workspace_fingerprint"]


def test_progress_snapshot_git_fingerprint_changes_with_dirty_worktree(kanban_home, tmp_path):
    """A commit isn't the only source of progress — uncommitted edits (the
    common case while a worker is mid-task) must also move the fingerprint,
    since ``git status --porcelain`` / ``git diff --stat`` feed it too."""
    import subprocess

    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    ws = tmp_path / "workspace"
    ws.mkdir()

    def _git(*args):
        subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True)

    _git("init", "-q")
    _git("config", "user.email", "test@example.com")
    _git("config", "user.name", "Test")
    (ws / "a.txt").write_text("one\n")
    _git("add", "a.txt")
    _git("commit", "-q", "-m", "first")

    snap1 = _kanban_progress_snapshot(tid, str(ws))

    (ws / "a.txt").write_text("one\ntwo\n")  # uncommitted edit

    snap2 = _kanban_progress_snapshot(tid, str(ws))
    assert snap2["workspace_fingerprint"] != snap1["workspace_fingerprint"]


def test_progress_snapshot_test_manifest_fingerprint_changes_with_new_test_file(kanban_home, tmp_path):
    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    ws = tmp_path / "workspace"
    (ws / "tests").mkdir(parents=True)
    (ws / "tests" / "test_a.py").write_text("def test_a(): pass\n")

    snap1 = _kanban_progress_snapshot(tid, str(ws))
    assert snap1["test_manifest_fingerprint"] is not None

    (ws / "tests" / "test_b.py").write_text("def test_b(): pass\n")

    snap2 = _kanban_progress_snapshot(tid, str(ws))
    assert snap2["test_manifest_fingerprint"] != snap1["test_manifest_fingerprint"]


def test_progress_snapshot_no_tests_dir_gives_none_manifest(kanban_home, tmp_path):
    from cli import _kanban_progress_snapshot

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", body="b", assignee="default")

    ws = tmp_path / "workspace"
    ws.mkdir()

    snap = _kanban_progress_snapshot(tid, str(ws))
    assert snap["test_manifest_fingerprint"] is None


def test_run_kanban_goal_loop_q_wires_progress_fn(monkeypatch, kanban_home):
    """The actual S4 fix: cli._run_kanban_goal_loop_q must hand a working
    progress_fn to goals.run_kanban_goal_loop, not leave it unwired."""
    import cli as cli_mod
    from hermes_cli import goals as goals_mod

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="goal task", body="do it", assignee="default",
            goal_mode=True, goal_max_turns=5,
        )
        kb.claim_task(conn, tid)

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)

    captured = {}

    def _fake_loop(**kwargs):
        captured.update(kwargs)
        return {"outcome": "stopped", "turns_used": 1, "reason": "test stub"}

    monkeypatch.setattr(goals_mod, "run_kanban_goal_loop", _fake_loop)

    fake_cli = SimpleNamespace(
        agent=SimpleNamespace(session_id="s"),
        conversation_history=[],
        session_id="s",
    )
    cli_mod._run_kanban_goal_loop_q(fake_cli, "first turn response")

    assert captured.get("progress_fn") is not None
    snap = captured["progress_fn"]()
    assert set(snap) == {
        "task_event_count", "workspace_fingerprint", "test_manifest_fingerprint",
    }
    assert snap["task_event_count"] is not None
    # No HERMES_KANBAN_WORKSPACE set → workspace-derived keys stay None.
    assert snap["workspace_fingerprint"] is None
    assert snap["test_manifest_fingerprint"] is None
# CLI judge gate tests (hermes kanban complete bypass fix)
# ---------------------------------------------------------------------------

class TestCLIJudgeGate:
    """hermes kanban complete must apply the same goal_mode judge gate as the
    kanban_complete tool (Issue #38367 sibling gap).

    Uses mocks for kb.get_task and kb.complete_task to avoid depending on the
    full kanban_db schema; the gate logic is the unit under test.
    """

    def _run(self, monkeypatch, *, goal_mode=True, judge_available=True,
             verdict="done", reason="", complete_ok=True, summary="done"):
        import argparse
        import types
        from unittest.mock import MagicMock
        from hermes_cli.kanban import _cmd_complete

        fake_task = types.SimpleNamespace(
            goal_mode=goal_mode,
            title="Finish report",
            body="acceptance: criteria",
        )
        fake_conn = MagicMock()
        complete_calls: list = []

        def fake_connect_closing():
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                yield fake_conn
            return _cm()

        def fake_complete_task(conn, tid, **kw):
            complete_calls.append(tid)
            return complete_ok

        monkeypatch.setattr("hermes_cli.kanban.kb.get_task", lambda conn, tid: fake_task)
        monkeypatch.setattr("hermes_cli.kanban.kb.complete_task", fake_complete_task)
        monkeypatch.setattr("hermes_cli.kanban.kb.connect_closing", fake_connect_closing)
        monkeypatch.setattr("hermes_cli.kanban._worker_run_id_for", lambda _: None)

        _aux_client = (object(), "judge-model") if judge_available else (None, None)
        monkeypatch.setattr(
            "agent.auxiliary_client.get_text_auxiliary_client",
            lambda name: _aux_client,
        )
        # Match the real judge_goal contract:
        # (verdict, reason, parse_failed, wait_directive, transport_failed)
        monkeypatch.setattr(
            "hermes_cli.goals.judge_goal",
            lambda **kw: (verdict, reason, False, None, False),
        )

        args = argparse.Namespace(task_ids=["t1"], summary=summary, result=None, metadata=None)
        return _cmd_complete(args), complete_calls

    def test_judge_rejects_premature_completion(self, monkeypatch):
        rc, complete_calls = self._run(
            monkeypatch, verdict="continue", reason="criteria not met"
        )
        assert rc != 0, "judge rejection must produce non-zero exit code"
        assert complete_calls == [], (
            "complete_task must NOT be invoked when the judge rejects"
        )


    def test_non_goal_mode_task_skips_gate(self, monkeypatch):
        """Plain (non-goal_mode) tasks are never sent to the judge."""
        rc, complete_calls = self._run(monkeypatch, goal_mode=False)
        assert rc == 0
        assert complete_calls == ["t1"]

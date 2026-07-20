"""Executive control loop (E), including acceptance tests 3 and 8.

The tests worth reading are the ones about *not* escalating. Noticing a
problem is easy; the hard part is that the system must try to fix it first,
must ask at most once, and must ask in a form that carries a recommendation.
Every one of those failures shows up as "the person has to do it themselves",
which is the complaint the whole programme is answering.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli import executive, kanban_db, work_promotion
from hermes_cli.executive import Action, Condition


@pytest.fixture
def board(tmp_path):
    conn = kanban_db.connect(tmp_path / "kanban.db")
    yield conn
    conn.close()


def _item(board, *, key, status="running", **contract):
    result = work_promotion.promote(
        board, origin_kind="cron", origin_key=key, title=key, payload={}, **contract)
    board.execute("UPDATE tasks SET status=? WHERE id=?", (status, result.task_id))
    board.commit()
    return result.task_id


def _touch(board, task_id, *, heartbeat=None, created=None, retries=None):
    """Shape a task's observable history.

    ``retries`` writes real ``task_runs`` rows rather than a counter column:
    the attempt count IS the number of runs, and there is no ``retries``
    column on ``tasks``. An earlier version of these tests invented one, which
    is how I noticed that ``classify`` was reading liveness off the wrong
    table.
    """
    if created is not None:
        board.execute("UPDATE tasks SET created_at=? WHERE id=?", (created, task_id))
    if retries is not None:
        for _ in range(retries):
            board.execute(
                "INSERT INTO task_runs (task_id, status, started_at) VALUES (?,?,?)",
                (task_id, "ended", int(time.time())))
    if heartbeat is not None:
        board.execute(
            "INSERT INTO task_runs (task_id, status, started_at, last_heartbeat_at) "
            "VALUES (?,?,?,?)", (task_id, "running", int(heartbeat), heartbeat))
    board.commit()


# ------------------------------------------------------------ classification


def test_a_silent_running_item_is_stale(board):
    task_id = _item(board, key="quiet")
    _touch(board, task_id, heartbeat=time.time() - 7200)
    row = board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert executive.classify(row, last_signal=time.time() - 7200) is Condition.STALE
    assert executive.scan(board)[0].condition is Condition.STALE


def test_a_recently_active_item_is_healthy(board):
    task_id = _item(board, key="busy")
    _touch(board, task_id, heartbeat=time.time() - 60)
    row = board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert executive.classify(row, last_signal=time.time() - 60) is Condition.HEALTHY
    assert executive.scan(board) == []


def test_a_running_item_that_never_reported_is_stale_not_healthy(board):
    """Absence of a heartbeat is the signal.

    Treating "no evidence of trouble" as "fine" is how silently dead work goes
    unnoticed -- the exact failure mode this loop exists to catch.
    """
    task_id = _item(board, key="never-spoke")
    _touch(board, task_id, heartbeat=None, created=time.time() - 7200)
    row = board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert executive.classify(row) is Condition.STALE


def test_blocked_is_checked_before_staleness(board):
    """A blocked item is supposed to be quiet.

    Classifying it as stale would produce a stream of findings for items that
    are behaving exactly as intended.
    """
    task_id = _item(board, key="blocked-one", status="blocked")
    _touch(board, task_id, heartbeat=time.time() - 999999)
    row = board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert executive.classify(row) is Condition.BLOCKED


def test_review_pending_is_its_own_condition(board):
    task_id = _item(board, key="in-review", status="review")
    row = board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert executive.classify(row) is Condition.REVIEW_PENDING


def test_done_items_are_never_findings(board):
    task_id = _item(board, key="finished", status="done")
    _touch(board, task_id, heartbeat=time.time() - 999999)
    row = board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    assert executive.classify(row) is Condition.HEALTHY


# ------------------------------------------------- recovery before escalation


def test_a_first_stall_is_repaired_not_escalated():
    """The core ordering rule.

    Escalating the first stall is precisely the behaviour that turned every
    hiccup into a question for the person.
    """
    finding = executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=0)
    assert executive.decide(finding) is Action.REPAIR


def test_a_repeated_stall_changes_strategy_before_escalating():
    """A third identical attempt is a loop, not persistence.

    Repeating the same approach burns the budget that would otherwise buy a
    different one.
    """
    finding = executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=2)
    assert executive.decide(finding, retry_budget=4) is Action.STRATEGY_CHANGE


def test_only_an_exhausted_budget_escalates():
    finding = executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=3)
    assert executive.decide(finding, retry_budget=3) is Action.ESCALATE


def test_waiting_for_a_reviewer_is_not_a_problem():
    """The system working as designed must not look like a fault."""
    finding = executive.Finding("w:1", "t_1", Condition.REVIEW_PENDING, "")
    assert executive.decide(finding) is Action.WAIT


# ================================================== Acceptance 8
# Decision quality
# Fixture: a real blocker with no repair path.
# PASS: exactly ONE package with situation/attempts/options/recommendation/
#       impact; no repeated pings (dedupe evidence).
# FAIL: a bare "how do you want to proceed?"


def test_a8_a_bare_question_is_not_a_well_formed_package():
    """The FAIL case, enforced rather than documented.

    A question without options and a recommendation hands the work back to the
    person. That is the output this module was built to stop producing.
    """
    bare = executive.DecisionPackage(coalesce_key="k", situation="Wie weiter?")
    assert bare.is_well_formed() is False


def test_a8_a_complete_package_carries_all_five_parts():
    package = executive.DecisionPackage(
        coalesce_key="k",
        situation="Der Upstream-Endpunkt antwortet seit 40 Minuten mit 502",
        attempts=["3x Retry mit Backoff", "Fallback-Endpunkt probiert"],
        options=["Warten bis der Anbieter behebt", "Auf den Spiegel umschalten"],
        recommendation="Auf den Spiegel umschalten -- der Anbieter nennt keine ETA",
        impact="Der Spiegel ist einen Tag alt; zwei Karten würden veraltete Daten sehen",
    )
    assert package.is_well_formed()
    rendered = package.render()
    for part in ("Lage:", "Bisherige Versuche:", "Optionen:", "Empfehlung:",
                 "Auswirkung:"):
        assert part in rendered


def test_a8_related_findings_produce_exactly_one_package(board):
    """Three items, one cause, one decision.

    Sending three messages makes the person do the grouping the system should
    have done -- and three notifications for one problem is how the channels
    became unusable.
    """
    findings = [
        executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=9),
        executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=9),
        executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=9),
    ]
    grouped = executive.coalesce(findings)
    assert len(grouped) == 1
    package = executive.build_decision_package(next(iter(grouped.values())))
    assert len(package.work_uids) == 3


def test_a8_the_same_problem_is_escalated_only_once(board):
    """The dedupe evidence the acceptance test asks for."""
    task_id = _item(board, key="hopeless", status="blocked")
    _touch(board, task_id, retries=99)

    first = executive.run_once(board, respect_quiet_hours=False)
    assert len(first.escalated) == 1

    second = executive.run_once(board, respect_quiet_hours=False)
    assert second.escalated == []
    assert len(second.suppressed) == 1


def test_a8_suppression_expires_so_it_does_not_become_blindness(board):
    """Suppression must not turn into never noticing again."""
    task_id = _item(board, key="recurring", status="blocked")
    _touch(board, task_id, retries=99)

    now = time.time()
    executive.run_once(board, now=now, respect_quiet_hours=False)
    still_quiet = executive.run_once(
        board, now=now + executive.SUPPRESSION_TTL_SECONDS - 60,
        respect_quiet_hours=False)
    assert still_quiet.escalated == []

    later = executive.run_once(
        board, now=now + executive.SUPPRESSION_TTL_SECONDS + 60,
        respect_quiet_hours=False)
    assert len(later.escalated) == 1


def test_a8_the_coalesce_key_ignores_the_attempt_count(board):
    """Otherwise every tick would look like a new problem.

    A key that changes as attempts increase defeats suppression completely,
    which is how a watcher turns into a spam source.
    """
    a = executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=1)
    b = executive.Finding("w:1", "t_1", Condition.STALE, "", attempts=7)
    assert a.coalesce_key == b.coalesce_key


# ------------------------------------------------------------- quiet hours


def test_non_critical_escalations_wait_for_quiet_hours_to_end(board):
    task_id = _item(board, key="not-urgent", status="blocked")
    _touch(board, task_id, retries=99)

    midnight = time.mktime(time.struct_time(
        (2026, 7, 21, 2, 0, 0, 1, 202, -1)))
    result = executive.run_once(board, now=midnight)
    assert result.escalated == []
    assert len(result.deferred_quiet_hours) == 1


def test_quiet_hours_do_not_hide_a_lost_item():
    """Critical conditions still get through.

    Quiet hours are about not being woken for something that can wait, not
    about suppressing the one class of finding that means work was destroyed.
    """
    finding = executive.Finding("w:1", "t_1", Condition.LOST, "", attempts=99)
    assert executive.decide(finding) is Action.ESCALATE


# ================================================== Acceptance 3 (E form)
# Commitment survival
# Fixture: a commitment due tomorrow, then a forced restart.
# PASS (E form): the commitment and its next step still exist afterwards.
#       Delivery guarantees are F4/F6 and are NOT asserted here.
# Negative: a finished commitment does not fire twice.


def test_a3_a_commitment_survives_a_restart(tmp_path):
    """The E form: it is still there, and it is still due.

    Nothing here claims a message arrived -- there is no core inbox until
    F3/F4, and asserting delivery would be claiming a guarantee that does not
    exist yet.
    """
    db = tmp_path / "kanban.db"
    due = int(time.time()) + 86400

    conn = kanban_db.connect(db)
    result = work_promotion.promote(
        conn, origin_kind="conversation", origin_key="promise-1",
        title="report back on the migration", payload={},
        commitment="Ich melde mich morgen mit dem Ergebnis",
        commitment_due_at=due)
    conn.commit()
    conn.close()

    # Forced restart: nothing of the previous process remains.
    conn = kanban_db.connect(db)
    row = conn.execute(
        "SELECT commitment, commitment_due_at, status FROM tasks WHERE id=?",
        (result.task_id,)).fetchone()
    conn.close()

    assert row["commitment"] == "Ich melde mich morgen mit dem Ergebnis"
    assert row["commitment_due_at"] == due
    assert row["status"] != "done"


def test_a3_a_due_commitment_is_noticed(board):
    task_id = _item(board, key="due-now", status="ready",
                    commitment="melde mich", commitment_due_at=int(time.time()) - 60)
    findings = executive.scan(board)
    due = [f for f in findings if f.condition is Condition.COMMITMENT_DUE]
    assert [f.task_id for f in due] == [task_id]


def test_a3_negative_a_finished_commitment_does_not_fire(board):
    """The negative case: done work must not keep reminding."""
    task_id = _item(board, key="kept", status="done",
                    commitment="melde mich", commitment_due_at=int(time.time()) - 60)
    findings = executive.scan(board)
    assert task_id not in {f.task_id for f in findings}


def test_a3_a_due_commitment_is_repaired_before_it_is_escalated(board):
    """Being due is a reason to act, not a reason to ask."""
    finding = executive.Finding("w:1", "t_1", Condition.COMMITMENT_DUE, "")
    assert executive.decide(finding) is Action.REPAIR


# ------------------------------------------------------------------- scan


def test_scan_reports_every_unhealthy_item_once(board):
    _item(board, key="a", status="blocked")
    _item(board, key="b", status="review")
    healthy = _item(board, key="c", status="running")
    _touch(board, healthy, heartbeat=time.time())

    findings = executive.scan(board)
    conditions = {f.condition for f in findings}
    assert Condition.BLOCKED in conditions
    assert Condition.REVIEW_PENDING in conditions
    assert healthy not in {f.task_id for f in findings}


def test_scan_produces_resolvable_work_uids(board):
    _item(board, key="a", status="blocked")
    for finding in executive.scan(board):
        assert finding.work_uid.startswith("kanban:")
        assert finding.work_uid.endswith(finding.task_id)

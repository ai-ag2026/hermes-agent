"""Acceptance tests 4, 5 and 7 in their D1 form (GESAMTSKIZZE v4.1 §9).

These are the plan's own acceptance criteria, written as executable tests
rather than prose. D1 covers them "bis Review-State, ohne Inbox-Garantie" --
there is no core inbox yet, so nothing here asserts that a result reached a
conversation. Test 3 (commitment survival) and 8 (decision quality) belong to
phase E and are deliberately absent.

Each test states its fixture, its PASS condition and its negative case in the
same shape the plan uses, so a reviewer can line them up without translating.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db, work_promotion


@pytest.fixture
def board_path(tmp_path):
    db = tmp_path / "board" / "kanban.db"
    db.parent.mkdir(parents=True)
    kanban_db.connect(db).close()
    return db


@pytest.fixture
def board(board_path):
    conn = kanban_db.connect(board_path)
    yield conn
    conn.close()


# ============================================================ Acceptance 4
# Cross-runtime work identity
# Fixture: a project uses delegate + kanban + cron.
# PASS: every run sits under one work graph (work_uid/parent); no second
#       semantic truth.
# Negative: a second adapter call with the same origin_key creates no second
#       item.


def test_a4_runtimes_share_one_work_graph(board, tmp_path):
    """Three producers, one graph, one identity scheme."""
    parent = work_promotion.promote(
        board, origin_kind="conversation", origin_key="proj-1",
        title="the project", payload={"p": 1})

    children = [
        work_promotion.promote(
            board, origin_kind="conversation", origin_key="deleg-1",
            title="delegated slice", payload={"d": 1}, parents=[parent.task_id]),
        work_promotion.promote(
            board, origin_kind="cron", origin_key="nightly-1",
            title="scheduled slice", payload={"c": 1}, parents=[parent.task_id]),
        work_promotion.promote(
            board, origin_kind="process", origin_key="bg-1",
            title="background slice", payload={"b": 1}, parents=[parent.task_id]),
    ]
    board.commit()

    # One identity scheme for all three: every item gets a well-formed work_uid.
    # (board_resolver.resolve was removed with hermes_cli/board_resolver.py —
    # audit decision 2026-07-31, same as test_work_promotion; git history
    # preserves the end-to-end resolution test.)
    board_uuid = kanban_db.get_board_uuid(board)
    for item in [parent, *children]:
        work_uid = kanban_db.build_work_uid(board_uuid, item.task_id)
        assert work_uid and item.task_id in work_uid

    # And one graph: each child hangs under the same parent.
    linked = {
        row[0] for row in board.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (parent.task_id,))
    }
    assert linked == {c.task_id for c in children}, \
        "runs from different runtimes did not land under one work graph"


def test_a4_negative_second_adapter_call_creates_no_second_item(board):
    """The negative case named in the plan."""
    first = work_promotion.promote(
        board, origin_kind="cron", origin_key="nightly-1",
        title="scheduled slice", payload={"c": 1})
    second = work_promotion.promote(
        board, origin_kind="cron", origin_key="nightly-1",
        title="scheduled slice", payload={"c": 1})

    assert second.task_id == first.task_id
    assert second.created is False
    assert board.execute(
        "SELECT COUNT(*) FROM tasks WHERE origin_key='nightly-1'").fetchone()[0] == 1


def test_a4_two_cores_do_not_share_one_key_space(board):
    """Scoping, which the W1 contract requires and I first got wrong.

    An unscoped unique key would deduplicate a second core's request into the
    first core's item -- accepting work and never doing it, dressed up as
    idempotency.
    """
    a = work_promotion.promote(
        board, origin_kind="cron", origin_key="job-1", title="core A job",
        payload={"n": 1}, owner_core_id="core-a")
    b = work_promotion.promote(
        board, origin_kind="cron", origin_key="job-1", title="core B job",
        payload={"n": 2}, owner_core_id="core-b")
    assert a.task_id != b.task_id


# ============================================================ Acceptance 5
# Review truth
# Fixture: an acceptance_required item whose run succeeded.
# PASS: work_state=review; done only after ACCEPT.
# FAIL: done without a review decision (mechanical guard test).
# Negative: an item that does not require review goes straight to done.


def test_a5_successful_run_on_a_review_item_lands_in_review_not_done(board):
    item = work_promotion.promote(
        board, origin_kind="cron", origin_key="risky", title="risky job",
        payload={}, acceptance_required=True)
    board.execute("UPDATE tasks SET status='running' WHERE id=?", (item.task_id,))
    board.commit()

    # The run succeeded -- but succeeding is not the same as being accepted.
    assert kanban_db.complete_task(board, item.task_id, result="run ok") is False
    assert kanban_db.get_task(board, item.task_id).status != "done"


def test_a5_done_only_after_an_accept_decision(board):
    item = work_promotion.promote(
        board, origin_kind="cron", origin_key="risky", title="risky job",
        payload={}, acceptance_required=True)
    board.execute("UPDATE tasks SET status='ready' WHERE id=?", (item.task_id,))
    board.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?,?,?,strftime('%s','now'))",
        (item.task_id, "review_decided", '{"decision": "ACCEPT"}'))
    board.commit()

    assert kanban_db.complete_task(board, item.task_id, result="run ok") is True
    assert kanban_db.get_task(board, item.task_id).status == "done"


def test_a5_fail_case_done_without_a_decision_is_mechanically_refused(board):
    """The plan calls this out as a *mechanical* guard test.

    A convention that says "reviewers should decide first" is not a guard. The
    refusal has to come from the code path every surface converges on.
    """
    item = work_promotion.promote(
        board, origin_kind="cron", origin_key="risky", title="risky job",
        payload={}, acceptance_required=True)
    board.execute("UPDATE tasks SET status='ready' WHERE id=?", (item.task_id,))
    board.commit()

    assert kanban_db.complete_task(board, item.task_id, result="sneaking past") is False
    kinds = [r[0] for r in board.execute(
        "SELECT kind FROM task_events WHERE task_id=?", (item.task_id,))]
    assert "completion_blocked_acceptance_required" in kinds


def test_a5_negative_item_without_review_requirement_goes_straight_to_done(board):
    item = work_promotion.promote(
        board, origin_kind="cron", origin_key="routine", title="routine job",
        payload={}, acceptance_required=False)
    board.execute("UPDATE tasks SET status='ready' WHERE id=?", (item.task_id,))
    board.commit()

    assert kanban_db.complete_task(board, item.task_id, result="fine") is True
    assert kanban_db.get_task(board, item.task_id).status == "done"


def test_a5_a_reject_does_not_open_the_door(board):
    item = work_promotion.promote(
        board, origin_kind="cron", origin_key="risky", title="risky job",
        payload={}, acceptance_required=True)
    board.execute("UPDATE tasks SET status='ready' WHERE id=?", (item.task_id,))
    board.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?,?,?,strftime('%s','now'))",
        (item.task_id, "review_decided", '{"decision": "NEEDS_REPAIR"}'))
    board.commit()

    assert kanban_db.complete_task(board, item.task_id, result="anyway") is False


# ============================================================ Acceptance 7
# No card pollution
# Fixture: a short parallel lookup (tier A).
# PASS: run evidence with a parent reference; no work item, no card.
# Negative: the same lookup plus a promise -> promotion.


def test_a7_ephemeral_work_creates_no_card(board):
    """Tier A: the parent reads the answer inside its own turn."""
    verdict = work_promotion.crosses_commitment_threshold()
    assert verdict.tier == "A"
    assert bool(verdict) is False

    before = board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    if verdict:  # pragma: no cover - the point is that this does not run
        work_promotion.promote(board, origin_kind="conversation",
                               origin_key="lookup", title="lookup", payload={})
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before


def test_a7_negative_the_same_lookup_plus_a_promise_is_promoted(board):
    """The negative case: a promise crosses the threshold."""
    verdict = work_promotion.crosses_commitment_threshold(promised_to_person=True)
    assert verdict.tier == "B"
    assert bool(verdict) is True
    assert any("promise" in r for r in verdict.reasons)

    result = work_promotion.promote(
        board, origin_kind="conversation", origin_key="lookup-with-promise",
        title="look it up and report back", payload={},
        commitment="I will report back", commitment_due_at=1800000000)
    assert result.created is True
    row = board.execute("SELECT commitment FROM tasks WHERE id=?",
                        (result.task_id,)).fetchone()
    assert row["commitment"] == "I will report back"


@pytest.mark.parametrize("kwargs,expected_tier", [
    ({}, "A"),
    ({"survives_turn": True}, "B"),
    ({"promised_to_person": True}, "B"),
    ({"detached_process": True}, "B"),
    ({"expects_artifacts": True}, "B"),
    ({"may_need_retry_or_review": True}, "B"),
    ({"standing": True}, "B"),
])
def test_a7_threshold_criteria(kwargs, expected_tier):
    """Each criterion alone is sufficient.

    Both directions are failures: promoting tier A buries the board in cards
    nobody reads, and a board nobody trusts gets ignored -- which costs more
    than the cards saved. Not promoting tier B loses the work outright.
    """
    assert work_promotion.crosses_commitment_threshold(**kwargs).tier == expected_tier


def test_a7_the_threshold_names_its_reason(board):
    """An explicit predicate cannot drift into promoting everything.

    Requiring the caller to name *why* work is adopted is what stops "just in
    case" promotion, which is how the board filled with noise before.
    """
    verdict = work_promotion.crosses_commitment_threshold(
        detached_process=True, expects_artifacts=True)
    assert len(verdict.reasons) == 2
    assert all(isinstance(r, str) and r for r in verdict.reasons)

"""Idempotent work promotion (D1).

The tests that matter are the two ways a "duplicate" can arrive:

* the same request retried -- must return the existing item quietly;
* a different request reusing the key -- must fail loudly.

Conflating them accepts a request and then never performs it, which is worse
than either a duplicate card or an error.
"""

from __future__ import annotations

import threading

import pytest

from hermes_cli import kanban_db, work_promotion


@pytest.fixture
def board(tmp_path):
    conn = kanban_db.connect(tmp_path / "kanban.db")
    yield conn
    conn.close()


# ------------------------------------------------------------- the basics


def test_first_promotion_creates_a_work_item(board):
    result = work_promotion.promote(
        board, origin_kind="cron", origin_key="job-42",
        title="nightly backup", payload={"job": 42})
    assert result.created is True
    row = board.execute(
        "SELECT origin_kind, origin_key, payload_digest, work_kind "
        "FROM tasks WHERE id=?", (result.task_id,)).fetchone()
    assert row["origin_kind"] == "cron"
    assert row["origin_key"] == "job-42"
    assert row["payload_digest"] == result.payload_digest
    assert row["work_kind"] == "one_shot"


def test_the_same_request_retried_returns_the_existing_item(board):
    first = work_promotion.promote(
        board, origin_kind="cron", origin_key="job-42",
        title="nightly backup", payload={"job": 42})
    second = work_promotion.promote(
        board, origin_kind="cron", origin_key="job-42",
        title="nightly backup", payload={"job": 42})

    assert second.created is False
    assert second.deduplicated is True
    assert second.task_id == first.task_id
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_key_order_in_the_payload_does_not_change_the_digest(board):
    """Otherwise every retry would look like a conflict."""
    a = work_promotion.promote(
        board, origin_kind="cron", origin_key="k", title="t",
        payload={"x": 1, "y": 2})
    b = work_promotion.promote(
        board, origin_kind="cron", origin_key="k", title="t",
        payload={"y": 2, "x": 1})
    assert b.task_id == a.task_id and b.created is False


# ------------------------------------------------- the conflict must be loud


def test_same_key_different_payload_raises(board):
    """The failure this module exists to prevent.

    Treating this as a retry would accept the second request and never
    perform it -- the caller gets a success and the work never happens.
    """
    first = work_promotion.promote(
        board, origin_kind="cron", origin_key="job-42",
        title="nightly backup", payload={"job": 42})

    with pytest.raises(work_promotion.OriginKeyConflict) as exc:
        work_promotion.promote(
            board, origin_kind="cron", origin_key="job-42",
            title="something else entirely", payload={"job": 99})

    assert exc.value.existing_task_id == first.task_id
    assert exc.value.incoming_digest != exc.value.existing_digest
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_the_conflict_message_names_both_digests(board):
    """The operator must be able to tell which request was discarded."""
    work_promotion.promote(board, origin_kind="cron", origin_key="k",
                           title="a", payload={"v": 1})
    with pytest.raises(work_promotion.OriginKeyConflict) as exc:
        work_promotion.promote(board, origin_kind="cron", origin_key="k",
                               title="b", payload={"v": 2})
    message = str(exc.value)
    assert exc.value.existing_digest in message
    assert exc.value.incoming_digest in message


def test_same_key_under_a_different_producer_is_a_separate_item(board):
    """A key is only meaningful inside its producer's namespace."""
    a = work_promotion.promote(board, origin_kind="cron", origin_key="42",
                               title="from cron", payload={"n": 1})
    b = work_promotion.promote(board, origin_kind="webhook", origin_key="42",
                               title="from webhook", payload={"n": 1})
    assert a.task_id != b.task_id
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2


# ---------------------------------------------------------------- validation


def test_unknown_origin_kind_is_rejected(board):
    with pytest.raises(ValueError, match="origin_kind"):
        work_promotion.promote(board, origin_kind="telepathy", origin_key="k",
                               title="t", payload={})


def test_missing_origin_key_is_rejected(board):
    """A promotion without a key cannot be deduplicated.

    Accepting one would quietly reintroduce duplicate cards for that producer,
    so callers with no stable key have to say so by using create_task.
    """
    with pytest.raises(ValueError, match="origin_key"):
        work_promotion.promote(board, origin_kind="cron", origin_key="",
                               title="t", payload={})


# ------------------------------------------------------- the contract fields


def test_promotion_populates_the_work_contract(board):
    result = work_promotion.promote(
        board, origin_kind="conversation", origin_key="conv-1",
        title="research the thing", payload={"q": "why"},
        work_kind="standing", owner_core_id="core-tars",
        origin_conversation_ref="sess-abc", acceptance_required=True,
        definition_of_done="a written answer with sources",
        commitment="I will report back", commitment_due_at=1800000000,
        tenant="default")

    row = board.execute("SELECT * FROM tasks WHERE id=?",
                        (result.task_id,)).fetchone()
    assert row["work_kind"] == "standing"
    assert row["owner_core_id"] == "core-tars"
    assert row["origin_conversation_ref"] == "sess-abc"
    assert row["acceptance_required"] == 1
    assert row["definition_of_done"] == "a written answer with sources"
    assert row["commitment"] == "I will report back"
    assert row["commitment_due_at"] == 1800000000
    assert row["route_tenant_snapshot"] == "default"
    assert row["authority_epoch"] == 0


def test_a_promoted_card_with_acceptance_required_cannot_self_complete(board):
    """D0 guard and D1 promotion must actually meet.

    Each is fine alone; the point is that a card promoted with
    ``acceptance_required=True`` really is refused by the completion path.
    """
    result = work_promotion.promote(
        board, origin_kind="cron", origin_key="needs-review",
        title="risky job", payload={}, acceptance_required=True)
    board.execute("UPDATE tasks SET status='ready' WHERE id=?", (result.task_id,))
    board.commit()

    assert kanban_db.complete_task(board, result.task_id, result="done") is False


# test_promoted_work_is_resolvable_by_work_uid was removed with
# hermes_cli/board_resolver.py (audit decision 2026-07-31: the module had zero
# production consumers and its only production entry was broken; git history
# preserves both). Promotion itself stays covered by every other test here.


# ----------------------------------------------------------------- the race


def test_concurrent_promotions_of_the_same_request_create_one_item(tmp_path):
    """The whole reason promotion is a single INSERT.

    Eight producers retry the same request simultaneously. The old
    check-then-insert loses this: every thread reads "no such key" and every
    thread inserts. Here the unique index decides, and the losers re-read and
    return the winner's item rather than erroring.
    """
    db = tmp_path / "kanban.db"
    kanban_db.connect(db).close()

    results, errors = [], []
    barrier = threading.Barrier(8)

    def producer() -> None:
        conn = kanban_db.connect(db)
        try:
            barrier.wait(timeout=10)
            results.append(work_promotion.promote(
                conn, origin_kind="cron", origin_key="the-job",
                title="contended", payload={"same": True}))
        except Exception as exc:  # pragma: no cover - surfaces real breakage
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=producer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"unexpected errors: {errors}"
    assert len(results) == 8
    assert len({r.task_id for r in results}) == 1, "more than one item was created"
    assert sum(1 for r in results if r.created) == 1, "more than one creator won"

    conn = kanban_db.connect(db)
    total = conn.execute("SELECT COUNT(*) FROM tasks WHERE origin_key='the-job'"
                         ).fetchone()[0]
    conn.close()
    assert total == 1

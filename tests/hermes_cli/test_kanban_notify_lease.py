"""Subscription delivery lease (B0, 2026-07-28) against a REAL SQLite board.

Why this file exists separately from the WebUI poller tests: those run against
a hand-written fake (CI there does not install hermes-agent). A fake cannot
prove that the CAS statements, the fences and the transaction boundaries do
what they claim — it can only prove that the poller calls them in the right
order. The guarantees themselves are tested here, on the real database.

What the lease replaces: until B0 the poller advanced ``last_event_id`` first
and delivered afterwards, undoing the advance from process memory when
delivery failed. A crash in between left the cursor past events nobody had
received, and the 2026-07-27 incident showed what that costs. Now the cursor
moves only after a confirmed hand-over.
"""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _subscribed_task(conn, *, chat_id="sess1"):
    task_id = kb.create_task(conn, title="lease subject", assignee="worker1")
    kb.add_notify_sub(conn, task_id=task_id, platform="webui", chat_id=chat_id)
    return task_id


def _start_cursor(conn, task_id, chat_id="sess1"):
    """Der Cursor, mit dem ein frisches Abo startet.

    Seit dem Upstream-Merge 30.07. schnappt ``add_notify_sub`` den Cursor auf
    ``MAX(task_events.id)`` — ein neues Abo ist AUFGEHOLT und spielt die
    Vorgeschichte der Task nicht nach (Boot-Burst #29905). Die Lease-Tests
    prüfen Bewegungen des Cursors, nicht seinen Absolutwert; also hier fragen
    statt 0 zu unterstellen.
    """
    return _cursor(conn, task_id, chat_id)["last_event_id"]


def _key(task_id, chat_id="sess1"):
    return {"task_id": task_id, "platform": "webui", "chat_id": chat_id}


def _cursor(conn, task_id, chat_id="sess1"):
    row = conn.execute(
        "SELECT last_event_id, active, lease_owner, lease_until, lease_version "
        "FROM kanban_notify_subs WHERE task_id=? AND platform='webui' AND chat_id=?",
        (task_id, chat_id),
    ).fetchone()
    return dict(row) if row is not None else None


def test_acquire_is_single_owner(board):
    task_id = _subscribed_task(board)

    start = _start_cursor(board, task_id)
    first = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)
    assert first is not None
    assert first["last_event_id"] == start

    second = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="B", now=1001)
    assert second is None, "a live lease must not be handed out twice"


def test_expired_lease_is_reclaimable(board):
    """The recovery path: an owner that died must not wedge the subscription."""
    task_id = _subscribed_task(board)
    kb.acquire_notify_sub_lease(board, **_key(task_id), owner="dead",
                                lease_seconds=60, now=1000)

    taken = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="fresh",
                                        now=1061)
    assert taken is not None, "an expired lease is reclaimable, not an error"
    assert taken["lease_version"] == 2, "and the fence moved with it"


def test_acquiring_does_not_move_the_cursor(board):
    """The whole point of B0."""
    task_id = _subscribed_task(board)
    kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)
    assert _cursor(board, task_id)["last_event_id"] == _start_cursor(board, task_id)


def test_commit_advances_the_cursor_and_frees_the_lease(board):
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)

    assert kb.commit_notify_sub_delivery(
        board, **_key(task_id), owner="A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        new_cursor=42,
    ) is True
    state = _cursor(board, task_id)
    assert state["last_event_id"] == 42
    assert state["lease_owner"] is None and state["lease_until"] is None


@pytest.mark.parametrize("wrong", ["owner", "lease_version", "generation"])
def test_commit_is_fenced(board, wrong):
    """A stale holder must not consume events that now belong to someone else."""
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)
    args = {
        "owner": "A",
        "generation": lease["generation"],
        "lease_version": lease["lease_version"],
    }
    args[wrong] = "someone-else" if wrong == "owner" else args[wrong] + 1

    assert kb.commit_notify_sub_delivery(
        board, **_key(task_id), new_cursor=42, **args
    ) is False
    assert _cursor(board, task_id)["last_event_id"] == _start_cursor(board, task_id), "cursor untouched"


def test_release_parks_without_consuming(board):
    """409/5xx: the events stay deliverable, the backoff is durable."""
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)

    assert kb.release_notify_sub_lease(
        board, **_key(task_id), owner="A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        retry_after_seconds=120, now=1000,
    ) is True
    state = _cursor(board, task_id)
    assert state["last_event_id"] == _start_cursor(board, task_id)
    assert state["lease_owner"] is None
    assert state["lease_until"] == 1120, "parked, not free"

    # Inside the park window nobody may take it — that IS the durable backoff.
    assert kb.acquire_notify_sub_lease(board, **_key(task_id), owner="B", now=1119) is None
    assert kb.acquire_notify_sub_lease(board, **_key(task_id), owner="B", now=1121) is not None


def test_release_without_backoff_frees_immediately(board):
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)
    kb.release_notify_sub_lease(
        board, **_key(task_id), owner="A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        retry_after_seconds=0, now=1000,
    )
    assert _cursor(board, task_id)["lease_until"] is None
    assert kb.acquire_notify_sub_lease(board, **_key(task_id), owner="B", now=1000) is not None


def test_retire_deactivates_and_records_in_one_transaction(board):
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)

    assert kb.retire_notify_sub_with_marker(
        board, **_key(task_id), owner="A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        reason="session_not_found",
    ) is True
    assert _cursor(board, task_id)["active"] == 0
    markers = [
        row for row in board.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? AND kind=?",
            (task_id, "notify_delivery_failed"),
        )
    ]
    assert len(markers) == 1, "deactivation without evidence is exactly the drift we banned"
    assert "session_not_found" in (markers[0]["payload"] or "")


def test_retire_is_fenced_too(board):
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)

    assert kb.retire_notify_sub_with_marker(
        board, **_key(task_id), owner="not-A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        reason="session_not_found",
    ) is False
    assert _cursor(board, task_id)["active"] == 1
    assert board.execute(
        "SELECT COUNT(*) c FROM task_events WHERE task_id=? AND kind='notify_delivery_failed'",
        (task_id,),
    ).fetchone()["c"] == 0, "a rejected retire must not leave a marker either"


def test_inactive_subscription_is_not_leasable(board):
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)
    kb.retire_notify_sub_with_marker(
        board, **_key(task_id), owner="A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        reason="session_not_found",
    )
    assert kb.acquire_notify_sub_lease(board, **_key(task_id), owner="B", now=2000) is None


def test_lease_columns_survive_a_legacy_table_rebuild(board, monkeypatch):
    """The columns must survive the REBUILD path, not just the ADD path.

    TARS' B0 review, P1 — and my own test was worthless before: it asserted the
    columns exist on a database the fixture had just created with the current
    schema. That can never fail. The dangerous case is a legacy board with a
    TEXT ``last_event_id``: the migration adds the lease columns, then
    ``_rebuild_drifted_tables`` recreates the table from a HAND-WRITTEN DDL —
    and if that DDL does not carry them, they are gone again and the poller
    fail-closes this board into permanent silence.

    So build the legacy shape for real and drive the migration over it.
    """
    board.execute("DROP TABLE kanban_notify_subs")
    board.execute(
        "CREATE TABLE kanban_notify_subs ("
        " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,"
        " notifier_profile TEXT, created_at INTEGER NOT NULL,"
        " last_event_id TEXT,"                     # the legacy shape
        " active INTEGER NOT NULL DEFAULT 1,"
        " generation INTEGER NOT NULL DEFAULT 1,"
        " PRIMARY KEY (task_id, platform, chat_id, thread_id))"
    )
    board.execute(
        "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, thread_id,"
        " created_at, last_event_id) VALUES ('t_legacy','webui','sess1','',1,'7')"
    )
    board.commit()

    kb.init_db()          # runs the migration + rebuild over the legacy table

    cols = {row["name"] for row in board.execute("PRAGMA table_info(kanban_notify_subs)")}
    assert {"lease_owner", "lease_until", "lease_version"} <= cols, (
        "the rebuild DDL dropped the lease columns again"
    )
    row = board.execute(
        "SELECT last_event_id, lease_version FROM kanban_notify_subs "
        "WHERE task_id='t_legacy'"
    ).fetchone()
    assert row["last_event_id"] == 7, "the legacy TEXT cursor was carried over"
    assert row["lease_version"] == 0

    # And the lease actually works on the rebuilt table.
    assert kb.acquire_notify_sub_lease(
        board, task_id="t_legacy", platform="webui", chat_id="sess1",
        owner="A", now=1000,
    ) is not None


def test_a_deactivated_subscription_cannot_be_committed(board):
    """P2: remove_notify_sub must invalidate a live lease, not just deactivate."""
    task_id = _subscribed_task(board)
    lease = kb.acquire_notify_sub_lease(board, **_key(task_id), owner="A", now=1000)

    kb.remove_notify_sub(board, **_key(task_id))

    assert kb.commit_notify_sub_delivery(
        board, **_key(task_id), owner="A",
        generation=lease["generation"], lease_version=lease["lease_version"],
        new_cursor=99,
    ) is False, "a holder must not write into a deactivated generation"
    assert _cursor(board, task_id)["last_event_id"] == _start_cursor(board, task_id)

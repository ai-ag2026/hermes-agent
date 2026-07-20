"""Durable terminal-state ledger for background processes (P0).

Why this module exists
----------------------
``ProcessRegistry`` tracked the whole life of a background process **only in
memory**. ``_move_to_finished()`` moved the session into ``self._finished``
(a dict) and pushed a notification onto ``self.completion_queue``
(a ``queue.Queue``); the on-disk checkpoint ``processes.json`` then actively
*excluded* finished sessions (``if not s.exited``). If the gateway died
between a process finishing and something consuming the notification, the
result was gone — not merely undelivered, but nonexistent. There was no row
to recover from.

That is the same loss class Phase B closed one layer higher, where results
were persisted but never delivered. Here persistence was missing entirely,
which is why this slice runs *before* D0: D0/D1 would otherwise build on a
producer layer that is known to lose work.

Contract (ADR-8 §4)
-------------------
1. Persist the run **at spawn**: stable ``process_id``, session/routing
   reference, pid **plus process start time** (against pid reuse), state
   ``running``.
2. Terminalise **exactly once**: ``completed|failed|killed``, exit code and
   reason, a bounded redacted output reference.
3. Terminal run **and** ``process_outbox(pending)`` in **one** transaction.
4. Restart recovery: pid+start time alive -> stays ``running``; owner
   provably dead with no terminal row -> ``unknown``. **Never invent success,
   never auto-retry an ``unknown``.**
5. Delivery recovery: drain ``pending`` normally; a crash during
   ``attempting`` becomes ``ambiguous``, never silently ``pending`` or
   ``delivered``. Claim/lease guards against double pickup.
6. Retention: an open outbox row is **never** deleted by a cap — backpressure
   and alarm instead. ``unknown`` is cleaned only by an explicit rule.
7. Crash-point tests cover: before terminal commit, after terminal commit but
   before drain, during delivery, pid reuse, ``unknown`` recovery, and
   retention of open obligations.

Storage
-------
A profile-local ``processes.db`` rather than a table in ``state.db``. Two
reasons: ``state.db`` is already 7.6 GB and ADR-8 flags appending further
outbox load there as an open ops question, and the upstream cron work
(``cron/executions.db``) establishes profile-local durable ledgers as the
house pattern for producer-owned run truth.

The path is resolved **at call time**, not at import. ``process_registry``
and ``cron/jobs.py`` both freeze their paths at import, which is exactly why
test isolation had to patch module constants after the fact and why a
production cron store got polluted on 2026-07-19/20. This module does not
repeat that mistake.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Bounded snapshot of process output kept in the ledger. Enough to reconstruct
# what happened without turning the ledger into a log store.
MAX_OUTPUT_SNAPSHOT_CHARS = 8_000

# Terminal rows may be pruned above this count. Rows with an unsettled outbox
# obligation are exempt -- see _prune_locked().
MAX_TERMINAL_RUNS = 2_000

# Soft cap for unsettled obligations. Exceeding it raises an alarm; it never
# licenses deletion.
MAX_PENDING_OBLIGATIONS = 500

# A claim older than this is treated as abandoned by its owner.
CLAIM_LEASE_SECONDS = 300

RUN_STATES = ("running", "completed", "failed", "killed", "unknown")
TERMINAL_STATES = ("completed", "failed", "killed", "unknown")
DELIVERY_STATES = ("pending", "attempting", "delivered", "ambiguous")

_DB_LOCK = threading.RLock()
_SCHEMA_READY: set = set()


def ledger_path() -> Path:
    """Resolve the ledger path **at call time**.

    Deliberately not a module constant: freezing this at import is the exact
    defect that let tests write into the production cron store.

    But call-time resolution alone is not enough for a process's *lifecycle*.
    A background process is spawned on one thread and terminalised later on
    its reader thread -- possibly after the test that spawned it has torn down
    its ``HERMES_HOME`` redirect. Resolving again at that moment sends the
    terminal write to a different store than the spawn write.

    That is not hypothetical: during the 2026-07-20 full-suite run exactly one
    row appeared in the production ``~/.hermes/processes.db``, with NULL
    command and cwd -- the signature of a terminal write with no matching
    spawn row. My fix for the import-freeze leak had created a
    lifetime-mismatch leak.

    So callers that span a process lifetime must capture this ONCE at spawn
    and pass it back in as ``db_path``. See ``ProcessRegistry._ledger_*``.
    """
    from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "processes.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS process_runs (
    process_id       TEXT PRIMARY KEY,
    session_id       TEXT,
    session_key      TEXT,
    command          TEXT,
    cwd              TEXT,
    task_id          TEXT,
    pid              INTEGER,
    pid_scope        TEXT,
    host_start_time  REAL,
    owner_pid        INTEGER,
    owner_start_time REAL,
    state            TEXT NOT NULL,
    exit_code        INTEGER,
    completion_reason TEXT,
    termination_source TEXT,
    output_snapshot  TEXT,
    routing_json     TEXT,
    started_at       REAL,
    terminalised_at  REAL,
    updated_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS process_outbox (
    obligation_id    TEXT PRIMARY KEY,
    process_id       TEXT NOT NULL,
    delivery_state   TEXT NOT NULL,
    attempts         INTEGER NOT NULL DEFAULT 0,
    claim_owner      TEXT,
    claim_generation INTEGER NOT NULL DEFAULT 0,
    lease_expires_at REAL,
    payload_json     TEXT NOT NULL,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    settled_at       REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_outbox_one_per_process
    ON process_outbox(process_id);
CREATE INDEX IF NOT EXISTS idx_outbox_open
    ON process_outbox(delivery_state) WHERE delivery_state != 'delivered';
CREATE INDEX IF NOT EXISTS idx_runs_state ON process_runs(state);
"""


@contextmanager
def _connect(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Open the ledger with durability settings and a real busy timeout.

    ``BEGIN IMMEDIATE`` is used by callers that write, so that the terminal
    row and its outbox obligation land in one transaction with a writer lock
    held from the start -- ADR-8 requires the same guarantee the kanban
    ``write_txn()`` provides, not the weaker implicit DBAPI transaction that
    ``async_delegations`` uses today.
    """
    path = Path(db_path) if db_path is not None else ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    try:
        key = str(path)
        if key not in _SCHEMA_READY:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(_SCHEMA)
            _SCHEMA_READY.add(key)
        else:
            conn.execute("PRAGMA synchronous=FULL")
        yield conn
    finally:
        conn.close()


@contextmanager
def _write_txn(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _own_start_time() -> Optional[float]:
    try:
        from gateway.status import get_process_start_time

        return get_process_start_time(os.getpid())
    except Exception:
        return None


def _pid_alive(pid: Optional[int], start_time: Optional[float]) -> bool:
    """Liveness with pid-reuse protection.

    Fail-safe: when liveness cannot be determined the process counts as
    **alive**. Declaring a live process dead would let recovery terminalise a
    run that is still producing output -- inventing an outcome, which the
    contract forbids.
    """
    if not pid:
        return False
    try:
        from gateway.status import _pid_exists, get_process_start_time
    except Exception:
        return True
    try:
        if not _pid_exists(pid):
            return False
    except Exception:
        return True
    if start_time is None:
        return True
    try:
        current = get_process_start_time(pid)
    except Exception:
        return True
    if current is None:
        return True
    # Same pid, different start time => the pid was recycled; the original
    # process is gone.
    return abs(current - start_time) < 1.0


# ---------------------------------------------------------------- spawn side


def record_spawn(
    process_id: str,
    *,
    session_id: Optional[str] = None,
    session_key: Optional[str] = None,
    command: Optional[str] = None,
    cwd: Optional[str] = None,
    task_id: Optional[str] = None,
    pid: Optional[int] = None,
    pid_scope: Optional[str] = None,
    host_start_time: Optional[float] = None,
    routing: Optional[Dict[str, Any]] = None,
    started_at: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Persist a run at spawn time (contract §1).

    Idempotent: re-recording the same ``process_id`` leaves an existing row
    untouched, so a retried spawn path cannot resurrect a terminalised run.
    """
    now = time.time()
    with _DB_LOCK, _connect(db_path) as conn:
        with _write_txn(conn):
            conn.execute(
                """INSERT INTO process_runs (
                       process_id, session_id, session_key, command, cwd,
                       task_id, pid, pid_scope, host_start_time, owner_pid,
                       owner_start_time, state, routing_json, started_at,
                       updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?, 'running', ?,?,?)
                   ON CONFLICT(process_id) DO NOTHING""",
                (process_id, session_id, session_key, command, cwd, task_id,
                 pid, pid_scope, host_start_time, os.getpid(),
                 _own_start_time(), json.dumps(routing or {}),
                 started_at if started_at is not None else now, now),
            )


# ------------------------------------------------------------- terminal side


def record_terminal(
    process_id: str,
    *,
    state: str,
    exit_code: Optional[int] = None,
    completion_reason: Optional[str] = None,
    termination_source: Optional[str] = None,
    output: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    notify: bool = True,
    db_path: Optional[Path],
) -> bool:
    """Terminalise a run and enqueue its obligation in ONE transaction.

    ``db_path`` is a REQUIRED keyword with no default. A default would let a
    caller omit it by accident and silently fall back to a globally resolved
    path -- which is the leak. Making it required moves the mistake from
    runtime to the call site, where it is visible.

    Contract §2 and §3. Returns ``True`` when this call performed the
    terminalisation, ``False`` when the run was already terminal — the caller
    uses that to decide whether to emit a notification, which is how duplicate
    ``[IMPORTANT: ...]`` messages are avoided.

    The single-statement guard ``WHERE state='running'`` makes terminal states
    immutable: a second caller (``kill_process`` racing the reader thread)
    updates zero rows and gets ``False``.
    """
    if state not in TERMINAL_STATES:
        raise ValueError(f"not a terminal state: {state!r}")
    if db_path is None:
        # FAIL-CLOSED. Falling back to a globally resolved path here is what
        # produced the production leak: a background thread outliving its
        # caller's environment writes to whatever HERMES_HOME now says, and
        # `_connect()` will happily create the directory, turning a misroute
        # into a formally valid second database instead of a visible error.
        #
        # Refusing loses this outcome -- but an unbound terminal write had
        # already lost it, by putting it in a store nobody reads. A loud
        # refusal is recoverable; a silent stray database is not.
        logger.error(
            "process ledger: refusing to terminalise %s without a bound "
            "store. The caller must bind one at spawn; writing to a "
            "globally-resolved path is how test data reaches production.",
            process_id,
        )
        return False
    now = time.time()
    snapshot = (output or "")[-MAX_OUTPUT_SNAPSHOT_CHARS:]

    with _DB_LOCK, _connect(db_path) as conn:
        with _write_txn(conn):
            cur = conn.execute(
                """UPDATE process_runs
                      SET state=?, exit_code=?, completion_reason=?,
                          termination_source=?, output_snapshot=?,
                          terminalised_at=?, updated_at=?
                    WHERE process_id=? AND state='running'""",
                (state, exit_code, completion_reason, termination_source,
                 snapshot, now, now, process_id),
            )
            if cur.rowcount != 1:
                existing = conn.execute(
                    "SELECT state FROM process_runs WHERE process_id=?",
                    (process_id,),
                ).fetchone()
                if existing is not None:
                    # Already terminal -- a second caller (kill racing the
                    # reader) must not overwrite the first outcome.
                    return False
                # No spawn row at all: the spawn-side write failed or this
                # producer never recorded one. Losing the outcome here would
                # be precisely the failure this module exists to prevent, so
                # the run is created directly in its terminal state.
                logger.warning(
                    "process ledger: terminalising %s without a spawn row -- "
                    "recording the outcome anyway; the spawn-side write was "
                    "lost", process_id,
                )
                conn.execute(
                    """INSERT INTO process_runs (
                           process_id, state, exit_code, completion_reason,
                           termination_source, output_snapshot, owner_pid,
                           owner_start_time, terminalised_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (process_id, state, exit_code, completion_reason,
                     termination_source, snapshot, os.getpid(),
                     _own_start_time(), now, now),
                )
            if notify:
                # Same transaction as the state change -- a post-commit hook
                # would be an observer, not a delivery guarantee (ADR-8).
                conn.execute(
                    """INSERT INTO process_outbox (
                           obligation_id, process_id, delivery_state,
                           payload_json, created_at, updated_at)
                       VALUES (?,?, 'pending', ?,?,?)
                       ON CONFLICT(process_id) DO NOTHING""",
                    (f"obl_{uuid.uuid4().hex[:16]}", process_id,
                     json.dumps(payload or {}), now, now),
                )
        _prune_locked(conn)
    return True


# ------------------------------------------------------------- delivery side


def claim_next(owner: str, limit: int = 1) -> List[Dict[str, Any]]:
    """Claim deliverable obligations under a lease (contract §5).

    Claimable rows are ``pending``, plus ``attempting`` rows whose lease has
    expired (their owner died mid-delivery). ``ambiguous`` rows are **not**
    claimable automatically -- resolving them needs an idempotent ack or an
    operator decision, never a silent retry.
    """
    now = time.time()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _connect() as conn:
        with _write_txn(conn):
            rows = conn.execute(
                """SELECT obligation_id, process_id, payload_json,
                          claim_generation, attempts
                     FROM process_outbox
                    WHERE delivery_state='pending'
                       OR (delivery_state='attempting'
                           AND lease_expires_at IS NOT NULL
                           AND lease_expires_at < ?)
                    ORDER BY created_at ASC LIMIT ?""",
                (now, limit),
            ).fetchall()
            for obligation_id, process_id, payload_json, generation, attempts in rows:
                cur = conn.execute(
                    """UPDATE process_outbox
                          SET delivery_state='attempting', claim_owner=?,
                              claim_generation=?, lease_expires_at=?,
                              attempts=attempts+1, updated_at=?
                        WHERE obligation_id=? AND claim_generation=?""",
                    (owner, generation + 1, now + CLAIM_LEASE_SECONDS, now,
                     obligation_id, generation),
                )
                if cur.rowcount == 1:
                    claimed.append({
                        "obligation_id": obligation_id,
                        "process_id": process_id,
                        "payload": json.loads(payload_json or "{}"),
                        "claim_generation": generation + 1,
                        "attempts": attempts + 1,
                    })
    return claimed


def mark_delivered(obligation_id: str, claim_generation: int) -> bool:
    """Settle an obligation. CAS on the generation the caller claimed under."""
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        with _write_txn(conn):
            cur = conn.execute(
                """UPDATE process_outbox
                      SET delivery_state='delivered', settled_at=?,
                          updated_at=?, claim_owner=NULL, lease_expires_at=NULL
                    WHERE obligation_id=? AND claim_generation=?
                      AND delivery_state='attempting'""",
                (now, now, obligation_id, claim_generation),
            )
            return cur.rowcount == 1


def release_claim(obligation_id: str, claim_generation: int) -> bool:
    """Return a failed delivery to ``pending`` for another consumer."""
    now = time.time()
    with _DB_LOCK, _connect() as conn:
        with _write_txn(conn):
            cur = conn.execute(
                """UPDATE process_outbox
                      SET delivery_state='pending', claim_owner=NULL,
                          lease_expires_at=NULL, updated_at=?
                    WHERE obligation_id=? AND claim_generation=?
                      AND delivery_state='attempting'""",
                (now, obligation_id, claim_generation),
            )
            return cur.rowcount == 1


# ---------------------------------------------------------------- recovery


def recover_after_restart() -> Dict[str, int]:
    """Reconcile ledger state with reality after a gateway restart.

    Contract §4 and §5. Two independent reconciliations:

    * A ``running`` row whose owning gateway is provably gone becomes
      ``unknown``. It never becomes ``completed`` -- the outcome genuinely is
      not known, and recording a guess would be worse than recording
      ignorance. No obligation is enqueued and nothing is retried.
    * An ``attempting`` obligation left behind by a crashed consumer becomes
      ``ambiguous``: the delivery may or may not have reached its target.
      Silently returning it to ``pending`` would risk a duplicate; silently
      marking it ``delivered`` would risk a loss. Both are decisions this
      layer is not entitled to make.
    """
    now = time.time()
    result = {"marked_unknown": 0, "marked_ambiguous": 0, "still_running": 0}
    with _DB_LOCK, _connect() as conn:
        with _write_txn(conn):
            rows = conn.execute(
                """SELECT process_id, owner_pid, owner_start_time
                     FROM process_runs WHERE state='running'"""
            ).fetchall()
            for process_id, owner_pid, owner_start_time in rows:
                if _pid_alive(owner_pid, owner_start_time):
                    result["still_running"] += 1
                    continue
                cur = conn.execute(
                    """UPDATE process_runs
                          SET state='unknown', completion_reason=?,
                              terminalised_at=?, updated_at=?
                        WHERE process_id=? AND state='running'""",
                    ("owner process gone before terminal state was recorded",
                     now, now, process_id),
                )
                result["marked_unknown"] += cur.rowcount

            cur = conn.execute(
                """UPDATE process_outbox
                      SET delivery_state='ambiguous', claim_owner=NULL,
                          lease_expires_at=NULL, updated_at=?
                    WHERE delivery_state='attempting'""",
                (now,),
            )
            result["marked_ambiguous"] = cur.rowcount
    if result["marked_unknown"] or result["marked_ambiguous"]:
        logger.warning(
            "process ledger recovery: %d run(s) marked unknown, %d "
            "obligation(s) marked ambiguous -- neither is retried "
            "automatically; resolve via idempotent ack or operator decision",
            result["marked_unknown"], result["marked_ambiguous"],
        )
    return result


# ---------------------------------------------------------------- retention


def _prune_locked(conn: sqlite3.Connection) -> None:
    """Bound terminal history without ever dropping an open obligation.

    Contract §6. This is the Phase-B retention lesson applied one layer down:
    a cap may delete acknowledged history only. When unsettled obligations
    exceed the soft cap the answer is backpressure and an alarm, never
    deletion -- deleting them is precisely the data loss this module exists
    to prevent.
    """
    open_count = conn.execute(
        "SELECT COUNT(*) FROM process_outbox WHERE delivery_state!='delivered'"
    ).fetchone()[0]
    if open_count > MAX_PENDING_OBLIGATIONS:
        logger.warning(
            "process outbox backpressure: %d unsettled obligations exceed the "
            "soft cap of %d; rows are retained -- drain the consumer or "
            "investigate stalled delivery",
            open_count, MAX_PENDING_OBLIGATIONS,
        )

    placeholders = ",".join("?" * len(TERMINAL_STATES))
    terminal_count = conn.execute(
        f"SELECT COUNT(*) FROM process_runs WHERE state IN ({placeholders})",
        TERMINAL_STATES,
    ).fetchone()[0]
    excess = terminal_count - MAX_TERMINAL_RUNS
    if excess <= 0:
        return
    with _write_txn(conn):
        conn.execute(
            f"""DELETE FROM process_runs WHERE process_id IN (
                  SELECT r.process_id FROM process_runs r
                  LEFT JOIN process_outbox o ON o.process_id = r.process_id
                  WHERE r.state IN ({placeholders})
                    AND (o.process_id IS NULL
                         OR o.delivery_state = 'delivered')
                  ORDER BY r.terminalised_at ASC LIMIT ?)""",
            (*TERMINAL_STATES, excess),
        )
        conn.execute(
            """DELETE FROM process_outbox WHERE delivery_state='delivered'
                 AND process_id NOT IN (SELECT process_id FROM process_runs)"""
        )


# ------------------------------------------------------------------ queries


def get_run(process_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM process_runs WHERE process_id=?", (process_id,)
        ).fetchone()
        return dict(row) if row else None


def get_obligation(process_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK, _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM process_outbox WHERE process_id=?", (process_id,)
        ).fetchone()
        return dict(row) if row else None


def counts() -> Dict[str, int]:
    """Operational summary for status surfaces and tests."""
    with _DB_LOCK, _connect() as conn:
        out: Dict[str, int] = {}
        for state, n in conn.execute(
            "SELECT state, COUNT(*) FROM process_runs GROUP BY state"
        ):
            out[f"runs_{state}"] = n
        for state, n in conn.execute(
            "SELECT delivery_state, COUNT(*) FROM process_outbox "
            "GROUP BY delivery_state"
        ):
            out[f"outbox_{state}"] = n
        return out


def reset_schema_cache() -> None:
    """Drop the per-path schema memo. Tests redirect HERMES_HOME per test."""
    _SCHEMA_READY.clear()

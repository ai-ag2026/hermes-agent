"""Crash-point tests for the durable background-process ledger (P0).

The contract in ADR-8 §4.7 names six scenarios that must be covered:
before the terminal commit, after the terminal commit but before the drain,
during delivery, pid reuse, ``unknown`` recovery, and retention of open
obligations. Each has a test below, plus the invariants they depend on.

The point of these tests is not that the happy path works. It is that every
crash point either preserves the result or records honest ignorance -- and
that nothing anywhere invents an outcome.
"""

from __future__ import annotations

import pytest

from tools import process_ledger as pl


@pytest.fixture(autouse=True)
def _ledger_home(tmp_path, monkeypatch):
    """Point the ledger at a per-test home.

    ``ledger_path()`` resolves at call time, so redirecting ``HERMES_HOME`` is
    enough -- no module-constant patching needed. That is deliberate: the
    import-time freeze in ``cron/jobs.py`` and ``process_registry.py`` is what
    let tests write into production stores.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pl.reset_schema_cache()
    yield
    pl.reset_schema_cache()


def _await_terminal(process_id, timeout=15.0):
    """Wait for the ledger row to reach a terminal state.

    Terminalisation happens on the registry's reader thread, so
    ``wait()`` returning "exited" does not imply the ledger has been written
    yet. Polling here tests the guarantee that actually matters -- the row
    arrives -- without asserting on an internal ordering the registry never
    promised.
    """
    import time as _t

    deadline = _t.time() + timeout
    while _t.time() < deadline:
        run = pl.get_run(process_id)
        if run and run["state"] in pl.TERMINAL_STATES:
            return run
        _t.sleep(0.05)
    return pl.get_run(process_id)


def _spawn(process_id="p1", **kw):
    pl.record_spawn(process_id, command="sleep 1", session_id="s1", **kw)
    return process_id


# ------------------------------------------------------- the loss path itself


def test_terminal_state_survives_process_death():
    """The defect this module exists to fix.

    Before P0 the terminal state lived only in an in-memory dict and a Queue,
    and the checkpoint file actively excluded finished sessions. Death between
    exit and pickup destroyed the result. Now a fresh reader -- standing in
    for a restarted gateway -- still finds it.
    """
    _spawn("p1")
    assert pl.record_terminal("p1", state="completed", exit_code=0,
                              output="the result", payload={"r": 1}) is True

    pl.reset_schema_cache()  # simulate a cold process attaching to the store
    run = pl.get_run("p1")
    assert run["state"] == "completed"
    assert run["exit_code"] == 0
    assert "the result" in run["output_snapshot"]
    assert pl.get_obligation("p1")["delivery_state"] == "pending"


def test_crash_before_terminal_commit_leaves_run_recoverable():
    """Crash point 1: process exited, terminal state never written.

    The run must not be silently dropped and must not be guessed as success.
    Recovery classifies it ``unknown`` -- and enqueues nothing, because there
    is no outcome to deliver.
    """
    _spawn("p1", pid=999_999)
    # Owner recorded as a pid that cannot be alive.
    with pl._connect() as conn:
        conn.execute(
            "UPDATE process_runs SET owner_pid=?, owner_start_time=? "
            "WHERE process_id='p1'", (999_998, 1.0))

    result = pl.recover_after_restart()

    assert result["marked_unknown"] == 1
    run = pl.get_run("p1")
    assert run["state"] == "unknown"
    assert pl.get_obligation("p1") is None, "unknown must not enqueue delivery"


def test_crash_after_terminal_commit_before_drain_keeps_obligation():
    """Crash point 2: result committed, nobody consumed it yet.

    This is the exact scenario that used to lose data. The obligation must
    still be claimable after a restart.
    """
    _spawn("p1")
    pl.record_terminal("p1", state="completed", exit_code=0, payload={"r": 1})

    pl.recover_after_restart()

    claimed = pl.claim_next("consumer-a")
    assert len(claimed) == 1
    assert claimed[0]["process_id"] == "p1"
    assert claimed[0]["payload"] == {"r": 1}


def test_crash_during_delivery_becomes_ambiguous_not_pending():
    """Crash point 3: consumer died mid-delivery.

    Returning the row to ``pending`` risks a duplicate; marking it
    ``delivered`` risks a loss. Neither call is this layer's to make, so the
    row becomes ``ambiguous`` and is excluded from automatic claiming.
    """
    _spawn("p1")
    pl.record_terminal("p1", state="completed", payload={"r": 1})
    claimed = pl.claim_next("consumer-a")
    assert claimed  # consumer now holds the claim, then "dies"

    pl.recover_after_restart()

    assert pl.get_obligation("p1")["delivery_state"] == "ambiguous"
    assert pl.claim_next("consumer-b") == [], \
        "ambiguous must never be auto-retried"


def test_expired_lease_is_reclaimable_but_delivered_is_not():
    """A consumer that stalls without crashing must not block delivery forever."""
    _spawn("p1")
    pl.record_terminal("p1", state="completed", payload={"r": 1})
    first = pl.claim_next("consumer-a")[0]

    assert pl.claim_next("consumer-b") == [], "lease is still valid"

    with pl._connect() as conn:  # expire the lease
        conn.execute("UPDATE process_outbox SET lease_expires_at=1.0")

    second = pl.claim_next("consumer-b")
    assert len(second) == 1
    assert second[0]["claim_generation"] > first["claim_generation"]

    # The original owner's stale generation can no longer settle the row.
    assert pl.mark_delivered(first["obligation_id"], first["claim_generation"]) is False
    assert pl.mark_delivered(second[0]["obligation_id"],
                             second[0]["claim_generation"]) is True
    assert pl.claim_next("consumer-c") == []


def test_pid_reuse_does_not_keep_a_dead_run_alive():
    """Crash point 4: the owner pid was recycled by an unrelated process.

    Matching on pid alone would see "alive" and leave the run ``running``
    forever. The start-time fingerprint catches it.
    """
    import os

    _spawn("p1")
    with pl._connect() as conn:
        # Our own pid, but a start time that cannot belong to it.
        conn.execute(
            "UPDATE process_runs SET owner_pid=?, owner_start_time=? "
            "WHERE process_id='p1'", (os.getpid(), 1.0))

    assert pl.recover_after_restart()["marked_unknown"] == 1
    assert pl.get_run("p1")["state"] == "unknown"


def test_live_owner_is_left_running():
    """The mirror image: a genuinely live owner must not be terminalised.

    Fail-safe direction matters. Declaring a live process dead would let
    recovery invent an outcome for work still in flight.
    """
    _spawn("p1")  # record_spawn stores our real pid and start time
    result = pl.recover_after_restart()
    assert result["still_running"] == 1
    assert result["marked_unknown"] == 0
    assert pl.get_run("p1")["state"] == "running"


def test_terminalisation_happens_exactly_once():
    """kill_process racing the reader thread must not double-notify.

    The single-statement ``WHERE state='running'`` guard makes the second
    caller a no-op, which is what the caller uses to suppress the duplicate
    ``[IMPORTANT: ...]`` message.
    """
    _spawn("p1")
    assert pl.record_terminal("p1", state="completed", exit_code=0) is True
    assert pl.record_terminal("p1", state="killed", exit_code=137) is False

    run = pl.get_run("p1")
    assert run["state"] == "completed", "terminal state must be immutable"
    assert run["exit_code"] == 0


def test_spawn_is_idempotent_and_cannot_resurrect_a_terminal_run():
    _spawn("p1")
    pl.record_terminal("p1", state="failed", exit_code=1)
    pl.record_spawn("p1", command="sleep 1")
    assert pl.get_run("p1")["state"] == "failed"


def test_retention_never_deletes_an_open_obligation():
    """Crash point 6, and the Phase-B lesson one layer down.

    A cap may delete acknowledged history only. Unsettled obligations survive
    even when that leaves the table above the cap -- deleting them is exactly
    the loss this module prevents.
    """
    monkey_cap = 5
    original = pl.MAX_TERMINAL_RUNS
    pl.MAX_TERMINAL_RUNS = monkey_cap
    try:
        for i in range(20):
            pid = f"delivered_{i}"
            _spawn(pid)
            pl.record_terminal(pid, state="completed", payload={"i": i})
            claim = pl.claim_next(f"c{i}")[0]
            pl.mark_delivered(claim["obligation_id"], claim["claim_generation"])

        for i in range(3):
            pid = f"open_{i}"
            _spawn(pid)
            pl.record_terminal(pid, state="completed", payload={"i": i})

        # Force one more prune pass.
        _spawn("trigger")
        pl.record_terminal("trigger", state="completed", payload={})

        for i in range(3):
            assert pl.get_run(f"open_{i}") is not None, \
                "a run with an unsettled obligation was pruned"
            assert pl.get_obligation(f"open_{i}")["delivery_state"] == "pending"

        c = pl.counts()
        assert c.get("outbox_pending", 0) >= 3
    finally:
        pl.MAX_TERMINAL_RUNS = original


def test_backpressure_warns_instead_of_deleting(caplog):
    original = pl.MAX_PENDING_OBLIGATIONS
    pl.MAX_PENDING_OBLIGATIONS = 2
    try:
        for i in range(4):
            pid = f"p{i}"
            _spawn(pid)
            pl.record_terminal(pid, state="completed", payload={})
        with caplog.at_level("WARNING"):
            _spawn("px")
            pl.record_terminal("px", state="completed", payload={})
        assert any("backpressure" in r.message for r in caplog.records)
        assert pl.counts().get("outbox_pending", 0) == 5, \
            "backpressure must retain, never delete"
    finally:
        pl.MAX_PENDING_OBLIGATIONS = original


def test_unknown_run_is_not_reclassified_on_a_second_recovery():
    """Crash point 5: repeated restarts must be stable.

    ``unknown`` is terminal. A later recovery pass must not churn it, and
    must not decide retroactively that it succeeded.
    """
    _spawn("p1")
    with pl._connect() as conn:
        conn.execute("UPDATE process_runs SET owner_pid=?, owner_start_time=?",
                     (999_998, 1.0))
    pl.recover_after_restart()
    first = pl.get_run("p1")

    second_pass = pl.recover_after_restart()
    assert second_pass["marked_unknown"] == 0
    assert pl.get_run("p1")["state"] == "unknown"
    assert pl.get_run("p1")["terminalised_at"] == first["terminalised_at"]


def test_output_snapshot_is_bounded():
    _spawn("p1")
    pl.record_terminal("p1", state="completed", output="x" * 50_000)
    assert len(pl.get_run("p1")["output_snapshot"]) == pl.MAX_OUTPUT_SNAPSHOT_CHARS


def test_non_terminal_state_is_rejected():
    _spawn("p1")
    with pytest.raises(ValueError):
        pl.record_terminal("p1", state="running")


def test_ledger_path_resolves_at_call_time(tmp_path, monkeypatch):
    """Guards against reintroducing the import-time path freeze.

    ``cron/jobs.py`` and ``process_registry.py`` bind their paths at import,
    which is why a production cron store was polluted on 2026-07-19/20 and why
    the isolation fixture has to patch module constants. This module must stay
    free of that defect.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "a"))
    first = pl.ledger_path()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "b"))
    assert pl.ledger_path() != first


# ------------------------------------------------- registry integration (P0)


def test_registry_terminal_reaches_the_ledger():
    """End-to-end: a real ProcessRegistry run leaves a durable row.

    This is the integration the unit tests above cannot prove: that
    ``_move_to_finished`` actually calls the ledger, with a payload a
    consumer could act on.
    """
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 0.4 && echo p0-integration")
    session.notify_on_complete = True
    assert registry.wait(session.id, timeout=30)["status"] == "exited"

    run = _await_terminal(session.id)
    assert run is not None, "registry did not record the run in the ledger"
    assert run["state"] == "completed"
    assert run["exit_code"] == 0

    obligation = pl.get_obligation(session.id)
    assert obligation is not None
    assert obligation["delivery_state"] == "pending"

    claimed = pl.claim_next("integration-consumer")
    assert len(claimed) == 1
    assert claimed[0]["payload"]["session_id"] == session.id


def test_registry_without_notify_still_records_the_outcome():
    """"Nobody asked to be told" must not mean "it never happened"."""
    from tools.process_registry import ProcessRegistry

    registry = ProcessRegistry()
    session = registry.spawn_local("echo quiet")
    assert registry.wait(session.id, timeout=30)["status"] == "exited"

    run = _await_terminal(session.id)
    assert run is not None and run["state"] == "completed"
    assert pl.get_obligation(session.id) is None, \
        "no notification requested => no delivery obligation"


def test_terminal_without_spawn_row_still_records_the_outcome(caplog):
    """A lost spawn-side write must not cost the result.

    If ``record_spawn`` failed (disk full, permissions, a bug), the naive
    ``UPDATE ... WHERE state='running'`` would match zero rows and the outcome
    would vanish -- reproducing the very loss this module prevents, just with
    extra steps. The terminal write therefore creates the row itself and says
    so loudly.
    """
    with caplog.at_level("WARNING"):
        assert pl.record_terminal("orphan", state="failed", exit_code=2,
                                  payload={"r": "x"}) is True
    run = pl.get_run("orphan")
    assert run["state"] == "failed" and run["exit_code"] == 2
    assert pl.get_obligation("orphan")["delivery_state"] == "pending"
    assert any("without a spawn row" in r.message for r in caplog.records)


def test_the_terminal_write_follows_the_spawn_even_if_home_moves(tmp_path, monkeypatch):
    """The leak found by the 2026-07-20 full-suite run.

    A background process is spawned on one thread and terminalised later on
    its reader thread. If the ledger path is re-resolved at that moment and
    ``HERMES_HOME`` has since changed -- which is exactly what happens when a
    test's fixture tears down while the process is still running -- the
    terminal write lands in a *different* store than the spawn write. The
    production ledger then grows a row with NULL command and cwd: the
    signature of a terminal half with no spawn half.

    Call-time resolution fixed the import-freeze leak and created this one.
    The store is now bound to the session at spawn.
    """
    from tools.process_registry import ProcessRegistry

    spawn_home = tmp_path / "spawn-home"
    other_home = tmp_path / "other-home"
    spawn_home.mkdir()
    other_home.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(spawn_home))
    pl.reset_schema_cache()
    registry = ProcessRegistry()
    session = registry.spawn_local("sleep 0.6 && echo moved")

    # The fixture that spawned it tears down while the process still runs.
    monkeypatch.setenv("HERMES_HOME", str(other_home))

    assert registry.wait(session.id, timeout=30)["status"] == "exited"
    _await_terminal_at(spawn_home / "processes.db", session.id)

    assert not (other_home / "processes.db").exists(), \
        "the terminal write followed HERMES_HOME instead of its own session"


def _await_terminal_at(db_path, process_id, timeout=15.0):
    import sqlite3
    import time as _t

    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if db_path.exists():
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                row = con.execute(
                    "SELECT state FROM process_runs WHERE process_id=?",
                    (process_id,)).fetchone()
            except sqlite3.Error:
                row = None
            finally:
                con.close()
            if row and row[0] in pl.TERMINAL_STATES:
                return
        _t.sleep(0.05)
    raise AssertionError(f"no terminal row for {process_id} in {db_path}")

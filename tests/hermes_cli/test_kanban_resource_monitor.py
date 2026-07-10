import json
import os
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_proc_probe_rejects_reused_pid(tmp_path):
    from hermes_cli.kanban_resource_monitor import probe_process

    proc = tmp_path / "proc"
    pid_dir = proc / "42"
    pid_dir.mkdir(parents=True)
    (pid_dir / "stat").write_text("42 (worker) D " + "0 " * 18 + "900 0\n")
    (pid_dir / "status").write_text("Name:\tworker\nState:\tD (disk sleep)\n")
    (pid_dir / "wchan").write_text("__mem_cgroup_handle_over_high\n")

    stale = probe_process(42, expected_start_ticks=899, proc_root=proc)
    assert stale["identity_matches"] is False
    current = probe_process(42, expected_start_ticks=900, proc_root=proc)
    assert current["identity_matches"] is True
    assert current["state"] == "D"
    assert current["wchan"] == "__mem_cgroup_handle_over_high"


def test_cgroup_probe_reports_bounded_deltas_and_non_linux_fallback(tmp_path):
    from hermes_cli.kanban_resource_monitor import probe_cgroup

    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.current").write_text("900\n")
    (cg / "memory.high").write_text("1000\n")
    (cg / "memory.max").write_text("2000\n")
    (cg / "memory.swap.current").write_text("25\n")
    (cg / "memory.swap.max").write_text("100\n")
    (cg / "memory.events").write_text("high 7\noom 1\noom_kill 0\n")
    (cg / "memory.pressure").write_text("some avg10=1.25 avg60=0.50 avg300=0.10 total=500\nfull avg10=0.10 avg60=0.05 avg300=0.01 total=20\n")

    sample = probe_cgroup(cg, previous={"events": {"high": 5}, "pressure": {"some_total": 400}})
    assert sample["supported"] is True
    assert sample["events_delta"]["high"] == 2
    assert sample["pressure_delta"]["some_total"] == 100
    assert sample["high_ratio"] == 0.9
    assert sample["swap_current"] == 25
    assert sample["swap_max"] == 100
    assert sample["swap_ratio"] == 0.25
    assert probe_cgroup(cg, platform="darwin") == {"supported": False, "reason": "unsupported_platform"}


def test_auto_activity_does_not_reset_semantic_progress(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="separate clocks", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid)
        run_id = kb.latest_run(conn, tid).id
        monkeypatch.setattr(kb.time, "time", lambda: 100)
        assert kb.heartbeat_worker(conn, tid, expected_run_id=run_id, semantic=False)
        row = conn.execute("SELECT last_activity_at, last_semantic_progress_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert row["last_activity_at"] == 100
        assert row["last_semantic_progress_at"] is None
        monkeypatch.setattr(kb.time, "time", lambda: 120)
        assert kb.heartbeat_worker(conn, tid, note="checkpoint", expected_run_id=run_id, semantic=True)
        row = conn.execute("SELECT last_activity_at, last_semantic_progress_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert tuple(row) == (120, 120)


def test_sustained_memcg_d_state_blocks_fail_closed(kanban_home, monkeypatch):
    import hermes_cli.kanban_resource_monitor as rm

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="wedged", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 42, start_ticks=900)
        run_id = kb.latest_run(conn, tid).id
        conn.execute("UPDATE task_runs SET d_state_since=? WHERE id=?", (10, run_id))
        conn.commit()

        monkeypatch.setattr(rm, "probe_process", lambda *a, **k: {"supported": True, "identity_matches": True, "state": "D", "wchan": "__mem_cgroup_handle_over_high", "start_ticks": 900})
        # detect_resource_stalls reads the cgroup snapshot once per tick (not
        # per row) and computes deltas locally, so tests mock
        # read_cgroup_snapshot with absolute counters rather than probe_cgroup.
        monkeypatch.setattr(rm, "read_cgroup_snapshot", lambda *a, **k: {"supported": True, "current": 1000, "high": 1000, "high_ratio": 1.0, "events": {"high": 9}, "pressure": {"some_avg10": 2.0, "some_total": 20}, "swap_current": 0})
        monkeypatch.setattr(kb.time, "time", lambda: 100)
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

        blocked = kb.detect_resource_stalls(
            conn, cgroup_path="/unused", d_state_seconds=60,
            psi_some_avg10=1.0, high_event_delta=1,
            signal_fn=lambda pid, sig: None,
        )
        assert blocked == [tid]
        assert kb.get_task(conn, tid).status == "blocked"
        event = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='resource_stalled'", (tid,)).fetchone()
        assert event is not None
        assert json.loads(event["payload"])["process"]["wchan"] == "__mem_cgroup_handle_over_high"
        run = kb.latest_run(conn, tid)
        assert run.outcome == "resource_stalled"
        blocked_event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'", (tid,)
        ).fetchone()
        assert json.loads(blocked_event["payload"])["reason"] == "resource_stalled"


def test_surviving_kill_is_typed_termination_pending(kanban_home, monkeypatch):
    import hermes_cli.kanban_resource_monitor as rm

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unkillable", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 42, start_ticks=900)
        run_id = kb.latest_run(conn, tid).id
        conn.execute("UPDATE task_runs SET d_state_since=10 WHERE id=?", (run_id,))
        conn.commit()
        monkeypatch.setattr(rm, "probe_process", lambda *a, **k: {"supported": True, "identity_matches": True, "state": "D"})
        monkeypatch.setattr(rm, "read_cgroup_snapshot", lambda *a, **k: {"supported": True, "high_ratio": 1.0, "events": {"high": 1}, "pressure": {}})
        monkeypatch.setattr(kb.time, "time", lambda: 100)
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: True)

        assert kb.detect_resource_stalls(conn, cgroup_path="/unused", d_state_seconds=60, signal_fn=lambda *a: None) == [tid]
        pending = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='termination_pending'", (tid,)
        ).fetchone()
        assert pending is not None
        assert json.loads(pending["payload"])["pid"] == 42
        row = conn.execute("SELECT termination_pending_since FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert row[0] == 100
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.claim_expires is None


def test_monitor_errors_are_sampled_into_events_and_logs(kanban_home, monkeypatch, caplog):
    import hermes_cli.kanban_resource_monitor as rm

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="monitor errors", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 42, start_ticks=900)
        monkeypatch.setattr(rm, "probe_process", lambda *a, **k: {"supported": True, "identity_matches": True, "state": "S", "errors": ["status:PermissionError"]})
        monkeypatch.setattr(rm, "read_cgroup_snapshot", lambda *a, **k: {"supported": True, "events": {}, "pressure": {}, "errors": ["memory.events:PermissionError"]})

        for _ in range(3):
            assert kb.detect_resource_stalls(conn, cgroup_path="/unused") == []

        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='monitoring_error' ORDER BY id", (tid,)
        ).fetchall()
        assert [json.loads(row["payload"])["count"] for row in events] == [1, 2]
        assert caplog.text.count("kanban resource monitor error") == 2


def test_null_start_ticks_never_auto_blocks(kanban_home, monkeypatch, caplog):
    """A run row with no worker_start_ticks can't be identity-verified against
    its PID (a reused PID could belong to an unrelated process), so it must
    never be auto-blocked -- even if probe_process/probe_cgroup would
    otherwise report a sustained stall. Fund 5 repair."""
    import hermes_cli.kanban_resource_monitor as rm

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no start ticks", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid)
        # _set_worker_pid without start_ticks, and force worker_start_ticks
        # back to NULL to simulate a spawn that raced or predates the column.
        kb._set_worker_pid(conn, tid, 42, start_ticks=None)
        run_id = kb.latest_run(conn, tid).id
        conn.execute(
            "UPDATE task_runs SET worker_start_ticks=NULL, d_state_since=10 WHERE id=?",
            (run_id,),
        )
        conn.commit()

        # Even if the process/cgroup probes would otherwise scream "stalled",
        # identity is unverifiable so detect_resource_stalls must skip it.
        called = {"probe_process": False}

        def _probe_process(*a, **k):
            called["probe_process"] = True
            return {"supported": True, "identity_matches": True, "state": "D"}

        monkeypatch.setattr(rm, "probe_process", _probe_process)
        monkeypatch.setattr(rm, "read_cgroup_snapshot", lambda *a, **k: {"supported": True, "high_ratio": 1.0, "events": {}, "pressure": {"some_avg10": 5.0}})
        monkeypatch.setattr(kb.time, "time", lambda: 200)

        for _ in range(3):
            assert kb.detect_resource_stalls(conn, cgroup_path="/unused", d_state_seconds=60) == []

        assert called["probe_process"] is False, (
            "identity is unverifiable without worker_start_ticks -- probe_process "
            "must not even be called, let alone used to justify a block"
        )
        assert kb.get_task(conn, tid).status == "running"

        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='monitoring_error' ORDER BY id",
            (tid,),
        ).fetchall()
        assert [json.loads(row["payload"])["count"] for row in events] == [1, 2]
        assert [json.loads(row["payload"])["errors"] for row in events] == [
            ["worker_start_ticks:missing"]
        ] * 2
        assert caplog.text.count("worker_start_ticks") >= 2


def test_resource_monitor_recovers_from_transient_d_state(kanban_home, monkeypatch):
    import hermes_cli.kanban_resource_monitor as rm

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="recovered", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 42, start_ticks=900)
        run_id = kb.latest_run(conn, tid).id
        conn.execute("UPDATE task_runs SET d_state_since=10 WHERE id=?", (run_id,))
        conn.commit()
        monkeypatch.setattr(rm, "probe_process", lambda *a, **k: {"supported": True, "identity_matches": True, "state": "S", "wchan": "ep_poll", "start_ticks": 900})
        monkeypatch.setattr(rm, "read_cgroup_snapshot", lambda *a, **k: {"supported": True, "events": {}, "pressure": {}, "high_ratio": 0.1})
        assert kb.detect_resource_stalls(conn, cgroup_path="/unused", d_state_seconds=60) == []
        row = conn.execute("SELECT d_state_since FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert row[0] is None
        assert kb.get_task(conn, tid).status == "running"

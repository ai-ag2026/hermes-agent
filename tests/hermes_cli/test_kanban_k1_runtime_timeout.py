"""K-1 (2026-07-14) regression: enforce_max_runtime must not crash the dispatcher tick.

enforce_max_runtime parks a runtime-capped approved-action attempt with
attention_type="transient", reason_code="runtime_timeout". That pair was missing
from VALID_BLOCK_REASON_CODES + BLOCK_CAUSE_SCOPE_ENUMS, so _block_cause_fingerprint
raised ValueError and crashed the whole board tick, stranding the task. Fixed by
adding the enum pair + a defensive fall-through to the normal timeout requeue.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_runtime_timeout_is_a_valid_block_cause(kanban_home):
    assert "runtime_timeout" in kb.VALID_BLOCK_REASON_CODES
    assert ("transient", "runtime_timeout") in kb.BLOCK_CAUSE_SCOPE_ENUMS


def test_block_cause_fingerprint_runtime_timeout_no_crash(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="rt")
        # Pre-fix this raised ValueError("unsupported block reason code").
        fp = kb._block_cause_fingerprint(
            conn, task_id=t, attention_type="transient", reason_code="runtime_timeout"
        )
        assert isinstance(fp, str) and len(fp) == 64


def test_enforce_max_runtime_times_out_without_crash(kanban_home, monkeypatch):
    """A running task past its runtime cap is timed out (requeued), never crashing."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="slow", assignee="worker", max_runtime_seconds=1)
        claimed = kb.claim_task(conn, t)
        assert claimed is not None
        # Force the heartbeat/claim to look long-expired and the pid dead.
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at=?, claim_expires=?, worker_pid=? WHERE id=?",
            (1, 1, 2147480000, t),
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET started_at=1 WHERE task_id=?", (t,))
        # Must not raise; the task leaves 'running'.
        kb.enforce_max_runtime(conn)
        assert kb.get_task(conn, t).status != "running"

from __future__ import annotations

import asyncio
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import (
    AttentionStormMetrics, _collect_attention_storm_metrics_readonly,
    _normalize_backpressure_shadow_config, _resolve_backpressure_shadow_config,
    _should_emit_shadow_warning, evaluate_backpressure_shadow,
)
from hermes_cli import kanban_db as kb


def _board(tmp_path: Path) -> sqlite3.Connection:
    path = tmp_path / "shadow.db"
    kb.init_db(path)
    return sqlite3.connect(path)


def _task(conn: sqlite3.Connection, task_id: str, *, status: str = "blocked", recurrence: int = 0, cause: str | None = None) -> None:
    conn.execute("INSERT INTO tasks (id,title,status,created_at,workspace_kind,block_recurrences,block_cause_fingerprint) VALUES (?,?,?,?,?,?,?)", (task_id, task_id, status, 1, "scratch", recurrence, cause))


def _typed(conn: sqlite3.Connection, task_id: str, kind: str, created: int, version: int = 1) -> int:
    cur = conn.execute("INSERT INTO task_attentions (task_id,type,cause_fingerprint,summary,created_at,version) VALUES (?,?,?,?,?,?)", (task_id, kind, f"safe-{kind}-{created}", "safe", created, version))
    return int(cur.lastrowid)


def test_default_off_normal_zero_denominator_and_threshold_order() -> None:
    hot = AttentionStormMetrics(99, 99, 99, 0.9, 99, 0, 99)
    assert evaluate_backpressure_shadow(hot).action == "none"
    normal = AttentionStormMetrics(blocked_run_ratio_15m=None)
    assert evaluate_backpressure_shadow(normal, {"shadow_enabled": True}).action == "none"
    cfg = {"shadow_enabled": True, "active_human_required_threshold": 1, "new_attention_5m_threshold": 1, "new_attention_15m_threshold": 1, "blocked_run_ratio_15m_threshold": 0.5, "recurrent_blocks_threshold": 1, "duplicate_suppressed_threshold": 1}
    decision = evaluate_backpressure_shadow(AttentionStormMetrics(1, 1, 1, 0.5, 1, 0, 1), cfg)
    assert decision.reasons == ("active_human_required", "new_attention_5m", "new_attention_15m", "blocked_run_ratio_15m", "recurrent_blocks", "duplicate_suppressed_last_tick")
    assert decision.fingerprint == "pause_new_fanout:" + ",".join(decision.reasons)


@pytest.mark.parametrize("raw", [None, "false", [], {"shadow_enabled": "true"}, {"shadow_enabled": 1}, {"blocked_run_ratio_15m_threshold": float("nan")}, {"blocked_run_ratio_15m_threshold": float("inf")}, {"active_human_required_threshold": -1}])
def test_config_is_strict_and_malformed_safe(raw) -> None:
    cfg = _normalize_backpressure_shadow_config(raw)
    assert cfg.shadow_enabled is False
    assert 1 <= cfg.active_human_required_threshold <= 100000
    assert math.isfinite(cfg.blocked_run_ratio_15m_threshold)
    assert 0 <= cfg.blocked_run_ratio_15m_threshold <= 1
    assert cfg.new_attention_15m_threshold >= cfg.new_attention_5m_threshold


def test_config_clamps_and_loader_error_disables() -> None:
    cfg = _normalize_backpressure_shadow_config({"shadow_enabled": True, "active_human_required_threshold": 999999, "new_attention_5m_threshold": 50, "new_attention_15m_threshold": 2, "blocked_run_ratio_15m_threshold": 2, "warning_cooldown_seconds": 0})
    assert (cfg.shadow_enabled, cfg.active_human_required_threshold, cfg.new_attention_15m_threshold, cfg.blocked_run_ratio_15m_threshold, cfg.warning_cooldown_seconds) == (True, 100000, 50, 1.0, 1)
    assert _resolve_backpressure_shadow_config(lambda: (_ for _ in ()).throw(RuntimeError())).shadow_enabled is False


def test_collector_currentness_boundaries_recurrence_and_ratio(tmp_path: Path) -> None:
    now = 10_000
    conn = _board(tmp_path)
    _task(conn, "at5", recurrence=2, cause="canonical")
    _typed(conn, "at5", "decision", now - 300)
    _task(conn, "at15")
    _typed(conn, "at15", "review", now - 900)
    _task(conn, "old")
    _typed(conn, "old", "protocol", now - 901)
    _task(conn, "done", status="done")
    _typed(conn, "done", "decision", now)
    _task(conn, "superseded")
    _typed(conn, "superseded", "decision", now)
    _typed(conn, "superseded", "transient", now + 1)  # newest is not human-required
    conn.execute("INSERT INTO task_runs (task_id,status,started_at,ended_at,outcome) VALUES (?,?,?,?,?)", ("at5", "done", 1, now - 900, "blocked"))
    conn.execute("INSERT INTO task_runs (task_id,status,started_at,ended_at,outcome) VALUES (?,?,?,?,?)", ("old", "done", 1, now - 901, "blocked"))
    conn.commit(); conn.close()
    metrics = _collect_attention_storm_metrics_readonly(tmp_path / "shadow.db", now)
    assert metrics is not None
    assert (metrics.active_human_required, metrics.new_attention_5m, metrics.new_attention_15m, metrics.recurrent_blocks) == (3, 1, 2, 1)
    assert metrics.blocked_run_ratio_15m == 1.0


def test_collector_readonly_trace_delivery_gauge_and_bad_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = 10_000
    conn = _board(tmp_path)
    _task(conn, "live")
    attention_id = _typed(conn, "live", "decision", now)
    conn.execute("INSERT INTO kanban_attention_deliveries (task_id,attention_id,attention_version,platform,chat_id,updated_at,state) VALUES (?,?,?,?,?,?,?)", ("live", attention_id, 1, "test", "1", now, "delivered"))
    conn.execute("INSERT INTO kanban_attention_deliveries (task_id,attention_id,attention_version,platform,chat_id,updated_at,state) VALUES (?,?,?,?,?,?,?)", ("live", attention_id, 2, "test", "2", now, "cancelled"))
    before = {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in ("tasks", "task_attentions", "task_pending_actions", "task_runs", "kanban_attention_deliveries", "task_events")}
    conn.commit(); conn.close()
    traced: list[str] = []
    real_connect = sqlite3.connect
    def traced_connect(*args, **kwargs):
        c = real_connect(*args, **kwargs); c.set_trace_callback(traced.append); return c
    monkeypatch.setattr("gateway.kanban_watchers.sqlite3.connect", traced_connect)
    metrics = _collect_attention_storm_metrics_readonly(tmp_path / "shadow.db", now)
    assert metrics and metrics.current_delivery_identities_materialized == 1
    assert _collect_attention_storm_metrics_readonly(tmp_path / "missing.db", now) is None
    verify = sqlite3.connect(tmp_path / "shadow.db")
    after = {t: verify.execute(f"SELECT * FROM {t}").fetchall() for t in before}
    assert after == before
    assert not any(token in " ".join(traced).upper() for token in ("BEGIN IMMEDIATE", "INSERT", "UPDATE", "DELETE"))


@pytest.mark.parametrize("filename", ["shadow #?.db", "shadow % space.db"])
def test_collector_uri_escapes_readonly_db_path(tmp_path: Path, filename: str) -> None:
    now = 10_000
    path = tmp_path / filename
    kb.init_db(path)
    conn = sqlite3.connect(path)
    _task(conn, "live")
    _typed(conn, "live", "decision", now)
    conn.commit()
    before = {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in ("tasks", "task_attentions", "task_pending_actions", "task_runs", "kanban_attention_deliveries", "task_events")}
    conn.close()

    metrics = _collect_attention_storm_metrics_readonly(path, now)

    verify = sqlite3.connect(path)
    after = {table: verify.execute(f"SELECT * FROM {table}").fetchall() for table in before}
    verify.close()
    assert metrics is not None
    assert metrics.active_human_required == 1
    assert after == before


def test_warning_dedupe_and_safe_aggregate_log(caplog) -> None:
    cfg = {"shadow_enabled": True, "active_human_required_threshold": 1}
    one = evaluate_backpressure_shadow(AttentionStormMetrics(active_human_required=1), cfg)
    changed = evaluate_backpressure_shadow(AttentionStormMetrics(new_attention_5m=1), {**cfg, "new_attention_5m_threshold": 1})
    seen: dict[tuple[str, str], int] = {}
    assert _should_emit_shadow_warning(seen, "board-a", one, 100, 300)
    assert _should_emit_shadow_warning(seen, "board-b", one, 100, 300)
    assert not _should_emit_shadow_warning(seen, "board-a", one, 101, 300)
    # 2026-07-13 (Claude review G5): a CHANGED reason-set on the SAME board within
    # the cooldown is now also suppressed. The cooldown keys on (board, action),
    # not the flapping reason fingerprint, so a sustained storm whose triggered
    # reasons wobble around their thresholds can no longer re-fire the aggregate
    # warning every tick (the anti-spam intent). Previously this asserted a fresh
    # warning here — that encoded the storm-fragmentation bug.
    assert not _should_emit_shadow_warning(seen, "board-a", changed, 101, 300)
    assert _should_emit_shadow_warning(seen, "board-a", one, 400, 300)
    from gateway.kanban_watchers import _log_shadow_warning
    _log_shadow_warning(one)
    text = caplog.text
    for forbidden in ("task-SECRET", "profile-SECRET", "payload-SECRET", "workspace-SECRET", "token-SECRET"):
        assert forbidden not in text


def test_pause_decision_does_not_skip_auto_decompose_or_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Behavioral single-tick proof: a hot shadow decision cannot gate the real calls."""
    from gateway.run import GatewayRunner
    import gateway.kanban_watchers as watchers
    import hermes_cli.config as config
    import hermes_cli.kanban_db as db

    path = tmp_path / "watcher.db"
    kb.init_db(path)
    calls: list[str] = []
    runner = object.__new__(GatewayRunner)
    runner._running = True
    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1, "auto_decompose": True, "backpressure_shadow": {"shadow_enabled": True, "active_human_required_threshold": 1}}})
    monkeypatch.setattr(watchers, "_acquire_singleton_lock", lambda _path: (None, "unavailable"))
    monkeypatch.setattr(db, "list_boards", lambda include_archived=False: [{"slug": "test", "db_path": str(path)}])
    monkeypatch.setattr(db, "read_board_metadata", lambda slug: {"slug": slug})
    monkeypatch.setattr(db, "kanban_db_path", lambda board=None: path)
    monkeypatch.setattr(db, "reap_worker_zombies", lambda: calls.append("reap") or [])
    monkeypatch.setattr(db, "connect", lambda board=None: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(db, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(db, "has_spawnable_review", lambda conn: False)
    result = SimpleNamespace(spawned=["spawn-stub"], crashed=[], timed_out=[], auto_blocked=[], reclaimed=0, promoted=0)
    monkeypatch.setattr(db, "dispatch_once", lambda *a, **kw: calls.append("dispatch") or result)
    import hermes_cli.kanban_decompose as decomp
    monkeypatch.setattr(decomp, "list_triage_ids", lambda: ["triage"])
    monkeypatch.setattr(decomp, "decompose_task", lambda *a, **kw: calls.append("decompose") or SimpleNamespace(ok=True, fanout=False, child_ids=[]))

    async def immediate_to_thread(fn, *args, **kwargs):
        value = fn(*args, **kwargs)
        if getattr(fn, "__name__", "") == "_ready_nonempty":
            runner._running = False
        return value
    async def immediate_sleep(_delay):
        return None
    monkeypatch.setattr(watchers.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(watchers.asyncio, "sleep", immediate_sleep)
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3))
    assert calls == ["reap", "decompose", "dispatch"]


def test_duplicate_suppression_shadow_is_board_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A hot notifier outcome on board A must not trigger board B's shadow decision."""
    from gateway.run import GatewayRunner
    import gateway.kanban_watchers as watchers
    import hermes_cli.config as config
    import hermes_cli.kanban_db as db

    board_a = tmp_path / "board-a.db"
    board_b = tmp_path / "board-b.db"
    kb.init_db(board_a)
    kb.init_db(board_b)
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_shadow_duplicate_suppressed_last_tick = {"board-a": 1, "board-b": 0}
    decisions = []
    monkeypatch.setattr(config, "load_config", lambda: {
        "kanban": {
            "dispatch_in_gateway": True,
            "dispatch_interval_seconds": 1,
            "auto_decompose": False,
            "backpressure_shadow": {"shadow_enabled": True, "duplicate_suppressed_threshold": 1},
        }
    })
    monkeypatch.setattr(watchers, "_acquire_singleton_lock", lambda _path: (None, "unavailable"))
    monkeypatch.setattr(db, "list_boards", lambda include_archived=False: [
        {"slug": "Board-A", "db_path": str(board_a)},
        {"slug": "board-b", "db_path": str(board_b)},
    ])
    monkeypatch.setattr(db, "read_board_metadata", lambda slug: {"slug": slug})
    monkeypatch.setattr(db, "kanban_db_path", lambda board=None: board_a)
    monkeypatch.setattr(db, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(db, "connect", lambda board=None: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(db, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(db, "has_spawnable_review", lambda conn: False)
    monkeypatch.setattr(db, "dispatch_once", lambda *a, **kw: SimpleNamespace(spawned=[], crashed=[], timed_out=[], auto_blocked=[], reclaimed=0, promoted=0))
    monkeypatch.setattr(watchers, "_collect_attention_storm_metrics_readonly", lambda path, now: AttentionStormMetrics())
    monkeypatch.setattr(watchers, "_should_emit_shadow_warning", lambda _seen, _board_slug, decision, *_args: decision.action == "pause_new_fanout")
    monkeypatch.setattr(watchers, "_log_shadow_warning", decisions.append)

    async def immediate_to_thread(fn, *args, **kwargs):
        value = fn(*args, **kwargs)
        if getattr(fn, "__name__", "") == "_ready_nonempty":
            runner._running = False
        return value

    async def immediate_sleep(_delay):
        return None

    monkeypatch.setattr(watchers.asyncio, "to_thread", immediate_to_thread)
    monkeypatch.setattr(watchers.asyncio, "sleep", immediate_sleep)
    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3))

    assert len(decisions) == 1
    assert decisions[0].reasons == ("duplicate_suppressed_last_tick",)

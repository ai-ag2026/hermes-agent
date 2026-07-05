"""Tests for hermes_cli/orchestration_budget.py (CC-PARITY-A4)."""

from __future__ import annotations

import sqlite3

from hermes_cli.orchestration_budget import Budget, resolve_budget, spent


def _seed_telemetry(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE llm_api_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            session_id TEXT,
            total_tokens INTEGER,
            estimated_cost_usd REAL
        )"""
    )
    conn.executemany(
        "INSERT INTO llm_api_calls (created_at, session_id, total_tokens, estimated_cost_usd) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestSpent:
    def test_missing_db_returns_zero(self, tmp_path):
        assert spent(db_path=tmp_path / "nope.sqlite") == 0.0

    def test_sums_tokens_and_cost(self, tmp_path):
        db = tmp_path / "t.sqlite"
        _seed_telemetry(db, [
            (100.0, "s1", 1000, 0.5),
            (200.0, "s1", 2000, 1.0),
            (300.0, "s2", 4000, 2.0),
        ])
        assert spent(metric="tokens", db_path=db) == 7000.0
        assert spent(metric="cost", db_path=db) == 3.5

    def test_session_and_since_filters(self, tmp_path):
        db = tmp_path / "t.sqlite"
        _seed_telemetry(db, [
            (100.0, "s1", 1000, 0.5),
            (200.0, "s1", 2000, 1.0),
            (300.0, "s2", 4000, 2.0),
        ])
        assert spent(metric="tokens", session_id="s1", db_path=db) == 3000.0
        assert spent(metric="tokens", since=200.0, db_path=db) == 6000.0


class TestBudgetScaling:
    def test_full_budget_keeps_base(self):
        b = Budget(1000, "tokens")
        assert b.scale(12, 0) == 12
        assert b.fraction_remaining(0) == 1.0

    def test_half_spent_halves(self):
        b = Budget(1000, "tokens")
        assert b.scale(12, 500) == 6
        assert b.exhausted(500) is False

    def test_exhausted_hits_floor(self):
        b = Budget(1000, "tokens")
        assert b.scale(12, 1000, floor=1) == 1
        assert b.scale(12, 2000, floor=0) == 0
        assert b.exhausted(1000) is True

    def test_over_base_clamped(self):
        b = Budget(1000, "tokens")
        assert b.scale(4, 0) == 4  # never exceeds base


class TestResolveBudget:
    def test_none_when_unset(self):
        assert resolve_budget({}) is None
        assert resolve_budget({"kanban": {}}) is None
        assert resolve_budget({"kanban": {"orchestration_budget": {}}}) is None

    def test_tokens_budget(self):
        b = resolve_budget({"kanban": {"orchestration_budget": {"tokens": 50000, "window_seconds": 600}}})
        assert b is not None and b.metric == "tokens" and b.total == 50000.0
        assert b.window_seconds == 600.0

    def test_cost_budget(self):
        b = resolve_budget({"kanban": {"orchestration_budget": {"cost": 5.0}}})
        assert b is not None and b.metric == "cost" and b.total == 5.0

    def test_malformed_is_none(self):
        assert resolve_budget({"kanban": {"orchestration_budget": {"tokens": "abc"}}}) is None
        assert resolve_budget({"kanban": {"orchestration_budget": {"tokens": 0}}}) is None

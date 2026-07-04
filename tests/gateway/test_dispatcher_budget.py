"""CC-PARITY-A4 follow-up: budget-scaled dispatcher concurrency cap."""

from __future__ import annotations

from gateway import kanban_watchers as kw


def test_no_budget_leaves_max_spawn_unchanged():
    assert kw._budget_scaled_max_spawn({}, 4) == 4
    assert kw._budget_scaled_max_spawn({"max_spawn": 4}, None) is None
    assert kw._budget_scaled_max_spawn({"orchestration_budget": {}}, 6) == 6


def test_budget_scales_and_hard_stops(monkeypatch):
    cfg = {"orchestration_budget": {"tokens": 1000}}
    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", lambda **k: 0.0)
    assert kw._budget_scaled_max_spawn(cfg, 4) == 4      # full budget
    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", lambda **k: 500.0)
    assert kw._budget_scaled_max_spawn(cfg, 4) == 2      # half spent
    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", lambda **k: 1000.0)
    assert kw._budget_scaled_max_spawn(cfg, 4) == 0      # exhausted -> hard stop


def test_budget_bounds_unbounded_dispatcher(monkeypatch):
    cfg = {"orchestration_budget": {"tokens": 1000}}
    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", lambda **k: 0.0)
    assert kw._budget_scaled_max_spawn(cfg, None) == 8   # budget_base_spawn default
    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", lambda **k: 1000.0)
    assert kw._budget_scaled_max_spawn(cfg, None) == 0


def test_custom_base_spawn(monkeypatch):
    cfg = {"orchestration_budget": {"tokens": 1000}, "budget_base_spawn": 4}
    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", lambda **k: 0.0)
    assert kw._budget_scaled_max_spawn(cfg, None) == 4


def test_telemetry_error_fails_open(monkeypatch):
    cfg = {"orchestration_budget": {"tokens": 1000}}

    def _boom(**kwargs):
        raise RuntimeError("telemetry down")

    monkeypatch.setattr("hermes_cli.orchestration_budget.spent", _boom)
    # never block the dispatcher on a telemetry hiccup
    assert kw._budget_scaled_max_spawn(cfg, 4) == 4

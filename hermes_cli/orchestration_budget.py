"""Budget-aware orchestration scaling (CC-PARITY-A4).

A small, dependency-free primitive that reads *actual* spend from the
tars-model-telemetry sqlite (``~/.hermes/ops/model-telemetry/model_telemetry.sqlite``,
table ``llm_api_calls``) and scales an orchestration fan-out (decompose depth,
swarm width) toward a token/cost target — mirroring the ``budget.remaining()``
scaling of the Claude Code Workflow harness.

Everything here is opt-in: with no ``kanban.orchestration_budget`` configured,
``resolve_budget`` returns ``None`` and callers change nothing. All reads are
best-effort and read-only — a missing/locked telemetry DB yields spend 0.0, so
budgeting never crashes the orchestration hot path.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Optional


def telemetry_db_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "ops" / "model-telemetry" / "model_telemetry.sqlite"


def spent(
    *,
    metric: str = "tokens",
    session_id: Optional[str] = None,
    since: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> float:
    """Sum spend from telemetry. ``metric`` is ``tokens`` (total_tokens) or
    ``cost`` (estimated_cost_usd). Optionally scope to a session and/or a
    ``created_at >= since`` epoch window. Best-effort: returns 0.0 on any error
    or when the telemetry DB is absent."""
    path = Path(db_path) if db_path is not None else telemetry_db_path()
    if not path.exists():
        return 0.0
    column = "total_tokens" if metric == "tokens" else "estimated_cost_usd"
    query = f"SELECT COALESCE(SUM({column}), 0) FROM llm_api_calls WHERE 1=1"
    params: list[Any] = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    if since is not None:
        query += " AND created_at >= ?"
        params.append(since)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = conn.execute(query, params).fetchone()
            return float(row[0] or 0.0)
        finally:
            conn.close()
    except Exception:
        return 0.0


class Budget:
    """A token/cost ceiling with proportional fan-out scaling + hard stop."""

    def __init__(self, total: float, metric: str = "tokens",
                 window_seconds: Optional[float] = None):
        self.total = float(total)
        self.metric = metric
        self.window_seconds = window_seconds

    def remaining(self, spent_value: float) -> float:
        return max(0.0, self.total - float(spent_value))

    def fraction_remaining(self, spent_value: float) -> float:
        if self.total <= 0:
            return 1.0
        return max(0.0, min(1.0, (self.total - float(spent_value)) / self.total))

    def exhausted(self, spent_value: float) -> bool:
        return self.remaining(spent_value) <= 0

    def scale(self, base: int, spent_value: float, *, floor: int = 0) -> int:
        """Scale a base fan-out count by the remaining budget fraction, clamped
        to ``[floor, base]``. ``floor`` lets a caller keep making minimal
        progress (floor=1) or hard-stop entirely (floor=0) when exhausted."""
        if base <= 0:
            return 0
        scaled = math.ceil(base * self.fraction_remaining(spent_value))
        return max(floor, min(base, scaled))


def resolve_budget(cfg: Any) -> Optional[Budget]:
    """Build a Budget from config ``kanban.orchestration_budget: {tokens|cost: N,
    window_seconds?: M}``. Returns None when unset/malformed (neutral)."""
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    ob = kanban_cfg.get("orchestration_budget")
    if not isinstance(ob, dict):
        return None
    window = ob.get("window_seconds")
    try:
        window = float(window) if window is not None else None
    except (TypeError, ValueError):
        window = None
    for metric, key in (("tokens", "tokens"), ("cost", "cost")):
        raw = ob.get(key)
        if raw is None:
            continue
        try:
            total = float(raw)
        except (TypeError, ValueError):
            return None
        if total > 0:
            return Budget(total, metric, window)
    return None

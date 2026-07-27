"""Kanban board watcher methods for GatewayRunner.

Extracted verbatim from ``gateway/run.py`` (god-file decomposition Phase 3).
These are the background-loop methods that subscribe to kanban boards, deliver
notifications/artifacts, and drive the multi-agent dispatcher. They use only
``self`` state, so they live on a mixin that ``GatewayRunner`` inherits — the
``self._kanban_*`` call sites resolve identically via the MRO, making this a
behavior-neutral move that lifts ~1,000 LOC out of run.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from urllib.parse import quote
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

from agent.i18n import t

# Match the logger run.py uses (logging.getLogger(__name__) where __name__ ==
# "gateway.run") so extracted log records keep their original logger name.
logger = logging.getLogger("gateway.run")


@dataclass(frozen=True)
class AttentionStormMetrics:
    active_human_required: int = 0
    new_attention_5m: int = 0
    new_attention_15m: int = 0
    blocked_run_ratio_15m: float | None = None
    recurrent_blocks: int = 0
    current_delivery_identities_materialized: int = 0
    duplicate_suppressed_last_tick: int = 0


@dataclass(frozen=True)
class BackpressureShadowConfig:
    shadow_enabled: bool = False
    active_human_required_threshold: int = 20
    new_attention_5m_threshold: int = 8
    new_attention_15m_threshold: int = 20
    blocked_run_ratio_15m_threshold: float = 0.60
    recurrent_blocks_threshold: int = 5
    duplicate_suppressed_threshold: int = 50
    warning_cooldown_seconds: int = 300


@dataclass(frozen=True)
class BackpressureShadowDecision:
    action: Literal["none", "pause_new_fanout"]
    reasons: tuple[str, ...]
    metrics: AttentionStormMetrics

    @property
    def fingerprint(self) -> str:
        return f"{self.action}:{','.join(self.reasons)}"


def _normalize_backpressure_shadow_config(raw: Mapping[str, Any] | None) -> BackpressureShadowConfig:
    """Strict, fail-safe normalization for the report-only shadow namespace."""
    raw = raw if isinstance(raw, Mapping) else {}
    def integer(name: str, default: int, maximum: int = 100000) -> int:
        value = raw.get(name, default)
        if isinstance(value, bool):
            return default
        try:
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return min(max(value, 1), maximum)
    ratio = raw.get("blocked_run_ratio_15m_threshold", 0.60)
    if isinstance(ratio, bool):
        ratio = 0.60
    try:
        ratio = float(ratio)
    except (TypeError, ValueError, OverflowError):
        ratio = 0.60
    if not math.isfinite(ratio):
        ratio = 0.60
    five = integer("new_attention_5m_threshold", 8)
    return BackpressureShadowConfig(
        shadow_enabled=bool(raw.get("shadow_enabled")) if isinstance(raw.get("shadow_enabled"), bool) else False,
        active_human_required_threshold=integer("active_human_required_threshold", 20),
        new_attention_5m_threshold=five,
        new_attention_15m_threshold=max(integer("new_attention_15m_threshold", 20), five),
        blocked_run_ratio_15m_threshold=min(max(ratio, 0.0), 1.0),
        recurrent_blocks_threshold=integer("recurrent_blocks_threshold", 5),
        duplicate_suppressed_threshold=integer("duplicate_suppressed_threshold", 50),
        warning_cooldown_seconds=integer("warning_cooldown_seconds", 300, 86400),
    )


def _resolve_backpressure_shadow_config(load_config: Callable[[], Any]) -> BackpressureShadowConfig:
    try:
        cfg = load_config()
        kanban = cfg.get("kanban", {}) if isinstance(cfg, Mapping) else {}
        shadow = kanban.get("backpressure_shadow", {}) if isinstance(kanban, Mapping) else {}
    except Exception:
        shadow = {}
    return _normalize_backpressure_shadow_config(shadow)


def evaluate_backpressure_shadow(metrics: AttentionStormMetrics, config: Mapping[str, Any] | None = None) -> BackpressureShadowDecision:
    """Return report-only advice; callers must not use it to control dispatch."""
    cfg = _normalize_backpressure_shadow_config(config)
    if not cfg.shadow_enabled:
        return BackpressureShadowDecision("none", (), metrics)
    reasons: list[str] = []
    if metrics.active_human_required >= cfg.active_human_required_threshold:
        reasons.append("active_human_required")
    if metrics.new_attention_5m >= cfg.new_attention_5m_threshold:
        reasons.append("new_attention_5m")
    if metrics.new_attention_15m >= cfg.new_attention_15m_threshold:
        reasons.append("new_attention_15m")
    if metrics.blocked_run_ratio_15m is not None and metrics.blocked_run_ratio_15m >= cfg.blocked_run_ratio_15m_threshold:
        reasons.append("blocked_run_ratio_15m")
    if metrics.recurrent_blocks >= cfg.recurrent_blocks_threshold:
        reasons.append("recurrent_blocks")
    if metrics.duplicate_suppressed_last_tick >= cfg.duplicate_suppressed_threshold:
        reasons.append("duplicate_suppressed_last_tick")
    return BackpressureShadowDecision("pause_new_fanout" if reasons else "none", tuple(reasons), metrics)


def _collect_attention_storm_metrics_readonly(db_path: str | Path, now: int) -> AttentionStormMetrics | None:
    """Collect authoritative aggregates via a read-only SQLite connection only."""
    conn: sqlite3.Connection | None = None
    try:
        path = Path(db_path).expanduser().resolve()
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True, timeout=1.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=1000")
        row = conn.execute("""
            WITH candidates AS (
              SELECT x.id,x.task_id,x.type,x.created_at,x.version,t.block_recurrences,t.block_cause_fingerprint,
                     a.state action_state,a.expires_at,a.version action_version
              FROM task_attentions x JOIN tasks t ON t.id=x.task_id LEFT JOIN task_pending_actions a ON a.id=x.action_id
              WHERE t.status NOT IN ('done','archived') AND ((x.type='exact_action' AND a.state IN ('pending','approved') AND a.expires_at>?) OR (x.type IN ('capability','transient') AND x.action_id IS NOT NULL AND a.state='approved' AND a.expires_at>?) OR (x.type IN ('decision','protocol','review','loop_triage','capability','transient') AND x.action_id IS NULL))
            ), current AS (
              SELECT * FROM (SELECT candidates.*,ROW_NUMBER() OVER (PARTITION BY task_id ORDER BY id DESC) rn FROM candidates) WHERE rn=1
            ), human AS (
              SELECT *,CASE WHEN type='exact_action' THEN action_version ELSE version END delivery_version FROM current WHERE (type='exact_action' AND action_state='pending') OR type IN ('decision','protocol','review','capability')
            )
            SELECT COUNT(*) active,COALESCE(SUM(created_at>=?-300),0) new5,COALESCE(SUM(created_at>=?-900),0) new15,COALESCE(SUM(block_recurrences>=2 AND block_cause_fingerprint IS NOT NULL),0) recurrent,(SELECT COUNT(*) FROM human h JOIN kanban_attention_deliveries d ON d.task_id=h.task_id AND d.attention_id=h.id AND d.attention_version=h.delivery_version WHERE d.state IN ('pending','sending','delivered')) materialized FROM human
        """, (now, now, now, now)).fetchone()
        runs = conn.execute("SELECT COUNT(*) ended,COALESCE(SUM(outcome='blocked'),0) blocked FROM task_runs WHERE ended_at IS NOT NULL AND ended_at>=?", (now - 900,)).fetchone()
        ended = int(runs["ended"])
        return AttentionStormMetrics(int(row["active"]), int(row["new5"]), int(row["new15"]), int(runs["blocked"]) / ended if ended else None, int(row["recurrent"]), int(row["materialized"]))
    except (sqlite3.Error, OSError, ValueError):
        return None
    finally:
        if conn is not None:
            conn.close()


def _should_emit_shadow_warning(last_warning: dict[tuple[str, str], int], board_slug: str, decision: BackpressureShadowDecision, now: int, cooldown: int) -> bool:
    if decision.action != "pause_new_fanout":
        return False
    # 2026-07-13 (Claude review G5): key the cooldown by (board, action), NOT by
    # the exact triggered reason-set fingerprint. During a sustained storm whose
    # reason set flaps tick-to-tick (e.g. blocked_run_ratio_15m oscillating around
    # its threshold), each distinct combination got its own cooldown clock, so the
    # aggregate warning re-fired far more often than warning_cooldown_seconds —
    # defeating the storm report's own anti-spam intent. action is constant
    # ("pause_new_fanout") past the guard above, so this is effectively per-board.
    key = (board_slug, decision.action)
    previous = last_warning.get(key)
    if previous is not None and now - previous < cooldown:
        return False
    last_warning[key] = now
    return True


def _log_shadow_warning(decision: BackpressureShadowDecision) -> None:
    """A deliberately aggregate-only warning boundary for shadow diagnostics."""
    m = decision.metrics
    logger.warning(
        "kanban backpressure shadow action=%s reasons=%s active=%d new5=%d new15=%d ratio=%s recurrent=%d materialized=%d duplicate_suppressed_last_tick=%d",
        decision.action, ",".join(decision.reasons), m.active_human_required,
        m.new_attention_5m, m.new_attention_15m, m.blocked_run_ratio_15m,
        m.recurrent_blocks, m.current_delivery_identities_materialized,
        m.duplicate_suppressed_last_tick,
    )


_DECOMPOSE_FAILED_EVENT = "decompose_attempt_failed"
_DECOMPOSE_GAVE_UP_EVENT = "decompose_gave_up"


def _resolve_decompose_max_attempts(load_config: Callable[[], Any]) -> int:
    """Live-resolve ``kanban.decompose_max_attempts`` (0 = unbegrenzt).

    K-10 (Vollaudit 2026-07-16): der Key stand in der Config, hatte aber
    keinen Consumer — gescheiterte Triage-Karten wurden jeden Tick erneut
    decomposed (Aux-LLM-Spend ohne Ende, keine Eskalation). Wie die anderen
    auto-decompose-Flags pro Tick frisch gelesen; fail-safe auf 0
    (verhaltensneutral), negative/kaputte Werte → 0.
    """
    try:
        cfg = load_config()
    except Exception:
        return 0
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        value = int(kcfg.get("decompose_max_attempts", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def _count_failed_decompose_attempts(kb_module: Any, task_id: str) -> int:
    """Durable Zählung fehlgeschlagener Auto-Decompose-Versuche."""
    conn = kb_module.connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, _DECOMPOSE_FAILED_EVENT),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _record_failed_decompose_attempt(
    kb_module: Any, task_id: str, reason: str,
) -> int:
    """Fehlversuch als Event festhalten; liefert die neue Gesamtzahl."""
    conn = kb_module.connect()
    try:
        with kb_module.write_txn(conn):
            kb_module._append_event(
                conn, task_id, _DECOMPOSE_FAILED_EVENT,
                {"reason": (reason or "")[:300]},
            )
        row = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, _DECOMPOSE_FAILED_EVENT),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _record_decompose_gave_up_once(
    kb_module: Any, task_id: str, *, attempts: int, limit: int,
) -> bool:
    """Einmaliges ``decompose_gave_up``-Event (idempotent). True = neu.

    Die Karte bleibt in triage; den Human-Ping übernimmt der
    Attention-Cron nach dem Grace-Fenster (Automation-first 2026-07-16).
    """
    conn = kb_module.connect()
    try:
        exists = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
            (task_id, _DECOMPOSE_GAVE_UP_EVENT),
        ).fetchone()
        if exists:
            return False
        with kb_module.write_txn(conn):
            kb_module._append_event(
                conn, task_id, _DECOMPOSE_GAVE_UP_EVENT,
                {"attempts": attempts, "limit": limit},
            )
        return True
    finally:
        conn.close()


def _resolve_auto_decompose_settings(
    load_config: Callable[[], Any],
) -> "tuple[bool, int]":
    """Resolve the live (enabled, per_tick) auto-decompose settings.

    Read fresh from config on every dispatcher tick (#49638) so that flipping
    ``kanban.auto_decompose: false`` to STOP runaway fan-out takes effect on the
    next tick instead of requiring a gateway restart. Auto-decompose is a
    safety toggle — a user who sees it create and launch tasks they didn't
    intend reaches for this flag to halt it, and a stale boot-captured value
    silently ignoring that change is the bug reported in #49638.

    Fails **safe**: if the config read raises, return ``(False, 3)`` — a
    transient read error must never re-enable a feature the user turned off,
    nor fall back to the burst-prone default-on behaviour. ``per_tick`` is
    clamped to ``>= 1``.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    enabled = bool(kcfg.get("auto_decompose", True))
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    if per_tick < 1:
        per_tick = 1
    return enabled, per_tick


def _resolve_operator_authors(load_config: Callable[[], Any]) -> "frozenset[str]":
    """Resolve ``kanban.operator_authors`` — read live each tick, same as
    :func:`_resolve_auto_decompose_settings`, so an operator can widen/narrow
    the list without a gateway restart. Falls back to the config default
    (``{"claude-code", "manfred"}``) on any read error or malformed value —
    the auto-decomposer must never come back on for a card it was told to
    leave alone just because a config read glitched.
    """
    default = frozenset({"claude-code", "manfred"})
    try:
        cfg = load_config()
    except Exception:
        return default
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    raw = kcfg.get("operator_authors", default)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return default
    authors = frozenset(str(a) for a in raw if a)
    return authors


# Triage card ids the auto-decomposer has already logged a skip for
# (Kanban-Krise 2026-07-10 geparkter Repair-Punkt 8). Module-level and
# process-lifetime: a card stays logged once even across dispatcher ticks,
# so an operator card sitting untouched in triage doesn't spam the log
# every tick. Resets on gateway restart, which is fine — worst case is one
# extra log line.
_operator_owned_skip_logged: "set[str]" = set()


def _filter_operator_owned_triage_ids(
    triage_ids: "list[str]",
    *,
    board_slug: str,
    operator_authors: "frozenset[str]",
    kb_module: Any,
) -> "list[str]":
    """Drop triage ids created_by an operator author from the AUTO-decompose
    batch; explicit decomposition (CLI / decompose_task with another author)
    is a separate call path and is untouched by this filter.

    Fails open on lookup errors (returns ``triage_ids`` unchanged) — a DB
    hiccup here must not silently wedge auto-decompose for an entire board;
    worst case an operator card slips through on that one tick, same risk
    profile as before this fix existed.
    """
    if not triage_ids or not operator_authors:
        return triage_ids
    conn = None
    try:
        conn = kb_module.connect(board=board_slug)
        placeholders = ",".join(["?"] * len(triage_ids))
        rows = conn.execute(
            f"SELECT id, created_by FROM tasks WHERE id IN ({placeholders})",
            tuple(triage_ids),
        ).fetchall()
        created_by_map = {r["id"]: r["created_by"] for r in rows}
    except Exception:
        logger.debug(
            "kanban auto-decompose: operator-author lookup failed on board %s",
            board_slug, exc_info=True,
        )
        return triage_ids
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    kept: "list[str]" = []
    for tid in triage_ids:
        created_by = created_by_map.get(tid)
        if created_by and created_by in operator_authors:
            if tid not in _operator_owned_skip_logged:
                _operator_owned_skip_logged.add(tid)
                logger.info(
                    "kanban auto-decompose [%s]: %s skipped — operator-created "
                    "(created_by=%r); left in triage for a human decision",
                    board_slug, tid, created_by,
                )
            continue
        kept.append(tid)
    return kept


# ---------------------------------------------------------------------------
# pm-supervisor tick (Autonomie-Umbau Baustein B, 2026-07-17)
#
# ``needs_input`` / ``gave_up`` / ``decompose_gave_up`` cards previously went
# straight to a human ping once they landed in ``blocked``/``triage``. This
# tick gives an autonomous "product manager" pass at them FIRST: an aux-LLM
# call chooses one action from a small, bound set; trusted ``kb.*`` helpers
# execute it. The LLM never mutates the board directly — see
# ``_decide_pm_action`` / ``_execute_pm_decision`` below.
#
# Shape mirrors the auto-decompose tick on purpose (config resolver read
# live every tick -> per-tick cap -> operator-authors exemption -> durable
# attempt counter -> idempotent gave-up event), but the tick body itself is
# module-level (not a closure inside ``_kanban_dispatcher_watcher``) so it —
# and every helper it calls — can be unit-tested directly with an injected
# ``call_llm_fn`` instead of only being reachable through the live gateway
# loop.
# ---------------------------------------------------------------------------

_PM_SUPERVISOR_ATTEMPTED_EVENT = "pm_supervisor_attempted"
_PM_SUPERVISOR_GAVE_UP_EVENT = "pm_supervisor_gave_up"
_PM_SUPERVISOR_ESCALATED_EVENT = "pm_supervisor_escalated"
_PM_SUPERVISOR_ACTOR = "pm-supervisor"

_PM_VALID_ACTIONS = frozenset({
    "answer_and_requeue", "clarify_dod", "reassign",
    "decompose", "close_obsolete", "escalate",
})

_PM_FALLBACK_MEMO = {
    "situation": "pm-supervisor could not resolve this card autonomously",
    "options": [],
    "recommendation": "manual triage required",
    "cost_of_ignoring": "card stays stuck until a human looks",
}


def _resolve_pm_supervisor_settings(
    load_config: Callable[[], Any],
) -> "tuple[bool, int, int]":
    """Resolve (enabled, per_tick, max_attempts), live, every tick.

    Default is OFF (``pm_supervisor_enabled`` absent -> ``False``) — unlike
    auto-decompose (a pre-existing default-on behaviour this project must
    stay compatible with), the pm-supervisor is a NEW autonomy step the
    operator opts into. Fails safe to disabled on any config-read error,
    same reasoning as :func:`_resolve_auto_decompose_settings`: a transient
    read glitch must never silently turn ON a feature that mutates the
    board unattended.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3, 2
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    enabled = bool(kcfg.get("pm_supervisor_enabled", False))
    try:
        per_tick = int(kcfg.get("pm_supervisor_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    if per_tick < 1:
        per_tick = 1
    try:
        max_attempts = int(kcfg.get("pm_supervisor_max_attempts", 2) or 2)
    except (TypeError, ValueError):
        max_attempts = 2
    if max_attempts < 1:
        max_attempts = 1
    return enabled, per_tick, max_attempts


def _count_pm_supervisor_attempts(kb_module: Any, task_id: str) -> int:
    """Durable count of pm-supervisor touches on this card (K-10 pattern)."""
    conn = kb_module.connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _record_pm_supervisor_attempt(
    kb_module: Any, task_id: str, *, action: str, outcome: str,
) -> int:
    """Record one pm-supervisor touch as a durable, auditable event.

    Called exactly once per executed action, success or failure alike —
    this is the auditability invariant: every pm mutation is traceable to
    an event (this) plus a human-readable comment (written by the action
    executor itself). Returns the new total attempt count.
    """
    conn = kb_module.connect()
    try:
        with kb_module.write_txn(conn):
            kb_module._append_event(
                conn, task_id, _PM_SUPERVISOR_ATTEMPTED_EVENT,
                {"action": action, "outcome": outcome},
            )
        row = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, _PM_SUPERVISOR_ATTEMPTED_EVENT),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _record_pm_supervisor_gave_up_once(
    kb_module: Any, task_id: str, *, attempts: int, limit: int,
) -> bool:
    """Idempotent ``pm_supervisor_gave_up`` event. True = newly recorded.

    The card is left exactly where it is (status untouched) once the
    attempt limit is hit — the regular attention/human escalation path
    takes over from here, same handoff as the decompose loop-breaker.
    """
    conn = kb_module.connect()
    try:
        exists = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1",
            (task_id, _PM_SUPERVISOR_GAVE_UP_EVENT),
        ).fetchone()
        if exists:
            return False
        with kb_module.write_txn(conn):
            kb_module._append_event(
                conn, task_id, _PM_SUPERVISOR_GAVE_UP_EVENT,
                {"attempts": attempts, "limit": limit},
            )
        return True
    finally:
        conn.close()


def _list_pm_supervisor_candidate_ids(kb_module: Any, *, board_slug: str) -> "list[str]":
    """Cards eligible for an autonomous pm pass on this board.

    Two DISTINCT markers feed this, because ``block_reason_code`` is a real
    ``tasks`` column but ``decompose_gave_up`` never lands there — it is
    recorded ONLY as a ``task_events`` row by
    :func:`_record_decompose_gave_up_once` (that card was already sitting in
    triage and never went through ``block_task`` at all). So:

    * ``block_kind = 'needs_input'`` on a ``blocked`` OR ``triage`` row —
      covers plain needs_input blocks, the spawn-retry ``gave_up`` breaker
      (which always pairs ``block_reason_code='gave_up'`` with
      ``block_kind='needs_input'``, see ``kanban_db.py`` ~13835), and the
      generic unblock-loop breaker's triage route (which preserves
      ``block_kind`` when it reroutes ``blocked`` -> ``triage`` at
      ``BLOCK_RECURRENCE_LIMIT``).
    * a ``triage`` row with a ``decompose_gave_up`` event — the
      auto-decompose loop-breaker's marker, which never touches
      ``block_kind``/``block_reason_code``.

    Excludes, fail-safe, right here (defense in depth — the per-candidate
    handler re-checks these too right before executing, since the LLM
    round-trip between selection and execution takes real time):
    ``human_gate=1``, and any card with a live pending exact-action
    approval. Fails open to an empty list on any DB error — a lookup
    hiccup here must not crash the whole tick.
    """
    try:
        conn = kb_module.connect(board=board_slug)
    except Exception:
        return []
    try:
        rows = conn.execute(
            """
            SELECT t.id FROM tasks t
             WHERE t.status IN ('triage', 'blocked')
               AND COALESCE(t.human_gate, 0) = 0
               AND (
                     t.block_kind = 'needs_input'
                     OR (
                          t.status = 'triage'
                          AND EXISTS (
                                SELECT 1 FROM task_events e
                                 WHERE e.task_id = t.id AND e.kind = ?
                              )
                        )
                   )
               AND NOT EXISTS (
                     SELECT 1 FROM task_pending_actions a
                      WHERE a.task_id = t.id AND a.state IN ('pending', 'approved')
                        AND a.expires_at > ?
                   )
             ORDER BY t.created_at ASC
            """,
            (_DECOMPOSE_GAVE_UP_EVENT, int(time.time())),
        ).fetchall()
        return [r["id"] for r in rows]
    except Exception:
        logger.debug(
            "pm-supervisor: candidate lookup failed on board %s", board_slug, exc_info=True,
        )
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass(frozen=True)
class PmSupervisorDecision:
    """One bound-action decision from the aux LLM. The LLM never executes
    anything itself — it only fills this in; trusted ``kb.*`` helpers do
    the actual mutation. See ``_execute_pm_decision``."""

    action: str
    comment: str = ""
    assignee: Optional[str] = None
    memo: Optional[dict] = None
    severity: str = "routine"


_PM_SYSTEM_PROMPT = """You are the pm-supervisor for the Hermes Agent Kanban board.

A card is stuck: a worker asked a question and blocked (needs_input), or
gave up after repeated failed attempts (gave_up / decompose_gave_up). Your
job is to try to unstick it autonomously BEFORE a human is paged, using
the context you're given (title, body, block reason, recent comments,
current assignee, the available profile roster).

You will be given the card's title/body, its status and block reason, the
last several comments, the current assignee, and the roster of profiles
work can be routed to.

Output a single JSON object with this exact shape:

  {
    "action": "<one of: answer_and_requeue, clarify_dod, reassign, decompose, close_obsolete, escalate>",
    "comment": "<text appropriate to the chosen action -- see below>",
    "assignee": "<profile name from the roster, only for action=reassign, else omit/null>",
    "memo": {
      "situation": "<1-2 sentences, only for action=escalate>",
      "options": ["<option 1>", "<option 2>", ...],
      "recommendation": "<your recommendation>",
      "cost_of_ignoring": "<what happens if nobody looks at this>"
    },
    "severity": "<critical|routine, only for action=escalate>"
  }

Action semantics — pick EXACTLY ONE:
  - "answer_and_requeue": you can answer the worker's question yourself
    from the given context. "comment" is that answer; the card goes back
    to the queue.
  - "clarify_dod": the worker's real problem is an ambiguous definition of
    done. "comment" restates a concrete, checkable DoD; the card goes back
    to the queue.
  - "reassign": a different profile is clearly better suited. "assignee"
    names it (must be from the roster); "comment" explains why.
  - "decompose": the card needs to be broken into smaller pieces before
    anyone can make progress on it. Only valid for a card currently in
    the triage column.
  - "close_obsolete": the card is no longer relevant (superseded,
    duplicate, or the underlying need is gone). "comment" states why.
  - "escalate": none of the above apply — this genuinely needs a human
    decision. Fill in "memo" and "severity". Use "critical" only for
    something time-sensitive or high-blast-radius; everything else is
    "routine" (collected into the operator's regular digest, not paged
    immediately).

When you are not confident which action applies, or the context is too
thin to safely answer/reassign/close, choose "escalate" with
severity="routine" — do NOT guess at closing or reassigning a card you
don't understand.

No preamble, no closing remarks, no code fences. Output only the JSON
object.
"""

_PM_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Status: {status}
Block kind: {block_kind}
Current assignee: {assignee}

Recent comments (oldest first):
{comments}

Available profiles (for action=reassign):
{roster}

Default assignee (fallback when nothing fits): {default_assignee}
"""


def _pm_extract_json_blob(raw: str) -> "Optional[dict]":
    """Lenient JSON-object extraction from an LLM response.

    Deliberately a local copy of ``kanban_decompose._extract_json_blob``'s
    fence-strip + outer-braces + ``json.loads`` approach rather than an
    import — the two call sites belong to different modules and this repo's
    convention (see that function's neighbours) keeps each aux-LLM response
    parser self-contained. A shared ``json_blob`` utility would be a
    reasonable follow-up if a third caller shows up.
    """
    if not raw:
        return None
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(val, dict):
        return None
    return val


def _decide_pm_action(
    call_llm_fn: Callable[..., Any],
    *,
    task: Any,
    comments: "list[Any]",
    profiles_roster: "list[dict]",
    default_assignee: str,
) -> PmSupervisorDecision:
    """One aux-LLM call -> a bound-action decision.

    Never raises. Any failure mode (API error, malformed JSON, unparsable
    response, unknown/missing action) degrades to ``escalate``
    severity="routine" — fail SAFE to a human, never silently closes,
    reassigns, or no-ops a card it didn't understand (invariant 5).
    """
    recent = list(comments)[-8:]
    comments_text = "\n".join(
        f"- [{getattr(c, 'author', '?')}] {getattr(c, 'body', '')}" for c in recent
    ) or "(no comments)"
    roster_text = "\n".join(
        f"  - {p.get('name')}: {p.get('description') or '(no description)'}"
        for p in profiles_roster
    ) or "  (no profiles installed)"
    user_msg = _PM_USER_TEMPLATE.format(
        task_id=getattr(task, "id", "?"),
        title=(getattr(task, "title", "") or "")[:400],
        body=(getattr(task, "body", "") or "(no body)")[:4000],
        status=getattr(task, "status", "?"),
        block_kind=getattr(task, "block_kind", None) or "(none)",
        assignee=getattr(task, "assignee", None) or "(unassigned)",
        comments=comments_text,
        roster=roster_text,
        default_assignee=default_assignee or "(none configured)",
    )

    def _fail_safe(situation: str) -> PmSupervisorDecision:
        memo = dict(_PM_FALLBACK_MEMO)
        memo["situation"] = situation
        return PmSupervisorDecision(action="escalate", severity="routine", memo=memo)

    try:
        resp = call_llm_fn(
            task="kanban_pm_supervisor",
            messages=[
                {"role": "system", "content": _PM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=1500,
            timeout=120,
        )
    except Exception as exc:
        logger.info(
            "pm-supervisor: LLM call failed for %s (%s)", getattr(task, "id", "?"), exc,
        )
        return _fail_safe(f"pm-supervisor LLM call failed: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _pm_extract_json_blob(raw)
    if parsed is None:
        return _fail_safe("pm-supervisor received malformed/unparsable JSON from the aux LLM")

    action = parsed.get("action")
    if action not in _PM_VALID_ACTIONS:
        return _fail_safe(f"pm-supervisor LLM returned an unknown action {action!r}")

    comment = parsed.get("comment")
    comment = comment.strip() if isinstance(comment, str) else ""
    assignee = parsed.get("assignee")
    assignee = assignee.strip() if isinstance(assignee, str) and assignee.strip() else None
    memo = parsed.get("memo")
    memo = memo if isinstance(memo, dict) else None
    severity = parsed.get("severity")
    severity = severity if severity in ("critical", "routine") else "routine"
    return PmSupervisorDecision(
        action=action, comment=comment, assignee=assignee, memo=memo, severity=severity,
    )


def _pm_execute_escalate(
    kb_module: Any, conn: Any, task_id: str, decision: PmSupervisorDecision,
) -> "tuple[bool, str]":
    """``escalate``: no status change — the card stays exactly where it is.

    Writes a human-readable comment (deduplicated via ``add_comment_once``,
    since a repeatedly-stuck card escalating with a similar memo each pass
    must not spam identical comments) plus a ``pm_supervisor_escalated``
    event carrying the structured memo as a loose payload key. Immediate
    vs. digest-collected delivery for severity=critical/routine is left to
    the attention/notifier cron to interpret from this event — this tick
    deliberately does not touch the notification pipeline itself (see the
    handover notes: that pipeline is a separate, carefully-gated surface).
    """
    memo = decision.memo if isinstance(decision.memo, dict) else dict(_PM_FALLBACK_MEMO)
    situation = str(memo.get("situation") or _PM_FALLBACK_MEMO["situation"])[:1000]
    recommendation = str(memo.get("recommendation") or "")[:1000]
    comment_body = situation
    if recommendation:
        comment_body += f"\n\nRecommendation: {recommendation}"
    kb_module.add_comment_once(conn, task_id, _PM_SUPERVISOR_ACTOR, comment_body.strip())
    with kb_module.write_txn(conn):
        kb_module._append_event(
            conn, task_id, _PM_SUPERVISOR_ESCALATED_EVENT,
            {"pm_memo": memo, "severity": decision.severity},
        )
    return True, f"escalate:{decision.severity}"


def _execute_pm_decision(
    kb_module: Any, task_id: str, decision: PmSupervisorDecision, *, board_slug: str,
) -> "tuple[bool, str]":
    """Execute exactly one bound action through trusted ``kb.*`` helpers.

    Returns ``(ok, outcome_label)`` for the caller's audit event. Every
    branch re-checks fresh DB state right before mutating — the LLM call
    that produced ``decision`` can take real wall-clock time, and a human
    or another process may have already moved the card in the meantime.
    That re-check is defense in depth on top of the candidate-selection
    filters, not a replacement for them.
    """
    conn = kb_module.connect(board=board_slug)
    try:
        task = kb_module.get_task(conn, task_id)
        if task is None:
            return False, "task_gone"
        if task.status not in ("triage", "blocked"):
            return False, "status_changed"
        if task.human_gate:
            return False, "human_gate_set"
        if kb_module.get_pending_action(conn, task_id) is not None:
            return False, "pending_action_appeared"

        action = decision.action

        if action in ("answer_and_requeue", "clarify_dod", "reassign"):
            comment_body = decision.comment or f"pm-supervisor: {action}"
            kb_module.add_comment(conn, task_id, _PM_SUPERVISOR_ACTOR, comment_body)
            if action == "reassign":
                if not decision.assignee:
                    return False, "reassign_missing_assignee"
                # Defense in depth (on top of the filtered roster): never let the
                # supervisor route a card to a human-driven profile — the
                # dispatcher would refuse to spawn it and it would park silently.
                try:
                    _hd = kb_module.human_driven_profiles()
                except Exception:
                    _hd = frozenset()
                if decision.assignee.strip().lower() in _hd:
                    return False, "reassign_to_human_driven"
                if not kb_module.assign_task(conn, task_id, decision.assignee):
                    return False, "reassign_failed"
            attn = kb_module.get_current_attention(conn, task_id)
            result = kb_module.transition_task_status_with_attention(
                conn, task_id=task_id, status="ready", actor=_PM_SUPERVISOR_ACTOR,
                expected_attention_id=(attn.id if attn else None),
                expected_attention_version=(attn.version if attn else None),
            )
            ok = getattr(result, "status", None) == "transitioned"
            return ok, (action if ok else f"{action}_failed:{getattr(result, 'status', '?')}")

        if action == "decompose":
            if task.status != "triage":
                # decompose_task only accepts triage-status cards -- a
                # blocked candidate that the LLM chose "decompose" for is a
                # structurally invalid decision. Fail safe to escalate
                # rather than silently dropping it (extension of
                # invariant 5 to "inapplicable", not just "unparsable").
                return _pm_execute_escalate(
                    kb_module, conn, task_id,
                    replace(
                        decision, action="escalate", severity="routine",
                        memo={
                            **_PM_FALLBACK_MEMO,
                            "situation": "pm-supervisor chose decompose on a non-triage card",
                        },
                    ),
                )
            if decision.comment:
                kb_module.add_comment(conn, task_id, _PM_SUPERVISOR_ACTOR, decision.comment)
            from hermes_cli import kanban_decompose as _decomp
            conn.close()  # decompose_task opens its own connection(s)
            outcome = _decomp.decompose_task(task_id, author=_PM_SUPERVISOR_ACTOR)
            return outcome.ok, ("decompose" if outcome.ok else f"decompose_failed:{outcome.reason}")

        if action == "close_obsolete":
            comment_body = decision.comment or "pm-supervisor: closing as obsolete"
            kb_module.add_comment(conn, task_id, _PM_SUPERVISOR_ACTOR, comment_body)
            ok = kb_module.archive_task(conn, task_id, actor=_PM_SUPERVISOR_ACTOR)
            return ok, ("close_obsolete" if ok else "close_obsolete_failed")

        if action == "escalate":
            return _pm_execute_escalate(kb_module, conn, task_id, decision)

        # Unreachable given _decide_pm_action's validation (action is always
        # one of _PM_VALID_ACTIONS by construction) -- fail safe anyway.
        return _pm_execute_escalate(
            kb_module, conn, task_id, replace(decision, action="escalate", severity="routine"),
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _pm_supervisor_handle_candidate(
    *, kb_module: Any, load_config: Callable[[], Any], call_llm_fn: Callable[..., Any],
    board_slug: str, task_id: str,
) -> None:
    """Decide + execute + record for exactly one candidate card."""
    conn = kb_module.connect(board=board_slug)
    try:
        task = kb_module.get_task(conn, task_id)
        if task is None:
            return
        comments = kb_module.list_comments(conn, task_id)
    finally:
        conn.close()

    try:
        from hermes_cli import profiles as _profiles_mod
        # Exclude human-driven profiles (default/work) from the reassign roster
        # the aux-LLM sees — mirrors kanban_decompose._build_roster. Otherwise
        # the supervisor could reassign a stuck card to a human-driven profile,
        # which the dispatcher then refuses to spawn, silently parking it in
        # 'ready' with no attention signal (worse than the original block).
        try:
            _hd = kb_module.human_driven_profiles()
        except Exception:
            _hd = frozenset()
        roster = [
            {"name": p.name, "description": (p.description or "").strip()}
            for p in _profiles_mod.list_profiles()
            if p.name not in _hd
        ]
    except Exception:
        roster = []
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    default_assignee = (kcfg.get("default_assignee") or "").strip() or (task.assignee or "")

    decision = _decide_pm_action(
        call_llm_fn, task=task, comments=comments,
        profiles_roster=roster, default_assignee=default_assignee,
    )
    ok, outcome = _execute_pm_decision(kb_module, task_id, decision, board_slug=board_slug)
    _record_pm_supervisor_attempt(kb_module, task_id, action=decision.action, outcome=outcome)


def _pm_supervisor_tick(
    *, kb_module: Any, load_config: Callable[[], Any], call_llm_fn: Callable[..., Any],
    per_tick: int, max_attempts: int, operator_authors: "frozenset[str]",
) -> int:
    """Run the pm-supervisor for up to ``per_tick`` candidates across all
    boards. Returns the number of candidates actually handled this tick
    (attempt-limited gave-up skips don't consume the per-tick budget, same
    as the auto-decompose tick's ``attempted`` counter)."""
    try:
        boards = kb_module.list_boards(include_archived=False)
    except Exception:
        boards = [kb_module.read_board_metadata(kb_module.DEFAULT_BOARD)]
    handled = 0
    for b in boards:
        if handled >= per_tick:
            break
        slug = b.get("slug") or kb_module.DEFAULT_BOARD
        prev_env = os.environ.get("HERMES_KANBAN_BOARD")
        try:
            os.environ["HERMES_KANBAN_BOARD"] = slug
            candidate_ids = _list_pm_supervisor_candidate_ids(kb_module, board_slug=slug)
            if candidate_ids:
                candidate_ids = _filter_operator_owned_triage_ids(
                    candidate_ids, board_slug=slug,
                    operator_authors=operator_authors, kb_module=kb_module,
                )
            for tid in candidate_ids:
                if handled >= per_tick:
                    break
                prior = _count_pm_supervisor_attempts(kb_module, tid)
                if prior >= max_attempts:
                    if _record_pm_supervisor_gave_up_once(
                        kb_module, tid, attempts=prior, limit=max_attempts,
                    ):
                        logger.warning(
                            "pm-supervisor [%s]: %s gave up after %d attempts (limit %d)",
                            slug, tid, prior, max_attempts,
                        )
                    continue
                handled += 1
                try:
                    _pm_supervisor_handle_candidate(
                        kb_module=kb_module, load_config=load_config, call_llm_fn=call_llm_fn,
                        board_slug=slug, task_id=tid,
                    )
                except Exception:
                    logger.exception("pm-supervisor: candidate handling crashed on %s", tid)
        finally:
            if prev_env is None:
                os.environ.pop("HERMES_KANBAN_BOARD", None)
            else:
                os.environ["HERMES_KANBAN_BOARD"] = prev_env
    return handled


def _root_dispatch_frozen() -> bool:
    """O-1 (2026-07-14): global dispatch freeze kill-switch.

    The per-gateway dispatch gate reads ``load_config()`` — which is PROFILE-scoped
    (no merge with root). Setting ``kanban.dispatch_in_gateway: false`` only in the
    root ``~/.hermes/config.yaml`` therefore left a profile gateway (``-p work`` or
    any of the other profiles that default to True) free to dispatch and silently
    break the freeze. This reads the ROOT config directly — ``~/.hermes/config.yaml``,
    independent of ``HERMES_HOME``/``HERMES_PROFILE`` — and reports whether it
    EXPLICITLY freezes dispatch. Applied as an AND-condition to every gateway lane, so
    one root setting freezes all of them. Fails OPEN (returns False) on any read error
    so a transient/broken root config never wedges an otherwise-enabled dispatcher —
    the profile gate remains the primary control.

    A2/A3 (2026-07-14): delegate to the single source of truth in ``kanban_db`` so the
    root-config read goes through its test-injectable ``_ROOT_CONFIG_PATH_OVERRIDE``
    seam. Behaviour-neutral in prod (the seam is None → reads the real
    ``~/.hermes/config.yaml``); in tests the conftest points the seam at the isolated
    HERMES_HOME, so this watcher copy honours the freeze hermetically too."""
    try:
        from hermes_cli import kanban_db as _kb
        return _kb._root_dispatch_frozen()
    except Exception:
        return False


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Take an exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway process machine-wide may run the embedded kanban
    dispatcher: concurrent dispatchers double the reclaim frequency (each
    runs its own ``release_stale_claims`` → promote → dispatch loop), double
    claim-attempt events in the event log, and — with ``wal_autocheckpoint=0`` —
    concurrent manual WAL checkpoints can corrupt index pages. The
    ``dispatch_in_gateway`` config flag is the primary control; this lock is the
    backstop that survives config drift and same-profile restart races.

    Delegates to :func:`gateway.status._try_acquire_file_lock` (``fcntl`` on
    POSIX, ``msvcrt`` on Windows) so the guard is cross-platform.

    Returns ``(handle, "held")`` on success — the caller keeps the file handle
    for the process lifetime and **must** release it via
    :func:`_release_singleton_lock` when done. ``(None, "contended")`` when
    another process holds the lock (caller must NOT dispatch). ``(None,
    "unavailable")`` when locking cannot be performed (non-POSIX filesystem
    without flock, or the status.py helpers are unimportable) — caller falls
    back to config-only control.
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a dispatcher singleton lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    try:
        from gateway.status import _release_file_lock
        _release_file_lock(handle)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        For each subscription row, fetches ``task_events`` newer than the
        stored cursor with kind in the terminal set (``completed``,
        ``blocked``, ``gave_up``, ``crashed``, ``timed_out``). Sends one
        message per new event to ``(platform, chat_id, thread_id)``,
        then advances the cursor. When a task reaches a terminal state
        (``completed`` / ``archived``), the subscription is removed.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the
        WAL lock. Failures in one tick don't stop subsequent ticks.

        **Multi-board:** iterates every board discovered on disk per
        tick. Subscriptions live inside each board's own DB and cannot
        cross boards, so delivery semantics are unchanged — this is
        purely a fan-out of the single-DB poll.
        """
        # Gate: only the dispatch-owning gateway opens kanban DBs for notifier polling.
        # Non-dispatch gateways have no subscriptions to deliver — all kanban state lives
        # in the dispatch owner's per-board DBs. This prevents N-gateway -shm contention.
        # TODO: gate per-board when per-board dispatcher_owner tracking lands.
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban notifier: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban notifier: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return
        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban notifier: cannot load config (%s); disabled", exc)
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True) or _root_dispatch_frozen():
            logger.info(
                "kanban notifier: disabled via kanban.dispatch_in_gateway=false "
                "(profile or O-1 root freeze)"
            )
            return
        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        # "status" covers dashboard drag-drop and `_set_status_direct()`
        # writes — surface those transitions to subscribers too.
        TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked")
        # Subscriptions are removed only when the task reaches a truly final
        # status (done / archived). We used to also unsub on any terminal
        # event kind (gave_up / crashed / timed_out / blocked), but that
        # silently dropped the user out of the loop whenever the dispatcher
        # respawned the task: a worker that crashes, gets reclaimed, runs
        # again, and crashes a second time would only notify on the first
        # crash because the subscription was deleted after the first event.
        # Same shape as the reblock-after-unblock cycle that PR #22941
        # fixed for `blocked`. Keeping the subscription alive until the
        # task is genuinely done lets the cursor (advanced atomically by
        # claim_unseen_events_for_sub) handle dedup, and any retry-loop
        # event reaches the user.
        # Per-subscription send-failure counter. Adapter.send raising
        # means the chat is dead (deleted, bot kicked, etc.) — after N
        # consecutive send failures the sub is dropped so we don't spin
        # against a dead chat every 5 seconds forever.
        MAX_SEND_FAILURES = 3
        sub_fail_counts: dict[tuple, int] = getattr(
            self, "_kanban_sub_fail_counts", {}
        )
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None)
        if not notifier_profile:
            notifier_profile = self._active_profile_name()
            self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        while self._running:
            # The dispatcher reads these per-board values only as current-tick,
            # process-local Delivery outcomes; they are never reconstructed from rows.
            self._kanban_shadow_duplicate_suppressed_last_tick = {}
            try:
                def _collect():
                    duplicate_suppressed_by_board: dict[str, int] = {}
                    deliveries: list[dict] = []
                    attention_deliveries: list[dict] = []
                    # Enumerate every board on disk, but poll each resolved DB
                    # path once. Multiple slugs can point at the same DB when
                    # HERMES_KANBAN_DB pins the board path; without this guard
                    # one gateway could collect the same subscription/event
                    # more than once before advancing the cursor.
                    try:
                        boards = _kb.list_boards(include_archived=False)
                    except Exception:
                        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                    seen_db_paths: set[str] = set()
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        try:
                            board_slug = _kb._normalize_board_slug(slug) or _kb.DEFAULT_BOARD
                        except (TypeError, ValueError):
                            continue
                        duplicate_suppressed_by_board[board_slug] = 0
                        db_path = board_meta.get("db_path")
                        try:
                            resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                        except Exception:
                            resolved_db_path = f"slug:{slug}"
                        if resolved_db_path in seen_db_paths:
                            logger.debug(
                                "kanban notifier: skipping duplicate board slug %s for DB %s",
                                slug, resolved_db_path,
                            )
                            continue
                        seen_db_paths.add(resolved_db_path)
                        try:
                            conn = _kb.connect(board=slug)
                        except Exception as exc:
                            logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
                            continue
                        try:
                            # `connect()` runs the schema + idempotent migration
                            # on first open per process, so an explicit
                            # `init_db()` here would be redundant. Worse:
                            # `init_db()` deliberately busts the per-process
                            # cache and re-runs the migration on a *second*
                            # connection, which races the first and used to
                            # log a benign but noisy `duplicate column name`
                            # traceback (and intermittent "database is locked"
                            # — issue #21378) on every gateway start against
                            # a legacy DB. `_add_column_if_missing` now
                            # tolerates that race, but we still skip the
                            # redundant call to avoid the wasted work.

                            # Human-Gate v1 (human-gate-design.md): issue +
                            # ntfy-push a fresh one-time token for every
                            # blocked, human_gate=1 card that doesn't
                            # currently have a redeemable one. Deliberately
                            # NOT gated behind `subs`/`active_platforms` —
                            # a gate must fire even when nobody subscribed
                            # via a chat platform (e.g. `kanban block
                            # --human-gate` run from a bare terminal, or
                            # `kanban gate <id> on` for an already-blocked
                            # card). Covers both the fresh-block case and
                            # the re-block-after-unblock rotation case,
                            # since unblock_task only clears the hash on a
                            # successful, gate-satisfying unblock.
                            try:
                                self._kanban_issue_pending_gate_tokens(conn, board=slug)
                            except Exception as exc:
                                logger.warning(
                                    "kanban notifier: gate token scan failed for board %s: %s",
                                    slug, exc,
                                )

                            subs = _kb.list_notify_subs(conn)
                            # Current attention is Core-owned state, never inferred
                            # from event history. Synchronize all subscribed tasks once
                            # per board/tick, then lease per channel below.
                            if subs:
                                sync_outcome = _kb.sync_attention_deliveries(
                                    conn,
                                    task_ids=[sub["task_id"] for sub in subs],
                                )
                                duplicate_suppressed_by_board[board_slug] += int(
                                    sync_outcome.get("suppressed_duplicate", 0) or 0
                                )
                            if not subs:
                                logger.debug("kanban notifier: board %s has no subscriptions", slug)
                            for sub in subs:
                                # A watcher claims only its own profile's subscriptions.
                                # Legacy NULL/empty ownership remains the default profile.
                                owner_profile = (sub.get("notifier_profile") or "default").strip() or "default"
                                if owner_profile != notifier_profile:
                                    continue
                                platform = (sub.get("platform") or "").lower()
                                try:
                                    plat = _Platform(platform)
                                except ValueError:
                                    continue
                                # Resolve through the profile-bound chokepoint: within
                                # this owning watcher, a secondary adapter may deliver,
                                # but never via the default adapter as fallback.
                                if self._authorization_adapter(plat, owner_profile) is None:
                                    continue
                                current_attention = _kb.get_current_attention(conn, sub["task_id"])
                                if current_attention is not None and current_attention.requires_human_action:
                                    attention_claim = _kb.claim_attention_delivery(
                                        conn,
                                        task_id=sub["task_id"],
                                        attention_id=current_attention.id,
                                        attention_version=current_attention.version,
                                        platform=sub["platform"],
                                        chat_id=sub["chat_id"],
                                        thread_id=sub.get("thread_id") or "",
                                    )
                                    if attention_claim is not None:
                                        attention_deliveries.append({
                                            "sub": sub,
                                            "delivery": attention_claim,
                                            "attention": current_attention,
                                            "task": _kb.get_task(conn, sub["task_id"]),
                                            "board": slug,
                                        })
                                suppress_blocked = _kb.has_current_attention_delivery(
                                    conn,
                                    task_id=sub["task_id"],
                                    platform=sub["platform"],
                                    chat_id=sub["chat_id"],
                                    thread_id=sub.get("thread_id") or "",
                                )
                                old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                    conn,
                                    task_id=sub["task_id"],
                                    platform=sub["platform"],
                                    chat_id=sub["chat_id"],
                                    thread_id=sub.get("thread_id") or "",
                                    kinds=TERMINAL_KINDS,
                                )
                                if not events:
                                    continue
                                task = _kb.get_task(conn, sub["task_id"])
                                logger.debug(
                                    "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                    len(events), sub["task_id"], slug, old_cursor, cursor,
                                )
                                deliveries.append({
                                    "sub": sub,
                                    "old_cursor": old_cursor,
                                    "cursor": cursor,
                                    "events": events,
                                    "task": task,
                                    "board": slug,
                                    "suppress_blocked": suppress_blocked,
                                })
                        finally:
                            conn.close()
                    return {
                        "events": deliveries,
                        "attentions": attention_deliveries,
                        "duplicate_suppressed_by_board": duplicate_suppressed_by_board,
                    }

                collected = await asyncio.to_thread(_collect)
                outcomes = collected.get("duplicate_suppressed_by_board", {})
                self._kanban_shadow_duplicate_suppressed_last_tick = (
                    dict(outcomes) if isinstance(outcomes, Mapping) else {}
                )
                deliveries = collected["events"]
                attention_deliveries = collected["attentions"]
                for attention_item in attention_deliveries:
                    # 2026-07-13 (Claude review G3): isolate each attention item.
                    # Previously an unexpected error anywhere in this body (e.g. a
                    # malformed item shape reaching int(attention.id)) aborted the
                    # WHOLE tick and starved the terminal-event loop below, delaying
                    # every unrelated completed/blocked/crashed ping by a full tick.
                    # Self-heals next tick (the lease/cursor is not advanced), so a
                    # per-item guard converts a tick-wide stall into a one-item skip.
                    try:
                        sub = attention_item["sub"]
                        delivery = attention_item["delivery"]
                        attention = attention_item["attention"]
                        board_slug = attention_item["board"]
                        platform_str = (sub["platform"] or "").lower()
                        try:
                            plat = _Platform(platform_str)
                        except ValueError:
                            await asyncio.to_thread(self._kanban_finish_attention_delivery, delivery, False, board_slug)
                            continue
                        adapter = self._authorization_adapter(plat, sub.get("notifier_profile") or None)
                        if adapter is None:
                            await asyncio.to_thread(self._kanban_finish_attention_delivery, delivery, False, board_slug)
                            continue
                        # Revalidate immediately before the external side effect. A
                        # resolved/replaced projection is cancelled, never sent.
                        if not await asyncio.to_thread(self._kanban_attention_is_current, delivery, board_slug):
                            continue
                        task = attention_item["task"]
                        title = (task.title if task else sub["task_id"])[:120]
                        text = (
                            f"Attention for Kanban {sub['task_id']}: {title}\n"
                            f"{attention.type} ({attention.state}) — {attention.summary}"
                        )
                        metadata: dict[str, Any] = {
                            "kanban_attention_identity": {
                                "board": str(board_slug).strip().lower(),
                                "attention_id": int(attention.id),
                                "attention_version": int(attention.version),
                            },
                        }
                        if sub.get("thread_id"):
                            metadata["thread_id"] = sub["thread_id"]
                        try:
                            await adapter.send(sub["chat_id"], text, metadata=metadata)
                        except Exception:
                            # Exception prose can contain platform/user secrets. Keep
                            # durable state and logs to the closed safe code only.
                            await asyncio.to_thread(self._kanban_finish_attention_delivery, delivery, False, board_slug)
                            logger.warning(
                                "kanban attention delivery retry identity=%s/%s/%s",
                                board_slug, attention.id, attention.version,
                            )
                        else:
                            await asyncio.to_thread(self._kanban_finish_attention_delivery, delivery, True, board_slug)
                    except Exception:
                        # Never let one malformed attention item abort the tick and
                        # starve the terminal-event loop below. Message kept generic
                        # so no platform/user prose can leak into logs.
                        logger.warning("kanban attention delivery: item skipped this tick (unexpected error)")
                        continue
                for d in deliveries:
                    sub = d["sub"]
                    task = d["task"]
                    board_slug = d.get("board")
                    platform_str = (sub["platform"] or "").lower()
                    try:
                        plat = _Platform(platform_str)
                    except ValueError:
                        # Unknown platform string; skip and advance cursor so
                        # we don't replay forever.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        continue
                    sub_profile = sub.get("notifier_profile") or ""
                    # Route via the SAME chokepoint the authorization path uses
                    # (gateway/authz_mixin.py::_authorization_adapter): a stamped
                    # profile with its own adapter-registry entry must be served
                    # by THAT profile's same-platform adapter and must NOT silently
                    # fall back to the default profile's adapter — otherwise a
                    # secondary profile's task notification is delivered by the
                    # wrong bot (the cross-profile mis-delivery this whole change
                    # exists to fix). The helper returns None only when the profile
                    # (or default) genuinely has no adapter for the platform.
                    adapter = self._authorization_adapter(plat, sub_profile or None)
                    if adapter is None:
                        logger.debug(
                            "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                            platform_str, sub["task_id"],
                        )
                        await asyncio.to_thread(
                            self._kanban_rewind,
                            sub,
                            d["cursor"],
                            d.get("old_cursor", 0),
                            board_slug,
                        )
                        continue
                    title = (task.title if task else sub["task_id"])[:120]
                    board_tag = f"[{board_slug}] " if board_slug else ""
                    for ev in d["events"]:
                        kind = ev.kind
                        # The durable live attention is the human request. A
                        # historical blocked event still advances the cursor but
                        # must not create a second prompt or wake.
                        if kind == "blocked" and d.get("suppress_blocked"):
                            continue
                        # Identity prefix: attribute terminal pings to the
                        # worker that did the work. Makes fleets (where one
                        # chat subscribes to many tasks) legible at a glance.
                        who = (task.assignee if task and task.assignee else None)
                        tag = f"@{who} " if who else ""
                        if kind == "completed":
                            # Prefer the run's summary (the worker's
                            # intentional human-facing handoff, carried
                            # in the event payload), then fall back to
                            # task.result for legacy rows written before
                            # runs shipped.
                            handoff = ""
                            payload_summary = None
                            if ev.payload and ev.payload.get("summary"):
                                payload_summary = str(ev.payload["summary"])
                            if payload_summary:
                                lines = payload_summary.strip().splitlines()
                                h = lines[0][:200] if lines else payload_summary[:200]
                                handoff = f"\n{h}"
                            elif task and task.result:
                                lines = task.result.strip().splitlines()
                                r = lines[0][:160] if lines else task.result[:160]
                                handoff = f"\n{r}"
                            msg = (
                                f"✔ {board_tag}{tag}Kanban {sub['task_id']} done"
                                f" — {title}{handoff}"
                            )
                        elif kind == "blocked":
                            reason = ""
                            if ev.payload and ev.payload.get("reason"):
                                reason = f": {str(ev.payload['reason'])[:160]}"
                            msg = f"⏸ {board_tag}{tag}Kanban {sub['task_id']} blocked{reason}"
                        elif kind == "gave_up":
                            err = ""
                            if ev.payload and ev.payload.get("error"):
                                err = f"\n{str(ev.payload['error'])[:200]}"
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} gave up "
                                f"after repeated spawn failures{err}"
                            )
                        elif kind == "crashed":
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} worker crashed "
                                f"(pid gone); dispatcher will retry"
                            )
                        elif kind == "timed_out":
                            limit = 0
                            if ev.payload and ev.payload.get("limit_seconds"):
                                limit = int(ev.payload["limit_seconds"])
                            msg = (
                                f"⏱ {board_tag}{tag}Kanban {sub['task_id']} timed out "
                                f"(max_runtime={limit}s); will retry"
                            )
                        elif kind == "status":
                            new_status = ""
                            if ev.payload and ev.payload.get("status"):
                                new_status = str(ev.payload["status"])
                            msg = f"🔄 {board_tag}{tag}Kanban {sub['task_id']} → {new_status}"
                        else:
                            # archived / unblocked are claimed by TERMINAL_KINDS
                            # (so the cursor advances past them and they can't
                            # wedge a later completed/blocked event behind an
                            # unclaimed row) but are intentionally SILENT: an
                            # archive needs no user ping, and unblocked is an
                            # internal transition. They are also excluded from
                            # _WAKE_KINDS below, so they never wake the creator.
                            continue
                        metadata: dict[str, Any] = {}
                        if sub.get("thread_id"):
                            metadata["thread_id"] = sub["thread_id"]
                        sub_key = (
                            sub["task_id"], sub["platform"],
                            sub["chat_id"], sub.get("thread_id") or "",
                        )
                        try:
                            await adapter.send(
                                sub["chat_id"], msg, metadata=metadata,
                            )
                            logger.debug(
                                "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                                kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                            )
                            # After delivering the text notification, surface
                            # any artifact paths the worker referenced in
                            # ``kanban_complete(summary=..., artifacts=[...])``
                            # (or the legacy ``result`` field) as native
                            # uploads. ``extract_local_files`` finds bare
                            # absolute paths in the summary;
                            # ``send_document`` / ``send_image_file`` uploads
                            # them. Only fires on the ``completed`` event so
                            # we never spam attachments on retries.
                            if kind == "completed":
                                try:
                                    await self._deliver_kanban_artifacts(
                                        adapter=adapter,
                                        chat_id=sub["chat_id"],
                                        metadata=metadata,
                                        event_payload=getattr(ev, "payload", None),
                                        task=task,
                                    )
                                except Exception as art_exc:
                                    logger.debug(
                                        "kanban notifier: artifact delivery for %s failed: %s",
                                        sub["task_id"], art_exc,
                                    )
                            # Reset the failure counter on success.
                            sub_fail_counts.pop(sub_key, None)
                        except Exception as exc:
                            fails = sub_fail_counts.get(sub_key, 0) + 1
                            sub_fail_counts[sub_key] = fails
                            logger.warning(
                                "kanban notifier: send failed for %s on %s "
                                "(attempt %d/%d): %s",
                                sub["task_id"], platform_str, fails,
                                MAX_SEND_FAILURES, exc,
                            )
                            if fails >= MAX_SEND_FAILURES:
                                logger.warning(
                                    "kanban notifier: dropping subscription "
                                    "%s on %s after %d consecutive send failures",
                                    sub["task_id"], platform_str, fails,
                                )
                                await asyncio.to_thread(self._kanban_unsub, sub, board_slug)
                                sub_fail_counts.pop(sub_key, None)
                            else:
                                await asyncio.to_thread(
                                    self._kanban_rewind,
                                    sub,
                                    d["cursor"],
                                    d.get("old_cursor", 0),
                                    board_slug,
                                )
                            # Rewind the pre-send claim on transient failure so
                            # a later tick can retry. After too many failures,
                            # dropping the subscription is the terminal action.
                            break
                    else:
                        # All events delivered; advance cursor. The cursor
                        # is the dedup mechanism — it prevents re-delivery
                        # of the same event on subsequent ticks.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        # Unsubscribe only when the task has reached a truly
                        # final status (done / archived). For blocked /
                        # gave_up / crashed / timed_out the subscription is
                        # kept alive so the user gets notified again if the
                        # dispatcher respawns the task and it cycles into the
                        # same state. See the longer comment on TERMINAL_KINDS
                        # above for the failure mode this prevents.
                        task_terminal = task and task.status in {"done", "archived"}
                        _WAKE_KINDS = ("completed", "gave_up", "crashed", "timed_out", "blocked")
                        _wake_kinds = {
                            ev.kind for ev in d["events"]
                            if ev.kind in _WAKE_KINDS
                            and not (ev.kind == "blocked" and d.get("suppress_blocked"))
                        }
                        if _wake_kinds:
                            try:
                                _session_key = getattr(task, "session_id", None) or ""
                                if _session_key:
                                    _title = (task.title if task else sub["task_id"])[:120]
                                    _assignee = task.assignee if task else ""
                                    _parts = []
                                    if "completed" in _wake_kinds: _parts.append(t("gateway.kanban.wake.completed"))
                                    if "gave_up" in _wake_kinds: _parts.append(t("gateway.kanban.wake.gave_up"))
                                    if "crashed" in _wake_kinds: _parts.append(t("gateway.kanban.wake.crashed"))
                                    if "timed_out" in _wake_kinds: _parts.append(t("gateway.kanban.wake.timed_out"))
                                    if "blocked" in _wake_kinds: _parts.append(t("gateway.kanban.wake.blocked"))
                                    _status = t("gateway.kanban.wake.status_joiner").join(_parts) or t("gateway.kanban.wake.status_default")
                                    _synth = t(
                                        "gateway.kanban.wake.message",
                                        task_id=sub["task_id"],
                                        status=_status,
                                        title=_title,
                                        assignee=_assignee,
                                        board=board_slug,
                                    )
                                    from gateway.session import SessionSource
                                    from gateway.platforms.base import MessageEvent, MessageType
                                    # KNOWN LIMITATION (tracked follow-up): the
                                    # subscription row does not persist the
                                    # creator's chat_type, and it is not carried
                                    # on the session-context bridge, so we cannot
                                    # faithfully reconstruct the creator's real
                                    # session key here. build_session_key() keys
                                    # DMs (":dm:<chat_id>") on a wholly different
                                    # shape from group/thread, so any hardcoded
                                    # value mis-routes some creators. "group" is
                                    # the least-surprising default for the
                                    # dashboard/group flows this wake primarily
                                    # serves; DM-originated creators are handled
                                    # by the follow-up that stamps + persists
                                    # chat_type end-to-end. handle_message()
                                    # get_or_create_session's the target, so a
                                    # mismatch degrades to "wake lands in a fresh
                                    # group session" — never an exception.
                                    _source = SessionSource(
                                        platform=plat,
                                        chat_id=sub["chat_id"],
                                        chat_type="group",
                                        thread_id=sub.get("thread_id") or None,
                                        user_id=sub.get("user_id"),
                                        profile=sub_profile or None,
                                    )
                                    _synth_event = MessageEvent(
                                        text=_synth,
                                        message_type=MessageType.TEXT,
                                        source=_source,
                                        internal=True,
                                    )
                                    await adapter.handle_message(_synth_event)
                                    logger.info(
                                        "kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                                        sub["task_id"], platform_str, sub["chat_id"], sub_profile or "default", _wake_kinds,
                                    )
                            except Exception as _wk_err:
                                # Best-effort: the notification itself already
                                # delivered and the cursor has advanced, so a
                                # broken wake path must not wedge the tick — but
                                # log at WARNING with a traceback rather than
                                # DEBUG so a persistently-failing wake is visible
                                # in normal logs instead of silently no-op'ing.
                                logger.warning(
                                    "kanban notifier: wakeup injection failed for %s: %s",
                                    sub["task_id"], _wk_err, exc_info=True,
                                )
                        if task_terminal:
                            await asyncio.to_thread(
                                self._kanban_unsub, sub, board_slug,
                            )
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            # Sleep with cancellation checks.
            for _ in range(int(max(1, interval))):
                if not self._running:
                    return
                await asyncio.sleep(1)

    def _kanban_attention_is_current(self, delivery: Mapping[str, Any], board: Optional[str]) -> bool:
        """Fail closed before an external attention send; no event history."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.sync_attention_deliveries(conn, task_ids=[str(delivery["task_id"])])
            current = _kb.get_current_attention(conn, str(delivery["task_id"]))
            subscribed = conn.execute(
                "SELECT 1 FROM kanban_notify_subs WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=? "
                "AND active=1 AND generation=?",
                (str(delivery["task_id"]), delivery["platform"], delivery["chat_id"],
                 delivery.get("thread_id") or "", int(delivery["subscription_generation"])),
            ).fetchone()
            return bool(
                subscribed is not None
                and current is not None
                and current.id == int(delivery["attention_id"])
                and current.version == int(delivery["attention_version"])
                and current.requires_human_action
            )
        finally:
            conn.close()

    def _kanban_finish_attention_delivery(
        self, delivery: Mapping[str, Any], success: bool, board: Optional[str],
    ) -> bool:
        """Persist the outcome using the exact opaque lease generation."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.finish_attention_delivery(conn, delivery, success=success)
        finally:
            conn.close()

    def _kanban_advance(
        self, sub: dict, cursor: int, board: Optional[str] = None,
    ) -> None:
        """Sync helper: advance a subscription's cursor. Runs in to_thread.

        ``board`` scopes the DB connection to the board that owns this
        subscription. Unsub cursors in one board can't touch another's.
        """
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.advance_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                new_cursor=cursor,
            )
        finally:
            conn.close()

    def _kanban_issue_pending_gate_tokens(
        self, conn, *, board: Optional[str] = None,
    ) -> None:
        """Provision gate notifications for every gated card on this board.

        Two steps, both board-agnostic (they fire even when no chat platform
        subscribed):
          1. Subscribe the durable operator channel to every blocked,
             human_gate card (``ensure_ops_channel_gate_subs``) so a gate
             reaches the operator on ANY board via the normal notifier
             delivery — this replaced the old ntfy push.
          2. Issue a one-time CLI ``--token`` for cards without one (kept for
             the interactive rescue path; ntfy delivery is off by default).

        Sync helper called from ``_kanban_notifier_watcher``'s per-board
        ``_collect()`` on the SAME connection it already opened for that
        board this tick — no extra connect.
        """
        from hermes_cli import kanban_db as _kb
        # (1a) Durable ops-channel subscription for gated cards (idempotent).
        try:
            _kb.ensure_ops_channel_gate_subs(conn, board=board)
        except Exception as exc:
            logger.warning(
                "kanban notifier: ops-channel gate subscribe failed for board %s: %s",
                board, exc,
            )
        # (1b) Inherit the default board's subscriber channels onto this
        # (non-default) board's gated cards, so the operator's cockpit is
        # notified about gates on every board ("all boards use default's
        # settings"). No-op on default; idempotent.
        try:
            _kb.mirror_default_board_subs_to_gated(conn, board=board)
        except Exception as exc:
            logger.warning(
                "kanban notifier: default-sub mirror failed for board %s: %s",
                board, exc,
            )
        # (2) One-time CLI token issuance (ntfy push gated behind config).
        rows = conn.execute(
            "SELECT id FROM tasks WHERE status = 'blocked' AND human_gate = 1 "
            "AND gate_token_hash IS NULL"
        ).fetchall()
        for row in rows:
            tid = row["id"]
            try:
                _kb.issue_and_notify_gate_token(conn, tid, board=board)
            except Exception as exc:
                logger.warning(
                    "kanban notifier: gate token issue failed for %s: %s",
                    tid, exc,
                )

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.remove_notify_sub(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
            )
        finally:
            conn.close()

    def _kanban_rewind(
        self,
        sub: dict,
        claimed_cursor: int,
        old_cursor: int,
        board: Optional[str] = None,
    ) -> None:
        """Sync helper: undo a claimed notification cursor after send failure."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.rewind_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                claimed_cursor=claimed_cursor,
                old_cursor=old_cursor,
            )
        finally:
            conn.close()

    async def _deliver_kanban_artifacts(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        event_payload: Optional[dict],
        task,
    ) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
        file paths through the completion event so downstream humans get
        the deliverable as a native upload instead of a path printed in
        chat.

        Sources scanned, in priority order:
          1. ``event_payload['artifacts']`` (explicit list — preferred)
          2. ``event_payload['summary']`` (truncated first line)
          3. ``task.result`` (legacy fallback)

        Files are deduplicated, missing files are silently skipped (the
        path may have been mentioned for reference only), and delivery
        errors are logged but do not break the notifier loop.
        """
        from pathlib import Path as _Path

        candidates: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            if not path:
                return
            expanded = os.path.expanduser(path)
            if expanded in seen:
                return
            if not os.path.isfile(expanded):
                return
            seen.add(expanded)
            candidates.append(expanded)

        # 1. Explicit artifacts list in payload.
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, str):
                        _add(item)

            # 2. Paths embedded in the payload summary.
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                paths, _ = adapter.extract_local_files(summary)
                for p in paths:
                    _add(p)

        # 3. Legacy: paths embedded in task.result.
        if task is not None and getattr(task, "result", None):
            result_text = str(task.result)
            paths, _ = adapter.extract_local_files(result_text)
            for p in paths:
                _add(p)

        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        if not candidates:
            return

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

        from urllib.parse import quote as _quote

        # Partition images so they ride a single send_multiple_images call
        # on platforms that support batch image uploads (Signal/Slack RPCs).
        image_paths = [p for p in candidates if _Path(p).suffix.lower() in _IMAGE_EXTS]
        other_paths = [p for p in candidates if _Path(p).suffix.lower() not in _IMAGE_EXTS]

        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(
                    chat_id=chat_id, images=batch, metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: image batch upload failed: %s", exc,
                )

        for path in other_paths:
            ext = _Path(path).suffix.lower()
            try:
                if ext in _VIDEO_EXTS:
                    await adapter.send_video(
                        chat_id=chat_id, video_path=path, metadata=metadata,
                    )
                else:
                    await adapter.send_document(
                        chat_id=chat_id, file_path=path, metadata=metadata,
                    )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: artifact upload (%s) failed: %s",
                    path, exc,
                )

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
        When true, the gateway hosts the single dispatcher for this profile:
        no separate `hermes kanban daemon` process needed. When false, the
        loop exits immediately and an external daemon is expected.

        Each tick calls :func:`kanban_db.dispatch_once` inside
        ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
        event loop. Failures in one tick don't stop subsequent ticks —
        same pattern as `_kanban_notifier_watcher`.

        Shutdown: the loop checks ``self._running`` between ticks; gateway
        stop() flips it to False and cancels pending tasks, and the
        in-flight ``to_thread`` returns on its own after the current
        ``dispatch_once`` call finishes (typically <1ms on an idle board).
        """
        # Read config once at boot. If the user flips the flag later, they
        # restart the gateway; same pattern as every other background
        # watcher here. Honours HERMES_KANBAN_DISPATCH_IN_GATEWAY env var
        # as an escape hatch (false-y value disables without editing YAML).
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return

        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True) or _root_dispatch_frozen():
            logger.info(
                "kanban dispatcher: disabled via kanban.dispatch_in_gateway=false "
                "(profile or O-1 root freeze)"
            )
            return

        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return

        # Single-dispatcher backstop. dispatch_in_gateway defaults to true, so a
        # new profile gateway (or a same-profile restart race) can silently
        # start a second dispatcher; concurrent dispatchers double reclaim
        # frequency, double claim-attempt events, and — with
        # wal_autocheckpoint=0 — concurrent manual WAL checkpoints can corrupt
        # index pages. The lock lives at the machine-global kanban root
        # (shared across profiles by design), so it serialises ALL gateways.
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            logger.info(
                "kanban dispatcher: another gateway already holds the dispatcher "
                "lock (%s); this gateway will NOT dispatch.", _lock_path,
            )
            return
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            logger.info("kanban dispatcher: holding singleton dispatcher lock (%s)", _lock_path)
        else:
            logger.warning(
                "kanban dispatcher: advisory lock unavailable at %s; proceeding "
                "on config control alone.", _lock_path,
            )

        try:
            interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
        except (ValueError, TypeError):
            logger.warning(
                "kanban dispatcher: invalid dispatch_interval_seconds=%r, using default 60",
                kanban_cfg.get("dispatch_interval_seconds"),
            )
            interval = 60.0
        interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

        # Read max_spawn config to limit concurrent kanban tasks
        max_spawn = kanban_cfg.get("max_spawn", None)
        if max_spawn is not None:
            logger.info(f"kanban dispatcher: max_spawn={max_spawn}")

        # Cap the number of simultaneously running tasks so slow workers
        # (local LLMs, resource-constrained hosts) don't pile up and time
        # out. When set, the dispatcher skips spawning when the board
        # already has this many tasks in 'running' status.
        raw_max_in_progress = kanban_cfg.get("max_in_progress", None)
        max_in_progress = None
        if raw_max_in_progress is not None:
            try:
                max_in_progress = int(raw_max_in_progress)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress=%r; ignoring",
                    raw_max_in_progress,
                )
                max_in_progress = None
            else:
                if max_in_progress < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress=%r is below 1; ignoring",
                        raw_max_in_progress,
                    )
                    max_in_progress = None
                else:
                    logger.info(f"kanban dispatcher: max_in_progress={max_in_progress}")

        raw_failure_limit = kanban_cfg.get("failure_limit", _kb.DEFAULT_FAILURE_LIMIT)
        try:
            failure_limit = int(raw_failure_limit)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT
        if failure_limit < 1:
            logger.warning(
                "kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT

        # Read stale_timeout_seconds — 0 disables stale detection.
        raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
        resource_monitor = kanban_cfg.get("resource_monitor", {})
        if not isinstance(resource_monitor, dict):
            resource_monitor = {}
        try:
            stale_timeout_seconds = int(raw_stale or 0)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
                "disabling stale detection",
                raw_stale,
            )
            stale_timeout_seconds = 0

        # Read kanban.default_assignee — fallback profile for tasks
        # created without an explicit assignee (e.g. via the dashboard).
        # When set, the dispatcher applies it to unassigned ready tasks
        # instead of skipping them indefinitely (#27145). Empty string
        # (the schema default) means "no fallback, keep skipping" —
        # backward-compatible with existing installs.
        default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
        if default_assignee:
            logger.info(
                "kanban dispatcher: default_assignee=%r (unassigned ready tasks "
                "will route to this profile)",
                default_assignee,
            )

        # Read kanban.max_in_progress_per_profile — per-profile concurrency
        # cap (#21582). When set, no single profile gets more than N
        # workers running at once, even if the global max_in_progress
        # would allow it. Prevents one profile's local model / API quota
        # / browser pool from being overwhelmed by a fan-out.
        raw_per_profile = kanban_cfg.get("max_in_progress_per_profile", None)
        max_in_progress_per_profile = None
        if raw_per_profile is not None:
            try:
                max_in_progress_per_profile = int(raw_per_profile)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress_per_profile=%r; ignoring",
                    raw_per_profile,
                )
                max_in_progress_per_profile = None
            else:
                if max_in_progress_per_profile < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress_per_profile=%r is below 1; ignoring",
                        raw_per_profile,
                    )
                    max_in_progress_per_profile = None
                else:
                    logger.info(
                        "kanban dispatcher: max_in_progress_per_profile=%d",
                        max_in_progress_per_profile,
                    )

        # Initial delay so the gateway finishes wiring adapters before the
        # dispatcher spawns workers (those workers may hit gateway notify
        # subscriptions etc.). Matches the notifier watcher's delay.
        await asyncio.sleep(5)

        # Health telemetry mirrored from `_cmd_daemon`: warn when ready
        # queue is non-empty but spawns are 0 for N consecutive ticks —
        # usually means broken PATH, missing venv, or credential loss.
        HEALTH_WINDOW = 6
        bad_ticks = 0
        last_warn_at = 0
        shadow_last_warning: dict[tuple[str, str], int] = {}
        # Avoid hot-looping corrupt-looking board DBs, but do not suppress
        # same-fingerprint retries forever: transient WAL/open races can
        # surface as "database disk image is malformed" for one tick.
        CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300
        disabled_corrupt_boards: dict[
            str, tuple[tuple[str, int | None, int | None], float]
        ] = {}

        def _board_db_fingerprint(slug: str) -> tuple[str, int | None, int | None]:
            path = _kb.kanban_db_path(slug)
            try:
                resolved = str(path.expanduser().resolve())
            except Exception:
                resolved = str(path)
            try:
                stat = path.stat()
            except OSError:
                return (resolved, None, None)
            return (resolved, stat.st_mtime_ns, stat.st_size)

        def _is_corrupt_board_db_error(exc: Exception) -> bool:
            corrupt_guard_error = getattr(_kb, "KanbanDbCorruptError", None)
            if corrupt_guard_error is not None and isinstance(exc, corrupt_guard_error):
                return True
            if not isinstance(exc, sqlite3.DatabaseError):
                return False
            msg = str(exc).lower()
            return (
                "file is not a database" in msg
                or "database disk image is malformed" in msg
            )

        def _tick_once_for_board(slug: str) -> "Optional[object]":
            """Run one dispatch_once for a specific board.

            Runs in a worker thread via `asyncio.to_thread`. `board=slug`
            is passed through `dispatch_once` so `resolve_workspace` and
            `_default_spawn` see the right paths. The per-board DB is
            opened explicitly so concurrent boards never share a
            connection handle or accidentally claim across each other.
            """
            conn = None
            fingerprint = _board_db_fingerprint(slug)
            disabled_entry = disabled_corrupt_boards.get(slug)
            if disabled_entry is not None:
                disabled_fingerprint, disabled_at = disabled_entry
                age = time.monotonic() - disabled_at
                if (
                    disabled_fingerprint == fingerprint
                    and age < CORRUPT_BOARD_RETRY_AFTER_SECONDS
                ):
                    return None
                if disabled_fingerprint == fingerprint:
                    logger.info(
                        "kanban dispatcher: board %s database fingerprint unchanged "
                        "after %.0fs quarantine; retrying dispatch",
                        slug,
                        age,
                    )
                else:
                    logger.info(
                        "kanban dispatcher: board %s database changed; retrying dispatch",
                        slug,
                    )
                disabled_corrupt_boards.pop(slug, None)
            try:
                conn = _kb.connect(board=slug)
                # `connect()` runs the schema + idempotent migration on
                # first open per process; the previous explicit
                # `init_db()` call here busted the per-process cache and
                # re-ran the migration on a second connection, racing
                # the first. See the matching comment in
                # `_kanban_notifier_watcher` and issue #21378.
                return _kb.dispatch_once(
                    conn,
                    board=slug,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                    stale_timeout_seconds=stale_timeout_seconds,
                    default_assignee=default_assignee,
                    max_in_progress_per_profile=max_in_progress_per_profile,
                    resource_monitor=resource_monitor,
                )
            except sqlite3.DatabaseError as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            except Exception as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _tick_once() -> "list[tuple[str, Optional[object]]]":
            """Run one dispatch_once per board. Returns (slug, result) pairs.

            Enumerating boards on every tick keeps the dispatcher honest
            when users create a new board mid-run: no restart required,
            the next tick picks it up automatically.
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            out: list[tuple[str, "Optional[object]"]] = []
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                out.append((slug, _tick_once_for_board(slug)))
            return out

        def _ready_nonempty() -> bool:
            """Cheap probe: is there at least one ready+assigned+unclaimed
            task on ANY board whose assignee maps to a real Hermes profile
            (i.e. one the dispatcher would actually spawn for)?

            Tasks assigned to control-plane lanes (e.g. ``orion-cc``,
            ``orion-research``) are pulled by terminals via
            ``claim_task`` directly and never spawnable, so a queue full
            of those is "correctly idle", not "stuck". Filtering them out
            here keeps the stuck-warn fire only on real failures (broken
            PATH, missing venv, credential loss for a real Hermes profile).
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                conn = None
                try:
                    conn = _kb.connect(board=slug)
                    if _kb.has_spawnable_ready(conn):
                        return True
                    if _kb.has_spawnable_review(conn):
                        return True
                except Exception:
                    continue
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
            return False

        # Auto-decompose: turn fresh triage tasks into ready workgraphs
        # before the dispatcher fans out workers. Gated by
        # ``kanban.auto_decompose`` (default True). Capped by
        # ``kanban.auto_decompose_per_tick`` (default 3) so a bulk-load
        # of triage tasks doesn't burst-spend the aux LLM in one tick;
        # remainder defers to subsequent ticks.
        #
        # The flag is re-read from config EVERY tick (#49638) rather than
        # captured once at boot. Auto-decompose is a safety toggle: a user who
        # sees it fan out and run tasks they didn't intend reaches for
        # ``kanban.auto_decompose: false`` to STOP it — and that must take
        # effect on the next tick, not require a gateway restart. (Reported:
        # auto-decompose created and launched destructive tasks while the user
        # was still typing the task description, and the flag "couldn't be
        # disabled" because the gateway had captured its boot-time value.)
        # pm-supervisor: give needs_input/gave_up/decompose_gave_up cards one
        # autonomous pass BEFORE they page a human. Off by default
        # (``kanban.pm_supervisor_enabled``); re-read live every tick, same
        # reasoning as auto-decompose above. Runs BEFORE auto-decompose so a
        # card the pm-supervisor routes to "decompose" is picked up by the
        # very next block in this same tick, not a tick later.
        def _read_pm_supervisor_settings() -> "tuple[bool, int, int]":
            return _resolve_pm_supervisor_settings(_load_config)

        def _pm_supervisor_tick_once(per_tick: int, max_attempts: int) -> int:
            try:
                from agent.auxiliary_client import call_llm as _call_llm
            except Exception as exc:
                logger.warning(
                    "pm-supervisor: auxiliary client import failed (%s); skipping", exc,
                )
                return 0
            operator_authors = _resolve_operator_authors(_load_config)
            return _pm_supervisor_tick(
                kb_module=_kb, load_config=_load_config, call_llm_fn=_call_llm,
                per_tick=per_tick, max_attempts=max_attempts,
                operator_authors=operator_authors,
            )

        def _read_auto_decompose_settings() -> tuple[bool, int]:
            """Re-resolve (enabled, per_tick) from current config each tick."""
            return _resolve_auto_decompose_settings(_load_config)

        def _auto_decompose_tick(auto_decompose_per_tick: int) -> int:
            """Run the auto-decomposer for up to N triage tasks across all
            boards. Returns the number of triage tasks that were
            successfully decomposed or specified this tick.
            """
            try:
                from hermes_cli import kanban_decompose as _decomp
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "kanban auto-decompose: import failed (%s); skipping", exc,
                )
                return 0
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            attempted = 0
            successes = 0
            # K-10: Versuchslimit pro Karte, live aus der Config (0 = aus).
            max_attempts = _resolve_decompose_max_attempts(_load_config)
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                if attempted >= auto_decompose_per_tick:
                    break
                # Pin this board for the duration of the call — same
                # pattern as the dashboard specify endpoint. The
                # decomposer module connects with no board kwarg and
                # relies on the env var.
                prev_env = os.environ.get("HERMES_KANBAN_BOARD")
                try:
                    os.environ["HERMES_KANBAN_BOARD"] = slug
                    try:
                        triage_ids = _decomp.list_triage_ids()
                    except Exception as exc:
                        logger.debug(
                            "kanban auto-decompose: list_triage_ids failed on board %s (%s)",
                            slug, exc,
                        )
                        triage_ids = []
                    if triage_ids:
                        operator_authors = _resolve_operator_authors(_load_config)
                        triage_ids = _filter_operator_owned_triage_ids(
                            triage_ids,
                            board_slug=slug,
                            operator_authors=operator_authors,
                            kb_module=_kb,
                        )
                    for tid in triage_ids:
                        if attempted >= auto_decompose_per_tick:
                            break
                        if max_attempts:
                            prior = _count_failed_decompose_attempts(_kb, tid)
                            if prior >= max_attempts:
                                # Ausgeschöpft: nicht mehr retryen (kein
                                # Aux-Spend, kein per-tick-Budget) — Karte
                                # bleibt sichtbar in triage, Ping macht der
                                # Attention-Cron nach dem Grace-Fenster.
                                if _record_decompose_gave_up_once(
                                    _kb, tid, attempts=prior, limit=max_attempts,
                                ):
                                    logger.warning(
                                        "kanban auto-decompose [%s]: %s gave up "
                                        "after %d failed attempts (limit %d)",
                                        slug, tid, prior, max_attempts,
                                    )
                                continue
                        attempted += 1
                        try:
                            outcome = _decomp.decompose_task(
                                tid, author="auto-decomposer",
                            )
                        except Exception:
                            logger.exception(
                                "kanban auto-decompose: decompose_task crashed on %s",
                                tid,
                            )
                            if max_attempts:
                                _record_failed_decompose_attempt(
                                    _kb, tid, "decompose_task crashed",
                                )
                            continue
                        if outcome.ok:
                            successes += 1
                            if outcome.fanout and outcome.child_ids:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → %d children",
                                    slug, tid, len(outcome.child_ids),
                                )
                            else:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → single task (no fanout)",
                                    slug, tid,
                                )
                        else:
                            # Common no-op reasons (no aux client configured) shouldn't
                            # spam logs every tick. Log at debug.
                            logger.debug(
                                "kanban auto-decompose [%s]: %s skipped: %s",
                                slug, tid, outcome.reason,
                            )
                            if max_attempts:
                                _record_failed_decompose_attempt(
                                    _kb, tid, outcome.reason or "not ok",
                                )
                finally:
                    if prev_env is None:
                        os.environ.pop("HERMES_KANBAN_BOARD", None)
                    else:
                        os.environ["HERMES_KANBAN_BOARD"] = prev_env
            return successes

        try:
            maintenance_interval = float(
                kanban_cfg.get("maintenance_interval_seconds", 120) or 120
            )
        except (ValueError, TypeError):
            maintenance_interval = 120.0
        maintenance_interval = max(maintenance_interval, 10.0)

        # Maintenance runs on a DEDICATED single-worker executor, not the
        # default to_thread pool, so shutdown can actually wait for it.
        # ``task.cancel()`` unblocks the awaiting coroutine but does NOT stop
        # the OS thread: an aux-LLM round keeps mutating the board for up to
        # 180 s afterwards. Releasing the singleton dispatcher lock in that
        # window let a fresh gateway take over while the old maintenance was
        # still writing (TARS review 2026-07-27, P1).
        maintenance_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="kanban-maintenance",
        )
        maintenance_stop = threading.Event()
        maintenance_inflight: "list[concurrent.futures.Future]" = []

        async def _run_maintenance(fn, *fn_args):
            """Submit to the dedicated executor, keeping the future reachable.

            The future is registered ONCE and only ever removed by the drain
            once it has actually completed. Removing it in a ``finally`` was
            wrong: cancelling the awaiting coroutine runs that ``finally``
            while the executor thread keeps going, so the drain saw an empty
            list and released the lock under a still-writing thread — the
            same P1 the first repair round meant to close (TARS re-review).
            """
            future = maintenance_executor.submit(fn, *fn_args)
            maintenance_inflight.append(future)
            return await asyncio.wrap_future(future)

        def _drain_maintenance(timeout: float) -> bool:
            """Signal stop and wait for in-flight maintenance. True if idle."""
            maintenance_stop.set()
            # Prune only genuinely finished work; a cancelled awaiter leaves
            # its future here on purpose.
            for done_future in [f for f in list(maintenance_inflight) if f.done()]:
                try:
                    maintenance_inflight.remove(done_future)
                except ValueError:
                    pass
            pending = [f for f in list(maintenance_inflight) if not f.done()]
            if not pending:
                return True
            _, not_done = concurrent.futures.wait(pending, timeout=timeout)
            for finished in [f for f in pending if f.done()]:
                try:
                    maintenance_inflight.remove(finished)
                except ValueError:
                    pass
            return not not_done

        def _finish_dispatcher(drain_timeout: float) -> None:
            """Release the singleton lock ONLY once maintenance is quiet.

            If a round is still in flight we deliberately keep holding the
            lock: the flock is dropped by the kernel when this process exits,
            so worst case we block a same-process watcher restart — far
            better than two gateways mutating one board concurrently.
            """
            drained = _drain_maintenance(drain_timeout)
            maintenance_executor.shutdown(wait=False)
            handle = self._kanban_dispatcher_lock_handle
            if drained:
                _release_singleton_lock(handle)
                self._kanban_dispatcher_lock_handle = None
                return
            # Not drained: releasing now would hand the board to a second
            # dispatcher mid-write. Holding it until process exit avoids that
            # but wedges every in-process watcher restart (TARS re-review).
            # Instead hand the release to a watchdog that waits for the actual
            # thread to finish — the window is bounded by the aux-LLM timeout,
            # not by the process lifetime.
            self._kanban_dispatcher_lock_handle = None

            def _release_after_maintenance() -> None:
                try:
                    pending = [f for f in list(maintenance_inflight) if not f.done()]
                    if pending:
                        concurrent.futures.wait(pending)
                except Exception:
                    logger.warning(
                        "kanban maintenance: watchdog wait failed; releasing "
                        "the dispatcher lock anyway", exc_info=True,
                    )
                _release_singleton_lock(handle)
                logger.info(
                    "kanban maintenance finished after shutdown — dispatcher "
                    "singleton lock released"
                )

            logger.warning(
                "kanban maintenance still in flight after %.0fs — dispatcher "
                "singleton lock stays held until it finishes (no second "
                "dispatcher may take over mid-write)",
                drain_timeout,
            )
            threading.Thread(
                target=_release_after_maintenance,
                name="kanban-maintenance-lock-release",
                daemon=True,
            ).start()

        async def _maintenance_loop() -> None:
            # pm-supervisor + auto-decompose call the aux LLM synchronously
            # (timeouts 120 s / 180 s). Inline in the dispatch tick they ran
            # BEFORE dispatch_once, so one slow aux answer delayed claim/spawn
            # on every board by up to ~900 s (audit 2026-07-27). As a sibling
            # task on its own cadence they overlap the dispatch tick instead
            # of preceding it; the tick itself stays pure claim/spawn work.
            # Lives strictly inside the dispatcher's singleton-lock scope, so
            # concurrent gateways cannot double-decompose. Toggles are
            # re-read every round (#49638) exactly as before.
            while self._running and not maintenance_stop.is_set():
                try:
                    _pm_enabled, _pm_per_tick, _pm_max_attempts = _read_pm_supervisor_settings()
                    if _pm_enabled and not maintenance_stop.is_set():
                        await _run_maintenance(
                            _pm_supervisor_tick_once, _pm_per_tick, _pm_max_attempts,
                        )
                    _ad_enabled, _ad_per_tick = _read_auto_decompose_settings()
                    if _ad_enabled and not maintenance_stop.is_set():
                        await _run_maintenance(_auto_decompose_tick, _ad_per_tick)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "kanban maintenance: pm-supervisor/auto-decompose round failed"
                    )
                slept = 0.0
                while (
                    slept < maintenance_interval
                    and self._running
                    and not maintenance_stop.is_set()
                ):
                    await asyncio.sleep(min(1.0, maintenance_interval - slept))
                    slept += 1.0

        maintenance_task = asyncio.create_task(_maintenance_loop())

        logger.info(
            "kanban dispatcher: embedded in gateway (interval=%.1fs, "
            "maintenance=%.1fs)", interval, maintenance_interval,
        )
        while self._running:
            try:
                # Reap zombie children before per-board work so a board DB
                # failure cannot block cleanup of unrelated workers.
                pids = await asyncio.to_thread(_kb.reap_worker_zombies)
                if pids:
                    logger.info(
                        "kanban dispatcher: reaped %d zombie worker(s), pids=%s",
                        len(pids),
                        pids,
                    )
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            # Shadow is report-only: per-board read-only diagnostics run before
            # the existing Reap → Auto-Decompose → Dispatch execution block.
            # Its decision is intentionally not consulted below.
            try:
                shadow_cfg = _resolve_backpressure_shadow_config(_load_config)
                if shadow_cfg.shadow_enabled:
                    boards = _kb.list_boards(include_archived=False)
                    duplicate_outcomes = getattr(
                        self, "_kanban_shadow_duplicate_suppressed_last_tick", {}
                    )
                    if not isinstance(duplicate_outcomes, Mapping):
                        duplicate_outcomes = {}
                    now = int(time.time())
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        try:
                            board_slug = _kb._normalize_board_slug(slug) or _kb.DEFAULT_BOARD
                        except (TypeError, ValueError):
                            continue
                        path = board_meta.get("db_path") or _kb.kanban_db_path(slug)
                        metrics = await asyncio.to_thread(_collect_attention_storm_metrics_readonly, path, now)
                        if metrics is None:
                            continue
                        metrics = AttentionStormMetrics(
                            metrics.active_human_required, metrics.new_attention_5m,
                            metrics.new_attention_15m, metrics.blocked_run_ratio_15m,
                            metrics.recurrent_blocks, metrics.current_delivery_identities_materialized,
                            int(duplicate_outcomes.get(board_slug, 0) or 0),
                        )
                        decision = evaluate_backpressure_shadow(metrics, shadow_cfg.__dict__)
                        if _should_emit_shadow_warning(
                            shadow_last_warning, board_slug, decision, now, shadow_cfg.warning_cooldown_seconds,
                        ):
                            _log_shadow_warning(decision)
            except Exception:
                # Shadow diagnostics never impact the execution loop and never
                # surface board/path/error text through the safe warning channel.
                pass

            try:
                # pm-supervisor and auto-decompose run in _maintenance_loop
                # (own cadence, sibling task above) — NOT here, so their aux-LLM
                # calls can never delay claim/spawn on this tick.
                results = await asyncio.to_thread(_tick_once)
                any_spawned = False
                for slug, res in (results or []):
                    if res is None:
                        continue
                    if getattr(res, "spawned", None):
                        any_spawned = True
                    # Quiet by default — only log when something actually
                    # happened, so an idle gateway stays silent. Audit
                    # 2026-07-11 H5: "something" includes crash/timeout/
                    # reclaim/auto-block ticks, not just spawns — a crash
                    # wave without a simultaneous spawn was invisible (also
                    # to the mission-control log-tail escalation).
                    _spawned_n = len(getattr(res, "spawned", None) or [])
                    _crashed = len(res.crashed) if hasattr(getattr(res, "crashed", None), "__len__") else 0
                    _timed_out = len(res.timed_out) if hasattr(getattr(res, "timed_out", None), "__len__") else 0
                    _auto_blocked = len(res.auto_blocked) if hasattr(getattr(res, "auto_blocked", None), "__len__") else 0
                    _reclaimed = getattr(res, "reclaimed", 0) or 0
                    if _spawned_n or _crashed or _timed_out or _auto_blocked or _reclaimed:
                        logger.info(
                            "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
                            "crashed=%d timed_out=%d promoted=%d auto_blocked=%d",
                            slug,
                            _spawned_n,
                            _reclaimed,
                            _crashed,
                            _timed_out,
                            res.promoted,
                            _auto_blocked,
                        )
                # Health telemetry (aggregate across boards)
                ready_pending = await asyncio.to_thread(_ready_nonempty)
                if ready_pending and not any_spawned:
                    bad_ticks += 1
                else:
                    bad_ticks = 0
                if bad_ticks >= HEALTH_WINDOW:
                    now = int(time.time())
                    if now - last_warn_at >= 300:
                        # Distinguish "spawn machinery broken" from "dispatcher
                        # deliberately deferring via respawn guards" — the old
                        # blanket "check profile health" message sent operators
                        # chasing venv/credential ghosts when every skipped
                        # spawn was actually an explicit guard decision.
                        guard_counts: dict[str, int] = {}
                        for _slug, _res in (results or []):
                            for _tid, _greason in (
                                getattr(_res, "respawn_guarded", None) or []
                            ):
                                guard_counts[_greason] = (
                                    guard_counts.get(_greason, 0) + 1
                                )
                        if guard_counts:
                            logger.warning(
                                "kanban dispatcher idle-by-guard: ready queue "
                                "non-empty for %d consecutive ticks, 0 workers "
                                "spawned — deferred by respawn guards (%s). "
                                "Inspect the tasks' 'respawn_guarded' events; "
                                "profile health is NOT implicated.",
                                bad_ticks,
                                ", ".join(
                                    f"{r}×{n}"
                                    for r, n in sorted(guard_counts.items())
                                ),
                            )
                        else:
                            logger.warning(
                                "kanban dispatcher stuck: ready queue non-empty "
                                "for %d consecutive ticks but 0 workers spawned. "
                                "Check profile health (venv, PATH, credentials) "
                                "and `hermes kanban list --status ready`.",
                                bad_ticks,
                            )
                        last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                maintenance_task.cancel()
                # Cancellation path: the loop is going away now, so drain
                # synchronously with a short budget rather than leaking the
                # lock to a still-writing maintenance thread.
                _finish_dispatcher(10.0)
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
            # waits up to `interval` seconds for the current sleep to finish.
            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

        maintenance_task.cancel()
        # Normal stop: give an in-flight aux-LLM round room to finish (its own
        # timeouts are 120/180 s) before the lock changes hands.
        await asyncio.to_thread(_finish_dispatcher, 45.0)

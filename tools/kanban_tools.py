"""Kanban tools — structured tool-call surface for worker + orchestrator agents.

These tools are registered into the model's schema when the agent is
running under the dispatcher (env var ``HERMES_KANBAN_TASK`` set) or when
the active profile explicitly enables the ``kanban`` toolset for
orchestrator work. A normal ``hermes chat`` session still sees **zero**
kanban tools in its schema unless configured.

Why tools instead of just shelling out to ``hermes kanban``?

1. **Backend portability.** A worker whose terminal tool points at Docker
   / Modal / Singularity / SSH would run ``hermes kanban complete …``
   inside the container, where ``hermes`` isn't installed and the DB
   isn't mounted. Tools run in the agent's Python process, so they
   always reach ``~/.hermes/kanban.db`` regardless of terminal backend.

2. **No shell-quoting footguns.** Passing ``--metadata '{"x": [...]}'``
   through shlex+argparse is fragile. Structured tool args skip it.

3. **Better errors.** Tool-call failures return structured JSON the
   model can reason about, not stderr strings it has to parse.

Humans continue to use the CLI (``hermes kanban …``), the dashboard
(``hermes dashboard``), and the slash command (``/kanban …``) — all
three bypass the agent entirely. The tools are for dispatcher-spawned
worker handoffs and for configured orchestrator profiles that route work
through the board.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any, Optional

from agent.redact import redact_sensitive_text
from hermes_cli.goals import judge_goal
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delegated-subagent scoping (t_591dd454)
# ---------------------------------------------------------------------------
# A ``delegate_task`` subagent runs as a fresh AIAgent inside the SAME OS
# process as its parent (a thread from a dedicated per-child executor, see
# ``tools.delegate_tool._run_single_child``), not a separate process. Because
# ``os.environ`` is one process-wide dict, a subagent spawned from inside a
# dispatcher-spawned board worker previously saw the exact same
# ``HERMES_KANBAN_TASK`` / ``_RUN_ID`` / ``_CLAIM_LOCK`` the worker did, so
# ``_enforce_worker_task_ownership`` treated the subagent as if it WERE that
# worker: it could call ``kanban_complete`` / ``kanban_block`` /
# ``kanban_heartbeat`` on the parent's own task and the ownership check
# passed. On 2026-07-10 this let a read-only research subagent mark card S3
# complete while the real worker was still running and writing uncommitted
# changes to the shared worktree, which let the dispatcher promote S4 early
# with a second writer active.
#
# Fix: ``tools.delegate_tool`` marks the child's dedicated run-thread via
# ``agent.delegation_context.delegated_child_context()`` (upstream) *before*
# the child's conversation
# starts. Tool dispatch for that conversation — including tool calls the
# agent loop offloads onto further worker threads via
# ``tools.thread_context.propagate_context_to_thread`` — runs inside a
# ``contextvars.Context`` descended from that thread, so the marker is a
# ``contextvars.ContextVar`` (not ``threading.local``): it follows the
# subagent's calls into those nested threads without mutating the real
# process-wide ``os.environ`` and without affecting the parent thread or any
# concurrent sibling subagent (each gets its own dedicated thread).
#
# We deliberately do NOT touch ``_check_kanban_mode`` /
# ``_check_kanban_orchestrator_mode`` / ``_require_orchestrator_tool`` here:
# those still read the real (unstripped) ``HERMES_KANBAN_TASK`` and correctly
# keep classifying a delegated subagent as "worker-scoped" for the purpose of
# hiding orchestrator-only tools (``kanban_list`` / ``kanban_unblock``) from
# it. Stripping the env there too would flip that classification to
# "orchestrator" and grant the subagent MORE board access, not less.
BOARD_OWNERSHIP_ENV_KEYS = frozenset(
    {
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_BRANCH",
    }
)

# fork(tars) 30.07.2026: Unsere eigene Delegations-Schutzschicht ist hier
# ERSATZLOS ENTFALLEN — upstream hat sie mit agent/delegation_context.py und
# _reject_delegated_child_mutation() (weiter unten) selbst gebaut, semantisch
# deckungsgleich und breiter (Contextmanager mit Reset + Env-Scrubbing für
# Subprozesse). Geblieben ist nur _board_env: dafür gibt es kein Äquivalent.


def _board_env(name: str) -> Optional[str]:
    """``os.environ.get`` for board-ownership keys, honoring the subagent strip.

    Only use this for the keys in ``BOARD_OWNERSHIP_ENV_KEYS``. Other env
    reads (``HERMES_SESSION_ID``, ``HERMES_PROFILE``, ...) are unaffected and
    should keep using ``os.environ.get`` directly.
    """
    if _is_delegated_child_context() and name in BOARD_OWNERSHIP_ENV_KEYS:
        return None
    return os.environ.get(name)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

KANBAN_LIST_DEFAULT_LIMIT = 50
KANBAN_LIST_MAX_LIMIT = 200


def _profile_has_kanban_toolset() -> bool:
    # Uses load_config() which has mtime-based caching, so this adds
    # negligible overhead. The check_fn results are further TTL-cached
    # (~30s) by the tool registry.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        toolsets = cfg.get("toolsets", [])
        return "kanban" in toolsets
    except Exception:
        return False


def _is_delegated_child_context() -> bool:
    # Process-variant, not the ContextVar-only one: a delegate_task child that
    # spawns a subprocess loses the ContextVar across the fork but carries the
    # HERMES_DELEGATED_CHILD_CONTEXT env marker (set by the scrub helpers). The
    # ContextVar-only check returned False there, handing the subprocess the
    # full orchestrator surface (kanban_unblock/kanban_complete on any card).
    try:
        from agent.delegation_context import is_delegated_child_process_context

        return is_delegated_child_process_context()
    except Exception:
        return False


def _reject_delegated_child_mutation(tool_name: str) -> Optional[str]:
    """Deny Kanban mutations from delegate_task children.

    A delegate_task child runs in the same process as its parent, so stale or
    inherited HERMES_KANBAN_* env vars are not proof of dispatcher ownership.
    The child may summarize findings to its parent, but it must not complete,
    block, heartbeat, comment, create, link, or unblock board tasks directly.
    """
    if not _is_delegated_child_context():
        return None
    return tool_error(
        f"{tool_name} refused: delegate_task child agents are not Kanban "
        "run owners. Return findings to the parent agent; the dispatcher "
        "worker or an explicitly configured Kanban orchestrator must perform "
        "board mutations."
    )


def _check_kanban_mode() -> bool:
    """Task-lifecycle tools are available when:

    1. ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), OR
    2. The current profile has ``kanban`` in its toolsets config
       (orchestrator profiles like techlead that route work via Kanban).

    Humans running ``hermes chat`` without the kanban toolset see zero
    kanban tools. Workers spawned by the kanban dispatcher (gateway-
    embedded by default) and orchestrator profiles with the kanban
    toolset enabled see the Kanban lifecycle tool surface.
    """
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return True
    return _profile_has_kanban_toolset()


def _check_kanban_orchestrator_mode() -> bool:
    """Board-routing tools (kanban_list, kanban_unblock) are intentionally
    hidden from task workers.

    Dispatcher-spawned workers should close their own task via the
    lifecycle tools (complete/block/heartbeat), not enumerate or unblock
    board state. Profiles that explicitly opt into the kanban toolset
    and are NOT scoped to a single task are the orchestrator surface.
    """
    if _is_delegated_child_context():
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    return _profile_has_kanban_toolset()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _default_task_id(arg: Optional[str]) -> Optional[str]:
    """Resolve ``task_id`` arg or fall back to the env var the dispatcher set.

    Uses ``_board_env`` so a delegated subagent (see
    ``delegated_child_context``) never silently inherits the
    parent board worker's task id as a default.
    """
    if arg:
        return arg
    if _is_delegated_child_context():
        return None
    env_tid = _board_env("HERMES_KANBAN_TASK")
    return env_tid or None


def _completion_refusal_message(
    kb, conn, task_id: str, *, expected_run_id: Optional[int]
) -> str:
    """Explain a bare-``False`` refusal from ``complete_task``.

    The kernel refuses completion silently on several DISTINCT guards; the
    old catch-all message ("unknown id or already terminal") sent workers
    chasing the wrong cause. Live case (K-3b, Vollaudit 2026-07-16,
    t_5a43abe1): an unmatched review_requested — the reviewer crash-looped
    and never decided — blocked completion forever while the readback showed
    a healthy running card; the worker block-looped for hours. Diagnosis
    mirrors the guard ORDER in ``_complete_task_locked``; purely
    read-only, best effort (falls back to a generic line)."""
    try:
        task = kb.get_task(conn, task_id)
        if task is None:
            return f"could not complete {task_id}: unknown task id"
        if task.status not in {"running", "ready", "blocked"}:
            return (
                f"could not complete {task_id}: already terminal "
                f"(status={task.status}); no further completion is needed"
            )
        if kb._pending_review_request(conn, task_id) is not None:
            return (
                f"could not complete {task_id}: an open review handshake "
                "exists (review_requested without a review_decided). A card "
                "with a pending first-class review can only reach done via "
                "an explicit reviewer ACCEPT. Do NOT retry kanban_complete — "
                "either hand back to review (kanban_request_review) so the "
                "reviewer can decide, or ask the operator to resolve the "
                "stale handshake."
            )
        if (
            expected_run_id is not None
            and task.current_run_id != int(expected_run_id)
        ):
            return (
                f"could not complete {task_id}: run identity mismatch — this "
                f"worker owns run {expected_run_id}, the card's current run "
                f"is {task.current_run_id}. Another run superseded yours; "
                "stop and let the current owner finish."
            )
        if conn.execute(
            "SELECT 1 FROM task_pending_actions WHERE task_id = ? "
            "AND state = 'pending' LIMIT 1",
            (task_id,),
        ).fetchone() is not None:
            return (
                f"could not complete {task_id}: a pending exact action awaits "
                "operator approval; completion is only possible through the "
                "approval path."
            )
    except Exception:
        logger.warning(
            "completion refusal diagnosis failed for %s", task_id,
            exc_info=True,
        )
    return (
        f"could not complete {task_id}: refused by a completion guard "
        "(see the card's board events for the authoritative state)"
    )


def _worker_run_id(task_id: str) -> Optional[int]:
    """Return this worker's dispatcher run id when it is scoped to task_id."""
    if _board_env("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = _board_env("HERMES_KANBAN_RUN_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _stamp_worker_session_metadata(
    task_id: str, metadata: Optional[dict]
) -> Optional[dict]:
    """Add trusted worker session id metadata for this worker's own task."""
    if _board_env("HERMES_KANBAN_TASK") != task_id:
        return metadata
    session_id = os.environ.get("HERMES_SESSION_ID")
    if not session_id:
        return metadata
    stamped = dict(metadata or {})
    stamped["worker_session_id"] = session_id
    return stamped


def _enforce_worker_task_ownership(tid: str) -> Optional[str]:
    """Reject worker-driven destructive calls on foreign task IDs.

    A process spawned by the dispatcher has ``HERMES_KANBAN_TASK`` set
    to its own task id. Tools like ``kanban_complete`` / ``kanban_block``
    / ``kanban_heartbeat`` mutate run-lifecycle state, so a buggy or
    prompt-injected worker that passed an explicit ``task_id`` for some
    other task could corrupt sibling or cross-tenant runs (see #19534).

    Orchestrator profiles (kanban toolset enabled but **no**
    ``HERMES_KANBAN_TASK`` in env) aren't subject to this check — their
    job is routing, and they sometimes legitimately close out child
    tasks or reopen blocked ones. Workers are narrowly scoped to their
    one task.

    Note: every current caller of this function is a mutating handler
    that calls ``_reject_delegated_child_mutation`` first (see below), so a
    delegated subagent never reaches here at all — the ``not env_tid``
    branch below would otherwise (wrongly) treat a stripped env as
    "orchestrator, no restriction". This function's own env read is
    stripped anyway for defense in depth, but the real guarantee is the
    upfront reject in each handler.

    Returns ``None`` when the call is allowed, or a tool-error string
    when it must be rejected. Callers should ``return`` the error
    verbatim.
    """
    env_tid = _board_env("HERMES_KANBAN_TASK")
    if not env_tid:
        # Orchestrator or CLI context — no task-scope restriction.
        return None
    if tid != env_tid:
        return tool_error(
            f"worker is scoped to task {env_tid}; refusing to mutate "
            f"{tid}. Use kanban_comment to hand off information to other "
            f"tasks, or kanban_create to spawn follow-up work."
        )
    return None


def _connect(board: Optional[str] = None):
    """Import + connect lazily so the module imports cleanly in non-kanban
    contexts (e.g. test rigs that import every tool module).

    When ``board`` is provided it's forwarded to :func:`kb.connect`, which
    routes the connection to that board's sqlite file. ``None`` (the
    default) preserves the legacy resolution chain
    (``HERMES_KANBAN_DB`` → ``HERMES_KANBAN_BOARD`` env → current symlink
    → ``default``). Per-tool ``board`` lets a Telegram-side agent override
    the env-pinned active board without restarting Hermes.
    """
    from hermes_cli import kanban_db as kb
    return kb, kb.connect(board=board)


# Canonical definition lives in the kernel (audit 2026-07-11, Meta-Muster E:
# duplicated invariants drift). Imported lazily in _handle_block via the
# connected ``kb`` module: kb.GOAL_MODE_BLOCK_ALLOWED_KINDS.


def _goal_judge_available() -> bool:
    """True when an auxiliary client is configured for the goal judge.

    ``judge_goal`` is fail-open at the source: when no auxiliary model can
    be reached it returns a ``"continue"`` verdict that is indistinguishable
    from a real "not done yet" judgment. The completion gate must not treat
    that as a rejection, or an unconfigured/degraded auxiliary model would
    wedge every ``goal_mode`` worker (it could never close its own task).

    So we probe availability first and only enforce the gate when a judge is
    actually reachable. This mirrors the same client lookup ``judge_goal``
    performs internally.
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


# ---------------------------------------------------------------------------
# Runtime-activity → board-heartbeat bridge (#31752)
# ---------------------------------------------------------------------------
# When the agent ticks ``_touch_activity`` during normal work (between
# tool calls, mid-stream chunks, etc.), we want the kanban board's
# ``last_heartbeat_at`` columns to reflect that liveness so the dispatcher
# watchdog (which reads ``tasks.last_heartbeat_at``, not the agent's
# in-process timestamp) doesn't reclaim an actively-running worker as
# stale. The model is not required to call the explicit ``kanban_heartbeat``
# tool for this to work — that tool stays available for workers that want
# to attach a note or pre-emptively extend a claim across a known-long op.
#
# Constraints:
#   - Best-effort: never raise. The agent loop must not care if the bridge
#     fails (board missing, DB locked, etc.).
#   - Rate-limited to one DB write per 60s per-process; runtime activity
#     can tick on every chunk/tool result and we don't need that resolution.
#   - No-op outside dispatcher-spawned worker context (no ``HERMES_KANBAN_TASK``).
#   - No durable note on these auto-heartbeats; that's reserved for the
#     explicit tool which carries a model-supplied note.

_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS = 60.0
_auto_heartbeat_last_attempt: float = 0.0


def heartbeat_current_worker_from_env() -> bool:
    """Best-effort: extend the kanban claim + bump board heartbeat for the
    current dispatcher-spawned worker, using identity from env vars.

    Returns True if a write was attempted (whether or not it succeeded);
    False if the call was skipped (not a kanban worker, rate-limited, or
    swallowed exception). The boolean is informational — callers should
    not branch on it.

    Identity comes from:
      * ``HERMES_KANBAN_TASK`` — task id (required; absence means no-op)
      * ``HERMES_KANBAN_RUN_ID`` — pins the run row so we don't heartbeat
        a stale run that may have already been reclaimed
      * ``HERMES_KANBAN_CLAIM_LOCK`` — claim lock for ``heartbeat_claim``;
        falls back to the default ``_claimer_id()`` for locally-driven
        workers that never went through the dispatcher path

    Rate-limited via the module-level ``_auto_heartbeat_last_attempt``
    timestamp (monotonic clock); not thread-safe in the strict sense, but
    the worst case is one extra DB write per race, which is harmless.

    No-op inside a delegated subagent context: ``AIAgent._touch_activity``
    calls this unconditionally on every activity tick whenever
    ``HERMES_KANBAN_TASK`` is set in the process env, so without this guard
    a delegated subagent would silently extend the PARENT worker's claim
    just by being busy on its own unrelated goal — no explicit tool call
    required (see t_591dd454).
    """
    global _auto_heartbeat_last_attempt
    if _is_delegated_child_context():
        return False
    tid = os.environ.get("HERMES_KANBAN_TASK")
    if not tid:
        return False
    import time as _time
    now = _time.monotonic()
    if (now - _auto_heartbeat_last_attempt) < _AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS:
        return False
    _auto_heartbeat_last_attempt = now
    try:
        kb, conn = _connect()
        try:
            claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
            try:
                kb.heartbeat_claim(conn, tid, claimer=claim_lock)
            except Exception:
                logger.debug("auto-heartbeat: heartbeat_claim failed", exc_info=True)
            run_id_raw = os.environ.get("HERMES_KANBAN_RUN_ID")
            run_id: Optional[int]
            try:
                run_id = int(run_id_raw) if run_id_raw else None
            except (TypeError, ValueError):
                run_id = None
            try:
                kb.heartbeat_worker(
                    conn, tid, note=None, expected_run_id=run_id, semantic=False
                )
            except Exception:
                logger.debug("auto-heartbeat: heartbeat_worker failed", exc_info=True)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True
    except Exception:
        logger.debug("auto-heartbeat: bridge failed", exc_info=True)
        return False


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields})


def _normalize_profile(value: Any) -> Optional[str]:
    """Normalize CLI-compatible assignee sentinels for the tool surface."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "-", "null"}:
        return None
    return text


def _parse_bool_arg(args: dict, name: str, *, default: bool = False):
    value = args.get(name)
    if value is None:
        return default, None
    if isinstance(value, bool):
        return value, None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True, None
    if text in {"false", "0", "no"}:
        return False, None
    return default, f"{name} must be a boolean or 'true'/'false'"


def _require_orchestrator_tool(tool_name: str) -> Optional[str]:
    """Belt-and-suspenders runtime guard for orchestrator-only handlers.

    The check_fn (`_check_kanban_orchestrator_mode`) keeps these tools
    out of the worker schema entirely, but in case a stale registration
    or test harness routes a worker to one of them anyway, return a
    structured tool_error so the model gets a clear refusal instead of
    silently mutating board state from a worker context.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return tool_error(
            f"{tool_name} is orchestrator-only; dispatcher-spawned workers "
            "must use kanban_complete, kanban_block, kanban_heartbeat, or "
            "kanban_comment for their assigned task."
        )
    return None


def _task_summary_dict(kb, conn, task) -> dict[str, Any]:
    """Compact task shape for board-listing tools.

    Carries the outcome-bearing fields — trimmed ``result``, ``block_kind``
    and the last review decision — so an orchestrator can read the state of
    N cards from ONE listing. Without them the list said only "done"/"blocked"
    and every card needed a full ``kanban_show`` (26–36k chars each) just to
    learn what happened; that per-card round-trip is how orchestrators lost
    the thread mid-mission (incident + audit 2026-07-27).
    """
    parents = kb.parent_ids(conn, task.id)
    children = kb.child_ids(conn, task.id)
    summary = {
        "id": task.id,
        "title": task.title,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "project_id": task.project_id,
        "created_by": task.created_by,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "current_run_id": task.current_run_id,
        "model_override": task.model_override,
        "provider_override": task.provider_override,
        "parents": parents,
        "children": children,
        "parent_count": len(parents),
        "child_count": len(children),
    }
    result = (getattr(task, "result", None) or "").strip()
    if result:
        summary["result"] = result.splitlines()[0][:200]
    if getattr(task, "block_kind", None):
        summary["block_kind"] = task.block_kind
    try:
        row = conn.execute(
            "SELECT json_extract(payload, '$.decision') FROM task_events "
            "WHERE task_id = ? AND kind = 'review_decided' "
            "ORDER BY id DESC LIMIT 1",
            (task.id,),
        ).fetchone()
        if row is not None and row[0]:
            summary["last_review_decision"] = row[0]
    except Exception:
        pass
    return summary


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

_SHOW_SECTIONS = ("comments", "events", "runs", "worker_context")
# Compact-mode caps. The old unconditional dump (full body, ALL comments, 50
# events with full payloads, ALL runs with metadata, plus worker_context) was
# 26-36k chars per call; over the persistence threshold the generic result
# store cut it into an unparseable fragment mid-JSON. The default answer must
# fit an orchestrator's working set; anything deeper is one include= away.
_SHOW_COMPACT_EVENTS = 5
_SHOW_COMPACT_COMMENTS = 5
_SHOW_COMPACT_RUNS = 1
_SHOW_COMPACT_TEXT = 500


def _handle_show(args: dict, **kw) -> str:
    """Read a task's state: compact by default, deep sections on request."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    include_raw = args.get("include")
    compact = include_raw is None
    if compact:
        include = set(_SHOW_SECTIONS)
    else:
        if not isinstance(include_raw, (list, tuple)):
            return tool_error("include must be an array of section names")
        include = {str(s).strip().lower() for s in include_raw}
        if "all" in include:
            include = set(_SHOW_SECTIONS)
            compact = False
        else:
            unknown = include - set(_SHOW_SECTIONS)
            if unknown:
                return tool_error(
                    f"unknown include section(s) {sorted(unknown)}; "
                    f"valid: {list(_SHOW_SECTIONS)} or ['all']"
                )
    board = args.get("board")

    def _trim(text, limit=_SHOW_COMPACT_TEXT):
        if not compact or text is None:
            return text
        s = str(text)
        return s if len(s) <= limit else s[:limit] + "…"

    try:
        kb, conn = _connect(board=board)
        try:
            task = kb.get_task(conn, tid)
            if task is None:
                return tool_error(f"task {tid} not found")
            parents = kb.parent_ids(conn, tid)
            children = kb.child_ids(conn, tid)

            def _task_dict(t):
                return {
                    "id": t.id, "title": t.title, "body": _trim(t.body),
                    "assignee": t.assignee, "status": t.status,
                    "tenant": t.tenant, "priority": t.priority,
                    "workspace_kind": t.workspace_kind,
                    "workspace_path": t.workspace_path,
                    "created_by": t.created_by, "created_at": t.created_at,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                    "result": _trim(t.result),
                    "current_run_id": t.current_run_id,
                    "model_override": t.model_override,
                    "provider_override": t.provider_override,
                }

            def _run_dict(r):
                return {
                    "id": r.id, "profile": r.profile,
                    "status": r.status, "outcome": r.outcome,
                    "summary": _trim(r.summary), "error": _trim(r.error),
                    "metadata": None if compact else r.metadata,
                    "started_at": r.started_at, "ended_at": r.ended_at,
                }

            payload: dict[str, Any] = {
                "task": _task_dict(task),
                "parents": parents,
                "children": children,
            }
            if "comments" in include:
                comments = kb.list_comments(conn, tid)
                total = len(comments)
                if compact:
                    comments = comments[-_SHOW_COMPACT_COMMENTS:]
                payload["comments"] = [
                    {"author": c.author, "body": _trim(c.body),
                     "created_at": c.created_at}
                    for c in comments
                ]
                payload["comments_total"] = total
            if "events" in include:
                events = kb.list_events(conn, tid)
                total = len(events)
                # include= means "in full": no cap. Only the compact default
                # trims (the schema promised uncapped sections; TARS review).
                cap = _SHOW_COMPACT_EVENTS if compact else total
                payload["events"] = [
                    {"kind": e.kind, "payload": e.payload,
                     "created_at": e.created_at, "run_id": e.run_id}
                    for e in (events[-cap:] if cap else [])
                ]
                payload["events_total"] = total
            if "runs" in include:
                runs = kb.list_runs(conn, tid)
                total = len(runs)
                if compact:
                    runs = runs[-_SHOW_COMPACT_RUNS:]
                payload["runs"] = [_run_dict(r) for r in runs]
                payload["runs_total"] = total
            if "worker_context" in include and not compact:
                # The pre-formatted spawn-context block is large and mostly
                # duplicates the sections above; it ships only on request.
                payload["worker_context"] = kb.build_worker_context(conn, tid)
            if compact:
                payload["compact"] = True
                payload["hint"] = (
                    "Trimmed view (last "
                    f"{_SHOW_COMPACT_EVENTS} events/"
                    f"{_SHOW_COMPACT_COMMENTS} comments/"
                    f"{_SHOW_COMPACT_RUNS} run). Pass include=['all'] or a "
                    "subset of "
                    f"{list(_SHOW_SECTIONS)} for full sections."
                )
            return json.dumps(payload, indent=1)
        finally:
            conn.close()
    except ValueError as e:
        # Invalid board slug surfaces as ValueError from _normalize_board_slug.
        return tool_error(f"kanban_show: {e}")
    except Exception as e:
        logger.exception("kanban_show failed")
        return tool_error(f"kanban_show: {e}")


def _handle_list(args: dict, **kw) -> str:
    """List task summaries with the same core filters as the CLI."""
    guard = _require_orchestrator_tool("kanban_list")
    if guard:
        return guard
    assignee = args.get("assignee")
    status = args.get("status")
    tenant = args.get("tenant")
    include_archived, bool_error = _parse_bool_arg(args, "include_archived")
    if bool_error:
        return tool_error(bool_error)
    limit = args.get("limit")
    if limit is None:
        limit = KANBAN_LIST_DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return tool_error("limit must be an integer")
    if limit < 1:
        return tool_error("limit must be >= 1")
    if limit > KANBAN_LIST_MAX_LIMIT:
        return tool_error(f"limit must be <= {KANBAN_LIST_MAX_LIMIT}")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # Match CLI list: dependencies that cleared since the last
            # dispatcher tick should be visible to orchestrators immediately.
            promoted = kb.recompute_ready(conn)
            # Fetch one extra row so model-facing output can report that
            # a bounded listing was truncated without dumping the board.
            rows = kb.list_tasks(
                conn,
                assignee=assignee,
                status=status,
                tenant=tenant,
                include_archived=include_archived,
                limit=limit + 1,
            )
            truncated = len(rows) > limit
            tasks = rows[:limit]
            # indent=1: if the generic result store ever has to preview this,
            # it cuts at a newline boundary (an object edge) instead of raw-
            # slicing one endless JSON line mid-field (incident 2026-07-27).
            return json.dumps({
                "tasks": [_task_summary_dict(kb, conn, t) for t in tasks],
                "count": len(tasks),
                "limit": limit,
                "truncated": truncated,
                "next_limit": (
                    min(limit * 2, KANBAN_LIST_MAX_LIMIT)
                    if truncated and limit < KANBAN_LIST_MAX_LIMIT else None
                ),
                "promoted": promoted,
            }, indent=1)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_list: {e}")
    except Exception as e:
        logger.exception("kanban_list failed")
        return tool_error(f"kanban_list: {e}")


def _handle_complete(args: dict, **kw) -> str:
    """Mark the current task done with a structured handoff."""
    delegated_err = _reject_delegated_child_mutation("kanban_complete")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    # needs_input is a human-decision gate. complete_task() itself accepts
    # blocked tasks (a deliberate operator affordance via the CLI), but an
    # AGENT completing a needs_input-blocked card dissolves the gate without
    # any human in the loop — exactly what happened live on 2026-07-10
    # (t_614c91e9: auto-blocked 22:02, worker-completed 22:04, no unblock).
    # Agents must instead comment their evidence and leave the decision to
    # the operator, who unblocks (with reason) or completes via CLI.
    try:
        from hermes_cli import kanban_db as _kb_gate
        # Open the SAME board the completion targets, else get_task reads the
        # default board's DB, returns None for a named-board card, and the gate
        # silently does not fire (D4).
        with _kb_gate.connect_closing(board=args.get("board")) as _conn_gate:
            _task_gate = _kb_gate.get_task(_conn_gate, tid)
        if (
            _task_gate is not None
            and _task_gate.status == "blocked"
            and (_task_gate.block_kind or "") == "needs_input"
        ):
            return tool_error(
                f"{tid} is blocked with kind=needs_input — a human decision "
                "gate. kanban_complete is refused here: post your evidence "
                "as a comment and leave the card for the operator (who can "
                "unblock with a reason or complete it via the CLI)."
            )
    except Exception:
        # Fail open on gate-lookup errors: refusing ALL completions on a
        # transient DB hiccup would strand healthy workers; the gate is a
        # safety net, not the primary lifecycle path.
        pass
    summary = args.get("summary")
    metadata = args.get("metadata")
    result = args.get("result")
    if summary:
        summary = redact_sensitive_text(str(summary), force=True)
    if result:
        result = redact_sensitive_text(str(result), force=True)
    if metadata is not None and isinstance(metadata, dict):
        meta_json = json.dumps(metadata)
        meta_json = redact_sensitive_text(meta_json, force=True)
        try:
            metadata = json.loads(meta_json)
        except json.JSONDecodeError:
            pass
    created_cards = args.get("created_cards")
    artifacts = args.get("artifacts")
    if created_cards is not None:
        if isinstance(created_cards, str):
            # Accept a single id as a string for convenience.
            created_cards = [created_cards]
        if not isinstance(created_cards, (list, tuple)):
            return tool_error(
                f"created_cards must be a list of task ids, got "
                f"{type(created_cards).__name__}"
            )
        # Normalise: strings only, stripped, non-empty.
        created_cards = [
            str(c).strip() for c in created_cards if str(c).strip()
        ]
    if artifacts is not None:
        if isinstance(artifacts, str):
            # Accept a single path as a string for convenience.
            artifacts = [artifacts]
        if not isinstance(artifacts, (list, tuple)):
            return tool_error(
                f"artifacts must be a list of file paths, got "
                f"{type(artifacts).__name__}"
            )
        artifacts = [
            str(p).strip() for p in artifacts if str(p).strip()
        ]
        # Carry the artifact list inside metadata so it rides the
        # existing completed-event payload without a schema change at
        # the DB layer.  The gateway notifier reads payload['artifacts']
        # off the completion event and uploads each path as a native
        # attachment.
        if artifacts:
            if metadata is None:
                metadata = {}
            elif not isinstance(metadata, dict):
                return tool_error(
                    f"metadata must be an object/dict, got "
                    f"{type(metadata).__name__}"
                )
            # Don't overwrite an existing metadata.artifacts the worker
            # passed manually — merge instead.
            existing = metadata.get("artifacts")
            if isinstance(existing, (list, tuple)):
                merged: list[str] = []
                seen: set[str] = set()
                for item in list(existing) + artifacts:
                    s = str(item).strip()
                    if s and s not in seen:
                        seen.add(s)
                        merged.append(s)
                metadata["artifacts"] = merged
            else:
                metadata["artifacts"] = artifacts
    if not (summary or result):
        return tool_error(
            "provide at least one of: summary (preferred), result"
        )
    if metadata is not None and not isinstance(metadata, dict):
        return tool_error(
            f"metadata must be an object/dict, got {type(metadata).__name__}"
        )
    metadata = _stamp_worker_session_metadata(tid, metadata)
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # Goal-mode pre-completion judge gate (Issue #38367).
            # Prevent workers from bypassing the auxiliary judge by
            # calling kanban_complete before acceptance criteria are met.
            # Only enforce when a judge is actually reachable — see
            # _goal_judge_available for why an unavailable judge fails open.
            # Skipped when the kernel worker gates cover this process
            # (audit 2026-07-11 C1): complete_task runs the same judge for
            # worker-scoped processes, and judging twice would double the
            # LLM cost per completion. This tool-side check remains for
            # agent surfaces WITHOUT worker scope (orchestrator profiles),
            # which the kernel deliberately does not gate.
            _kernel_gated = kb._worker_scope()[0] is not None
            task = kb.get_task(conn, tid)
            if (
                not _kernel_gated
                and task and task.goal_mode and _goal_judge_available()
            ):
                verdict = "done"
                reason = ""
                try:
                    # judge_goal returns (verdict, reason, parse_failed,
                    # wait_directive, transport_failed) — see
                    # hermes_cli/goals.py. Unpacking fewer raises ValueError,
                    # which the defensive handler below swallows, leaving
                    # verdict="done" and silently disabling the gate.
                    verdict, reason, _, _, _ = judge_goal(
                        goal=f"{task.title}\n\n{task.body or ''}".strip(),
                        last_response=(summary or result or "").strip(),
                    )
                except Exception as judge_exc:
                    # Defensive: judge_goal swallows its own errors, but if
                    # it ever raises, fail open rather than wedge the worker.
                    logger.warning(
                        "goal judge check failed, allowing completion: %s",
                        judge_exc,
                        exc_info=True,
                    )
                if verdict != "done":
                    return tool_error(
                        f"Goal completion rejected by judge: {reason}. "
                        f"To proceed, either: (1) provide explicit acceptance "
                        f"evidence in your summary matching the task's criteria, "
                        f"or (2) create continuation tasks with parents=[{tid}] "
                        f"and keep this task alive."
                    )

            try:
                ok = kb.complete_task(
                    conn, tid,
                    result=result, summary=summary, metadata=metadata,
                    created_cards=created_cards,
                    expected_run_id=_worker_run_id(tid),
                    board=board,
                )
            except kb.WorkerGateError as gate_err:
                return tool_error(f"kanban_complete refused: {gate_err}")
            except kb.CompletionEvidenceError as evidence_err:
                return json.dumps(
                    {
                        "success": False,
                        "error": f"kanban_complete blocked: {evidence_err}",
                        "kind": evidence_err.kind,
                        "details": evidence_err.details,
                        "task_id": tid,
                        "state_changed": False,
                        "retryable": True,
                    }
                )
            except kb.ArtifactPreservationError as artifact_err:
                return tool_error(
                    f"kanban_complete could not preserve the declared artifacts: "
                    f"{artifact_err}. Your task is still in-flight and its "
                    f"scratch workspace was kept. Fix the artifact path or "
                    f"storage error, then retry kanban_complete with the same handoff."
                )
            except kb.HallucinatedCardsError as hall_err:
                # Structured rejection — surface the phantom ids so the
                # worker can retry with a corrected list or drop the
                # field. Audit event already landed in the DB.
                #
                # The task itself was NOT mutated (the gate runs before
                # the write txn), so the worker can simply call
                # kanban_complete again. Spell that out — without it the
                # model often interprets a tool_error as a terminal
                # failure and either blocks or crashes the run instead
                # of retrying. See #22923.
                return tool_error(
                    f"kanban_complete blocked: the following created_cards "
                    f"do not exist or were not created by this worker: "
                    f"{', '.join(hall_err.phantom)}. "
                    f"Your task is still in-flight (no state change). "
                    f"Retry kanban_complete with the same summary/metadata "
                    f"and either drop these ids from created_cards, or pass "
                    f"created_cards=[] to skip the card-claim check entirely."
                )
            if not ok:
                return tool_error(
                    _completion_refusal_message(
                        kb, conn, tid, expected_run_id=_worker_run_id(tid)
                    )
                )
            run = kb.latest_run(conn, tid)
            return _ok(task_id=tid, run_id=run.id if run else None)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_complete: {e}")
    except Exception as e:
        logger.exception("kanban_complete failed")
        return tool_error(f"kanban_complete: {e}")


def _redact_review_payload(summary: Any, metadata: Any):
    clean_summary = redact_sensitive_text(str(summary), force=True) if summary else None
    clean_metadata = metadata
    if metadata is not None:
        if not isinstance(metadata, dict):
            return None, None, "metadata must be an object/dict"
        encoded = redact_sensitive_text(json.dumps(metadata), force=True)
        try:
            clean_metadata = json.loads(encoded)
        except json.JSONDecodeError:
            clean_metadata = metadata
    return clean_summary, clean_metadata, None


def _handle_request_review(args: dict, **kw) -> str:
    guard = _reject_delegated_child_mutation("kanban_request_review")
    if guard:
        return guard
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error("task_id is required (or set HERMES_KANBAN_TASK in the env)")
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    reviewer = args.get("reviewer")
    if not reviewer or not str(reviewer).strip():
        return tool_error("reviewer is required")
    summary, metadata, payload_error = _redact_review_payload(
        args.get("summary"), args.get("metadata")
    )
    if payload_error:
        return tool_error(payload_error)
    if not summary:
        return tool_error("summary is required — provide the implementation handoff")
    try:
        kb, conn = _connect(board=args.get("board"))
        try:
            # Third exit from the goal loop, same class as kanban_block (#38696):
            # request_task_review sets status='review', which run_kanban_goal_loop
            # treats as terminal, and _accept_task_review then writes the
            # completion in SQL past the judge. A goal_mode worker that fails the
            # completion judge must not escape this way — route it back through
            # kanban_complete (judge-gated) or kanban_block (allowed kinds).
            gate_task = kb.get_task(conn, tid)
            if gate_task is not None and gate_task.goal_mode:
                return tool_error(
                    "goal_mode tasks cannot request review to leave the loop. "
                    "Call kanban_complete instead — the completion judge evaluates "
                    "it; or kanban_block with an allowed kind if an external "
                    "blocker genuinely prevents progress."
                )
            ok = kb.request_task_review(
                conn,
                tid,
                reviewer=str(reviewer),
                summary=summary,
                metadata=_stamp_worker_session_metadata(tid, metadata),
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(
                    f"could not request review for {tid} (stale run or invalid state)"
                )
            task = kb.get_task(conn, tid)
            return _ok(task_id=tid, status=task.status if task else None)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_request_review: {e}")
    except Exception as e:
        logger.exception("kanban_request_review failed")
        return tool_error(f"kanban_request_review: {e}")


def _handle_review_decide(args: dict, **kw) -> str:
    guard = _reject_delegated_child_mutation("kanban_review_decide")
    if guard:
        return guard
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error("task_id is required (or set HERMES_KANBAN_TASK in the env)")
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    decision = str(args.get("decision") or "").strip().upper()
    summary, metadata, payload_error = _redact_review_payload(
        args.get("summary"), args.get("metadata")
    )
    if payload_error:
        return tool_error(payload_error)
    if not summary:
        return tool_error("summary is required — provide review evidence")
    try:
        kb, conn = _connect(board=args.get("board"))
        try:
            if decision not in kb.VALID_REVIEW_DECISIONS:
                return tool_error(
                    f"decision must be one of {sorted(kb.VALID_REVIEW_DECISIONS)}"
                )
            ok = kb.decide_task_review(
                conn,
                tid,
                decision=decision,
                summary=summary,
                metadata=_stamp_worker_session_metadata(tid, metadata),
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(
                    f"could not decide review for {tid} (stale run or invalid state)"
                )
            task = kb.get_task(conn, tid)
            return _ok(
                task_id=tid,
                decision=decision,
                status=task.status if task else None,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_review_decide: {e}")
    except Exception as e:
        logger.exception("kanban_review_decide failed")
        return tool_error(f"kanban_review_decide: {e}")


def _handle_block(args: dict, **kw) -> str:
    """Transition the task to blocked with a reason a human will read."""
    delegated_err = _reject_delegated_child_mutation("kanban_block")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    reason = args.get("reason")
    if not reason or not str(reason).strip():
        return tool_error("reason is required — explain what input you need")
    reason = redact_sensitive_text(str(reason), force=True)
    kind = args.get("kind")
    # Layman-notification contract (2026-07-11): every block that surfaces
    # to a human MUST carry a plain-language summary + concrete operator
    # action. The Telegram relay leads with these; the technical ``reason``
    # stays on the board. 'dependency' never reaches a human, so it is
    # exempt. Enforced here (worker tool surface) and deliberately NOT in
    # the kernel/CLI, so human operators can still block without ceremony.
    human_summary = str(args.get("human_summary") or "").strip()
    human_action = str(args.get("human_action") or "").strip()
    if kind != "dependency" and (not human_summary or not human_action):
        return tool_error(
            "human_summary und human_action sind Pflicht (außer bei "
            "kind='dependency'): human_summary = 1-3 deutsche Sätze in "
            "Alltagssprache OHNE Pfade/IDs/Fachbegriffe, was das Problem "
            "ist; human_action = ein Satz, was der Operator konkret tun "
            "soll. Rufe kanban_block erneut mit beiden Feldern auf."
        )
    human_summary = (
        redact_sensitive_text(human_summary, force=True) if human_summary else None
    )
    human_action = (
        redact_sensitive_text(human_action, force=True) if human_action else None
    )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        if kind is not None and kind not in kb.VALID_BLOCK_KINDS:
            conn.close()
            return tool_error(
                f"kind must be one of {sorted(kb.VALID_BLOCK_KINDS)} (or omit it)"
            )
        # Goal-mode block gate (Issue #38696, sibling of the kanban_complete
        # judge gate in #38367). kanban_block is a second exit path out of
        # the goal loop — run_kanban_goal_loop() treats ANY `blocked` status
        # as terminal, identically to `done`, regardless of kind. Without
        # this, a worker that learns kanban_complete is gated can just call
        # kanban_block(reason="anything") to escape the loop instead.
        # Restrict goal_mode tasks to the kinds that represent a genuine
        # external blocker the worker cannot resolve itself; `capability`
        # and `transient` (or an unset kind) route back through
        # kanban_complete, which the judge now gates.
        task = kb.get_task(conn, tid)
        if (
            task
            and task.goal_mode
            and kind not in kb.GOAL_MODE_BLOCK_ALLOWED_KINDS
        ):
            conn.close()
            return tool_error(
                f"goal_mode tasks can only block with kind in "
                f"{sorted(kb.GOAL_MODE_BLOCK_ALLOWED_KINDS)} (got {kind!r}). "
                f"If the task is actually finished or cannot proceed for "
                f"another reason, call kanban_complete instead — the "
                f"completion judge will evaluate it."
            )
        try:
            try:
                ok = kb.block_task(
                    conn, tid,
                    reason=reason,
                    kind=kind,
                    expected_run_id=_worker_run_id(tid),
                    human_summary=human_summary,
                    human_action=human_action,
                )
            except kb.WorkerGateError as gate_err:
                return tool_error(f"kanban_block refused: {gate_err}")
            if not ok:
                return tool_error(
                    f"could not block {tid} (unknown id or not in "
                    f"running/ready)"
                )
            run = kb.latest_run(conn, tid)
            # Tell the worker where the task actually landed so it doesn't
            # assume it's sitting in 'blocked' when routing sent it elsewhere.
            landed = kb.get_task(conn, tid)
            return _ok(
                task_id=tid,
                run_id=run.id if run else None,
                status=landed.status if landed else "blocked",
                block_kind=kind,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_block: {e}")
    except Exception as e:
        logger.exception("kanban_block failed")
        return tool_error(f"kanban_block: {e}")


def _handle_heartbeat(args: dict, **kw) -> str:
    """Signal that the worker is still alive during a long operation.

    Extends the claim TTL via ``heartbeat_claim`` AND records a heartbeat
    event via ``heartbeat_worker``. Without the ``heartbeat_claim`` half,
    a diligent worker that loops this tool while a single tool call
    blocks the agent for >DEFAULT_CLAIM_TTL_SECONDS still gets reclaimed
    by ``release_stale_claims`` — which is exactly the trap that
    ``heartbeat_claim``'s docstring warns against.
    """
    delegated_err = _reject_delegated_child_mutation("kanban_heartbeat")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    note = args.get("note")
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # Extend the claim TTL first. The dispatcher pins
            # HERMES_KANBAN_CLAIM_LOCK in the worker env at spawn time
            # (see _default_spawn in kanban_db.py); falling back to the
            # default _claimer_id() covers locally-driven workers that
            # never went through the dispatcher path.
            claim_lock = os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
            kb.heartbeat_claim(conn, tid, claimer=claim_lock)

            ok = kb.heartbeat_worker(
                conn,
                tid,
                note=note,
                expected_run_id=_worker_run_id(tid),
            )
            if not ok:
                return tool_error(
                    f"could not heartbeat {tid} (unknown id or not running)"
                )
            return _ok(task_id=tid)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_heartbeat: {e}")
    except Exception as e:
        logger.exception("kanban_heartbeat failed")
        return tool_error(f"kanban_heartbeat: {e}")


def _handle_comment(args: dict, **kw) -> str:
    """Append a comment to a task's thread."""
    # kanban_comment has no ownership check by design (#19713 — cross-task
    # commenting is the deliberate handoff channel between legitimate
    # workers), so the delegated-subagent reject must be explicit here;
    # nothing else in this handler would otherwise stop it.
    delegated_err = _reject_delegated_child_mutation("kanban_comment")
    if delegated_err:
        return delegated_err
    tid = args.get("task_id")
    if not tid:
        return tool_error(
            "task_id is required (use the current task id if that's what "
            "you mean — pulls from env but kept explicit here)"
        )
    body = args.get("body")
    if not body or not str(body).strip():
        return tool_error("body is required")
    body = redact_sensitive_text(str(body), force=True)
    # Author is intentionally derived from the worker's own runtime
    # identity, NOT from caller-supplied args. Comments are injected
    # into the next worker's system prompt by ``build_worker_context``
    # as ``**{author}** (timestamp): {body}`` — accepting an
    # ``args["author"]`` override let a worker forge a comment from
    # an authoritative-looking name like ``hermes-system`` and poison
    # the future-worker context with what reads as a system directive.
    # Cross-task commenting itself remains unrestricted (see #19713) —
    # comments are the deliberate handoff channel between tasks.
    author = os.environ.get("HERMES_PROFILE") or "worker"
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            cid = kb.add_comment(conn, tid, author=author, body=str(body))
            return _ok(task_id=tid, comment_id=cid)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_comment: {e}")
    except Exception as e:
        logger.exception("kanban_comment failed")
        return tool_error(f"kanban_comment: {e}")


def _handle_attach(args: dict, **kw) -> str:
    """Attach an inline (base64) file to a task.

    Mirrors the dashboard's upload endpoint for the agent surface: decode
    the payload, enforce the shared size cap, write it under the per-task
    attachments dir, and record the metadata row — all via
    ``kanban_db.store_attachment_bytes`` so the three surfaces stay in lockstep.
    """
    # Required BEFORE the ownership check: a delegated subagent runs with a
    # stripped env, which the ownership helper reads as "orchestrator, no
    # restriction" — without this reject a subagent could attach bytes to ANY
    # card (documented invariant in _enforce_worker_task_ownership; gap found
    # in audit 2026-07-27).
    guard = _reject_delegated_child_mutation("kanban_attach")
    if guard:
        return guard
    from hermes_cli import kanban_db as kb

    delegated_err = _reject_delegated_child_mutation("kanban_attach")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    filename = args.get("filename")
    if not filename or not str(filename).strip():
        return tool_error("filename is required")
    content_b64 = args.get("content_base64")
    if not content_b64 or not str(content_b64).strip():
        return tool_error("content_base64 is required")
    import base64
    import binascii
    try:
        data = base64.b64decode(str(content_b64), validate=True)
    except (binascii.Error, ValueError) as e:
        return tool_error(f"content_base64 is not valid base64: {e}")
    content_type = args.get("content_type")
    board = args.get("board")
    try:
        _, conn = _connect(board=board)
        try:
            att_id = kb.store_attachment_bytes(
                conn,
                tid,
                str(filename),
                data,
                content_type=content_type,
                uploaded_by="agent",
                board=board,
            )
            return _ok(task_id=tid, attachment_id=att_id, size=len(data))
        finally:
            conn.close()
    except kb.AttachmentTooLarge as e:
        return tool_error(f"kanban_attach: {e}")
    except ValueError as e:
        return tool_error(f"kanban_attach: {e}")
    except Exception as e:
        logger.exception("kanban_attach failed")
        return tool_error(f"kanban_attach: {e}")


_MAX_ATTACH_URL_REDIRECTS = 5


def _download_url_with_cap(url: str, max_bytes: int) -> tuple[bytes, Optional[str]]:
    """Fetch ``url`` over http(s) with SSRF guarding, capped at ``max_bytes``.

    Every hop — the initial URL and each redirect target — is validated with
    ``tools.url_safety.is_safe_url`` before it is fetched, so a
    model-controlled URL (or a public host 302ing to one) cannot reach
    loopback, private/CGNAT ranges, or cloud metadata endpoints. Redirects
    are followed manually (``follow_redirects=False``) so each Location is
    re-checked, mirroring ``tools.skills_hub._guarded_http_get``.

    Returns ``(data, content_type)``. Raises ``ValueError`` for a non-http(s)
    scheme, an SSRF-blocked target, too many redirects, or a body that
    overruns the cap (the caller maps it to a clean tool error). Reads in
    chunks so an oversize response is rejected without buffering the whole
    thing.
    """
    from urllib.parse import urljoin, urlparse

    import httpx

    from tools.url_safety import is_safe_url

    current_url = url
    for _ in range(_MAX_ATTACH_URL_REDIRECTS + 1):
        scheme = (urlparse(current_url).scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported URL scheme {scheme!r}; only http/https are allowed"
            )
        if not is_safe_url(current_url):
            raise ValueError(
                f"URL blocked by SSRF protection (private/internal address): {current_url}"
            )
        chunks: list[bytes] = []
        total = 0
        with httpx.stream(
            "GET",
            current_url,
            headers={"User-Agent": "hermes-kanban/attach"},
            timeout=30,
            follow_redirects=False,
        ) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    raise ValueError(f"redirect without Location header from {current_url}")
                current_url = urljoin(current_url, location)
                continue
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip() or None
            for chunk in resp.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)
        return b"".join(chunks), content_type
    raise ValueError(f"too many redirects fetching {url}")


def _handle_attach_url(args: dict, **kw) -> str:
    """Attach a file fetched server-side from a URL.

    The agent passes a URL; Hermes downloads it (with the shared size cap)
    and stores it as a real attachment. Useful when the agent has a link
    rather than the bytes. Only http/https URLs are accepted.
    """
    # Same reasoning as kanban_attach — and stricter here, because this
    # variant performs a server-side download on the agent's behalf.
    guard = _reject_delegated_child_mutation("kanban_attach_url")
    if guard:
        return guard
    from hermes_cli import kanban_db as kb

    delegated_err = _reject_delegated_child_mutation("kanban_attach_url")
    if delegated_err:
        return delegated_err
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    url = args.get("url")
    if not url or not str(url).strip():
        return tool_error("url is required")
    url = str(url).strip()
    filename = args.get("filename") or args.get("title")
    if not filename or not str(filename).strip():
        # Derive a name from the URL path's leaf component.
        from urllib.parse import unquote, urlparse
        leaf = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip()
        filename = leaf or "download"
    content_type = args.get("content_type")
    board = args.get("board")
    try:
        data, fetched_ct = _download_url_with_cap(url, kb.KANBAN_ATTACHMENT_MAX_BYTES)
    except ValueError as e:
        return tool_error(f"kanban_attach_url: {e}")
    except Exception as e:
        logger.exception("kanban_attach_url download failed")
        return tool_error(f"kanban_attach_url: failed to fetch {url}: {e}")
    try:
        _, conn = _connect(board=board)
        try:
            att_id = kb.store_attachment_bytes(
                conn,
                tid,
                str(filename),
                data,
                content_type=content_type or fetched_ct,
                uploaded_by="agent",
                board=board,
            )
            return _ok(task_id=tid, attachment_id=att_id, size=len(data))
        finally:
            conn.close()
    except kb.AttachmentTooLarge as e:
        return tool_error(f"kanban_attach_url: {e}")
    except ValueError as e:
        return tool_error(f"kanban_attach_url: {e}")
    except Exception as e:
        logger.exception("kanban_attach_url failed")
        return tool_error(f"kanban_attach_url: {e}")


def _handle_attachments(args: dict, **kw) -> str:
    """List a task's attachments (read-only; no ownership restriction)."""
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            if kb.get_task(conn, tid) is None:
                return tool_error(f"task {tid} not found")
            atts = kb.list_attachments(conn, tid)
            return json.dumps({
                "ok": True,
                "task_id": tid,
                "attachments": [
                    {
                        "id": a.id,
                        "filename": a.filename,
                        "content_type": a.content_type,
                        "size": a.size,
                        "uploaded_by": a.uploaded_by,
                        "stored_path": a.stored_path,
                        "created_at": a.created_at,
                    }
                    for a in atts
                ],
            }, indent=1)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_attachments: {e}")
    except Exception as e:
        logger.exception("kanban_attachments failed")
        return tool_error(f"kanban_attachments: {e}")


_ARTIFACT_READ_DEFAULT_CHARS = 20_000
_ARTIFACT_READ_MAX_CHARS = 60_000


def _handle_artifacts(args: dict, **kw) -> str:
    """List a task's durable completion artifacts; optionally read one.

    Closes the biggest orchestration gap from the 2026-07-27 audit: durable
    artifacts are the reliable worker→orchestrator handoff channel, but the
    tool surface could neither list nor read them — callers had to guess
    storage paths and use read_file. Listing returns the recorded manifest
    (path, sha256, size); ``read`` returns an artifact's text content with
    integrity verdict, so a lost or tampered file is reported instead of
    silently served.
    """
    tid = _default_task_id(args.get("task_id"))
    if not tid:
        return tool_error(
            "task_id is required (or set HERMES_KANBAN_TASK in the env)"
        )
    # Reading a foreign task's artifact bytes is a cross-tenant information
    # leak: without these guards any worker could pass an arbitrary task_id and
    # read up to 60k chars of another tenant's durable artifact. The reject
    # must precede the ownership check — per its contract, the latter treats a
    # delegated child's stripped env as "orchestrator, no restriction".
    delegated_err = _reject_delegated_child_mutation("kanban_artifacts")
    if delegated_err:
        return delegated_err
    ownership_err = _enforce_worker_task_ownership(tid)
    if ownership_err:
        return ownership_err
    board = args.get("board")
    read_id = args.get("read")
    max_chars = args.get("max_chars")
    try:
        max_chars = int(max_chars) if max_chars is not None else _ARTIFACT_READ_DEFAULT_CHARS
    except (TypeError, ValueError):
        return tool_error("max_chars must be an integer")
    max_chars = max(1, min(max_chars, _ARTIFACT_READ_MAX_CHARS))
    try:
        kb, conn = _connect(board=board)
        try:
            if kb.get_task(conn, tid) is None:
                return tool_error(f"task {tid} not found")
            artifacts = kb.list_task_artifacts(conn, tid)
            listing = [
                {
                    "id": a["id"],
                    "name": Path(a["durable_path"]).name,
                    "durable_path": a["durable_path"],
                    "sha256": a["sha256"],
                    "size": a["size"],
                    "content_type": a["content_type"],
                    "producer_run_id": a["producer_run_id"],
                }
                for a in artifacts
            ]
            if read_id is None:
                return json.dumps({
                    "ok": True,
                    "task_id": tid,
                    "artifacts": listing,
                    "hint": (
                        "Pass read=<id> to fetch an artifact's content "
                        "(integrity-checked against the recorded sha256)."
                    ) if listing else None,
                }, indent=1)

            try:
                read_id = int(read_id)
            except (TypeError, ValueError):
                return tool_error("read must be an artifact id from the listing")
            match = next((a for a in artifacts if int(a["id"]) == read_id), None)
            if match is None:
                return tool_error(
                    f"artifact {read_id} not found on task {tid}; call "
                    "kanban_artifacts without 'read' to list valid ids"
                )
            path = Path(match["durable_path"])
            # Defense in depth: durable paths are written by the trusted
            # completion pipeline and always live under the board's artifact
            # root. A row edited to point elsewhere (e.g. at ~/.hermes/.env)
            # must not turn this reader into an arbitrary-file oracle that
            # bypasses the live-config path guard.
            #
            # TOCTOU: validating a resolved path and then re-opening it by
            # name leaves a window in which a same-uid process swaps the
            # target (TARS review 2026-07-27). So OPEN FIRST (O_NOFOLLOW
            # rejects a symlinked final component), then validate the file we
            # actually hold via its fd, and read only from that fd — the
            # bytes hashed are exactly the bytes returned.
            try:
                root = Path(kb.completion_artifacts_root(board)).resolve()
            except Exception:
                return tool_error(
                    f"kanban_artifacts: cannot resolve the board's artifact root"
                )
            fd = None
            try:
                fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode):
                    return tool_error(
                        f"artifact {read_id} is not a regular file; refusing to read"
                    )
                # Identify what we are actually holding, independent of the
                # name we opened it under.
                try:
                    opened = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()
                except OSError:
                    # No /proc (some containers): re-resolving the NAME would
                    # reintroduce the very TOCTOU this path exists to close —
                    # the name can point somewhere else than the open fd. Fail
                    # closed instead (TARS re-review 2026-07-27).
                    return tool_error(
                        f"artifact {read_id}: cannot verify the identity of the "
                        "opened file (no /proc); refusing to read"
                    )
                try:
                    opened.relative_to(root)
                except ValueError:
                    return tool_error(
                        f"artifact {read_id} path is outside the board's durable "
                        "artifact store; refusing to read"
                    )
                with os.fdopen(fd, "rb") as handle:
                    fd = None  # ownership moved to the file object
                    data = handle.read()
            except FileNotFoundError:
                return json.dumps({
                    "ok": False,
                    "task_id": tid,
                    "artifact_id": read_id,
                    "integrity": "missing",
                    "error": (
                        "durable file is gone from disk — the recorded bytes "
                        "cannot be recovered from the kanban store"
                    ),
                }, indent=1)
            except OSError as exc:
                return tool_error(
                    f"artifact {read_id} could not be opened safely: {exc.strerror}"
                )
            finally:
                if fd is not None:
                    os.close(fd)

            digest = hashlib.sha256(data).hexdigest()
            if digest != match["sha256"]:
                # FAIL CLOSED: returning the bytes alongside a mismatch verdict
                # hands possibly-tampered content (prompt injection) to the
                # model and relies on it to act on a field it can ignore. A
                # detected integrity violation must withhold the content.
                return json.dumps({
                    "ok": False,
                    "task_id": tid,
                    "artifact_id": read_id,
                    "name": path.name,
                    "integrity": "sha256_mismatch",
                    "expected_sha256": match["sha256"],
                    "actual_sha256": digest,
                    "error": (
                        "artifact bytes do not match the recorded manifest — "
                        "content withheld. The file was modified or replaced "
                        "after completion; treat it as untrusted and do not "
                        "act on it. Recover from backup if the content matters."
                    ),
                }, indent=1)
            text = data.decode("utf-8", errors="replace")
            truncated = len(text) > max_chars
            return json.dumps({
                "ok": True,
                "task_id": tid,
                "artifact_id": read_id,
                "name": path.name,
                "integrity": "ok",
                "size": match["size"],
                "truncated": truncated,
                "content": text[:max_chars],
                "next_max_chars": (
                    min(max_chars * 2, _ARTIFACT_READ_MAX_CHARS)
                    if truncated and max_chars < _ARTIFACT_READ_MAX_CHARS
                    else None
                ),
            }, indent=1)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_artifacts: {e}")
    except Exception as e:
        logger.exception("kanban_artifacts failed")
        return tool_error(f"kanban_artifacts: {e}")


def _handle_create(args: dict, **kw) -> str:
    """Create a child task. Orchestrator workers use this to fan out.

    ``parents`` can be a list of task ids; dependency-gated promotion
    works as usual.
    """
    # Board mutation stays with workers/orchestrators; a delegated subagent
    # hands results back to its parent, it does not spawn board work itself
    # (same gap family as attach/link, audit 2026-07-27).
    delegated_err = _reject_delegated_child_mutation("kanban_create")
    if delegated_err:
        return delegated_err
    title = args.get("title")
    if not title or not str(title).strip():
        return tool_error("title is required")
    assignee = args.get("assignee")
    if not assignee:
        return tool_error(
            "assignee is required — name the profile that should execute this "
            "task (the dispatcher will only spawn tasks with an assignee)"
        )
    body = args.get("body")
    parents = args.get("parents") or []
    tenant = args.get("tenant") or os.environ.get("HERMES_TENANT")
    # Stamp the originating session id when the agent loop runs under
    # ACP (which sets HERMES_SESSION_ID before invoking tools). NULL on
    # CLI / dashboard paths and on legacy hosts that don't set the env.
    # Prefer the request-scoped api_server origin binding: HERMES_SESSION_ID
    # is clobbered with a subagent's internal id whenever a child agent is
    # constructed in-process (agent_init calls set_current_session_id), which
    # would stamp — and later wake — the wrong session.
    from tools.async_delegation import _current_origin_session_id

    session_id = (
        args.get("session_id")
        or _current_origin_session_id()
        or os.environ.get("HERMES_SESSION_ID")
    )
    priority = args.get("priority")
    # Resolve workspace. Workspace sharing is always explicit: omitted fields
    # mean a fresh scratch workspace, even when a dispatcher-spawned worker
    # creates the task. Reusing a parent's literal path would let a child
    # mutate review evidence or race the parent's checkout (#67567).
    #
    # Project identity is the one safe context to inherit implicitly. The DB
    # resolves a project-linked scratch request into a fresh per-task worktree,
    # preserving the repository/branch convention without sharing a checkout.
    workspace_kind = args.get("workspace_kind")
    workspace_path = args.get("workspace_path")
    project_id = args.get("project") or args.get("project_id")
    project_source_task_id = None
    _inherit_project = workspace_kind is None and workspace_path is None
    if workspace_kind is None:
        workspace_kind = "scratch"
    triage, bool_error = _parse_bool_arg(args, "triage")
    if bool_error:
        return tool_error(bool_error)
    idempotency_key = args.get("idempotency_key")
    max_runtime_seconds = args.get("max_runtime_seconds")
    initial_status = args.get("initial_status") or "running"
    skills = args.get("skills")
    if isinstance(skills, str):
        # Accept a single skill name as a string for convenience.
        skills = [skills]
    if skills is not None and not isinstance(skills, (list, tuple)):
        return tool_error(
            f"skills must be a list of skill names, got {type(skills).__name__}"
        )
    goal_mode, goal_bool_error = _parse_bool_arg(args, "goal_mode")
    if goal_bool_error:
        return tool_error(goal_bool_error)
    allow_workspace_refs, workspace_refs_bool_error = _parse_bool_arg(
        args, "allow_workspace_refs"
    )
    if workspace_refs_bool_error:
        return tool_error(workspace_refs_bool_error)
    goal_max_turns = args.get("goal_max_turns")
    # task_class (quality-class model routing) and max_retries (per-card
    # circuit-breaker limit) were previously CLI-only: agent-created cards
    # could neither route to a stronger model nor bound their retries
    # (Audit 2026-07-10 — 100 % der Problemkarten kamen über diesen Pfad).
    task_class = str(args.get("task_class") or "").strip() or None
    max_retries = args.get("max_retries")
    model_override = args.get("model")
    provider_override = args.get("provider")
    if provider_override and not model_override:
        return tool_error("'provider' requires 'model' to be set as well")
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, (list, tuple)):
        return tool_error(
            f"parents must be a list of task ids, got {type(parents).__name__}"
        )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            # A project link is safe to inherit because ``create_task`` turns
            # it into a fresh per-task worktree. Never inherit the parent's
            # literal workspace kind/path; directory sharing must be explicit.
            if _inherit_project and project_id is None:
                _self_tid = os.environ.get("HERMES_KANBAN_TASK")
                if _self_tid:
                    _self_task = kb.get_task(conn, _self_tid)
                    if _self_task is not None and _self_task.project_id:
                        project_id = _self_task.project_id
                        project_source_task_id = _self_task.id
            _create_kwargs = dict(
                title=str(title).strip(),
                body=body,
                assignee=str(assignee),
                parents=tuple(parents),
                tenant=tenant,
                priority=int(priority) if priority is not None else 0,
                workspace_kind=str(workspace_kind),
                workspace_path=workspace_path,
                project_id=project_id,
                project_source_task_id=project_source_task_id,
                triage=triage,
                max_runtime_seconds=(
                    int(max_runtime_seconds)
                    if max_runtime_seconds is not None else None
                ),
                skills=skills,
                model_override=model_override,
                provider_override=provider_override,
                goal_mode=goal_mode,
                goal_max_turns=(
                    int(goal_max_turns) if goal_max_turns is not None else None
                ),
                initial_status=str(initial_status),
                created_by=os.environ.get("HERMES_PROFILE") or "worker",
                session_id=session_id,
                completion_contract=args.get("completion_contract"),
                task_class=task_class,
                max_retries=(
                    int(max_retries) if max_retries is not None else None
                ),
                allow_workspace_refs=allow_workspace_refs,
            )
            _acceptance = (
                bool(args["acceptance_required"])
                if args.get("acceptance_required") is not None else None
            )
            deduplicated = False
            if idempotency_key:
                # Idempotent promotion (D1, audit 2026-07-31): a keyed create
                # goes through work_promotion.promote(), whose single INSERT is
                # deduplicated by the unique origin index — replacing
                # create_task's racy check-then-insert for this surface. The
                # same key retried returns the existing card; the same key with
                # a DIFFERENT payload fails loudly instead of silently
                # discarding the new request.
                from hermes_cli import work_promotion as _wp
                try:
                    promo = _wp.promote(
                        conn,
                        origin_kind="conversation",
                        origin_key=str(idempotency_key),
                        payload={"title": _create_kwargs["title"], "body": body},
                        acceptance_required=bool(_acceptance) if _acceptance else False,
                        **_create_kwargs,
                    )
                except _wp.OriginKeyConflict as conflict:
                    return tool_error(f"kanban_create: {conflict}")
                new_tid = promo.task_id
                deduplicated = promo.deduplicated
            else:
                new_tid = kb.create_task(
                    conn, acceptance_required=_acceptance, **_create_kwargs
                )
            new_task = kb.get_task(conn, new_tid)
            subscribed = _maybe_auto_subscribe(conn, new_tid)
            return _ok(
                task_id=new_tid,
                status=new_task.status if new_task else None,
                workspace_kind=new_task.workspace_kind if new_task else None,
                workspace_path=new_task.workspace_path if new_task else None,
                project_id=new_task.project_id if new_task else None,
                subscribed=subscribed,
                deduplicated=deduplicated,
            )
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_create: {e}")
    except Exception as e:
        logger.exception("kanban_create failed")
        return tool_error(f"kanban_create: {e}")


def _notify_platform_has_consumer(platform: str) -> bool:
    """True only if some notifier actually drains subscriptions for ``platform``.

    ``subscribed: true`` is a promise that a consumer will deliver the task's
    terminal events. Writing a row for a platform nobody polls keeps that
    promise for nobody: the gateway notifier skips unknown platforms *before*
    the event claim, so the cursor stays at 0 forever and the orchestrator —
    trusting ``subscribed: true`` — never falls back to polling. That exact
    gap left 155 dead ``webui`` subscriptions with 372 undelivered events
    (incident 2026-07-27).

    Known consumers:
    - the gateway notifier serves a platform only if the ``Platform`` enum
      resolves it AND the gateway has an enabled adapter configured for it —
      a name the enum happens to know is not a delivery path (TARS review
      2026-07-27);
    - ``webui`` is served by the WebUI's in-process poller
      (hermes-webui ``api/kanban_notify_poller.py``), unless that poller is
      switched off via ``HERMES_WEBUI_KANBAN_NOTIFY``;
    - additional out-of-tree consumers can declare themselves via
      ``HERMES_KANBAN_NOTIFY_CONSUMER_PLATFORMS`` (comma-separated), so a
      custom poller does not need a code change here.

    Still an ASSERTION about configuration, not a liveness probe: it cannot
    tell whether the gateway process is up right now. A platform that is
    configured and enabled but whose gateway happens to be down still answers
    True; the audit rule ``undelivered_subscription`` is the backstop that
    surfaces a subscription whose events nobody drains.

    A plugin platform that the gateway connects WITHOUT appearing in the
    resolved platform config (measured: ``reachy``) answers False. Declare it
    in ``HERMES_KANBAN_NOTIFY_CONSUMER_PLATFORMS`` if such a session must
    auto-subscribe.

    ``tui`` deliberately resolves to False: no TUI consumer exists (the
    docstring that used to claim ``tui_gateway/server.py`` polls these rows
    was wrong — it never did).
    """
    p = (platform or "").strip().lower()
    if not p:
        return False
    if p == "webui":
        # The WebUI poller is the consumer, and it can be switched off by the
        # very env var its own module reads. Claiming a consumer while the
        # poller is disabled would re-create the original lie in a new place
        # (TARS review 2026-07-27).
        return os.environ.get(
            "HERMES_WEBUI_KANBAN_NOTIFY", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
    extra = os.environ.get("HERMES_KANBAN_NOTIFY_CONSUMER_PLATFORMS", "")
    if p in {tok.strip().lower() for tok in extra.split(",") if tok.strip()}:
        return True
    try:
        from gateway.config import Platform, load_gateway_config
        plat = Platform(p)
    except Exception:
        return False
    # A resolvable Platform proves only that the enum knows the name. The
    # gateway notifier delivers exclusively through a CONFIGURED, ENABLED
    # adapter, so that is what we require.
    #
    # Measured 2026-07-27 (correcting an earlier wrong call of mine):
    # ``load_gateway_config()`` runs ``discover_plugins()`` and folds
    # env-configured plugin platforms into ``platforms`` — with ``.env``
    # loaded, as every dispatcher-spawned worker has it, ntfy appears there.
    # The earlier "env adapters are invisible here" objection came from a
    # measurement in a bare process without ``.env``.
    try:
        pcfg = load_gateway_config().platforms.get(plat)
    except Exception:
        # FAIL CLOSED: an unreadable/broken gateway config is not evidence of
        # a consumer. Answering True here re-opened the original false
        # delivery promise through a side door (TARS re-review 2026-07-27).
        # subscribed=false costs a polling fallback; subscribed=true without a
        # consumer costs the result.
        logger.warning(
            "kanban notify: cannot read the gateway config to verify a "
            "consumer for platform %r — reporting no consumer", p,
        )
        return False
    if pcfg is None:
        return False
    return bool(getattr(pcfg, "enabled", True))


def _maybe_auto_subscribe(conn: Any, task_id: str) -> bool:
    """Auto-subscribe the calling session to task completion / block events.

    Returns True if a subscription row was written for a platform that has a
    live consumer, False otherwise (no session context, config gate disabled,
    no consumer for the platform, or best-effort failure). The caller surfaces
    this in the ``subscribed`` field of the kanban_create response so an
    orchestrator can decide whether to fall back to polling (kanban_list /
    kanban_show; subscription management exists only on the CLI as
    ``hermes kanban notify-subscribe``, there is NO tool for it) — which is
    why the value must mean "someone will deliver", not merely "a row was
    written".

    Gated by ``kanban.auto_subscribe_on_create`` in config.yaml (default
    True). Disable to mirror pre-feature behaviour, e.g. when the
    originating user/chat opted out via the per-platform notification
    toggle (see ``hermes dashboard``).

    Subscription paths:

    - **Gateway** (telegram/discord/slack/etc): ``HERMES_SESSION_PLATFORM``,
      ``HERMES_SESSION_CHAT_ID``, and ``HERMES_SESSION_CHAT_TYPE`` are set in
      ContextVars by the messaging gateway before agent dispatch. The
      notification poller already keys off these, so we just register a row.

    - **WebUI**: sessions run with ``HERMES_SESSION_PLATFORM=webui`` and the
      session id as chat_id; the WebUI's in-process poller
      (hermes-webui ``api/kanban_notify_poller.py``) drains these rows.

    - **TUI** (herm desktop / herm TUI): the platform/chat_id ContextVars are
      intentionally cleared, but ``HERMES_SESSION_KEY`` identifies the parent
      session, so the fallback below still derives ``platform="tui"``. There
      is currently NO TUI consumer, so the consumer gate returns False and no
      row is written; the fallback is kept so a future TUI poller only has to
      add its platform to the consumer set.

    - **CLI / cron / test / unattached**: no persistent delivery channel,
      no-op.

    Failure mode: any exception inside the function is logged at WARNING
    with the offending exception + diagnostic env vars and swallowed.
    We never want a notification bookkeeping failure to fail the
    kanban_create that the agent is mid-conversation about.
    """
    try:
        cfg = load_config()
        if not cfg_get(cfg, "kanban", "auto_subscribe_on_create", default=True):
            return False
    except Exception:
        # If config can't load we still default to True — this is the
        # user-friendly behaviour that mirrors the pre-gate implementation.
        pass

    platform = ""
    chat_id = ""
    try:
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        if not platform or not chat_id:
            # TUI / desktop fallback: platform/chat_id ContextVars are
            # cleared for TUI sessions, but the parent process exports
            # HERMES_SESSION_KEY into the subprocess env. Derive a "tui"
            # subscription target; whether a row is actually written is
            # decided by the consumer gate below.
            #
            # HERMES_SESSION_ID is intentionally NOT a fallback here:
            # it is set by ACP / the agent subprocess for telemetry
            # regardless of whether the parent is a TUI or a CLI, so
            # treating it as a notification target would auto-subscribe
            # every CLI invocation, which is exactly the over-eager
            # behaviour that got #19718 reverted upstream. The TUI
            # poller keys on HERMES_SESSION_KEY.
            session_key = (
                get_session_env("HERMES_SESSION_KEY", "")
                or os.environ.get("HERMES_SESSION_KEY", "")
            )
            if not session_key:
                return False  # CLI / cron / test — no persistent channel
            platform = "tui"
            chat_id = session_key
        if not _notify_platform_has_consumer(platform):
            # Honest answer instead of a dead-letter row: without a consumer
            # the subscription would sit at cursor 0 forever while the caller
            # believes delivery is arranged.
            logger.info(
                "_maybe_auto_subscribe: no notify consumer for platform=%r; "
                "not subscribing (caller should poll)", platform,
            )
            return False
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "") or None
        user_id = get_session_env("HERMES_SESSION_USER_ID", "") or None
        chat_type = get_session_env("HERMES_SESSION_CHAT_TYPE", "") or None
        message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "") or ""
        notifier_profile = (
            get_session_env("HERMES_SESSION_PROFILE", "")
            or os.environ.get("HERMES_PROFILE")
        )
        if not notifier_profile:
            try:
                from hermes_cli.profiles import get_active_profile_name
                notifier_profile = get_active_profile_name() or "default"
            except Exception:
                notifier_profile = "default"
        delivery_metadata: dict[str, Any] = {}
        if thread_id:
            delivery_metadata["thread_id"] = thread_id
        if chat_type:
            delivery_metadata["chat_type"] = chat_type
        if (
            platform.lower() == "telegram"
            and thread_id
            and (chat_type or "").lower() in {"dm", "direct", "private"}
        ):
            delivery_metadata["telegram_dm_topic_reply_fallback"] = True
            if str(thread_id) not in {"", "1"}:
                delivery_metadata["direct_messages_topic_id"] = str(thread_id)
            if message_id:
                delivery_metadata["telegram_reply_to_message_id"] = str(message_id)

        # Lazy-import to keep the module-level dependency light
        from hermes_cli import kanban_db as _kb
        _kb.add_notify_sub(
            conn, task_id=task_id,
            platform=platform, chat_id=chat_id,
            chat_type=chat_type,
            thread_id=thread_id, user_id=user_id,
            notifier_profile=notifier_profile,
            delivery_metadata=delivery_metadata or None,
        )
        return True
    except Exception as _exc:
        logger.warning(
            "_maybe_auto_subscribe failed: %r (platform=%r key_set=%r)",
            _exc, platform, bool(chat_id),
        )
        return False


def _handle_unblock(args: dict, **kw) -> str:
    """Transition a blocked task to ready, or todo while parents remain open."""
    delegated_err = _reject_delegated_child_mutation("kanban_unblock")
    if delegated_err:
        return delegated_err
    guard = _require_orchestrator_tool("kanban_unblock")
    if guard:
        return guard
    tid = args.get("task_id")
    if not tid:
        return tool_error("task_id is required")
    ownership_err = _enforce_worker_task_ownership(str(tid))
    if ownership_err:
        return ownership_err
    board = args.get("board")
    reason = str(args.get("reason") or "").strip() or None
    # Passed through, never generated or inspected here — this tool has no
    # way to SET or clear human_gate, only to redeem a token an operator
    # already received out-of-band via ntfy (Human-Gate v1).
    token = str(args.get("token") or "").strip() or None
    actor = os.environ.get("HERMES_PROFILE") or "orchestrator"
    try:
        kb, conn = _connect(board=board)
        try:
            # needs_input gates require a stated reason so the unblock is
            # attributable (who approved, what was decided) — parity with
            # the CLI gate (Audit 2026-07-10).
            task = kb.get_task(conn, str(tid))
            if (
                task is not None
                and task.status == "blocked"
                and (task.block_kind or "") == "needs_input"
                and not reason
            ):
                return tool_error(
                    f"{tid} is a needs_input block — pass reason= (who "
                    "approved, what was decided) to lift it"
                )
            # kb.unblock_task raises kb.GateTokenError (a ValueError
            # subclass) for a human_gate=1 card with a missing/wrong
            # token; the existing `except ValueError` below already
            # surfaces that as a tool_error with no extra handling needed.
            ok = kb.unblock_task(conn, str(tid), actor=actor, reason=reason, token=token)
            if not ok:
                return tool_error(f"could not unblock {tid} (not blocked or unknown)")
            task = kb.get_task(conn, str(tid))
            return _ok(task_id=str(tid), status=task.status if task else None)
        finally:
            conn.close()
    except ValueError as e:
        return tool_error(f"kanban_unblock: {e}")
    except Exception as e:
        logger.exception("kanban_unblock failed")
        return tool_error(f"kanban_unblock: {e}")


def _handle_link(args: dict, **kw) -> str:
    """Add a parent→child dependency edge after the fact."""
    delegated_err = _reject_delegated_child_mutation("kanban_link")
    if delegated_err:
        return delegated_err
    parent_id = args.get("parent_id")
    child_id = args.get("child_id")
    if not parent_id or not child_id:
        return tool_error("both parent_id and child_id are required")
    # Worker scope: a dispatcher-spawned worker may rewire the dependency
    # graph only where its own task is one endpoint (linking its card under a
    # parent, or a follow-up card under its own). Mutating edges between two
    # FOREIGN tasks can promote/hold arbitrary cards — the graph equivalent
    # of the foreign-mutation problem _enforce_worker_task_ownership exists
    # for. Orchestrators (no HERMES_KANBAN_TASK) stay unrestricted.
    env_tid = _board_env("HERMES_KANBAN_TASK")
    if env_tid and env_tid not in (parent_id, child_id):
        return tool_error(
            f"worker is scoped to task {env_tid}; refusing to link foreign "
            f"tasks {parent_id} -> {child_id}. Link edges that include your "
            "own task, or hand off via kanban_comment."
        )
    board = args.get("board")
    try:
        kb, conn = _connect(board=board)
        try:
            kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)
            return _ok(parent_id=parent_id, child_id=child_id)
        finally:
            conn.close()
    except ValueError as e:
        # Covers cycle + self-parent rejections
        return tool_error(f"kanban_link: {e}")
    except Exception as e:
        logger.exception("kanban_link failed")
        return tool_error(f"kanban_link: {e}")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_DESC_TASK_ID_DEFAULT = (
    "Task id. If omitted, defaults to HERMES_KANBAN_TASK from the env "
    "(the task the dispatcher spawned you to work on)."
)

_DESC_BOARD = (
    "Kanban board slug to target. When omitted, the call resolves the "
    "active board the usual way: HERMES_KANBAN_DB env → "
    "HERMES_KANBAN_BOARD env → the 'current' symlink under the kanban "
    "home → 'default'. Pass an explicit slug only when the caller (e.g. "
    "a Telegram routing layer) needs to override the env-pinned active "
    "board for this one call."
)


def _board_schema_prop() -> dict[str, str]:
    """Schema fragment for the optional ``board`` parameter.

    Centralised so a future tweak to the description / validation hint
    only has to land in one place.
    """
    return {"type": "string", "description": _DESC_BOARD}

KANBAN_SHOW_SCHEMA = {
    "name": "kanban_show",
    "description": (
        "Read a task's state — title, body, result, assignee, parent/child "
        "ids, recent events, comments and the latest run. COMPACT by "
        "default (last 5 events, 5 comments, 1 run, long texts trimmed) so "
        "the answer fits your working set; totals and a hint tell you when "
        "more exists. Pass include=['all'] (or a subset of comments/events/"
        "runs/worker_context) for full sections — 'worker_context' is the "
        "pre-formatted spawn-context block, only shipped on request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "include": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "all", "comments", "events", "runs", "worker_context",
                    ],
                },
                "description": (
                    "Sections to include IN FULL (uncapped). Omit for the "
                    "compact default view."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_ARTIFACTS_SCHEMA = {
    "name": "kanban_artifacts",
    "description": (
        "List a task's durable completion artifacts (the validated files a "
        "worker promoted on completion — the reliable handoff channel), or "
        "read one with read=<id>. Reading verifies the recorded sha256 and "
        "reports integrity: ok / sha256_mismatch / missing, so you never "
        "act on silently lost or altered evidence. Content is returned as "
        "text (max_chars, default 20000)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "read": {
                "type": "integer",
                "description": "Artifact id from the listing to read.",
            },
            "max_chars": {
                "type": "integer",
                "description": (
                    "Character budget for read content (default 20000, "
                    "max 60000). Truncation is flagged with next_max_chars."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_LIST_SCHEMA = {
    "name": "kanban_list",
    "description": (
        "List Kanban task summaries so an orchestrator profile can discover "
        "work to route. Supports the same core filters as the CLI: assignee, "
        "status, tenant, include_archived, and limit. Returns compact rows "
        "with ids, title, status, assignee, priority, parent/child ids, "
        "counts — plus the outcome fields (trimmed result, block_kind, "
        "last_review_decision), so one listing tells you what happened to N "
        "cards without per-card kanban_show calls. Bounded to 50 rows by "
        "default, 200 max, with truncation metadata. Also recomputes ready "
        "tasks before listing, matching the CLI. Orchestrator-only — "
        "dispatcher-spawned task workers never see this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "assignee": {
                "type": "string",
                "description": "Optional assignee/profile filter.",
            },
            "status": {
                "type": "string",
                "enum": [
                    "triage", "todo", "ready", "running",
                    "blocked", "done", "archived",
                ],
                "description": "Optional task status filter.",
            },
            "tenant": {
                "type": "string",
                "description": "Optional tenant/project namespace filter.",
            },
            "include_archived": {
                "type": "boolean",
                "description": "Include archived tasks. Defaults to false.",
            },
            "limit": {
                "type": "integer",
                "description": "Optional maximum rows to return (default 50, max 200).",
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_COMPLETE_SCHEMA = {
    "name": "kanban_complete",
    "description": (
        "Mark your current task done with a structured handoff for "
        "downstream workers and humans. Prefer ``summary`` for a "
        "human-readable 1-3 sentence description of what you did; put "
        "machine-readable facts in ``metadata`` (changed_files, "
        "tests_run, decisions, findings, etc). At least one of "
        "``summary`` or ``result`` is required. If you created new "
        "tasks via ``kanban_create`` during this run, list their ids "
        "in ``created_cards`` — the kernel verifies them so phantom "
        "references are caught before they leak into downstream "
        "automation. If you produced deliverable files (charts, PDFs, "
        "spreadsheets, generated images), list their absolute paths "
        "in ``artifacts`` — the gateway notifier will upload them as "
        "native attachments to the human who subscribed to the task, "
        "so the deliverable lands in their chat alongside the summary "
        "instead of being a path they have to fetch by hand."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "summary": {
                "type": "string",
                "description": (
                    "Human-readable handoff, 1-3 sentences. Appears in "
                    "Run History on the dashboard and in downstream "
                    "workers' context."
                ),
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Free-form dict of structured facts about this "
                    "attempt — {\"changed_files\": [...], \"tests_run\": 12, "
                    "\"findings\": [...]}. Surfaced to downstream "
                    "workers alongside ``summary``."
                ),
            },
            "result": {
                "type": "string",
                "description": (
                    "Short result log line (legacy field, maps to "
                    "task.result). Use ``summary`` instead when "
                    "possible; this exists for compatibility with "
                    "callers that still set --result on the CLI."
                ),
            },
            "created_cards": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional structured manifest of task ids you "
                    "created via ``kanban_create`` during this run. "
                    "The kernel verifies each id exists and was "
                    "created by this worker's profile; any phantom "
                    "id blocks the completion with an error listing "
                    "what went wrong (auditable in the task's events). "
                    "Only list ids you got back from a successful "
                    "``kanban_create`` call — do not invent or "
                    "remember ids from prose. Omit the field if you "
                    "did not create any cards."
                ),
            },
            "artifacts": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of absolute paths to deliverable "
                    "files you produced during this run — generated "
                    "charts, PDFs, spreadsheets, images, archives. "
                    "Examples: [\"/tmp/q3-revenue.png\", "
                    "\"/tmp/report.pdf\"]. The gateway notifier "
                    "uploads each path as a native attachment to the "
                    "subscribed chat (images embed inline, everything "
                    "else uploads as a file) so the deliverable "
                    "lands with the completion notification. Skip "
                    "intermediate scratch files and references that "
                    "are not the deliverable. The path must exist "
                    "on disk at completion. Files inside a managed scratch "
                    "workspace are copied to durable task attachments before "
                    "cleanup; a missing declared scratch artifact keeps the "
                    "task in-flight so you can fix the path and retry."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_REQUEST_REVIEW_SCHEMA = {
    "name": "kanban_request_review",
    "description": (
        "Hand the current implementation run to an independent reviewer on the "
        "same card. The implementation run closes, the card enters the review "
        "lane, and no dependency child is created. Repeated requests are idempotent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": _DESC_TASK_ID_DEFAULT},
            "reviewer": {
                "type": "string",
                "description": "Reviewer profile that should claim the review lane.",
            },
            "summary": {
                "type": "string",
                "description": "Implementation handoff and concrete verification evidence.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional structured changed-files/tests/evidence manifest.",
            },
            "board": _board_schema_prop(),
        },
        "required": ["reviewer", "summary"],
    },
}

KANBAN_REVIEW_DECIDE_SCHEMA = {
    "name": "kanban_review_decide",
    "description": (
        "Close the current review run atomically with ACCEPT, NEEDS_REPAIR, or "
        "BLOCK. ACCEPT alone completes the card; NEEDS_REPAIR returns it to the "
        "implementer; BLOCK surfaces it for human action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": _DESC_TASK_ID_DEFAULT},
            "decision": {
                "type": "string",
                "enum": ["ACCEPT", "NEEDS_REPAIR", "BLOCK"],
            },
            "summary": {
                "type": "string",
                "description": "Review verdict with concrete file/line and test evidence.",
            },
            "metadata": {
                "type": "object",
                "description": "Optional structured review findings and checks.",
            },
            "board": _board_schema_prop(),
        },
        "required": ["decision", "summary"],
    },
}

KANBAN_BLOCK_SCHEMA = {
    "name": "kanban_block",
    "description": (
        "Stop work on this task and route it according to WHY you're stuck. "
        "Set ``kind`` to say which: 'dependency' (waiting on another task — "
        "goes to todo and auto-resumes when that task finishes, no human "
        "needed), 'needs_input' (you need a human decision/answer), "
        "'capability' (a hard wall: no access, missing credentials, an action "
        "no agent can do), or 'transient' (a flaky failure that may clear). "
        "``reason`` is shown to the human on the board. If a task keeps "
        "getting unblocked and re-blocked for the same reason, it is "
        "auto-escalated to triage. Use for genuine blockers only — don't "
        "block on things you can resolve yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "reason": {
                "type": "string",
                "description": (
                    "What you need answered or what stopped you, in one or "
                    "two sentences. Don't paste the whole conversation; the "
                    "human has the board and can ask follow-ups via comments."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["dependency", "needs_input", "capability", "transient"],
                "description": (
                    "Why you're blocked. 'dependency' waits in todo and "
                    "resumes automatically; the others surface to a human. "
                    "Omit only if none apply."
                ),
            },
            "human_summary": {
                "type": "string",
                "description": (
                    "REQUIRED unless kind='dependency'. 1-3 kurze deutsche "
                    "Sätze für einen Nicht-Techniker: Was ist das Problem, "
                    "in Alltagssprache? KEINE Pfade, IDs, Stacktraces oder "
                    "Fachbegriffe — die stehen schon in ``reason``. Diese "
                    "Zusammenfassung wird dem Operator per Telegram "
                    "gepusht."
                ),
            },
            "human_action": {
                "type": "string",
                "description": (
                    "REQUIRED unless kind='dependency'. Ein Satz: Was soll "
                    "der Operator (Manfred) jetzt konkret tun? Z.B. eine "
                    "Entscheidung treffen, einen Zugang freischalten, eine "
                    "Datei bereitstellen. Wenn er nichts tun kann/muss, "
                    "schreibe das explizit (dann ist kanban_block aber "
                    "vermutlich das falsche Werkzeug)."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["reason"],
    },
}

KANBAN_HEARTBEAT_SCHEMA = {
    "name": "kanban_heartbeat",
    "description": (
        "Signal that you're still alive during a long operation "
        "(training, encoding, large crawls). Call every few minutes so "
        "humans see liveness separately from PID checks. Pure side "
        "effect — no work changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "note": {
                "type": "string",
                "description": (
                    "Optional short note describing current progress. "
                    "Shown in the event log."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_COMMENT_SCHEMA = {
    "name": "kanban_comment",
    "description": (
        "Append a comment to a task's thread. Use for durable notes "
        "that should outlive this run (questions for the next worker, "
        "partial findings, rationale). Ephemeral reasoning doesn't "
        "belong here — use your normal response instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "Task id. Required (may be your own task or "
                    "another's — comment threads are per-task)."
                ),
            },
            "body": {
                "type": "string",
                "description": "Markdown-supported comment body.",
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id", "body"],
    },
}

KANBAN_ATTACH_SCHEMA = {
    "name": "kanban_attach",
    "description": (
        "Attach a file to a task by passing its bytes inline (base64). "
        "Use for genuine file artifacts the next worker or a human should "
        "be able to download — generated reports, images, exports. The "
        "file is stored as a real attachment (not a comment link) under "
        "the task's attachments dir, capped at 25 MB. Prefer "
        "kanban_attach_url when you only have a URL."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "filename": {
                "type": "string",
                "description": (
                    "File name to store it under (e.g. 'report.pdf'). "
                    "Directory components are stripped; only the leaf is kept."
                ),
            },
            "content_base64": {
                "type": "string",
                "description": "The file contents, base64-encoded. Max 25 MB decoded.",
            },
            "content_type": {
                "type": "string",
                "description": "Optional MIME type (e.g. 'application/pdf').",
            },
            "board": _board_schema_prop(),
        },
        "required": ["filename", "content_base64"],
    },
}

KANBAN_ATTACH_URL_SCHEMA = {
    "name": "kanban_attach_url",
    "description": (
        "Attach a file to a task by URL — Hermes downloads it server-side "
        "and stores it as a real attachment (capped at 25 MB). Use when "
        "you have a link rather than the bytes. Only http/https URLs are "
        "accepted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "url": {
                "type": "string",
                "description": "http(s) URL to fetch and store.",
            },
            "filename": {
                "type": "string",
                "description": (
                    "Optional name to store it under. Defaults to the URL "
                    "path's leaf component."
                ),
            },
            "content_type": {
                "type": "string",
                "description": (
                    "Optional MIME type override. Defaults to the "
                    "Content-Type the server returns."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["url"],
    },
}

KANBAN_ATTACHMENTS_SCHEMA = {
    "name": "kanban_attachments",
    "description": (
        "List the files attached to a task: id, filename, content_type, "
        "size, who uploaded it, and the absolute on-disk path you can read."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": _DESC_TASK_ID_DEFAULT,
            },
            "board": _board_schema_prop(),
        },
        "required": [],
    },
}

KANBAN_CREATE_SCHEMA = {
    "name": "kanban_create",
    "description": (
        "Create a new kanban task, optionally as a child of the current "
        "one (pass the current task id in ``parents``). Used by "
        "orchestrator workers to fan out — decompose work into child "
        "tasks with specific assignees, link them into a pipeline, "
        "then complete your own task. The dispatcher picks up the new "
        "tasks on its next tick and spawns the assigned profiles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short task title (required).",
            },
            "assignee": {
                "type": "string",
                "description": (
                    "Profile name that should execute this task "
                    "(e.g. 'researcher-a', 'reviewer', 'writer'). "
                    "Required — tasks without an assignee are never "
                    "dispatched."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "Opening post: full spec, acceptance criteria, "
                    "links. The assigned worker reads this as part of "
                    "its context."
                ),
            },
            "parents": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Parent task ids. The new task stays in 'todo' "
                    "until every parent reaches 'done'; then it "
                    "auto-promotes to 'ready'. Typical fan-in: list "
                    "all the researcher task ids when creating a "
                    "synthesizer task."
                ),
            },
            "tenant": {
                "type": "string",
                "description": (
                    "Optional namespace for multi-project isolation. "
                    "Defaults to HERMES_TENANT env if set."
                ),
            },
            "priority": {
                "type": "integer",
                "description": (
                    "Dispatcher tiebreaker. Higher = picked sooner "
                    "when multiple ready tasks share an assignee."
                ),
            },
            "workspace_kind": {
                "type": "string",
                "enum": ["scratch", "dir", "worktree"],
                "description": (
                    "Workspace flavor: 'scratch' (fresh tmp dir, "
                    "default), 'dir' (shared directory, requires "
                    "absolute workspace_path), 'worktree' (git worktree)."
                ),
            },
            "workspace_path": {
                "type": "string",
                "description": (
                    "Absolute path for 'dir' or 'worktree' workspace. "
                    "Relative paths are rejected at dispatch."
                ),
            },
            "allow_workspace_refs": {
                "type": "boolean",
                "description": (
                    "Explicit opt-out from the disposable-workspace handoff guard. "
                    "Use only for an intentional dependency on a concurrently running task."
                ),
            },
            "project": {
                "type": "string",
                "description": (
                    "Optional project id or slug to link the task to. When "
                    "set, the task becomes a git worktree under the project's "
                    "primary repo with a deterministic branch (project slug + "
                    "task id), instead of a random branch."
                ),
            },
            "triage": {
                "type": "boolean",
                "description": (
                    "If true, task lands in 'triage' instead of 'todo' "
                    "— a specifier profile is expected to flesh out "
                    "the body before work starts."
                ),
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "If a non-archived task with this key already "
                    "exists, return that task's id instead of creating "
                    "a duplicate. Useful for retry-safe automation."
                ),
            },
            "max_runtime_seconds": {
                "type": "integer",
                "description": (
                    "Per-task runtime cap. When exceeded, the "
                    "dispatcher SIGTERMs the worker and re-queues the "
                    "task with outcome='timed_out'."
                ),
            },
            "initial_status": {
                "type": "string",
                "enum": ["running", "blocked"],
                "description": (
                    "Initial card status. Use 'blocked' for tasks that "
                    "require immediate human ops (R3 gate) to skip the "
                    "brief running-to-blocked transition. Defaults to "
                    "'running', which preserves the usual dispatch path."
                ),
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Skill names to force-load into the dispatched "
                    "worker. The kanban lifecycle is already injected "
                    "automatically; use this to pin a task to a specialist "
                    "context — e.g. ['translation'] for a translation "
                    "task, ['github-code-review'] for a reviewer task. "
                    "The names must match skills installed on the "
                    "assignee's profile."
                ),
            },
            "goal_mode": {
                "type": "boolean",
                "description": (
                    "Run the dispatched worker in a goal loop. When true, "
                    "after each turn an auxiliary judge checks the worker's "
                    "response against this card's title/body; if the work "
                    "isn't done and budget remains, the worker keeps going "
                    "in the same session until the judge agrees it's "
                    "complete (or the goal-turn budget is exhausted, which "
                    "blocks the task for human review). Use this for "
                    "open-ended cards where one shot rarely finishes the "
                    "work. Defaults to false (classic single-shot worker)."
                ),
            },
            "goal_max_turns": {
                "type": "integer",
                "description": (
                    "Turn budget for goal_mode workers. Caps how many "
                    "continuation turns the worker may take before the task "
                    "is blocked for review. Ignored unless goal_mode is "
                    "true. Defaults to the goal-engine default (20)."
                ),
            },
            "completion_contract": {
                "type": "object",
                "properties": {
                    "tests_or_smokes": {"type": "boolean"},
                    "readback": {"type": "boolean"},
                    "artifacts": {"type": "boolean"},
                },
                "additionalProperties": False,
                "description": (
                    "Optional fail-closed evidence requirements. Required "
                    "fields must be present in kanban_complete metadata; "
                    "artifacts are validated and durably promoted before done."
                ),
            },
            "acceptance_required": {
                "type": "boolean",
                "description": (
                    "When true, this card cannot reach 'done' on the worker's "
                    "own say-so: an explicit review ACCEPT (via "
                    "kanban_request_review → kanban_review_decide) is "
                    "required before kanban_complete succeeds. Use for work "
                    "that MUST pass independent review — a free-text "
                    "'NEEDS_REPAIR' on a normal done card is invisible to "
                    "the dependency graph."
                ),
            },
            "task_class": {
                "type": "string",
                "description": (
                    "Quality class for model routing (e.g. 'hard' routes "
                    "the card to the planner-tier model). Omit for default "
                    "routing."
                ),
            },
            "max_retries": {
                "type": "integer",
                "description": (
                    "Per-card circuit-breaker limit: consecutive "
                    "crash/spawn failures before the card auto-blocks. "
                    "Omit to use the board default."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Pin the dispatched worker to this model instead of "
                    "the assignee profile's configured model. Use the "
                    "exact model name the target provider expects. Omit "
                    "to use the profile default."
                ),
            },
            "provider": {
                "type": "string",
                "description": (
                    "Provider the 'model' belongs to (e.g. 'openrouter', "
                    "'anthropic', 'nous'). Set this whenever the model "
                    "is not from the assignee profile's configured "
                    "provider — a model name alone is resolved against "
                    "the profile's provider and will fail if it belongs "
                    "to a different one. Requires 'model'."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["title", "assignee"],
    },
}

KANBAN_UNBLOCK_SCHEMA = {
    "name": "kanban_unblock",
    "description": (
        "Unblock a Kanban task. It moves to ready when all parents are done, "
        "or todo while any parent remains open. Orchestrator-only — only "
        "profiles with the kanban toolset can unblock routed work; "
        "dispatcher-spawned task workers never see this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Blocked task id to move to ready or parent-gated todo.",
            },
            "reason": {
                "type": "string",
                "description": (
                    "Why the block is lifted (who approved, what was "
                    "decided). REQUIRED for needs_input blocks — those are "
                    "human-decision gates and every lift must be "
                    "attributable."
                ),
            },
            "token": {
                "type": "string",
                "description": (
                    "One-time human-gate token. REQUIRED for cards marked "
                    "with a hard human gate (delivered to the operator via "
                    "ntfy, out of band) — this tool has no way to set or "
                    "remove that gate, only to redeem a token you were "
                    "given. Ignored for ungated cards."
                ),
            },
            "board": _board_schema_prop(),
        },
        "required": ["task_id"],
    },
}

KANBAN_LINK_SCHEMA = {
    "name": "kanban_link",
    "description": (
        "Add a parent→child dependency edge after both tasks already "
        "exist. The child won't promote to 'ready' until all parents "
        "are 'done'. Cycles and self-links are rejected."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "parent_id": {"type": "string", "description": "Parent task id."},
            "child_id":  {"type": "string", "description": "Child task id."},
            "board": _board_schema_prop(),
        },
        "required": ["parent_id", "child_id"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="kanban_show",
    toolset="kanban",
    schema=KANBAN_SHOW_SCHEMA,
    handler=_handle_show,
    check_fn=_check_kanban_mode,
    emoji="📋",
)

registry.register(
    name="kanban_list",
    toolset="kanban",
    schema=KANBAN_LIST_SCHEMA,
    handler=_handle_list,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="📋",
)

registry.register(
    name="kanban_artifacts",
    toolset="kanban",
    schema=KANBAN_ARTIFACTS_SCHEMA,
    handler=_handle_artifacts,
    check_fn=_check_kanban_mode,
    emoji="📋",
)

registry.register(
    name="kanban_complete",
    toolset="kanban",
    schema=KANBAN_COMPLETE_SCHEMA,
    handler=_handle_complete,
    check_fn=_check_kanban_mode,
    emoji="✔",
)

registry.register(
    name="kanban_request_review",
    toolset="kanban",
    schema=KANBAN_REQUEST_REVIEW_SCHEMA,
    handler=_handle_request_review,
    check_fn=_check_kanban_mode,
    emoji="🔎",
)

registry.register(
    name="kanban_review_decide",
    toolset="kanban",
    schema=KANBAN_REVIEW_DECIDE_SCHEMA,
    handler=_handle_review_decide,
    check_fn=_check_kanban_mode,
    emoji="⚖",
)

registry.register(
    name="kanban_block",
    toolset="kanban",
    schema=KANBAN_BLOCK_SCHEMA,
    handler=_handle_block,
    check_fn=_check_kanban_mode,
    emoji="⏸",
)

registry.register(
    name="kanban_heartbeat",
    toolset="kanban",
    schema=KANBAN_HEARTBEAT_SCHEMA,
    handler=_handle_heartbeat,
    check_fn=_check_kanban_mode,
    emoji="💓",
)

registry.register(
    name="kanban_comment",
    toolset="kanban",
    schema=KANBAN_COMMENT_SCHEMA,
    handler=_handle_comment,
    check_fn=_check_kanban_mode,
    emoji="💬",
)

registry.register(
    name="kanban_attach",
    toolset="kanban",
    schema=KANBAN_ATTACH_SCHEMA,
    handler=_handle_attach,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_attach_url",
    toolset="kanban",
    schema=KANBAN_ATTACH_URL_SCHEMA,
    handler=_handle_attach_url,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_attachments",
    toolset="kanban",
    schema=KANBAN_ATTACHMENTS_SCHEMA,
    handler=_handle_attachments,
    check_fn=_check_kanban_mode,
    emoji="📎",
)

registry.register(
    name="kanban_create",
    toolset="kanban",
    schema=KANBAN_CREATE_SCHEMA,
    handler=_handle_create,
    check_fn=_check_kanban_mode,
    emoji="➕",
)

registry.register(
    name="kanban_unblock",
    toolset="kanban",
    schema=KANBAN_UNBLOCK_SCHEMA,
    handler=_handle_unblock,
    check_fn=_check_kanban_orchestrator_mode,
    emoji="▶",
)

registry.register(
    name="kanban_link",
    toolset="kanban",
    schema=KANBAN_LINK_SCHEMA,
    handler=_handle_link,
    check_fn=_check_kanban_mode,
    emoji="🔗",
)

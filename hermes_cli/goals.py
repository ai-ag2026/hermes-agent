"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, Hermes feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message (which takes priority and pauses the goal loop).

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.

Design notes / invariants:

- The continuation prompt is just a normal user message appended to the
  session via ``run_conversation``. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- Judge failures are fail-OPEN: ``continue``. A broken judge must not wedge
  progress; the turn budget is the backstop.
- When a real user message arrives mid-loop it preempts the continuation
  prompt and also pauses the goal loop for that turn (we still re-judge
  after, so if the user's message happens to complete the goal the judge
  will say ``done``).
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
# Judge output budget. The freeform judge returns a one-line JSON verdict, but
# reasoning models (deepseek-v4, qwq, etc.) burn tokens on hidden reasoning
# before emitting the visible JSON — and the first /goal turn's prompt is
# larger than later turns, which pushes total reply length past tight caps.
# 200 tokens (the original default) reliably truncated the JSON on reasoning
# models, leaving '{"done": true, "reason": "The agent successfully' and
# triggering the auto-pause. 4096 covers reasoning + verdict on every model
# we've live-tested; override via auxiliary.goal_judge.max_tokens for
# specifically constrained setups.
DEFAULT_JUDGE_MAX_TOKENS = 4096
# Cap how much of the last response + recent messages we send to the judge.
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
# After this many consecutive judge *parse* failures (empty output / non-JSON),
# the loop auto-pauses and points the user at the goal_judge config. API /
# transport errors do NOT count toward this — those are transient. This guards
# against small models (e.g. deepseek-v4-flash) that cannot follow the strict
# JSON reply contract; without it the loop runs until the turn budget is
# exhausted with every reply shaped like `judge returned empty response` or
# `judge reply was not JSON`.
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 3
# Transport failures (API auth errors 401, timeouts, DNS, etc.) are also
# tracked and auto-pause the loop after this many consecutive failures.
# A broken/invalid API key returns 401 every call — the loop must not
# run until the turn budget, wasting every turn on an unreachable judge.
DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES = 5


CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)

# Used when the goal carries a structured completion contract. The contract
# block tells the agent exactly what "done" means, how to prove it, what not
# to break, what's in scope, and when to stop and ask — so it targets the
# verification surface instead of declaring victory loosely.
CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Completion contract:\n"
    "{contract_block}\n\n"
    "Continue working toward the outcome above. Take the next concrete step. "
    "Stay within the stated boundaries and do not violate the constraints. "
    "Before claiming the goal is done, satisfy the Verification criterion and "
    "show the concrete evidence (command output, file contents, test result). "
    "If you hit the stated stop condition or are otherwise blocked and need "
    "user input, say so clearly and stop."
)

# Used when the user has added one or more /subgoal criteria. Surfaced
# to the agent verbatim so it sees what to target on the next turn,
# and surfaced to the judge so the verdict considers them too.
CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Additional criteria the user added mid-loop:\n"
    "{subgoals_block}\n\n"
    "Continue working toward the goal AND all additional criteria. Take "
    "the next concrete step. If you believe the goal and every "
    "additional criterion are complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly "
    "and stop."
)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text, the agent's "
    "most recent response, and — when present — a list of background "
    "processes the agent has running. Decide one of three verdicts.\n\n"
    "DONE — the goal is fully satisfied:\n"
    "- The response explicitly confirms the goal was completed AND took the terminal "
    "step it implies (shipped / deployed / opened the PR / delivered), OR\n"
    "- The response clearly shows the final deliverable was produced, OR\n"
    "- The goal is genuinely blocked by something ONLY the user can resolve, AND the "
    "agent first exhausted every resourceful option and reached every step it could "
    "reach on its own. A merely cautious 'I could do more but I'll stop and ask', "
    "partial progress reported as complete, or stopping short of the obvious terminal "
    "step is NOT done — pick CONTINUE and push to finish.\n\n"
    "WAIT — the goal is NOT done, but the next step is to wait for async "
    "work to finish rather than act again. Choose this ONLY when the agent's "
    "progress is genuinely gated on something running on its own:\n"
    "- A background process listed below is still running AND the response "
    "shows the agent is waiting on its result (e.g. a CI poller, build, "
    "test run, deploy). If the process has a session id, return it in "
    "``wait_on_session`` — that releases when the process exits OR its "
    "watch_patterns trigger fires (use this for a long-lived watcher that "
    "signals mid-run and may never exit). Otherwise return its pid in "
    "``wait_on_pid`` (releases on exit only).\n"
    "- The agent says it is rate-limited / backing off / must wait a fixed "
    "period — return seconds in ``wait_for_seconds``.\n"
    "Picking WAIT parks the loop without burning a turn; it resumes "
    "automatically when the pid exits or the time elapses. Do NOT pick WAIT "
    "just because work remains — only when re-poking now would be pure "
    "busy-work because the agent can't progress until the async thing "
    "finishes.\n\n"
    "CONTINUE — not done, and there is a concrete next step the agent can "
    "take right now. This is the default when in doubt.\n"
    "- When the work is substantially complete, the concrete next step is the TERMINAL "
    "one (ship it / open the PR / deploy / deliver), NOT another round of tests, "
    "smokes, docs, or polish. Push the agent toward finishing the WHOLE goal.\n"
    "- If the response is re-verifying an already-passing result, polishing "
    "already-working code, or reporting partial progress as if complete, that is a "
    "signal to CONTINUE toward the terminal step — not to stop.\n\n"
    "Reply ONLY with a single JSON object on one line. Shapes:\n"
    '{"verdict": "done", "reason": "<one sentence>"}\n'
    '{"verdict": "continue", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_session": "<id>", "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_on_pid": <int>, "reason": "<one sentence>"}\n'
    '{"verdict": "wait", "wait_for_seconds": <int>, "reason": "<one sentence>"}\n'
    "The legacy shape {\"done\": <true|false>, \"reason\": \"...\"} is still "
    "accepted (true=done, false=continue)."
)


# Rendered into the judge prompt when the agent has background processes
# running. Gives the judge the context it needs to decide WAIT vs CONTINUE
# (and which pid to wait on) without it having to probe anything itself.
JUDGE_BACKGROUND_BLOCK_TEMPLATE = (
    "Background processes the agent currently has running (it may be waiting "
    "on one of these):\n{background_lines}\n\n"
)


JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Is the goal satisfied — done, continue, or wait?"
)

# Used when the user has added /subgoal criteria. The judge must
# evaluate ALL of them being met, not just the original goal.
JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Additional criteria the user added mid-loop (all must also be "
    "satisfied for the goal to be DONE):\n{subgoals_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision: For each numbered criterion above, find concrete "
    "evidence in the agent's response that the criterion is "
    "satisfied. Do not accept generic phrases like 'all requirements "
    "met' or 'implying it was done' — require specific evidence (a "
    "file contents excerpt, an output line, a command result). If "
    "ANY criterion lacks specific evidence in the response, the goal "
    "is NOT done — return CONTINUE (or WAIT if blocked on a listed "
    "background process).\n\n"
    "Is the goal AND every additional criterion satisfied?"
)


# Used when the goal carries a structured completion contract. The judge
# decides DONE strictly against the Verification criterion and refuses to
# accept completion when a constraint was violated.
JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Completion contract (the authoritative definition of done):\n"
    "{contract_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "{background_block}"
    "Current time: {current_time}\n\n"
    "Decision rules:\n"
    "- The goal is DONE only when the Verification criterion is satisfied AND "
    "the response shows concrete evidence of it (a command result, file "
    "contents excerpt, test/benchmark output) — not a claim like 'done' or "
    "'all tests pass' without evidence.\n"
    "- If any stated Constraint was violated, the goal is NOT done — CONTINUE.\n"
    "- If the response shows the agent is waiting on a listed background "
    "process to satisfy the Verification criterion (e.g. CI is the "
    "verification and it's still running), return WAIT on that process "
    "instead of re-poking — re-poking now would be pure busy-work.\n"
    "- If the response hits the goal's explicit Stop condition, treat it as DONE "
    "with the reason. A block that is merely cautious ('I could do more but I'll "
    "stop and ask'), partial progress reported as complete, or stopping short of "
    "the terminal step is NOT done — CONTINUE and push to finish. Only a genuine "
    "blocker that ONLY the user can resolve, after the agent exhausted its options "
    "and reached every step it could reach, is DONE.\n"
    "- Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Is the goal satisfied per its completion contract — done, continue, or wait?"
)


# System prompt for /goal draft — turns a plain-language objective into a
# structured completion contract the user can review before activating.
# Adapted from Codex's "let Codex draft the goal" guidance.
DRAFT_CONTRACT_SYSTEM_PROMPT = (
    "You turn a user's plain-language objective into a structured completion "
    "contract for an autonomous coding agent. The contract has five fields:\n"
    "- outcome: the single end state that must be true when done\n"
    "- verification: the specific test / command / artifact that PROVES the "
    "outcome (must be concrete and checkable)\n"
    "- constraints: what must NOT change or regress\n"
    "- boundaries: which files, dirs, tools, or systems are in scope\n"
    "- stop_when: the condition under which the agent should stop and ask "
    "for human input instead of pushing on\n\n"
    "Infer sensible, specific values from the objective and any project "
    "context implied by it. Prefer concrete verification (a named test "
    "command, a build, a benchmark) over vague phrases. Keep each field to "
    "one or two sentences. If a field genuinely cannot be inferred, use an "
    "empty string for it.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"outcome": "...", "verification": "...", "constraints": "...", '
    '"boundaries": "...", "stop_when": "..."}'
)


# ──────────────────────────────────────────────────────────────────────
# Completion contract
# ──────────────────────────────────────────────────────────────────────

# The five contract fields, in display order. Adapted from OpenAI Codex's
# "strong goal" guidance: a durable objective works best when it names what
# "done" means, how to prove it, what must not regress, what tools/paths are
# in bounds, and when to stop and ask. A bare free-form goal (no contract)
# stays fully supported — every field defaults empty and is simply omitted
# from the prompts when unset.
_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")

# Human labels for rendering and for the inline `field: value` parser.
_CONTRACT_LABELS = {
    "outcome": "Outcome",
    "verification": "Verification",
    "constraints": "Constraints",
    "boundaries": "Boundaries",
    "stop_when": "Stop when blocked",
}

# Inline-input aliases the user may type before a value, mapped to the
# canonical field name. e.g. `verify: tests pass` or `done when: ...`.
_CONTRACT_ALIASES = {
    "outcome": "outcome",
    "goal": "outcome",
    "done": "outcome",
    "done when": "outcome",
    "verification": "verification",
    "verify": "verification",
    "verified by": "verification",
    "evidence": "verification",
    "proof": "verification",
    "constraints": "constraints",
    "constraint": "constraints",
    "preserve": "constraints",
    "must not": "constraints",
    "do not change": "constraints",
    "boundaries": "boundaries",
    "boundary": "boundaries",
    "scope": "boundaries",
    "allowed": "boundaries",
    "files": "boundaries",
    "stop when": "stop_when",
    "stop_when": "stop_when",
    "blocked": "stop_when",
    "stop if blocked": "stop_when",
    "give up when": "stop_when",
}


@dataclass
class GoalContract:
    """Optional structured completion contract for a goal.

    Each field is free-form prose the user (or :func:`draft_contract`)
    supplies. Empty fields are omitted everywhere — a goal with no contract
    behaves exactly like the original free-form goal. The contract is woven
    into both the continuation prompt (so the agent targets the verification
    surface and respects constraints) and the judge prompt (so "done" is
    decided against evidence, not vibes).
    """

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""

    def is_empty(self) -> bool:
        return not any(getattr(self, f).strip() for f in _CONTRACT_FIELDS)

    def to_dict(self) -> Dict[str, str]:
        return {f: getattr(self, f) for f in _CONTRACT_FIELDS}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalContract":
        if not isinstance(data, dict):
            return cls()
        return cls(**{f: str(data.get(f) or "").strip() for f in _CONTRACT_FIELDS})

    def render_block(self) -> str:
        """Render non-empty contract fields as a labelled block. Empty
        contract → empty string (callers skip the section entirely)."""
        lines = []
        for f in _CONTRACT_FIELDS:
            val = getattr(self, f).strip()
            if val:
                lines.append(f"- {_CONTRACT_LABELS[f]}: {val}")
        return "\n".join(lines)


def parse_contract(text: str) -> Tuple[str, GoalContract]:
    """Split user-typed goal text into a headline + structured contract.

    Supports inline ``field: value`` lines so power users can type a full
    contract in one shot, e.g.::

        Migrate auth to JWT
        verify: the auth test suite passes
        constraints: keep the public /login response shape unchanged
        boundaries: only touch services/auth and its tests
        stop when: a schema change needs product sign-off

    The first non-field line(s) become the goal headline; recognized
    ``field:`` lines populate the contract. Lines for the same field are
    joined. Unrecognized prefixes stay part of the headline, so a plain
    free-form goal with an incidental colon (``Fix bug: the parser``)
    is NOT mangled — only lines whose prefix matches a known alias are
    pulled out. Returns ``(headline, contract)``.
    """
    if not text:
        return "", GoalContract()

    headline_parts: List[str] = []
    fields: Dict[str, List[str]] = {f: [] for f in _CONTRACT_FIELDS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        if ":" in line:
            prefix, _, value = line.partition(":")
            key = _CONTRACT_ALIASES.get(prefix.strip().lower())
            if key is not None and value.strip():
                fields[key].append(value.strip())
                matched = True
        if not matched:
            headline_parts.append(line)

    headline = " ".join(headline_parts).strip()
    contract = GoalContract(
        **{f: " ".join(v).strip() for f, v in fields.items()}
    )
    # If a headline was given but no explicit `outcome:` field, the headline
    # IS the outcome — don't duplicate it into the contract block (the goal
    # text already carries it), so leave outcome empty in that case.
    return headline, contract


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalState:
    """Serializable goal state stored per session."""

    goal: str
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # "done" | "continue" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    consecutive_parse_failures: int = 0       # judge-output parse failures in a row
    # Transport failures are API/auth/network errors.  Broken API keys return
    # 401 every call — track them separately so the loop auto-pauses instead
    # of burning every turn budget slot on an unreachable judge.
    consecutive_transport_failures: int = 0   # judge API/transport errors in a row
    # User-added criteria appended mid-loop via the /subgoal command.
    # When non-empty the judge prompt and continuation prompt both
    # include them so the agent works toward them and the judge factors
    # them into the verdict. Backwards-compatible: defaults to empty so
    # old state_meta rows load unchanged.
    subgoals: List[str] = field(default_factory=list)
    # Wait barrier: when the agent is blocked on long-running async work
    # (CI poller, build, test run, deploy, rate-limit cooldown) the goal loop
    # PARKS instead of being re-poked every turn into busy-work. Two barrier
    # kinds, set automatically by the judge (which now sees the live
    # background-process list and can return a ``wait`` verdict) or manually
    # via ``/goal wait``:
    #   • ``waiting_on_pid`` — park until that process exits.
    #   • ``waiting_on_session`` — park until that process_registry session's
    #     OWN trigger fires: it exits, OR (if it has watch_patterns) its
    #     pattern matches. Covers long-lived watchers/servers that signal
    #     mid-run via a trigger and may never exit. Preferred over raw pid
    #     when the agent set up a watch_patterns/notify_on_complete process.
    #   • ``waiting_until``  — park until this wall-clock epoch (time backoff).
    # While ANY is active, ``evaluate_after_turn`` short-circuits to
    # should_continue=False without burning a turn or calling the judge. The
    # barrier auto-clears when the pid exits / the trigger fires / the deadline
    # passes, then the next turn resumes normal judging. Cleared by that,
    # ``/goal unwait``, pause, resume, or clear. Backwards-compatible: old
    # state_meta rows load with no barrier.
    waiting_on_pid: Optional[int] = None
    waiting_on_session: Optional[str] = None
    waiting_until: float = 0.0
    waiting_reason: Optional[str] = None
    waiting_since: float = 0.0
    # Optional structured completion contract (outcome / verification /
    # constraints / boundaries / stop_when). Empty by default; a goal with
    # no contract behaves exactly like the original free-form goal.
    contract: GoalContract = field(default_factory=GoalContract)

    def to_json(self) -> str:
        data = asdict(self)
        # asdict already recursed GoalContract into a plain dict.
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_subgoals = data.get("subgoals") or []
        subgoals: List[str] = []
        if isinstance(raw_subgoals, list):
            subgoals = [str(s).strip() for s in raw_subgoals if str(s).strip()]
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            consecutive_parse_failures=int(data.get("consecutive_parse_failures", 0) or 0),
            consecutive_transport_failures=int(data.get("consecutive_transport_failures", 0) or 0),
            subgoals=subgoals,
            waiting_on_pid=(int(data["waiting_on_pid"]) if data.get("waiting_on_pid") else None),
            waiting_on_session=(str(data["waiting_on_session"]) if data.get("waiting_on_session") else None),
            waiting_until=float(data.get("waiting_until", 0.0) or 0.0),
            waiting_reason=data.get("waiting_reason"),
            waiting_since=float(data.get("waiting_since", 0.0) or 0.0),
            contract=GoalContract.from_dict(data.get("contract")),
        )

    # --- contract helpers -------------------------------------------------

    def has_contract(self) -> bool:
        return self.contract is not None and not self.contract.is_empty()

    # --- subgoals helpers -------------------------------------------------

    def render_subgoals_block(self) -> str:
        """Render the subgoals as a numbered ``- N. text`` block. Empty
        when no subgoals exist."""
        if not self.subgoals:
            return ""
        return "\n".join(f"- {i}. {text}" for i, text in enumerate(self.subgoals, start=1))


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    SessionDB has no built-in singleton, but opening a new connection per
    /goal call would thrash the file. We cache one instance per
    ``hermes_home`` path so profile switches still pick up the right DB.
    Defensive against import/instantiation failures so tests and
    non-standard launchers can still use the GoalManager.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_goal(session_id: str) -> Optional[GoalState]:
    """Load the goal for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("GoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return GoalState.from_json(raw)
    except Exception as exc:
        logger.warning("GoalManager: could not parse stored goal for %s: %s", session_id, exc)
        return None


def save_goal(session_id: str, state: GoalState) -> None:
    """Persist a goal to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("GoalManager: set_meta failed: %s", exc)


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


def migrate_goal_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /goal from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session,
    but ``load_goal`` does a flat ``goal:<session_id>`` lookup with no
    parent-lineage walk — so an active goal silently dies at the
    compaction boundary (#33618). Copy the goal onto the new session and
    archive the old row as ``cleared`` so exactly one active goal row
    exists per logical conversation (avoids the "two active goals"
    hazard of a pure copy).

    Returns True when a goal was migrated, False when there was nothing
    to migrate or the DB was unavailable. Best-effort and never raises —
    a failure here must not block compression.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_goal(old_session_id)
        if state is None or getattr(state, "status", None) == "cleared":
            return False
        # Don't clobber a goal already set on the child (e.g. a resumed
        # lineage that re-established its own goal).
        if load_goal(new_session_id) is not None:
            return False
        save_goal(new_session_id, state)
        # Archive the parent's row so it isn't double-counted as active.
        clear_goal(old_session_id)
        logger.debug(
            "GoalManager: migrated goal %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("GoalManager: goal migration failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently alive.

    Delegates to ``gateway.status._pid_exists`` — the canonical,
    cross-platform, footgun-safe liveness check (psutil with a ctypes /
    POSIX fallback). Critically this avoids ``os.kill(pid, 0)``, which on
    Windows is NOT a no-op: it routes to ``CTRL_C_EVENT`` and hard-kills the
    target's console process group (bpo-14484). Any error resolves to False
    (treat unknown as dead) so a stale barrier never wedges the loop — the
    worst case is the goal resumes one turn early, which is safe.
    """
    if not pid or pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(int(pid)))
    except Exception:
        pass
    # Last-resort fallback if gateway.status is unavailable: psutil directly.
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _session_waiting(session_id: str) -> bool:
    """Whether a goal parked on a process_registry session should stay parked.

    Delegates to ``process_registry.is_session_waiting`` — True while the
    session is running and (if it has watch_patterns) its trigger hasn't fired.
    Fail-safe: any import/registry error yields False (don't wait) so a stale
    barrier can never wedge the loop.
    """
    if not session_id:
        return False
    try:
        from tools.process_registry import process_registry

        return bool(process_registry.is_session_waiting(session_id))
    except Exception:
        return False


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _goal_judge_max_tokens() -> int:
    """Resolve auxiliary.goal_judge.max_tokens, falling back to the default.

    ``load_config()`` is cached on the config file's (mtime, size), so calling
    this once per judge turn is cheap. A non-positive or non-int value falls
    back to the default rather than crashing the goal loop.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = (
            (cfg.get("auxiliary") or {})
            .get("goal_judge", {})
            .get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS)
        )
        value = int(value)
        if value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_JUDGE_MAX_TOKENS


def _parse_judge_response(raw: str) -> Tuple[str, str, bool, Optional[Dict[str, Any]]]:
    """Parse the judge's reply. Fail-open on unusable output.

    Returns ``(verdict, reason, parse_failed, wait_directive)`` where:
      - ``verdict`` is ``"done"``, ``"continue"``, or ``"wait"``.
      - ``parse_failed`` is True when the judge returned output that couldn't
        be interpreted as the expected JSON verdict (empty body, prose,
        malformed JSON). Callers use it to auto-pause after N consecutive
        parse failures so a weak judge model doesn't silently burn the budget.
      - ``wait_directive`` is set only for ``verdict == "wait"``: a dict with
        ``{"pid": int}`` or ``{"seconds": int}`` (whichever the judge supplied).
        ``None`` otherwise. If a wait verdict carries neither a usable pid nor
        seconds, it is downgraded to ``continue`` (can't park on nothing).

    Accepts both the new ``{"verdict": ...}`` shape and the legacy
    ``{"done": <bool>}`` shape.
    """
    if not raw:
        return "continue", "judge returned empty response", True, None

    text = raw.strip()

    # Strip markdown code fences the model may wrap JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        # Peel off leading json/JSON/etc tag
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    # First try: parse the whole blob.
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        # Second try: pull the first JSON object out.
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return "continue", f"judge reply was not JSON: {_truncate(raw, 200)!r}", True, None

    reason = str(data.get("reason") or "").strip() or "no reason provided"

    # Determine verdict — prefer the explicit "verdict" field, fall back to
    # the legacy "done" boolean.
    verdict_raw = data.get("verdict")
    if isinstance(verdict_raw, str):
        verdict = verdict_raw.strip().lower()
    else:
        done_val = data.get("done")
        if isinstance(done_val, str):
            done = done_val.strip().lower() in {"true", "yes", "1", "done"}
        else:
            done = bool(done_val)
        verdict = "done" if done else "continue"

    if verdict not in {"done", "continue", "wait"}:
        verdict = "continue"

    if verdict != "wait":
        return verdict, reason, False, None

    # Wait verdict: extract a concrete directive (pid or seconds). Accept a
    # few key spellings the model might emit.
    def _first_int(*keys: str) -> Optional[int]:
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except (TypeError, ValueError):
                continue
        return None

    # Prefer a session-id directive (releases on the process's own trigger —
    # exit OR watch-pattern match), then pid (exit only), then seconds.
    sess = data.get("wait_on_session") or data.get("session_id") or data.get("wait_session")
    if isinstance(sess, str) and sess.strip():
        return "wait", reason, False, {"session_id": sess.strip()}
    pid = _first_int("wait_on_pid", "pid", "wait_pid")
    if pid is not None:
        return "wait", reason, False, {"pid": pid}
    seconds = _first_int("wait_for_seconds", "seconds", "wait_seconds")
    if seconds is not None:
        return "wait", reason, False, {"seconds": seconds}
    # Wait with no usable target — can't park on nothing; treat as continue.
    return "continue", f"{reason} (wait verdict had no target — continuing)", False, None


def _render_background_block(background_processes: Optional[List[Dict[str, Any]]]) -> str:
    """Render the live background-process list for the judge prompt.

    Each entry is a ``process_registry.list_sessions()`` dict. Only RUNNING
    processes are worth showing (an exited one is nothing to wait on). Returns
    an empty string when there's nothing running, so the judge prompt is
    byte-identical to the no-background case (no behavior change for the
    common path).
    """
    if not background_processes:
        return ""
    lines: List[str] = []
    for p in background_processes:
        if not isinstance(p, dict):
            continue
        if p.get("status") == "exited":
            continue
        pid = p.get("pid")
        if not pid:
            continue
        cmd = _truncate(str(p.get("command") or "").replace("\n", " ").strip(), 120)
        uptime = p.get("uptime_seconds")
        tail = _truncate(str(p.get("output_preview") or "").replace("\n", " ").strip(), 120)
        sid = p.get("session_id")
        line = f"- pid {pid}"
        if sid:
            line += f" / session {sid}"
        line += f": {cmd}"
        if uptime is not None:
            line += f" (running {uptime}s)"
        # Surface the process's own trigger so the judge can wait on a
        # mid-run signal (watch-pattern) or completion, not just exit.
        wps = p.get("watch_patterns")
        if wps:
            hit = " [already matched]" if p.get("watch_hit") else ""
            line += f" | watch_patterns={wps}{hit}"
        elif p.get("notify_on_complete"):
            line += " | notify_on_complete"
        if tail:
            line += f" | recent output: {tail}"
        lines.append(line)
    if not lines:
        return ""
    return JUDGE_BACKGROUND_BLOCK_TEMPLATE.format(background_lines="\n".join(lines))


def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    subgoals: Optional[List[str]] = None,
    background_processes: Optional[List[Dict[str, Any]]] = None,
    contract: Optional[GoalContract] = None,
) -> Tuple[str, str, bool, Optional[Dict[str, Any]], bool]:
    """Ask the auxiliary model whether the goal is satisfied.

    Returns ``(verdict, reason, parse_failed, wait_directive, transport_failed)`` where verdict
    is ``"done"``, ``"continue"``, ``"wait"``, or ``"skipped"`` (when the
    judge couldn't be reached). ``wait_directive`` is set only for ``"wait"``
    (``{"pid": int}`` or ``{"seconds": int}``); ``None`` otherwise.

    ``parse_failed`` is True only when the judge call succeeded but its output
    was unusable (empty or non-JSON). API/transport errors return False — they
    are transient and should fail-open silently.

    ``transport_failed`` is True only when the judge couldn't reach the API at
    all (auth 401, timeout, DNS, connection error).  Repeated transport
    failures signal a permanent config problem (e.g. invalid API key).  Callers
    use this flag to auto-pause after N consecutive transport failures (see
    ``DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES``). Callers use this flag to
    auto-pause after N consecutive parse failures (see
    ``DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES``).

    ``subgoals`` is an optional list of user-added criteria (from
    ``/subgoal``) factored into the verdict. ``background_processes`` is the
    live ``process_registry.list_sessions()`` snapshot; when the agent is
    waiting on one (a CI poller, build, etc.) the judge can return a ``wait``
    verdict naming its pid, parking the loop instead of re-poking.
    ``contract`` is an optional structured completion contract; when present
    the judge decides DONE strictly against its Verification criterion and
    refuses completion when a Constraint was violated. All three are additive
    — a contract, subgoals, and a background-process list can coexist in one
    judge prompt; when none are set, behavior is identical to the original
    free-form judge.

    This is deliberately fail-open: transport errors return ``("continue", ..., ..., None, True)``
    — the ``transport_failed=True`` flag lets callers track and auto-pause after
    N consecutive transport failures (see
    ``DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES``) so a permanently broken
    judge doesn't burn the entire turn budget.
    """
    if not goal.strip():
        return "skipped", "empty goal", False, None, False
    if not last_response.strip():
        # No substantive reply this turn — almost certainly not done yet.
        return "continue", "empty response (nothing to evaluate)", False, None, False

    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return "continue", "auxiliary client unavailable", False, None, False

    # Build the prompt. Priority: contract > subgoals > plain. When both a
    # contract and subgoals exist, the subgoals are appended into the
    # contract block as extra criteria so the judge sees a single source of
    # truth.
    clean_subgoals = [s.strip() for s in (subgoals or []) if s and s.strip()]
    background_block = _render_background_block(background_processes)
    current_time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    if contract is not None and not contract.is_empty():
        contract_block = contract.render_block()
        if clean_subgoals:
            extra = "\n".join(
                f"- Extra criterion {i}: {text}"
                for i, text in enumerate(clean_subgoals, start=1)
            )
            contract_block = f"{contract_block}\n{extra}"
        prompt = JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            contract_block=_truncate(contract_block, 2500),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    elif clean_subgoals:
        subgoals_block = "\n".join(
            f"- {i}. {text}" for i, text in enumerate(clean_subgoals, start=1)
        )
        prompt = JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            subgoals_block=_truncate(subgoals_block, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    else:
        prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )

    try:
        # Route through call_llm so auxiliary.goal_judge.* config
        # (provider/model/base_url, extra_body, reasoning_effort, retries)
        # all apply — the direct-create path dropped extra_body (#35566).
        resp = call_llm(
            task="goal_judge",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
        )
    except Exception as exc:
        logger.info("goal judge: API call failed (%s) — falling through to continue", exc)
        return "continue", f"judge error: {type(exc).__name__}", False, None, True

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    verdict, reason, parse_failed, wait_directive = _parse_judge_response(raw)
    logger.info(
        "goal judge: verdict=%s reason=%s%s",
        verdict, _truncate(reason, 120),
        f" wait={wait_directive}" if wait_directive else "",
    )
    return verdict, reason, parse_failed, wait_directive, False


def gather_background_processes(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the live background-process snapshot for the goal judge.

    Thin, fail-safe wrapper over ``process_registry.list_sessions(task_id)``.
    Returns only RUNNING processes (an exited one is nothing to wait on) and
    never raises — any import/registry failure yields ``[]`` so the goal loop
    degrades to its pre-wait-barrier behavior (judge just won't see processes).
    The drivers (CLI + gateway) call this and pass the result into
    ``GoalManager.evaluate_after_turn(background_processes=...)``.
    """
    try:
        from tools.process_registry import process_registry

        sessions = process_registry.list_sessions(task_id=task_id) or []
    except Exception as exc:
        logger.debug("gather_background_processes failed: %s", exc)
        return []
    return [s for s in sessions if isinstance(s, dict) and s.get("status") != "exited"]


def draft_contract(objective: str, *, timeout: float = DEFAULT_JUDGE_TIMEOUT) -> Optional[GoalContract]:
    """Expand a plain-language objective into a structured completion contract.

    Uses the ``goal_judge`` auxiliary task (main-model-first, cache-safe — it
    is a side LLM call, not a conversation turn). Returns a populated
    :class:`GoalContract` on success, or ``None`` when the auxiliary client is
    unavailable or the model's reply can't be parsed. Callers fall back to a
    bare free-form goal in that case, so a missing/weak aux model never blocks
    setting a goal.
    """
    objective = (objective or "").strip()
    if not objective:
        return None

    try:
        from agent.auxiliary_client import call_llm
    except Exception as exc:
        logger.debug("goal draft: auxiliary client import failed: %s", exc)
        return None

    try:
        # Route through call_llm — same #35566 fix as the judge call above.
        resp = call_llm(
            task="goal_judge",
            messages=[
                {"role": "system", "content": DRAFT_CONTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Objective:\n{_truncate(objective, 4000)}"},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
        )
    except Exception as exc:
        logger.info("goal draft: API call failed (%s)", exc)
        return None

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        logger.debug("goal draft: reply was not JSON: %r", _truncate(raw, 200))
        return None
    contract = GoalContract.from_dict(data)
    return None if contract.is_empty() else contract


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort: pull the first JSON object out of a model reply.

    Shares the fence-stripping + first-object fallback logic used by the
    judge parser, but returns the dict (or None) rather than a verdict.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None
    return data if isinstance(data, dict) else None


# ──────────────────────────────────────────────────────────────────────
# GoalManager — the orchestration surface CLI + gateway talk to
# ──────────────────────────────────────────────────────────────────────


class GoalManager:
    """Per-session goal state + continuation decisions.

    The CLI and gateway each hold one ``GoalManager`` per live session.

    Methods:

    - ``set(goal)`` — start a new standing goal.
    - ``clear()`` — remove the active goal.
    - ``pause()`` / ``resume()`` — explicit user controls.
    - ``status()`` — printable one-liner.
    - ``evaluate_after_turn(last_response)`` — call the judge, update state,
      and return a decision dict the caller uses to drive the next turn.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[GoalState] = load_goal(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def has_contract(self) -> bool:
        return self._state is not None and self._state.has_contract()

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in {"cleared",}:
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        sub = f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}" if s.subgoals else ""
        con = ", contract" if self.has_contract() else ""
        meta = f"{turns}{sub}{con}"
        if s.status == "active":
            if s.waiting_on_session and _session_waiting(s.waiting_on_session):
                wr = s.waiting_reason or f"session {s.waiting_on_session}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_on_pid and _pid_alive(s.waiting_on_pid):
                wr = s.waiting_reason or f"pid {s.waiting_on_pid}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_until and time.time() < s.waiting_until:
                remaining = int(s.waiting_until - time.time())
                wr = s.waiting_reason or f"{remaining}s"
                return f"⏳ Goal (parked {remaining}s — {wr}, {meta}): {s.goal}"
            return f"⊙ Goal (active, {meta}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {meta}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({meta}): {s.goal}"
        return f"Goal ({s.status}, {meta}): {s.goal}"

    # --- mutation -----------------------------------------------------

    def set(self, goal: str, *, max_turns: Optional[int] = None, contract: Optional[GoalContract] = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
            contract=contract if contract is not None else GoalContract(),
        )
        self._state = state
        save_goal(self.session_id, state)
        return state

    def set_contract(self, contract: GoalContract) -> Optional[GoalState]:
        """Attach or replace the completion contract on the active goal.

        Returns the updated state, or None when there is no goal to attach to.
        """
        if self._state is None:
            return None
        self._state.contract = contract or GoalContract()
        save_goal(self.session_id, self._state)
        return self._state

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        # A wait barrier is meaningless once paused — drop it.
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        save_goal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True) -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        # Resuming starts fresh — clear any stale barrier.
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_goal(self.session_id, self._state)
        self._state = None

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        save_goal(self.session_id, self._state)

    # --- /subgoal user controls ---------------------------------------

    def add_subgoal(self, text: str) -> str:
        """Append a user-added criterion to the active goal. Requires
        ``has_goal()``; raises ``RuntimeError`` otherwise.

        Returns the cleaned text so the caller can show it back to the user.
        """
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")
        self._state.subgoals.append(text)
        save_goal(self.session_id, self._state)
        return text

    def remove_subgoal(self, index_1based: int) -> str:
        """Remove a subgoal by 1-based index. Returns the removed text."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.subgoals):
            raise IndexError(
                f"index out of range (1..{len(self._state.subgoals)})"
            )
        removed = self._state.subgoals.pop(idx)
        save_goal(self.session_id, self._state)
        return removed

    def clear_subgoals(self) -> int:
        """Wipe all subgoals. Returns the previous count."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        prev = len(self._state.subgoals)
        self._state.subgoals = []
        save_goal(self.session_id, self._state)
        return prev

    def render_subgoals(self) -> str:
        """Public helper for the /subgoal slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.subgoals:
            return "(no subgoals — use /subgoal <text> to add criteria)"
        return self._state.render_subgoals_block()

    # --- /goal wait barrier -------------------------------------------

    def wait_on(self, pid: int, reason: str = "") -> GoalState:
        """Park the goal loop on a background process PID.

        While the PID is alive, ``evaluate_after_turn`` returns
        ``should_continue=False`` without burning a turn or calling the
        judge — the loop quiesces instead of re-poking the agent into busy
        work. The barrier auto-clears when the process exits. Requires an
        active goal. For a process with a watch_patterns/notify_on_complete
        trigger, prefer ``wait_on_session`` so a mid-run trigger (not just
        exit) releases the barrier.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        pid = int(pid)
        if pid <= 0:
            raise ValueError("pid must be a positive integer")
        self._state.waiting_on_pid = pid
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def wait_on_session(self, session_id: str, reason: str = "") -> GoalState:
        """Park the goal loop on a process_registry session's OWN trigger.

        Unlike ``wait_on`` (which releases only on PID exit), this releases
        when the session's trigger fires: it exits, OR — if it was started
        with ``watch_patterns`` — its pattern matches. This is the right
        barrier for a long-lived watcher/server/poller that signals mid-run
        and may never exit. Requires an active goal.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._state.waiting_on_session = session_id
        self._state.waiting_on_pid = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def wait_for_seconds(self, seconds: int, reason: str = "") -> GoalState:
        """Park the goal loop until ``seconds`` from now have elapsed.

        Time-based counterpart to ``wait_on`` — for backoff / cooldown waits
        where there's no process to track (e.g. the agent is rate-limited).
        The barrier auto-clears once the deadline passes. Requires an active
        goal.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError("seconds must be a positive integer")
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = time.time() + seconds
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def stop_waiting(self) -> bool:
        """Clear any active wait barrier (pid / session / time). Returns True
        if one was cleared."""
        if self._state is None:
            return False
        if (
            self._state.waiting_on_pid is None
            and self._state.waiting_on_session is None
            and not self._state.waiting_until
        ):
            return False
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        save_goal(self.session_id, self._state)
        return True

    def is_waiting(self) -> bool:
        """True iff a barrier is set AND not yet satisfied.

        Session barrier: active until the process exits or its watch-pattern
        trigger fires. Pid barrier: active while the process is alive. Time
        barrier: active until the deadline passes. Side effect: a satisfied
        barrier is cleared here (lazy auto-clear) so the next evaluation
        resumes normal judging.
        """
        s = self._state
        if s is None:
            return False
        if s.waiting_on_session is not None:
            if _session_waiting(s.waiting_on_session):
                return True
            self.stop_waiting()  # session exited or trigger fired
            return False
        if s.waiting_on_pid is not None:
            if _pid_alive(s.waiting_on_pid):
                return True
            self.stop_waiting()  # process gone
            return False
        if s.waiting_until:
            if time.time() < s.waiting_until:
                return True
            self.stop_waiting()  # deadline passed
            return False
        return False

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
        background_processes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False). Both increment
        ``turns_used`` because both consume model budget.

        ``background_processes`` is the live ``process_registry.list_sessions()``
        snapshot for this session. It's handed to the judge so it can decide
        to WAIT on an in-flight process (CI poller, build, ...) instead of
        re-poking the agent — the automatic counterpart to ``/goal wait``.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "wait" | "skipped" | "inactive"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        # Wait barrier: if the loop is parked (on a live process OR a time
        # deadline that hasn't passed), quiesce — do NOT burn a turn or call
        # the judge. Resumes automatically once the barrier clears.
        if self.is_waiting():
            if state.waiting_on_session is not None:
                tgt = f"session {state.waiting_on_session}"
            elif state.waiting_on_pid is not None:
                tgt = f"pid {state.waiting_on_pid}"
            else:
                remaining = max(0, int(state.waiting_until - time.time()))
                tgt = f"{remaining}s remaining"
            reason = state.waiting_reason or tgt
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "waiting",
                "reason": reason,
                "message": f"⏳ Goal parked — waiting on {tgt}: {reason}",
            }

        # Count the turn that just finished.
        state.turns_used += 1
        state.last_turn_at = time.time()

        verdict, reason, parse_failed, wait_directive, transport_failed = judge_goal(
            state.goal,
            last_response,
            subgoals=state.subgoals or None,
            background_processes=background_processes,
            contract=state.contract if state.has_contract() else None,
        )
        state.last_verdict = verdict
        state.last_reason = reason

        # Track consecutive judge parse failures. Reset on any usable reply,
        # including API / transport errors (parse_failed=False) so a flaky
        # network doesn't trip the auto-pause meant for bad judge models.
        if parse_failed:
            state.consecutive_parse_failures += 1
        else:
            state.consecutive_parse_failures = 0

        # Track consecutive transport failures separately — persistent API
        # errors (401 auth, DNS, timeout) signal a broken config, not
        # transient network flakiness.  Auto-pause after N consecutive
        # transport failures so a permanently broken judge doesn't burn
        # every turn budget slot on an unreachable API.
        if transport_failed:
            state.consecutive_transport_failures += 1
        else:
            state.consecutive_transport_failures = 0

        # WAIT verdict: the judge decided the agent is blocked on async work
        # and re-poking now would be busy-work. Set the barrier and park —
        # the turn we just counted stands (the judge call happened), but no
        # continuation fires. The loop resumes automatically when the pid
        # exits or the deadline passes (next evaluate_after_turn falls through
        # the is_waiting() short-circuit once the barrier clears).
        if verdict == "wait" and wait_directive:
            if wait_directive.get("session_id"):
                self.wait_on_session(str(wait_directive["session_id"]), reason=reason)
                tgt = f"session {wait_directive['session_id']}"
            elif wait_directive.get("pid"):
                self.wait_on(int(wait_directive["pid"]), reason=reason)
                tgt = f"pid {wait_directive['pid']}"
            else:
                self.wait_for_seconds(int(wait_directive["seconds"]), reason=reason)
                tgt = f"{wait_directive['seconds']}s"
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "wait",
                "reason": reason,
                "message": f"⏳ Goal parked (judge) — waiting on {tgt}: {reason}",
            }

        if verdict == "done":
            state.status = "done"
            save_goal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        # Auto-pause when the judge cannot reach the API at all N turns in a
        # row (401 auth, DNS failure, timeout).  Persistent transport failures
        # signal a broken configuration (e.g. invalid API key), not transient
        # flakiness.  Without this guard, a permanently broken judge burns
        # every turn budget slot on an unreachable API.
        if state.consecutive_transport_failures >= DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge API unreachable {state.consecutive_transport_failures} turns in a row "
                f"(check auxiliary.goal_judge provider/key in config.yaml)"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — judge API returned errors "
                    f"({state.consecutive_transport_failures} turns). "
                    "Check the goal_judge provider/key in ~/.hermes/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: deepseek\n"
                    "      model: deepseek-v4-flash\n"
                    "Then /goal resume to continue."
                ),
            }

        # Auto-pause when the judge model can't produce the expected JSON
        # verdict N turns in a row. Points the user at the goal_judge config
        # so they can route this side task to a model that follows the
        # contract (e.g. google/gemini-3-flash-preview). Without this guard,
        # weak judge models burn the entire turn budget returning prose or
        # empty strings.
        if state.consecutive_parse_failures >= DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge model returned unparseable output {state.consecutive_parse_failures} turns in a row"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — the judge model ({state.consecutive_parse_failures} turns) "
                    "isn't returning the required JSON verdict. Route the judge to a stricter "
                    "model in ~/.hermes/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: openrouter\n"
                    "      model: google/gemini-3-flash-preview\n"
                    "Then /goal resume to continue."
                ),
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        save_goal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        # Contract takes priority: it carries the verification surface and
        # constraints the agent must target. Subgoals fold in as extra
        # criteria appended to the contract block.
        if self._state.has_contract():
            contract_block = self._state.contract.render_block()
            if self._state.subgoals:
                extra = "\n".join(
                    f"- Extra criterion {i}: {text}"
                    for i, text in enumerate(self._state.subgoals, start=1)
                )
                contract_block = f"{contract_block}\n{extra}"
            return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                goal=self._state.goal,
                contract_block=contract_block,
            )
        if self._state.subgoals:
            return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal=self._state.goal,
                subgoals_block=self._state.render_subgoals_block(),
            )
        return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)

    def render_contract(self) -> str:
        """Public helper for the /goal show + /goal draft slash commands."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.has_contract():
            return "(no completion contract — set one with /goal draft <objective> or inline field: value lines)"
        return self._state.contract.render_block()


# ──────────────────────────────────────────────────────────────────────
# Kanban worker goal loop
# ──────────────────────────────────────────────────────────────────────

# Continuation prompt fed back to a kanban goal-mode worker that has not
# yet completed/blocked its task. The card's own acceptance criteria are
# the goal — the worker already has the full task body in its first turn,
# so we keep this short and point it back at the lifecycle contract.
KANBAN_GOAL_CONTINUATION_TEMPLATE = (
    "[Continuing toward this kanban task — judge says it is not done yet]\n"
    "Reason: {reason}\n\n"
    "Take the next concrete step toward completing the task. When the work "
    "is genuinely finished, call kanban_complete with a summary. If you are "
    "blocked and need human input, call kanban_block with a reason. Do not "
    "stop without calling one of them."
)

# Appended to an ordinary continuation prompt once the work budget is
# running low, so the worker gets an explicit heads-up before it hits the
# reserved closeout turn — a soft limit ahead of the hard one.
KANBAN_GOAL_SOFT_LIMIT_SUFFIX = (
    "\n\n[Budget notice: {used}/{max} turns used, {remaining} ordinary work "
    "turn(s) left before the reserved closeout turn. Start wrapping up now "
    "so you have something concrete to hand off.]"
)

# Fed when the judge believes the work is done but the worker never called
# kanban_complete / kanban_block. One explicit nudge to terminate the task
# the right way before the loop gives up. This consumes the reserved
# closeout turn — see ``run_kanban_goal_loop``.
KANBAN_GOAL_FINALIZE_TEMPLATE = (
    "[The work looks complete, but the task is still open]\n"
    "Reason: {reason}\n\n"
    "If the task is genuinely done, call kanban_complete now with a short "
    "summary of what you did. If something still blocks completion, call "
    "kanban_block with the reason instead."
)

# Fed on the reserved closeout turn when the work budget ran out before the
# judge ever saw a "done" response. This is a lifecycle-only turn: no new
# work, just terminate the task honestly with whatever evidence exists.
KANBAN_GOAL_CLOSEOUT_TEMPLATE = (
    "[Closeout — this is the reserved final turn for this task]\n"
    "Reason: {reason}\n\n"
    "Your ordinary work budget is used up. This turn is reserved exclusively "
    "for wrapping up: call kanban_complete with an honest summary of what you "
    "actually accomplished (partial progress counts — describe it plainly), "
    "or call kanban_block with the reason you could not finish. Do not start "
    "new work and do not reply with prose alone — call kanban_complete or "
    "kanban_block this turn."
)


DEFAULT_RESERVED_CLOSEOUT_TURNS = 1
DEFAULT_NO_PROGRESS_LIMIT = 2
DEFAULT_MAX_TRANSIENT_RETRIES = 3
DEFAULT_WAIT_POLL_SECONDS = 5.0
DEFAULT_MAX_WAIT_SECONDS = 900.0
# Overall safety valve across ALL run_turn retries in one loop, regardless
# of classification — guards against a pathological "always slightly
# different error message" defeating the repeat-fingerprint check forever.
_GOAL_MAX_TOTAL_TURN_RETRIES = 5
# How many recent judge verdicts ride along in emitted progress events.
# Bounded so the payload can never grow unbounded across a long-running loop.
_GOAL_VERDICT_HISTORY_LIMIT = 5
# Judge reasons are free text from an auxiliary model; cap what we persist.
_GOAL_PROGRESS_REASON_LIMIT = 200


class KanbanTurnError(Exception):
    """Raised by an injected ``run_turn`` to report a classified turn failure.

    Subclass with :class:`KanbanTransientTurnError` for infrastructure /
    cooldown failures the loop should retry. A plain ``KanbanTurnError`` (or
    any other exception ``run_turn`` happens to raise) is treated as a
    deterministic protocol failure — no blind retry, see
    ``_classify_turn_error``.
    """


class KanbanTransientTurnError(KanbanTurnError):
    """A run_turn failure classified as transient (rate limit, timeout, ...)."""


# Substring markers used to classify an *unclassified* exception (one that
# isn't already a ``KanbanTurnError``) as transient infrastructure trouble
# rather than a deterministic protocol failure. Best-effort only — callers
# that can tell the difference authoritatively should raise
# ``KanbanTransientTurnError`` / ``KanbanTurnError`` directly instead of
# relying on message sniffing.
_TRANSIENT_ERROR_MARKERS = (
    "rate limit", "rate_limit", "ratelimit", "429", "502", "503", "504",
    "timeout", "timed out", "temporarily unavailable", "connection reset",
    "connection error", "overloaded", "cooldown", "quota",
)


def _failure_fingerprint(exc: BaseException) -> str:
    """Bounded, content-free identifier for a run_turn failure.

    Used to detect a second *identical* failure so the loop can stop
    blind-retrying and escalate to triage instead of looping forever on the
    same defect. Never carries raw content — only a hash of the exception's
    class name and the first 200 characters of its message.
    """
    text = f"{type(exc).__name__}:{str(exc)[:200]}"
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _classify_turn_error(exc: BaseException) -> str:
    """Return ``"transient"`` or ``"deterministic"`` for a run_turn failure."""
    if isinstance(exc, KanbanTransientTurnError):
        return "transient"
    if isinstance(exc, KanbanTurnError):
        return "deterministic"
    text = str(exc).lower()
    if any(marker in text for marker in _TRANSIENT_ERROR_MARKERS):
        return "transient"
    return "deterministic"


def _response_fingerprint(response: str) -> str:
    """Bounded, content-free fingerprint of a worker response."""
    return hashlib.sha256((response or "").encode("utf-8", "replace")).hexdigest()[:16]


def _valid_wait_directive(directive: Optional[Dict[str, Any]]) -> bool:
    """Whether a judge-supplied wait directive names something concrete to
    park on. Anything else (empty dict, non-positive pid/seconds, garbage
    types) is invalid — the caller must fail safe rather than park forever.
    """
    if not isinstance(directive, dict):
        return False
    pid = directive.get("pid")
    if pid is not None:
        try:
            return int(pid) > 0
        except (TypeError, ValueError):
            return False
    seconds = directive.get("seconds")
    if seconds is not None:
        try:
            return int(seconds) > 0
        except (TypeError, ValueError):
            return False
    return False


def _park_on_wait_barrier(
    directive: Dict[str, Any],
    *,
    heartbeat_fn: Optional[Callable[[], None]],
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
    poll_seconds: float,
    max_wait_seconds: float,
    log: Callable[[str], None],
) -> None:
    """Block (via ``sleep_fn``) until a valid wait barrier clears.

    Polls at ``poll_seconds`` intervals and calls ``heartbeat_fn`` on every
    poll so a caller can keep a kanban claim TTL alive while parked. Bounded
    by ``max_wait_seconds`` overall — a barrier that never clears (a pid that
    never dies, a clock that never advances in a broken test) still returns
    instead of hanging the loop forever; the caller resumes normal judging
    one turn early, which is safe.
    """
    pid = directive.get("pid")
    seconds = directive.get("seconds")
    start = monotonic_fn()
    deadline = start + int(seconds) if seconds else None
    ceiling = start + max_wait_seconds
    while True:
        now = monotonic_fn()
        if pid is not None and not _pid_alive(int(pid)):
            return
        if deadline is not None and now >= deadline:
            return
        if now >= ceiling:
            log(
                f"kanban goal loop: wait barrier exceeded max_wait_seconds "
                f"({max_wait_seconds}s); resuming"
            )
            return
        if heartbeat_fn is not None:
            try:
                heartbeat_fn()
            except Exception as exc:
                log(f"kanban goal loop: heartbeat_fn failed while waiting ({exc})")
        sleep_fn(poll_seconds)


def run_kanban_goal_loop(
    *,
    task_id: str,
    goal_text: str,
    run_turn,
    task_status_fn,
    block_fn,
    max_turns: int = DEFAULT_MAX_TURNS,
    first_response: str = "",
    log=None,
    reserved_closeout_turns: int = DEFAULT_RESERVED_CLOSEOUT_TURNS,
    no_progress_limit: int = DEFAULT_NO_PROGRESS_LIMIT,
    max_transient_retries: int = DEFAULT_MAX_TRANSIENT_RETRIES,
    progress_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    emit_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    heartbeat_fn: Optional[Callable[[], None]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    wait_poll_seconds: float = DEFAULT_WAIT_POLL_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> Dict[str, Any]:
    """Drive a kanban worker through a Ralph-style goal loop.

    The dispatcher spawns a goal-mode worker exactly like a normal worker
    (``hermes -p <profile> chat -q "work kanban task <id>"``). The worker's
    first turn has already run by the time this is called; ``first_response``
    is that turn's reply. From here we:

    1. Check whether the worker already terminated the task (called
       ``kanban_complete`` / ``kanban_block``). If so, stop — nothing to do.
    2. Otherwise judge the latest response against ``goal_text`` (the card's
       title + body). ``continue`` → feed a continuation prompt and run
       another turn IN THE SAME SESSION via ``run_turn``. ``done`` but the
       task is still open → one explicit "call kanban_complete" nudge.
       ``wait`` with a concrete pid/seconds directive → park without
       spending a turn until the barrier clears; an invalid directive
       degrades to ``continue`` instead of parking forever.
    3. The last ``reserved_closeout_turns`` of the budget (default 1) are
       set aside for lifecycle-only wrap-up: once ordinary work turns run
       out (or the judge says "done"), the loop stops feeding ordinary
       continuation prompts and instead demands a ``kanban_complete`` /
       ``kanban_block`` call. Ordinary work never gets to spend that turn.
    4. Repeated, indistinguishable turns (same response, same verdict, no
       task/event/workspace/test delta reported via ``progress_fn``) are
       recognized as ``no_progress`` and blocked early — burning the rest
       of the budget on identical prose helps no one.
    5. ``run_turn`` failures are classified transient (retried, bounded) vs
       deterministic (retried once, not blindly). A second occurrence of the
       *same* failure fingerprint escalates straight to triage instead of
       retrying again.
    6. When the turn budget is exhausted and the worker still hasn't
       terminated the task, ``block_fn`` is invoked so the card lands in a
       sticky ``blocked`` state for human review (NOT a silent exit).

    This function performs NO SessionDB persistence — a worker process is
    ephemeral, so the turn budget lives in a local counter. It is fully
    decoupled from the CLI for testability: callers inject ``run_turn``
    (str -> str, may raise ``KanbanTurnError``/``KanbanTransientTurnError``
    or any exception), ``task_status_fn`` (() -> str|None), and ``block_fn``
    (reason: str -> None). ``progress_fn``, ``emit_progress``,
    ``heartbeat_fn``, ``sleep_fn``, and ``monotonic_fn`` are optional
    injection points with safe no-op / real-clock defaults; callers that
    want dispatcher-visible budget/progress telemetry or claim-TTL renewal
    while parked wire them in (see ``cli._run_kanban_goal_loop_q``).

    Returns a decision dict: ``{"outcome", "turns_used", "reason"}`` where
    outcome is one of ``"completed_by_worker"``, ``"blocked_by_worker"``,
    ``"blocked_budget"`` (exhausted before a closeout turn could even be
    attempted — a reserved_closeout_turns/max_turns misconfiguration),
    ``"blocked_no_progress"``, ``"blocked_terminal_step_not_taken"``,
    ``"blocked_resume_closeout"``, ``"blocked_triage"``, or ``"stopped"``.
    """

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    max_turns = int(max_turns or DEFAULT_MAX_TURNS)
    if max_turns < 1:
        max_turns = DEFAULT_MAX_TURNS
    reserved_closeout_turns = int(reserved_closeout_turns or 0)
    if reserved_closeout_turns < 0:
        reserved_closeout_turns = DEFAULT_RESERVED_CLOSEOUT_TURNS
    # Never reserve the entire budget — there must be at least one work turn
    # slot (turn 1, already spent on ``first_response``) to reserve *from*.
    reserved_closeout_turns = min(reserved_closeout_turns, max(0, max_turns - 1))
    work_budget = max_turns - reserved_closeout_turns
    no_progress_limit = max(1, int(no_progress_limit or DEFAULT_NO_PROGRESS_LIMIT))
    max_transient_retries = max(0, int(max_transient_retries or 0))

    last_response = first_response or ""
    # The first turn already consumed one unit of budget.
    turns_used = 1
    # Two independent one-shot/bounded mechanisms feed the same reserved
    # corridor concept but must not contaminate each other:
    #  - ``nudged_to_finalize``: the judge said "done" — always gets exactly
    #    one nudge to actually call kanban_complete/kanban_block, regardless
    #    of the forced-closeout corridor size (this is "free": the judge
    #    already believes the work is finished).
    #  - ``closeout_turns_used``: how many turns of the *forced* closeout
    #    corridor (entered when ordinary work budget runs out while the
    #    judge still says "continue") have been spent, bounded by
    #    ``reserved_closeout_turns``.
    nudged_to_finalize = False
    closeout_turns_used = 0
    no_progress_count = 0
    last_fingerprint: Optional[Tuple[Any, ...]] = None
    last_failure_fingerprint: Optional[str] = None
    transient_retry_count = 0
    total_retry_attempts = 0
    consecutive_parse_failures = 0
    verdict_history: List[str] = []

    def _emit(phase: str, *, reason: str = "", **extra: Any) -> None:
        if emit_progress is None:
            return
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "phase": phase,
            "turns_used": turns_used,
            "max_turns": max_turns,
            "work_budget": work_budget,
            "reserved_closeout_turns": reserved_closeout_turns,
            "closeout_turns_used": closeout_turns_used,
            "nudged_to_finalize": nudged_to_finalize,
            "no_progress_count": no_progress_count,
            "verdict_history": list(verdict_history),
            "reason": _truncate(reason, _GOAL_PROGRESS_REASON_LIMIT),
        }
        payload.update(extra)
        try:
            emit_progress(payload)
        except Exception as exc:
            _log(f"kanban goal loop: emit_progress failed ({exc})")

    def _block(message: str) -> None:
        try:
            block_fn(message)
        except Exception as exc:
            _log(f"kanban goal loop: block_fn failed ({exc})")

    def _has_evidence() -> bool:
        # Conservative default: something was actually produced, or the
        # loop had already reached a closeout/finalize attempt at least
        # once. Callers with real artifact/test-manifest visibility should
        # prefer wiring that signal in via ``progress_fn`` instead — this is
        # the fallback when they don't.
        return bool(last_response.strip()) or nudged_to_finalize or closeout_turns_used > 0

    def _run_turn_with_retries(prompt: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        nonlocal last_failure_fingerprint, transient_retry_count, total_retry_attempts
        while True:
            try:
                return (run_turn(prompt) or ""), None
            except Exception as exc:
                fingerprint = _failure_fingerprint(exc)
                kind = _classify_turn_error(exc)
                _log(
                    f"kanban goal loop: task {task_id} run_turn failed "
                    f"kind={kind} fingerprint={fingerprint}: {exc}"
                )
                # A SECOND occurrence of the exact same failure — same
                # exception class + message — is never blindly retried
                # again, no matter its classification: it needs a human or
                # a genuinely different strategy, not another attempt.
                repeat = (
                    last_failure_fingerprint is not None
                    and fingerprint == last_failure_fingerprint
                )
                last_failure_fingerprint = fingerprint
                _emit(
                    "turn_error",
                    reason=str(exc),
                    failure_kind=kind,
                    failure_fingerprint=fingerprint,
                    repeated_fingerprint=repeat,
                )

                if repeat:
                    _block(
                        f"Goal-mode worker hit the same {kind} failure twice in a "
                        f"row ({type(exc).__name__}: {_truncate(str(exc), 200)}); "
                        f"needs triage/re-scoping rather than another blind retry."
                    )
                    return None, {
                        "outcome": "blocked_triage",
                        "turns_used": turns_used,
                        "reason": f"repeated {kind} failure fingerprint",
                    }

                # A changed fingerprint (different message/class from the
                # immediately preceding failure) is a genuinely new symptom
                # — not a blind retry — so it earns its own attempt. The
                # overall ceiling below is the backstop against a
                # pathological "always slightly different error" loop.
                total_retry_attempts += 1
                if total_retry_attempts > _GOAL_MAX_TOTAL_TURN_RETRIES:
                    evidence = _has_evidence()
                    outcome = "blocked_resume_closeout" if evidence else "blocked_triage"
                    _block(
                        f"Goal-mode worker hit {total_retry_attempts} run_turn "
                        f"failures in a row (latest {type(exc).__name__}); "
                        + (
                            "resume from existing evidence."
                            if evidence
                            else "needs triage — no usable evidence yet."
                        )
                    )
                    return None, {
                        "outcome": outcome,
                        "turns_used": turns_used,
                        "reason": "retry ceiling exhausted",
                    }

                if kind == "transient":
                    transient_retry_count += 1
                    if transient_retry_count > max_transient_retries:
                        evidence = _has_evidence()
                        outcome = "blocked_resume_closeout" if evidence else "blocked_triage"
                        _block(
                            f"Goal-mode worker hit {transient_retry_count} transient "
                            f"failures in a row ({type(exc).__name__}); "
                            + (
                                "resume from existing evidence."
                                if evidence
                                else "needs triage — no usable evidence yet."
                            )
                        )
                        return None, {
                            "outcome": outcome,
                            "turns_used": turns_used,
                            "reason": "transient retries exhausted",
                        }
                    sleep_fn(min(2 ** transient_retry_count, 30))
                    continue

                # Deterministic (or unclassified) failure with a fresh
                # fingerprint: one bounded retry, gated only by the repeat
                # check and the overall ceiling above — never a blind loop
                # on the *same* defect.
                transient_retry_count = 0
                continue

    while True:
        # Did the worker terminate the task itself this turn?
        try:
            status = task_status_fn()
        except Exception as exc:
            _log(f"kanban goal loop: status check failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": "status check failed"}

        if status == "done":
            _log(f"kanban goal loop: task {task_id} completed by worker after {turns_used} turn(s)")
            return {"outcome": "completed_by_worker", "turns_used": turns_used, "reason": "worker completed the task"}
        if status == "blocked":
            _log(f"kanban goal loop: task {task_id} blocked by worker after {turns_used} turn(s)")
            return {"outcome": "blocked_by_worker", "turns_used": turns_used, "reason": "worker blocked the task"}
        if status not in ("running", "ready"):
            # Reclaimed / archived / unexpected — let the dispatcher own it.
            _log(f"kanban goal loop: task {task_id} status={status!r}; stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"status={status}"}

        # Still open — judge whether the latest response satisfies the card.
        # MERGE-ENTSCHEID (Rehearsal 23.07.): Upstreams Kommentar "no
        # wait-barrier concept" beschreibt UPSTREAM, nicht diesen Fork — unser
        # Loop parkt WAIT unten über _park_on_wait_barrier. Deshalb: Upstreams
        # 5-Tupel-Signatur übernommen (transport_failed neu getrennt), aber
        # WAIT NICHT auf CONTINUE gemappt und History-/Parse-Zähler behalten.
        # transport_failed zählt in den Parse-Zähler: vor dem Upstream-Split
        # war "Judge nicht erreichbar" in parse_failed enthalten; der Zähler
        # meint "Judge nicht nutzbar", nicht nur "Antwort unlesbar".
        verdict, reason, parse_failed, wait_directive, transport_failed = judge_goal(goal_text, last_response)
        verdict_history.append(verdict)
        del verdict_history[:-_GOAL_VERDICT_HISTORY_LIMIT]
        consecutive_parse_failures = (
            consecutive_parse_failures + 1 if (parse_failed or transport_failed) else 0
        )
        _log(f"kanban goal loop: turn {turns_used}/{max_turns} verdict={verdict} reason={_truncate(reason, 120)}")

        # ---- WAIT barrier: park without spending a turn -------------------
        if verdict == "wait" and _valid_wait_directive(wait_directive):
            _emit("wait", reason=reason, wait_directive=wait_directive)
            _park_on_wait_barrier(
                wait_directive,
                heartbeat_fn=heartbeat_fn,
                sleep_fn=sleep_fn,
                monotonic_fn=monotonic_fn,
                poll_seconds=wait_poll_seconds,
                max_wait_seconds=max_wait_seconds,
                log=_log,
            )
            _emit("wait_cleared", reason=reason)
            continue
        if verdict == "wait":
            # Judge said WAIT but supplied nothing concrete to park on
            # (or something malformed slipped past judge_goal's own
            # parsing). Fail safe: degrade to a normal continue turn
            # instead of looping forever on an unusable directive.
            _log(f"kanban goal loop: task {task_id} invalid wait directive {wait_directive!r}; degrading to continue")
            verdict = "continue"
            reason = f"invalid wait directive ignored: {reason}"

        # ---- judge model itself is unusable N turns in a row --------------
        if consecutive_parse_failures >= DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES:
            _emit("no_progress", reason="judge output unparseable")
            _block(
                f"Goal judge returned unparseable output "
                f"{consecutive_parse_failures} turns in a row; needs a working "
                f"judge model (auxiliary.goal_judge) or human review."
            )
            return {
                "outcome": "blocked_no_progress",
                "turns_used": turns_used,
                "reason": "judge parse failures exhausted",
            }

        # ---- progress fingerprint / no_progress classification -------------
        # Only meaningful while the judge still says "continue" — a "done"
        # verdict is handled by the closeout branch below regardless of
        # whether the text happens to repeat.
        if verdict != "done":
            extra: Dict[str, Any] = {}
            if progress_fn is not None:
                try:
                    extra = progress_fn() or {}
                except Exception as exc:
                    _log(f"kanban goal loop: progress_fn failed ({exc})")
                    extra = {}
            fingerprint = (
                _response_fingerprint(last_response),
                extra.get("task_event_count"),
                extra.get("workspace_fingerprint"),
                extra.get("test_manifest_fingerprint"),
                verdict,
                _truncate(reason, 120),
            )
            if fingerprint == last_fingerprint:
                no_progress_count += 1
            else:
                no_progress_count = 0
            last_fingerprint = fingerprint

            if no_progress_count >= no_progress_limit:
                _emit("no_progress", reason=reason)
                _block(
                    f"Goal-mode worker produced {no_progress_count + 1} turns in a "
                    f"row with no detectable state/artifact/test progress (last "
                    f"judge reason: {_truncate(reason, 200)}). Needs human review "
                    f"or re-scoping rather than more budget."
                )
                return {"outcome": "blocked_no_progress", "turns_used": turns_used, "reason": "no_progress"}

        # ---- decide the next prompt: ordinary work, closeout, or terminal --
        # ``closeout_kind`` marks what this iteration's prompt is, so the
        # right counter gets bumped below AFTER the hard-budget check
        # passes (never before — an attempt that never actually ran must
        # not count as "spent").
        closeout_kind: Optional[str] = None
        if verdict == "done":
            if nudged_to_finalize:
                _emit("terminal_step_not_taken", reason=reason)
                _block(
                    f"Goal-mode worker's output looked complete but it never "
                    f"called kanban_complete/kanban_block after a finalize nudge "
                    f"({reason})."
                )
                return {
                    "outcome": "blocked_terminal_step_not_taken",
                    "turns_used": turns_used,
                    "reason": "judged done, lifecycle call never made",
                }
            closeout_kind = "finalize"
            prompt = KANBAN_GOAL_FINALIZE_TEMPLATE.format(reason=_truncate(reason, 400))
        elif reserved_closeout_turns > 0 and turns_used >= work_budget:
            if closeout_turns_used >= reserved_closeout_turns:
                _emit("terminal_step_not_taken", reason=reason)
                _block(
                    f"Goal-mode worker exhausted its reserved closeout corridor "
                    f"({closeout_turns_used}/{reserved_closeout_turns} turns) "
                    f"without calling kanban_complete/kanban_block ({reason})."
                )
                return {
                    "outcome": "blocked_terminal_step_not_taken",
                    "turns_used": turns_used,
                    "reason": "closeout corridor exhausted",
                }
            closeout_kind = "forced"
            prompt = KANBAN_GOAL_CLOSEOUT_TEMPLATE.format(reason=_truncate(reason, 400))
        else:
            prompt = KANBAN_GOAL_CONTINUATION_TEMPLATE.format(reason=_truncate(reason, 400))
            if reserved_closeout_turns > 0:
                remaining = work_budget - turns_used
                if remaining <= max(1, work_budget // 4):
                    prompt += KANBAN_GOAL_SOFT_LIMIT_SUFFIX.format(
                        used=turns_used, max=max_turns, remaining=remaining
                    )
                    _emit("soft_limit", reason=reason, remaining_work_turns=remaining)
                else:
                    _emit("continue", reason=reason)
            else:
                _emit("continue", reason=reason)

        # ---- hard budget check BEFORE spending the turn --------------------
        if turns_used >= max_turns:
            already_attempted_closeout = nudged_to_finalize or closeout_turns_used > 0
            if already_attempted_closeout:
                # A closeout/finalize attempt already ran in a prior
                # iteration and the task is still open — terminal.
                _emit("terminal_step_not_taken", reason=reason)
                outcome = "blocked_terminal_step_not_taken"
                detail = "turn budget exhausted after a closeout/finalize attempt"
            else:
                # Never even got to attempt a closeout/finalize turn — a
                # max_turns/reserved_closeout_turns misconfiguration (or a
                # max_turns=1 card), not the worker's fault.
                _emit("blocked_budget", reason=reason)
                outcome = "blocked_budget"
                detail = "turn budget exhausted before any closeout attempt could be made"
            _block(
                f"Goal-mode worker exhausted its turn budget "
                f"({turns_used}/{max_turns}) without completing the task. "
                f"Last judge verdict: {_truncate(reason, 300)}"
            )
            return {"outcome": outcome, "turns_used": turns_used, "reason": detail}

        if closeout_kind == "finalize":
            nudged_to_finalize = True
            _emit("closeout", reason=reason, closeout_kind="finalize")
        elif closeout_kind == "forced":
            closeout_turns_used += 1
            _emit("closeout", reason=reason, closeout_kind="forced")

        # Run another turn in the same session, with classified retries.
        new_response, failure_result = _run_turn_with_retries(prompt)
        if failure_result is not None:
            return failure_result
        last_response = new_response or ""
        turns_used += 1


__all__ = [
    "GoalState",
    "GoalContract",
    "GoalManager",
    "parse_contract",
    "draft_contract",
    "CONTINUATION_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE",
    "JUDGE_USER_PROMPT_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE",
    "DRAFT_CONTRACT_SYSTEM_PROMPT",
    "KANBAN_GOAL_CONTINUATION_TEMPLATE",
    "KANBAN_GOAL_SOFT_LIMIT_SUFFIX",
    "KANBAN_GOAL_FINALIZE_TEMPLATE",
    "KANBAN_GOAL_CLOSEOUT_TEMPLATE",
    "KanbanTurnError",
    "KanbanTransientTurnError",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_RESERVED_CLOSEOUT_TURNS",
    "DEFAULT_NO_PROGRESS_LIMIT",
    "DEFAULT_MAX_TRANSIENT_RETRIES",
    "DEFAULT_WAIT_POLL_SECONDS",
    "DEFAULT_MAX_WAIT_SECONDS",
    "load_goal",
    "save_goal",
    "clear_goal",
    "migrate_goal_to_session",
    "judge_goal",
    "run_kanban_goal_loop",
]

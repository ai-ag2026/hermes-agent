"""SQLite-backed Kanban board for multi-profile, multi-project collaboration.

In a fresh install the board lives at ``<root>/kanban.db`` where
``<root>`` is the **shared Hermes root** (the parent of any active
profile). Profiles intentionally collapse onto a shared board: it IS
the cross-profile coordination primitive. A worker spawned with
``hermes -p <profile>`` joins the same board as the dispatcher that
claimed the task. The same applies to ``<root>/kanban/workspaces/`` and
``<root>/kanban/logs/``.

**Multiple boards (projects):** users can create additional boards to
separate unrelated streams of work (e.g. one per project / repo / domain).
Each board is a directory under ``<root>/kanban/boards/<slug>/`` with
its own ``kanban.db``, ``workspaces/``, and ``logs/``. All boards share
the profile's Hermes home but are otherwise isolated: a worker spawned
for a task on board ``atm10-server`` sees only that board's tasks,
cannot enumerate other boards, and its dispatcher ticks don't touch
other boards' DBs.

The first (and for single-project users, only) board is ``default``.
For back-compat its on-disk DB is ``<root>/kanban.db`` (not
``boards/default/kanban.db``), so installs that predate the boards
feature keep working with zero migration. See :func:`kanban_db_path`.

Board resolution order (highest precedence first, all optional):

* ``board=`` argument passed directly to :func:`connect` / :func:`init_db`
  (explicit — used by the CLI ``--board`` flag and the dashboard
  ``?board=...`` query param).
* ``HERMES_KANBAN_BOARD`` env var (used by the dispatcher to pin workers
  to the board their task lives on — workers cannot see other boards).
* ``HERMES_KANBAN_DB`` env var (pins the DB file path directly — legacy
  override still honoured; highest precedence when the file path itself
  is what the caller wants to force).
* ``<root>/kanban/current`` — a one-line text file holding the slug of
  the "currently selected" board. Written by ``hermes kanban boards
  switch <slug>``. When absent, the active board is ``default``.

In standard installs ``<root>`` is ``~/.hermes``. In Docker / custom
deployments where ``HERMES_HOME`` points outside ``~/.hermes`` (e.g.
``/opt/hermes``), ``<root>`` is ``HERMES_HOME``. Legacy env-var
overrides still work:

* ``HERMES_KANBAN_DB`` — pin the database file path directly.
* ``HERMES_KANBAN_WORKSPACES_ROOT`` — pin the workspaces root directly.
* ``HERMES_KANBAN_HOME`` — pin the umbrella root that anchors kanban
  paths. Useful for tests and unusual deployments.

The dispatcher injects ``HERMES_KANBAN_DB``,
``HERMES_KANBAN_WORKSPACES_ROOT``, and ``HERMES_KANBAN_BOARD`` into
worker subprocess env so workers converge on the exact DB the
dispatcher used to claim their task — even under unusual symlink or
Docker layouts.

Schema is intentionally small: tasks, task_links, task_comments,
task_events.  The ``workspace_kind`` field decouples coordination from git
worktrees so that research / ops / digital-twin workloads work alongside
coding workloads.  See ``docs/hermes-kanban-v1-spec.pdf`` for the full
design specification.

Concurrency strategy: WAL mode + ``BEGIN IMMEDIATE`` for write
transactions + compare-and-swap (CAS) updates on ``tasks.status`` and
``tasks.claim_lock``.  SQLite serializes writers via its WAL lock, so at
most one claimer can win any given task.  Losers observe zero affected
rows and move on -- no retry loops, no distributed-lock machinery.
The CAS coordination is **per-board** — each board is a separate DB,
so multi-board installs get the same atomicity guarantees without any
new locking.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import mimetypes
import os
import re
import random
import secrets
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import logging
import time
import urllib.parse
import urllib.request
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing
from toolsets import get_toolset_names

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"}
VALID_INITIAL_STATUSES = {"running", "blocked"}

# Typed block reasons. Distinguishes the two fundamentally different things a
# worker (or human) means by "blocked", so each can be routed differently
# instead of all landing in one undifferentiated ``blocked`` bucket that a cron
# unblocks → worker re-blocks → cron unblocks … forever.
#
#   * ``dependency``   — can't proceed until another task finishes. Routed to
#                        ``todo`` (NOT ``blocked``) so the existing
#                        parent-gating / ``recompute_ready`` machinery promotes
#                        it automatically once parents are done. No human, no
#                        cron, no retry storm.
#   * ``needs_input``  — needs a human decision/answer it cannot derive.
#   * ``capability``   — hit a hard wall (no access, missing creds, an action no
#                        AI agent can perform). Genuinely human-only.
#   * ``transient``    — a flaky/temporary failure that may clear on retry.
#
# ``needs_input`` and ``capability`` are "truly blocked": they go to ``blocked``
# for a human, and the unblock-loop breaker (see ``block_task`` /
# ``BLOCK_RECURRENCE_LIMIT``) escalates them to ``triage`` if a cron keeps
# unblocking them only to have the worker re-block for the same reason.
# ``None`` = legacy/un-typed block (treated as a generic human blocker).
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}

# Cause identities are deliberately a separate, small vocabulary from routing
# ``block_kind``.  Never derive these from prose: raw reasons commonly contain
# credentials, paths, or model text and are not a safe durable identity.
VALID_BLOCK_ATTENTION_TYPES = frozenset({"decision", "capability", "transient", "protocol", "review", "loop_triage"})
VALID_BLOCK_REASON_CODES = frozenset({
    "credential_choice", "publication_approval", "missing_capability",
    "external_transient", "goal_closeout_missing", "review_required",
})
BLOCK_CAUSE_VERSION = 1
# Typed causes may use only a tiny product vocabulary. Scope is identity
# metadata, never an operator message: accepting arbitrary values here would
# merely move a secret/path leak into an opaque fingerprint input.
BLOCK_CAUSE_SCOPE_ENUMS: dict[tuple[str, str], dict[str, frozenset[str]]] = {
    ("decision", "credential_choice"): {"required_decision": frozenset({"credential"})},
    ("decision", "publication_approval"): {"required_decision": frozenset({"publication"})},
    ("capability", "missing_capability"): {"capability": frozenset({"access", "credential", "tool"})},
    ("transient", "external_transient"): {"subject": frozenset({"external_service"})},
    ("protocol", "goal_closeout_missing"): {"protocol": frozenset({"goal_closeout"})},
    ("review", "review_required"): {"subject": frozenset({"review"})},
    ("loop_triage", "review_required"): {"subject": frozenset({"loop"})},
}

# Goal-mode tasks may only block with kinds that represent a genuine external
# blocker the worker cannot resolve itself; everything else must route through
# complete (where the goal judge gates). Canonical here (audit 2026-07-11,
# Meta-Muster E: this set previously lived only in tools/kanban_tools.py).
GOAL_MODE_BLOCK_ALLOWED_KINDS = frozenset({"dependency", "needs_input"})

# After a task has been blocked, unblocked, and re-blocked this many times for
# the same (truly-blocked) reason, the unblock-loop breaker stops trusting the
# unblocker (usually a cron) and routes the task to ``triage`` instead of back
# to ``blocked`` — breaking the infinite unblock↔re-block loop and forcing a
# human-in-the-loop decision. Mirrors the dispatcher's ``DEFAULT_FAILURE_LIMIT``
# spirit (default 2) but counts a different signal: manual unblock recurrences,
# not dispatcher spawn/crash/timeout failures.
BLOCK_RECURRENCE_LIMIT = 2
PENDING_ACTION_OPERATOR_SUMMARY = "An exact terminal action is awaiting approval."
VALID_WORKSPACE_KINDS = {"scratch", "worktree", "dir"}
KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())
_IS_WINDOWS = sys.platform == "win32"


def _fire_kanban_lifecycle_hook(event: str, task_id: str, **fields: Any) -> None:
    """Fire a kanban lifecycle plugin hook, fully best-effort.

    Called by the claim/complete/block transitions AFTER their write txn has
    committed, so plugin code never runs while a SQLite write lock is held and
    always observes durable board state. Any failure (plugins unavailable,
    a plugin raising, import error) is swallowed — a misbehaving observer must
    never break a board state transition.

    ``profile_name`` is resolved from the active HERMES_HOME so dispatcher- and
    worker-side hooks both carry the right profile without the caller plumbing
    it through.
    """
    try:
        from hermes_cli.plugins import invoke_hook
        from hermes_cli.profiles import get_active_profile_name
        try:
            profile_name = get_active_profile_name()
        except Exception:
            profile_name = "default"
        invoke_hook(event, task_id=task_id, profile_name=profile_name, **fields)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban lifecycle hook %s failed: %s", event, exc)


# A running task's claim is valid for 15 minutes by default; after that the
# next dispatcher tick reclaims it. Workers that outlive this window should
# call ``heartbeat_claim(task_id)`` periodically. In practice most kanban
# workloads either finish within 15m, set a longer claim explicitly, or use
# ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` to raise the default claim window for
# long single-call MCP workflows.
DEFAULT_CLAIM_TTL_SECONDS = 15 * 60

# If a worker's PID is still alive but its ``last_heartbeat_at`` is
# older than this when ``release_stale_claims`` runs, treat the worker
# as wedged and reclaim regardless of PID liveness (#29747 gap 3).
# This catches the logic-loop case where the process is technically
# running but not making observable progress.  ``_touch_activity``
# bridges chunk-level liveness into ``last_heartbeat_at`` via #31752,
# so any genuinely active worker keeps its heartbeat fresh as a side
# effect of normal API traffic.
DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS = 60 * 60

# Grace added to a claim when a reclaim is deferred because the previous
# host-local worker is still alive after a termination attempt. Releasing the
# claim in that state would spawn a duplicate alongside the surviving worker —
# the runaway seen when a cgroup memory.high throttle parks a worker in
# uninterruptible (D) state, where a pending SIGKILL cannot be delivered until
# the throttle lifts. Holding the claim a short grace and retrying next tick
# stops the duplication; once no duplicate is spawned the pressure eases, the
# signal lands, and the following tick reclaims cleanly.
RECLAIM_DEFER_GRACE_SECONDS = 120


def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Return the effective claim TTL, honoring the kanban env override.

    Explicit call-site values win. Otherwise a positive integer from
    ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` overrides the built-in default.
    Invalid or non-positive env values fall back silently so existing
    installs keep working.
    """
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    raw = os.environ.get("HERMES_KANBAN_CLAIM_TTL_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    return DEFAULT_CLAIM_TTL_SECONDS


# Grace period after a task transitions to ``running`` during which
# ``detect_crashed_workers`` skips the ``_pid_alive`` check. Covers the
# fork() → /proc-visibility window where liveness can transiently report
# False for a freshly-spawned worker. The 15-minute claim TTL still
# catches genuinely-crashed workers; this only suppresses false positives
# during the launch window.
DEFAULT_CRASH_GRACE_SECONDS = 30


# Sentinel exit code a kanban worker uses to signal "I bailed because the
# provider rate-limited / exhausted quota, not because the task failed."
# The dispatcher's reap classifier maps this to a ``rate_limited`` exit kind
# so ``detect_crashed_workers`` can release the task back to ``ready``
# WITHOUT counting a failure (the circuit breaker must never trip on a
# transient throttle). 75 == BSD ``EX_TEMPFAIL`` (sysexits.h) — the
# conventional "temporary failure, retry later" code, and well clear of the
# 0/1/2 codes the worker uses for success / generic failure / usage error.
KANBAN_RATE_LIMIT_EXIT_CODE = 75


def _resolve_crash_grace_seconds() -> int:
    """Return the crash-detection grace period in seconds.

    Reads ``HERMES_KANBAN_CRASH_GRACE_SECONDS`` from the environment;
    falls back to ``DEFAULT_CRASH_GRACE_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 restores immediate-reclaim
    behaviour (useful for tests).
    """
    raw = os.environ.get("HERMES_KANBAN_CRASH_GRACE_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_CRASH_GRACE_SECONDS


def _resolve_rate_limit_cooldown_seconds() -> int:
    """Return the rate-limit requeue cooldown in seconds.

    Reads ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` from the environment;
    falls back to ``DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 disables the cooldown (re-spawn on
    the next tick) — useful for tests that want to assert the task becomes
    spawnable again immediately.
    """
    raw = os.environ.get(
        "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", ""
    ).strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


def _resolve_resume_on_reclaim() -> bool:
    """Whether a re-claimed (crashed/stale) task RESUMES its previous worker
    session instead of starting fresh (event-sourcing resume, #2, opt-in default OFF).

    Read from ``kanban.resume_on_reclaim`` in config, overridable via the
    ``HERMES_KANBAN_RESUME_ON_RECLAIM`` env (``1``/``0``) for tests / dispatcher
    export. Default OFF = the long-standing fresh-start-on-reclaim behaviour, so
    no existing install changes behaviour until a board operator opts in.
    """
    raw = os.environ.get("HERMES_KANBAN_RESUME_ON_RECLAIM", "").strip()
    if raw:
        return raw not in ("0", "false", "no", "off", "")
    try:
        from hermes_cli.config import load_config

        return bool((load_config().get("kanban") or {}).get("resume_on_reclaim", False))
    except Exception:
        return False


def _resolve_resume_max_attempts() -> int:
    """Number of resume attempts allowed per worker session before a re-claim
    throws the (potentially poisoned) session away and starts fresh (#2). Default 1
    = resume once after the first crash; a second crash of that session goes fresh.

    Read from ``kanban.resume_max_attempts`` (env override
    ``HERMES_KANBAN_RESUME_MAX_ATTEMPTS``). Kept strictly below the
    ``consecutive_failures`` breaker so a task can never resume-loop forever: once
    the breaker trips the task blocks (needs_input) and is no longer dispatched.
    A value < 1 disables resume entirely (always fresh).
    """
    raw = os.environ.get("HERMES_KANBAN_RESUME_MAX_ATTEMPTS", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return 1
    try:
        from hermes_cli.config import load_config

        val = (load_config().get("kanban") or {}).get("resume_max_attempts", 1)
        return int(val)
    except Exception:
        return 1


def _resolve_worktree_auto_cleanup() -> bool:
    """Whether to auto-reap a task's git worktree on completion (opt-in, default OFF).

    Landscape-research #4 (per-task ephemeral worktree + auto-cleanup, à la Vibe
    Kanban). Read from ``kanban.worktree_auto_cleanup`` in config, overridable via
    the ``HERMES_KANBAN_WORKTREE_AUTO_CLEANUP`` env (``1``/``0``) for tests. Runs in
    the worker process (via complete_task), so config is read directly rather than
    relying on dispatcher env exports. Default OFF keeps behaviour identical to the
    long-standing "worktrees are intentionally preserved" contract until a board
    operator opts in. Safety-gating still applies even when enabled.
    """
    raw = os.environ.get("HERMES_KANBAN_WORKTREE_AUTO_CLEANUP", "").strip()
    if raw:
        return raw not in ("0", "false", "no", "off", "")
    try:
        from hermes_cli.config import load_config

        return bool((load_config().get("kanban") or {}).get("worktree_auto_cleanup", False))
    except Exception:
        return False


# Worker-context caps so build_worker_context() stays bounded on
# pathological boards (retry-heavy tasks, comment storms, giant
# summaries). Values chosen to fit a typical 100k-char LLM prompt with
# plenty of headroom. Each constant is tuned independently so users
# who need to relax one don't have to relax all of them.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # 4 KB per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # 8 KB per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # 2 KB per comment


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """Render the age of an epoch-seconds timestamp as a coarse, human-
    readable string like ``just now``, ``18h ago``, ``3d ago``.

    Workers read parent handoffs, comments, and prior-attempt summaries as
    if they describe *current* state. A bare absolute timestamp
    (``2026-06-25 14:30``) doesn't make an LLM reason about staleness — it
    reads the content as fact regardless of how old it is. A relative age
    ("18h ago") is the signal that prompts the worker to re-verify against
    the live source before acting on stale sibling work. Returns an empty
    string for missing/invalid timestamps so callers can append
    unconditionally.
    """
    if ts is None:
        return ""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 0:
        # Clock skew across machines/profiles — don't claim "in the future".
        return "just now"
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = delta // 60
        return f"{m}m ago"
    if delta < 86400:
        h = delta // 3600
        return f"{h}h ago"
    d = delta // 86400
    return f"{d}d ago"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_BOARD = "default"
_CURRENT_BOARD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_current_board_override",
    default=None,
)


@contextlib.contextmanager
def scoped_current_board(slug: str):
    """Temporarily pin the active board for the current context only."""
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(slug)
    try:
        yield
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)

# Slug validator: lowercase alphanumerics, digits, hyphens; 1–64 chars.
# Strict enough to stop traversal (`..`) and embedded path separators, loose
# enough that kebab-case names like ``atm10-server`` or ``hermes-agent``
# pass without fuss. Board names with display formatting (spaces, emoji)
# live in ``board.json``; the slug is just the directory name.
_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    if slug is None:
        return None
    s = str(slug).strip().lower()
    if not s:
        return None
    if not _BOARD_SLUG_RE.match(s):
        raise ValueError(
            f"invalid board slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )
    return s


def kanban_home() -> Path:
    """Return the shared Hermes root that anchors the kanban board.

    Resolution order:

    1. ``HERMES_KANBAN_HOME`` env var when set and non-empty (explicit
       override for tests and unusual deployments).
    2. ``get_default_hermes_root()``, which already returns ``<root>``
       when ``HERMES_HOME`` is ``<root>/profiles/<name>``, and returns
       ``HERMES_HOME`` directly for Docker / custom deployments.

    The kanban board is shared across profiles **by design** (see the
    module docstring). Resolving the kanban paths through the active
    profile's ``HERMES_HOME`` would silently fork the board per profile,
    which breaks the dispatcher / worker handoff.
    """
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def boards_root() -> Path:
    """Return ``<root>/kanban/boards`` — the parent of non-default board dirs.

    ``default`` is intentionally NOT under this directory — its DB lives at
    ``<root>/kanban.db`` for back-compat with pre-boards installs. This
    function returns the directory where *additional* named boards live,
    used by :func:`list_boards` to enumerate them.
    """
    return kanban_home() / "kanban" / "boards"


def current_board_path() -> Path:
    """Return the path to ``<root>/kanban/current``.

    One-line text file written by ``hermes kanban boards switch <slug>``
    to persist the user's board selection across CLI invocations. Absent
    by default (meaning: active board is ``default``).
    """
    return kanban_home() / "kanban" / "current"


def get_current_board() -> str:
    """Return the active board slug, honouring the resolution chain.

    Order (highest precedence first):

    1. ``HERMES_KANBAN_BOARD`` env var (set by the dispatcher on worker
       spawn, or manually for ad-hoc overrides).
    2. ``<root>/kanban/current`` on disk (set by ``hermes kanban boards
       switch``), but only when that board still exists.
    3. ``DEFAULT_BOARD`` (``"default"``).

    A malformed or stale slug at any step falls through to the next layer
    with a best-effort warning — the dispatcher must never crash because a
    user hand-edited a file or removed a board directory.
    """
    scoped = (_CURRENT_BOARD_OVERRIDE.get() or "").strip()
    if scoped:
        try:
            normed = _normalize_board_slug(scoped)
            if normed and board_exists(normed):
                return normed
        except ValueError:
            pass

    env = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    if env:
        try:
            normed = _normalize_board_slug(env)
            if normed and board_exists(normed):
                return normed
        except ValueError:
            pass
    try:
        f = current_board_path()
        if f.exists():
            val = f.read_text(encoding="utf-8").strip()
            if val:
                try:
                    normed = _normalize_board_slug(val)
                    if normed and board_exists(normed):
                        return normed
                except ValueError:
                    pass
    except OSError:
        pass
    return DEFAULT_BOARD


def set_current_board(slug: str) -> Path:
    """Persist ``slug`` as the active board. Returns the file written.

    Writes ``<root>/kanban/current``. The caller should validate the slug
    exists first (via :func:`board_exists`) — this function does not —
    so that ``hermes kanban boards switch <typo>`` returns an error
    instead of silently pointing at nothing.
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    path = current_board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normed + "\n", encoding="utf-8")
    return path


def clear_current_board() -> None:
    """Remove ``<root>/kanban/current`` so the active board reverts to ``default``."""
    try:
        current_board_path().unlink()
    except FileNotFoundError:
        pass


def board_dir(board: Optional[str] = None) -> Path:
    """Return the on-disk directory for ``board``.

    ``default`` is ``<root>/kanban/boards/default/`` **for metadata only**
    (board.json + workspaces/ + logs/). Its DB file stays at
    ``<root>/kanban.db`` for back-compat — see :func:`kanban_db_path`.

    All other boards live at ``<root>/kanban/boards/<slug>/`` with
    everything inside that directory including the ``kanban.db``.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return boards_root() / slug


def board_exists(board: Optional[str] = None) -> bool:
    """Return True if the board has persisted metadata or a DB on disk.

    ``default`` is considered to always exist — its DB is created
    on first :func:`connect` and there's no way for it to be missing
    in a configuration where the kanban feature is usable at all.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    if slug == DEFAULT_BOARD:
        return True
    d = board_dir(slug)
    return (d / "board.json").exists() or (d / "kanban.db").exists()


def kanban_db_path(board: Optional[str] = None) -> Path:
    """Return the path to the ``kanban.db`` for ``board``.

    Resolution (highest precedence first):

    1. ``HERMES_KANBAN_DB`` env var — pins the path directly. Honoured for
       back-compat and for the dispatcher→worker handoff (defense in
       depth: dispatcher injects this into worker env so workers are
       immune to any path-resolution disagreement).
    2. When ``board`` arg is None, the active board from
       :func:`get_current_board` is used.
    3. Board ``default`` → ``<root>/kanban.db`` (back-compat path).
       Other boards → ``<root>/kanban/boards/<slug>/kanban.db``.
    """
    override = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban.db"
    return board_dir(slug) / "kanban.db"


def workspaces_root(board: Optional[str] = None) -> Path:
    """Return the directory under which ``scratch`` workspaces are created.

    Anchored per-board so workspaces don't leak between projects.
    ``HERMES_KANBAN_WORKSPACES_ROOT`` pins the path directly (highest
    precedence) — the dispatcher injects this into worker env.

    ``default`` keeps the legacy path ``<root>/kanban/workspaces/`` so
    that existing scratch workspaces from before the boards feature are
    preserved. Other boards use ``<root>/kanban/boards/<slug>/workspaces/``.
    """
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "workspaces"
    return board_dir(slug) / "workspaces"


def attachments_root(board: Optional[str] = None) -> Path:
    """Return the directory under which task file attachments are stored.

    Mirrors :func:`worker_logs_dir` / :func:`workspaces_root`: anchored
    per-board so attachments don't leak between projects. Each task gets
    its own ``<root>/.../attachments/<task_id>/`` subdirectory.

    ``HERMES_KANBAN_ATTACHMENTS_ROOT`` pins the path directly (highest
    precedence) for tests and unusual deployments.

    ``default`` uses ``<root>/kanban/attachments/``; other boards use
    ``<root>/kanban/boards/<slug>/attachments/``.

    Workers (which run with full file-tool access) read attached files
    by the absolute path surfaced in :func:`build_worker_context`. On the
    local terminal backend — the default for kanban — that path resolves
    directly. Remote backends (Docker/Modal) need this directory mounted;
    see the kanban docs.
    """
    override = os.environ.get("HERMES_KANBAN_ATTACHMENTS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "attachments"
    return board_dir(slug) / "attachments"


def completion_artifacts_root(board: Optional[str] = None) -> Path:
    """Return the board-scoped durable store for completion evidence."""
    override = os.environ.get("HERMES_KANBAN_ARTIFACTS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "artifacts"
    return board_dir(slug) / "artifacts"


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Return the directory under which per-task worker logs are written.

    ``default`` keeps the legacy path ``<root>/kanban/logs/``. Other
    boards use ``<root>/kanban/boards/<slug>/logs/``. Logs follow the
    board — makes ``hermes kanban log`` unambiguous even when multiple
    boards have tasks with the same id.
    """
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "logs"
    return board_dir(slug) / "logs"


def board_metadata_path(board: Optional[str] = None) -> Path:
    """Return the path to ``board.json`` for ``board``.

    Stores display metadata (display name, description, icon, color,
    created_at). The on-disk slug is the canonical identity; this file
    is purely for presentation in the CLI / dashboard.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return board_dir(slug) / "board.json"


def _default_board_display_name(slug: str) -> str:
    """Turn a slug into a reasonable default display name.

    ``atm10-server`` → ``Atm10 Server``. Users can override via
    ``board.json`` but the default should look presentable in the
    dashboard without any follow-up editing.
    """
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part) or slug


def read_board_metadata(board: Optional[str] = None) -> dict:
    """Return ``board.json`` contents (or synthesized defaults).

    Never raises — a missing / malformed ``board.json`` falls back to a
    synthesised entry so the dashboard always has something to render.
    Includes the canonical ``slug`` and ``db_path`` so the caller
    doesn't need to reconstruct them.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta: dict[str, Any] = {
        "slug": slug,
        "name": _default_board_display_name(slug),
        "description": "",
        "icon": "",
        "color": "",
        "default_workdir": None,
        "created_at": None,
        "archived": False,
    }
    try:
        p = board_metadata_path(slug)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Never let the metadata file claim a different slug than
                # its directory — trust the filesystem.
                raw["slug"] = slug
                meta.update(raw)
    except (OSError, json.JSONDecodeError):
        pass
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def write_board_metadata(
    board: Optional[str],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    archived: Optional[bool] = None,
    default_workdir: Optional[str] = None,
) -> dict:
    """Create / update ``board.json`` for ``board``.

    Preserves any existing fields not mentioned in the call. Sets
    ``created_at`` on first write. Returns the resulting metadata dict.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta = read_board_metadata(slug)
    # Preserve existing DB-derived fields — they get re-computed each
    # read but shouldn't be written into board.json.
    meta.pop("db_path", None)
    if name is not None:
        meta["name"] = str(name).strip() or _default_board_display_name(slug)
    if description is not None:
        meta["description"] = str(description)
    if icon is not None:
        meta["icon"] = str(icon)
    if color is not None:
        meta["color"] = str(color)
    if archived is not None:
        meta["archived"] = bool(archived)
    if default_workdir is not None:
        meta["default_workdir"] = str(default_workdir) if default_workdir else None
    if not meta.get("created_at"):
        meta["created_at"] = int(time.time())
    path = board_metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def create_board(
    slug: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    default_workdir: Optional[str] = None,
) -> dict:
    """Create a new board directory + DB + metadata. Idempotent.

    Returns the resulting metadata. Raises :class:`ValueError` for a
    malformed slug; returns the existing metadata (not an error) if the
    board already exists — matching ``mkdir -p`` semantics.
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    meta = write_board_metadata(
        normed,
        name=name,
        description=description,
        icon=icon,
        color=color,
        default_workdir=default_workdir,
    )
    # Touch the DB so list_boards() sees it immediately.
    init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Enumerate all boards that exist on disk.

    Always includes ``default`` (even when the ``boards/default/``
    metadata dir doesn't exist, because its DB is at the legacy path).
    Other boards are discovered by scanning ``boards/`` for subdirectories
    that either contain a ``kanban.db`` or a ``board.json``.

    Returns a list of metadata dicts, sorted with ``default`` first and
    the rest alphabetically.
    """
    entries: list[dict] = []
    seen: set[str] = set()

    # Default board is always first.
    entries.append(read_board_metadata(DEFAULT_BOARD))
    seen.add(DEFAULT_BOARD)

    root = boards_root()
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            slug = child.name
            # Keep slug normalisation soft for discovery — but skip dirs
            # that don't parse as valid slugs so we don't surface junk.
            try:
                normed = _normalize_board_slug(slug)
            except ValueError:
                continue
            if not normed or normed in seen:
                continue
            has_db = (child / "kanban.db").exists()
            has_meta = (child / "board.json").exists()
            if not (has_db or has_meta):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


def remove_board(slug: str, *, archive: bool = True) -> dict:
    """Remove or archive a board.

    ``archive=True`` (default) moves the board's directory to
    ``<root>/kanban/boards/_archived/<slug>-<timestamp>/`` so the data
    is recoverable. ``archive=False`` deletes the directory outright.

    The ``default`` board cannot be removed — raises :class:`ValueError`.
    Returns a summary dict describing what happened (``{"slug", "action",
    "new_path"}``).
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    if normed == DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be removed")
    d = board_dir(normed)
    if not d.exists():
        raise ValueError(f"board {normed!r} does not exist")

    # If the user removed the currently-active board, revert to default.
    if get_current_board() == normed:
        clear_current_board()

    # A concurrent connect(board=normed) after the rename/delete recreates
    # an empty sqlite file via mkdir(exist_ok=True); the cache entry must be
    # dropped first so the schema init pass re-runs on that fresh file.
    _INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))

    if archive:
        archive_root = boards_root() / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        target = archive_root / f"{normed}-{ts}"
        # Avoid collision on rapid double-archives.
        suffix = 1
        while target.exists():
            target = archive_root / f"{normed}-{ts}-{suffix}"
            suffix += 1
        d.rename(target)
        return {"slug": normed, "action": "archived", "new_path": str(target)}
    else:
        import shutil
        shutil.rmtree(d)
        return {"slug": normed, "action": "deleted", "new_path": ""}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """In-memory view of a row from the ``tasks`` table."""

    id: str
    title: str
    body: Optional[str]
    assignee: Optional[str]
    status: str
    priority: int
    created_by: Optional[str]
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workspace_kind: str
    workspace_path: Optional[str]
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    tenant: Optional[str]
    branch_name: Optional[str] = None
    project_id: Optional[str] = None
    result: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Unified non-success counter. Incremented on any of:
    #   * spawn failure (dispatcher couldn't launch the worker)
    #   * timed_out outcome (worker exceeded max_runtime_seconds)
    #   * crashed outcome (worker PID vanished)
    # Reset to 0 only on a successful completion. See
    # ``_record_task_failure`` for the circuit-breaker trip rule.
    # (Pre-rename column: ``spawn_failures``.)
    consecutive_failures: int = 0
    worker_pid: Optional[int] = None
    # Short excerpt of the last failure's error text (any outcome, not
    # just spawn). Pre-rename column: ``last_spawn_error``.
    last_failure_error: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[int] = None
    current_run_id: Optional[int] = None
    workflow_template_id: Optional[str] = None
    current_step_key: Optional[str] = None
    # Force-loaded skills for the worker on this task (passed via
    # --skills). Stored as a JSON array of skill names. None = use only
    # the defaults; empty list = explicitly no extra skills.
    skills: Optional[list] = None
    model_override: Optional[str] = None
    # Optional quality class / task grade (e.g. "hard"). Free-form label
    # set manually (``--class`` on ``kanban create``/``edit``) or by a
    # plugin. Consumed by the decomposer to pick a class-specific planner
    # auxiliary role (see kanban_decompose.py). ``None`` = unclassified =
    # today's behaviour. Deliberately fully wired through ``from_row`` below.
    task_class: Optional[str] = None
    # Per-task reasoning-effort override (one of VALID_REASONING_EFFORTS, e.g.
    # "low"/"high"/"xhigh"). Written by the tars-workflow plugin for chained
    # goal_mode cards so cheap phases run on low effort and hard verify/judge
    # phases run high. When set, the dispatcher exports HERMES_REASONING_EFFORT
    # into the worker env; the worker's config loader lets that env value take
    # precedence over the profile's ``agent.reasoning_effort``. ``None`` (the
    # common case) = use the profile default = today's behaviour.
    effort: Optional[str] = None
    # Per-task override for the consecutive-failure circuit breaker.
    # The value is the failure count at which the breaker trips — e.g.
    # ``max_retries=1`` blocks on the first failure (zero retries),
    # ``max_retries=3`` blocks on the third (two retries allowed).
    # ``None`` (the common case) falls through to the dispatcher-level
    # ``kanban.failure_limit`` config, and then to ``DEFAULT_FAILURE_LIMIT``.
    # Name matches the ``--max-retries`` CLI flag on ``kanban create``.
    max_retries: Optional[int] = None
    # When True, the dispatched worker runs in a Ralph-style goal loop
    # (the same engine behind the ``/goal`` slash command): after each
    # turn an auxiliary judge model evaluates the worker's response
    # against this card's title/body (treated as the goal). If the judge
    # says "not done" and budget remains, the worker is fed a
    # continuation prompt IN THE SAME SESSION and keeps working until the
    # judge agrees, the goal-turn budget is exhausted (→ kanban_block),
    # or the worker explicitly blocks/completes. ``False`` (default) =
    # the classic single-shot worker. ``goal_max_turns`` bounds the loop.
    goal_mode: bool = False
    # Goal-loop turn budget for ``goal_mode`` workers. ``None`` falls
    # through to the goals engine default (``goals.DEFAULT_MAX_TURNS``).
    goal_max_turns: Optional[int] = None
    # Originating chat/agent session id, when the task was created from
    # within an agent loop that propagated ``HERMES_SESSION_ID``. NULL for
    # tasks created from the CLI, the dashboard, or any path that doesn't
    # set the env var. Lets clients render a per-session board without
    # relying on tenant + time-window heuristics.
    session_id: Optional[str] = None
    # Typed block reason (one of VALID_BLOCK_KINDS) or None for legacy/un-typed
    # blocks. Set by ``block_task``; preserved across unblock so a re-block for
    # the same kind is recognisable as an unblock↔re-block loop.
    block_kind: Optional[str] = None
    # Unblock-loop counter. See the column comment in SCHEMA_SQL and
    # ``BLOCK_RECURRENCE_LIMIT``. Reset only on successful completion.
    block_recurrences: int = 0
    completion_contract: Optional[dict] = None
    # Human-Gate v1 flag only — the token hash/issued_at columns are
    # deliberately NOT exposed here so they can never flow through
    # ``_task_to_dict`` / ``kanban show --json`` / any tool payload. Code
    # that needs to check or consume the hash reads it with a direct SQL
    # query (see ``unblock_task`` / ``issue_gate_token``).
    human_gate: bool = False
    # Event-sourcing resume (#2), in-memory only — NOT persisted columns and NOT
    # read by from_row. Populated by claim_task/claim_review_task so the spawn
    # path knows which resumable session id to pin for this run and whether this
    # claim should resume the previous run's conversation (re-claim after crash).
    worker_session_id: Optional[str] = None
    resume_requested: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        keys = set(row.keys())
        # Parse skills JSON blob if present
        skills_value: Optional[list] = None
        if "skills" in keys and row["skills"]:
            try:
                parsed = json.loads(row["skills"])
                if isinstance(parsed, list):
                    skills_value = [str(s) for s in parsed if s]
            except Exception:
                skills_value = None
        completion_contract: Optional[dict] = None
        if "completion_contract" in keys and row["completion_contract"]:
            try:
                parsed_contract = json.loads(row["completion_contract"])
                if isinstance(parsed_contract, dict):
                    completion_contract = parsed_contract
            except Exception:
                completion_contract = None
        return cls(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"],
            workspace_path=row["workspace_path"],
            branch_name=row["branch_name"] if "branch_name" in keys else None,
            project_id=row["project_id"] if "project_id" in keys else None,
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=row["tenant"] if "tenant" in keys else None,
            result=row["result"] if "result" in keys else None,
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            consecutive_failures=(
                row["consecutive_failures"] if "consecutive_failures" in keys
                # Pre-migration fallback: ``_migrate_add_optional_columns`` always
                # adds ``consecutive_failures`` now, so this branch is only reachable
                # on a DB that was never opened since pre-#20410 code ran. Keep for
                # belt-and-suspenders safety; in practice it is dead code post-migration.
                else (row["spawn_failures"] if "spawn_failures" in keys else 0)
            ),
            worker_pid=row["worker_pid"] if "worker_pid" in keys else None,
            last_failure_error=(
                row["last_failure_error"] if "last_failure_error" in keys
                # Same belt-and-suspenders fallback as consecutive_failures above.
                else (row["last_spawn_error"] if "last_spawn_error" in keys else None)
            ),
            max_runtime_seconds=(
                row["max_runtime_seconds"] if "max_runtime_seconds" in keys else None
            ),
            last_heartbeat_at=(
                row["last_heartbeat_at"] if "last_heartbeat_at" in keys else None
            ),
            current_run_id=(
                row["current_run_id"] if "current_run_id" in keys else None
            ),
            workflow_template_id=(
                row["workflow_template_id"] if "workflow_template_id" in keys else None
            ),
            current_step_key=(
                row["current_step_key"] if "current_step_key" in keys else None
            ),
            skills=skills_value,
            model_override=row["model_override"] if "model_override" in keys and row["model_override"] else None,
            task_class=row["task_class"] if "task_class" in keys and row["task_class"] else None,
            effort=row["effort"] if "effort" in keys and row["effort"] else None,
            max_retries=(
                row["max_retries"] if "max_retries" in keys else None
            ),
            goal_mode=(
                bool(row["goal_mode"]) if "goal_mode" in keys and row["goal_mode"] else False
            ),
            goal_max_turns=(
                row["goal_max_turns"] if "goal_max_turns" in keys and row["goal_max_turns"] else None
            ),
            session_id=(
                row["session_id"] if "session_id" in keys else None
            ),
            block_kind=(
                row["block_kind"] if "block_kind" in keys and row["block_kind"] else None
            ),
            block_recurrences=(
                int(row["block_recurrences"])
                if "block_recurrences" in keys and row["block_recurrences"] is not None
                else 0
            ),
            completion_contract=completion_contract,
            human_gate=(
                bool(row["human_gate"]) if "human_gate" in keys and row["human_gate"] else False
            ),
        )


@dataclass
class Run:
    """In-memory view of a ``task_runs`` row.

    A run is one attempt to execute a task — created on claim, closed
    on complete/block/crash/timeout/spawn_failure/reclaim. Multiple runs
    per task when retries happen. Carries the claim machinery, PID,
    heartbeat, and the structured handoff summary that downstream workers
    read via ``build_worker_context``.
    """

    id: int
    task_id: str
    profile: Optional[str]
    step_key: Optional[str]
    status: str
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    worker_pid: Optional[int]
    max_runtime_seconds: Optional[int]
    last_heartbeat_at: Optional[int]
    started_at: int
    ended_at: Optional[int]
    outcome: Optional[str]
    summary: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
        except Exception:
            meta = None
        return cls(
            id=int(row["id"]),
            task_id=row["task_id"],
            profile=row["profile"],
            step_key=row["step_key"],
            status=row["status"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            worker_pid=row["worker_pid"],
            max_runtime_seconds=row["max_runtime_seconds"],
            last_heartbeat_at=row["last_heartbeat_at"],
            started_at=int(row["started_at"]),
            ended_at=(int(row["ended_at"]) if row["ended_at"] is not None else None),
            outcome=row["outcome"],
            summary=row["summary"],
            metadata=meta,
            error=row["error"],
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int


@dataclass
class Attachment:
    """In-memory view of a row from the ``task_attachments`` table."""

    id: int
    task_id: str
    filename: str
    stored_path: str
    content_type: Optional[str]
    size: int
    uploaded_by: Optional[str]
    created_at: int


@dataclass
class PendingAction:
    """Durable, secret-free approval request bound to one worker action."""

    id: int
    task_id: str
    run_id: Optional[int]
    command_hash: str
    fingerprint: str
    mutation_kind: str
    summary: str
    profile: str
    workspace: str
    expires_at: int
    approved_at: Optional[int] = None
    consumed_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    state: str = "pending"
    version: int = 1
    updated_at: int = 0
    resolved_at: Optional[int] = None


@dataclass(frozen=True)
class Attention:
    """Safe current-operator projection of one exact action or typed blocker."""

    id: int
    task_id: str
    action_id: Optional[int]
    type: str
    summary: str
    created_at: int
    state: str
    version: int
    expires_at: Optional[int]
    requires_human_action: bool
    approvable: bool


@dataclass(frozen=True)
class AttentionDeliveryFinishResult:
    """Safe delivery outcome; contains no task, channel, or exception prose."""

    status: Literal["delivered", "retry", "suppressed_duplicate"]

    def __bool__(self) -> bool:
        return self.status != "suppressed_duplicate"


@dataclass(frozen=True)
class ResolvePendingActionResult:
    """Safe resolve outcome for HTTP 404/409/410 mapping without raw bindings."""

    status: str

    def __bool__(self) -> bool:
        return self.status == "resolved"


@dataclass(frozen=True)
class ApprovePendingActionResult:
    """Committed, redacted result of versioned exact-action approval."""

    status: Literal["approved", "not_found", "conflict", "gone"]
    task_id: str
    action_id: Optional[int] = None
    attention_id: Optional[int] = None
    attention_version: Optional[int] = None
    task_status: Optional[Literal["ready", "todo"]] = None

    def __bool__(self) -> bool:
        return self.status == "approved"


@dataclass(frozen=True)
class AttentionStatusTransitionResult:
    """Committed, redacted result of an attention-aware status transition."""

    status: Literal["transitioned", "not_found", "conflict", "gone"]
    task_id: str
    task_status: Optional[Literal["ready", "todo", "triage"]] = None
    attention_id: Optional[int] = None
    attention_version: Optional[int] = None

    def __bool__(self) -> bool:
        return self.status == "transitioned"


@dataclass(frozen=True)
class ResumeApprovedActionRetryResult:
    """Redacted result for the only explicit approved-grant retry escape."""

    status: Literal["resumed", "not_found", "conflict", "gone"]
    task_id: str
    attention_id: Optional[int] = None
    attention_version: Optional[int] = None
    task_status: Optional[Literal["ready", "todo"]] = None

    def __bool__(self) -> bool:
        return self.status == "resumed"


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int
    run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    -- Optional link to a first-class Project (hermes_cli/projects_db). When set,
    -- the task's worktree is anchored under the project's primary repo with a
    -- deterministic branch name instead of a random wt/<task-id> fallback.
    project_id           TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    -- Unified consecutive-failure counter. Incremented on spawn
    -- failure, timeout, or crash; reset only on successful completion.
    -- The circuit breaker in _record_task_failure trips when this
    -- exceeds DEFAULT_FAILURE_LIMIT consecutive non-successes.
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    -- Short excerpt of the most recent failure's error text.
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    -- Pointer into task_runs for the currently-active run (NULL if no
    -- run is in-flight). Denormalised for cheap reads.
    current_run_id       INTEGER,
    -- Forward-compat for v2 workflow routing. In v1 the kernel writes
    -- these when the task is opted into a template but otherwise ignores
    -- them; the dispatcher doesn't consult them for routing yet.
    workflow_template_id TEXT,
    current_step_key     TEXT,
    -- Force-loaded skills for the worker on this task, stored as JSON.
    -- Passed to the worker via `--skills`. NULL or empty array = no extras.
    skills               TEXT,
    -- Per-task model override. When set, the dispatcher passes -m <model>
    -- to the worker, overriding the profile's default model. NULL = use
    -- the profile default.
    model_override       TEXT,
    -- Optional quality class / task grade (e.g. "hard"). Free-form label;
    -- the decomposer maps it to a class-specific planner auxiliary role.
    -- NULL = unclassified = default behaviour.
    task_class           TEXT,
    -- Per-task reasoning-effort override. When set, the dispatcher exports
    -- HERMES_REASONING_EFFORT to the worker, overriding the profile's
    -- agent.reasoning_effort. NULL = use the profile default.
    effort               TEXT,
    -- Per-task override for the consecutive-failure circuit breaker.
    -- The value is the failure count at which the breaker trips — e.g.
    -- ``max_retries=1`` blocks on the first failure. NULL (the common
    -- case) falls through to the dispatcher-level ``kanban.failure_limit``
    -- config and then ``DEFAULT_FAILURE_LIMIT``.
    max_retries          INTEGER,
    -- When 1, the dispatched worker runs in a Ralph-style goal loop: an
    -- auxiliary judge re-evaluates the worker's response against the
    -- card title/body after each turn and feeds a continuation prompt
    -- back into the SAME session until the judge agrees the work is done
    -- or ``goal_max_turns`` is exhausted. NULL/0 = classic single-shot
    -- worker (the default).
    goal_mode            INTEGER NOT NULL DEFAULT 0,
    -- Goal-loop turn budget for ``goal_mode`` workers. NULL = use the
    -- goals-engine default.
    goal_max_turns       INTEGER,
    -- Originating chat/agent session id when the task was created from
    -- inside an agent loop that propagated ``HERMES_SESSION_ID``. NULL
    -- for tasks created from the CLI, dashboard, or any path that doesn't
    -- set the env var. Indexed so per-session list queries stay cheap on
    -- larger boards.
    session_id           TEXT,
    -- Typed block reason set by ``block_task`` (one of VALID_BLOCK_KINDS, or
    -- NULL for legacy/un-typed blocks). Drives routing: ``dependency`` never
    -- sits in ``blocked`` (goes to ``todo`` for parent-gating); the others go
    -- to ``blocked`` for a human. Preserved across unblock so a re-block for
    -- the SAME kind can be recognised as a loop.
    block_kind           TEXT,
    -- Unblock-loop counter. Incremented each time a task is re-blocked for the
    -- same truly-blocked reason after having been unblocked. When it reaches
    -- BLOCK_RECURRENCE_LIMIT the task is routed to ``triage`` instead of
    -- ``blocked`` so a cron can't spin it forever. Reset to 0 only on a
    -- successful completion — NOT on unblock (resetting on unblock is exactly
    -- the amnesia that let the loop run unbounded).
    block_recurrences    INTEGER NOT NULL DEFAULT 0,
    -- Versioned typed cause identity for the recurrence chain.  NULL is
    -- intentionally legacy/unknown and must never continue a prior chain.
    block_cause_fingerprint TEXT,
    block_reason_code     TEXT,
    block_cause_version   INTEGER,
    completion_contract  TEXT,
    -- Human-Gate v1 (2026-07-11, see human-gate-design.md). When 1, this
    -- card can only be unblocked with a one-time token pushed to the
    -- operator via ntfy — no tool surface and no worker session can set
    -- or clear this flag; only the interactive CLI (`kanban block
    -- --human-gate` / `kanban gate <id> on|off`) does. gate_token_hash
    -- stores ONLY the sha256 hex digest of the current token, never the
    -- plaintext; NULL means no token is currently redeemable (either
    -- none has been issued yet, or the last one was consumed/rotated),
    -- which makes unblock_task fail closed.
    human_gate           INTEGER NOT NULL DEFAULT 0,
    gate_token_hash       TEXT,
    gate_token_issued_at  INTEGER,
    gate_token_board      TEXT,
    gate_token_task_id    TEXT,
    gate_token_action     TEXT,
    -- Human-Gate v2: a grant is additionally bound to the canonical,
    -- versioned governance scope.  These fields are deliberately absent from
    -- Task and public projections; they are authorization state only.
    gate_scope_hash        TEXT,
    gate_scope_version     INTEGER,
    -- Internal-only declarations.  No worker tool accepts these fields.
    governance_target_ref  TEXT,
    governance_mutation_class TEXT,
    gate_failed_attempts  INTEGER NOT NULL DEFAULT 0,
    gate_failure_window_started_at INTEGER,
    gate_locked_until     INTEGER
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

-- Historical attempt record. Each time the dispatcher claims a task, a
-- new row is created here; claim state, PID, heartbeat, runtime cap,
-- and structured summary all live on the run, not the task. Multiple
-- rows per task id when the task was retried after crash/timeout/block.
-- v2 of the kanban schema will use ``step_key`` to drive per-stage
-- workflow routing; in v1 the column is nullable and unused (kernel
-- ignores it).
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    -- status: running | done | blocked | crashed | timed_out | failed | released
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    last_activity_at    INTEGER,
    last_semantic_progress_at INTEGER,
    worker_start_ticks  INTEGER,
    d_state_since       INTEGER,
    resource_sample     TEXT,
    termination_pending_since INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    -- outcome: completed | blocked | crashed | timed_out | spawn_failed |
    --          gave_up | reclaimed | (null while still running)
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS task_pending_actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Stable opaque public identity while live; retained only for task-bound history lookup.
    attention_id INTEGER,
    -- Exact resumed worker run that produced the technical retry blocker.
    -- Distinct from run_id, which remains the original approval-origin run.
    retry_origin_run_id INTEGER,
    task_id      TEXT NOT NULL,
    run_id       INTEGER,
    command_hash TEXT NOT NULL,
    fingerprint  TEXT,
    mutation_kind TEXT,
    summary      TEXT NOT NULL,
    profile      TEXT NOT NULL,
    workspace    TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    expires_at   INTEGER NOT NULL,
    approved_at  INTEGER,
    consumed_at  INTEGER,
    cancelled_at INTEGER,
    state        TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','approved','consumed','expired','cancelled','resolved')),
    version      INTEGER NOT NULL DEFAULT 1,
    updated_at   INTEGER NOT NULL DEFAULT 0,
    resolved_at  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_pending_actions_task
    ON task_pending_actions(task_id, consumed_at, expires_at);

CREATE TABLE IF NOT EXISTS task_attentions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id           TEXT NOT NULL,
    -- Ordinary typed blockers have no exact-action lifecycle.
    action_id         INTEGER UNIQUE,
    type              TEXT NOT NULL CHECK(type IN ('exact_action','decision','capability','transient','loop_triage','protocol','review')),
    cause_fingerprint TEXT NOT NULL,
    summary           TEXT NOT NULL,
    created_at        INTEGER NOT NULL,
    version           INTEGER NOT NULL DEFAULT 1,
    origin_run_id     INTEGER,
    UNIQUE(task_id, cause_fingerprint)
);

-- Files attached to a task (PDFs, images, source documents). The blob
-- lives on disk under ``attachments_root(board)/<task_id>/<stored_name>``;
-- this row carries metadata + the absolute ``stored_path`` so the
-- dashboard can list/download and ``build_worker_context`` can surface
-- the absolute path to the worker (which has full file-tool access). See
-- #35338.
CREATE TABLE IF NOT EXISTS task_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL DEFAULT 0,
    uploaded_by  TEXT,
    created_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_artifacts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    producer_run_id     INTEGER NOT NULL DEFAULT 0,
    original_path       TEXT NOT NULL,
    durable_path        TEXT NOT NULL,
    sha256              TEXT NOT NULL,
    size                INTEGER NOT NULL,
    content_type        TEXT,
    validated_at        INTEGER NOT NULL,
    retention_class     TEXT NOT NULL,
    UNIQUE(task_id, producer_run_id, original_path, sha256),
    UNIQUE(durable_path)
);

-- Subscription from a gateway source (platform + chat + thread) to a
-- task. The gateway's kanban-notifier watcher tails task_events and
-- pushes ``completed`` / ``blocked`` / ``spawn_auto_blocked`` events to
-- the original requester so human-in-the-loop workflows close the loop.
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    notifier_profile TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    active        INTEGER NOT NULL DEFAULT 1,
    generation    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_task         ON task_artifacts(task_id, producer_run_id);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_attachments_task      ON task_attachments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);

-- Durable, per-channel delivery projection for the *current* attention.
-- It deliberately does not use task_events: those are history, while an
-- attention can be resolved or superseded before a notifier gets a turn.
CREATE TABLE IF NOT EXISTS kanban_attention_deliveries (
    task_id           TEXT NOT NULL,
    attention_id      INTEGER NOT NULL,
    attention_version INTEGER NOT NULL,
    platform          TEXT NOT NULL,
    chat_id           TEXT NOT NULL,
    thread_id         TEXT NOT NULL DEFAULT '',
    notifier_profile  TEXT,
    subscription_generation INTEGER NOT NULL DEFAULT 1,
    state             TEXT NOT NULL DEFAULT 'pending'
                      CHECK(state IN ('pending','sending','delivered','cancelled')),
    attempts          INTEGER NOT NULL DEFAULT 0,
    -- Incremented on every claim. Finish operations must present this exact
    -- opaque generation so an expired claimant cannot finish a reclaimed lease.
    lease_version     INTEGER NOT NULL DEFAULT 0,
    lease_until       INTEGER,
    delivered_at      INTEGER,
    last_error        TEXT,
    updated_at        INTEGER NOT NULL,
    PRIMARY KEY(task_id, attention_id, attention_version, platform, chat_id, thread_id)
);
CREATE INDEX IF NOT EXISTS idx_attention_deliveries_claim
    ON kanban_attention_deliveries(state, lease_until);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()
_INIT_LOCK = threading.RLock()
_SQLITE_HEADER = b"SQLite format 3\x00"
DEFAULT_BUSY_TIMEOUT_MS = 120_000

# Bounded acquire for the cross-process init lock (#36644). The original bare
# blocking flock had no timeout, so a wedged holder blocked the dispatcher's
# next-tick connect forever. We retry a non-blocking acquire up to this
# deadline, polling at this interval, then proceed without the cross-process
# lock (the in-process _INIT_LOCK + idempotent init remain the backstop).
_INIT_LOCK_TIMEOUT_SECONDS = 10.0
_INIT_LOCK_POLL_SECONDS = 0.05
_ARTIFACT_LOCK_TIMEOUT_SECONDS = 30.0


def _resolve_busy_timeout_ms() -> int:
    """Return the SQLite busy timeout for Kanban connections.

    Kanban is the shared cross-profile dispatch bus, so worker stampedes are
    expected.  A long busy timeout lets SQLite serialize writers via WAL rather
    than surfacing transient ``database is locked`` failures during bursts.
    """
    raw = os.environ.get("HERMES_KANBAN_BUSY_TIMEOUT_MS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return DEFAULT_BUSY_TIMEOUT_MS


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a Kanban SQLite connection with consistent lock waiting."""
    busy_timeout_ms = _resolve_busy_timeout_ms()
    conn = sqlite3.connect(
        str(path),
        isolation_level=None,
        timeout=busy_timeout_ms / 1000.0,
    )
    # ``sqlite3.connect(timeout=...)`` normally maps to busy_timeout, but set
    # the PRAGMA explicitly so it is observable and survives future wrapper
    # changes. Parameter binding is not supported for PRAGMA assignments.
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return conn


@contextlib.contextmanager
def _cross_process_init_lock(path: Path):
    """Serialize first-connect WAL/schema/integrity setup across processes.

    ``_INIT_LOCK`` only protects threads inside one Python process. During a
    dispatcher burst, many worker processes can all hit a fresh/legacy board at
    once and each process has an empty ``_INITIALIZED_PATHS`` cache. This file
    lock keeps header validation, integrity probing, WAL activation, and
    additive migrations single-file/single-writer across the whole host while
    leaving normal post-init DB usage concurrent under SQLite WAL.

    The acquire is **bounded** (issue #36644): the original bare blocking
    ``flock(LOCK_EX)`` had no timeout, so a single process stalled inside the
    critical section (or a stale lock held by a wedged worker) blocked every
    other ``connect()`` — including the long-lived gateway dispatcher's
    next-tick connect — forever, with no traceback and no recovery short of a
    restart. We retry a non-blocking acquire up to a deadline and then fail
    closed. Init includes legacy table rebuilds, so proceeding without the
    process lock can race non-idempotent DDL and damage the board. A bounded,
    actionable failure is safer than either an unbounded hang or unlocked DDL.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".init.lock")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + _INIT_LOCK_TIMEOUT_SECONDS
        if _IS_WINDOWS:
            import msvcrt

            locking = getattr(msvcrt, "locking")
            nb_lock = getattr(msvcrt, "LK_NBLCK")
            while True:
                try:
                    handle.seek(0)
                    locking(handle.fileno(), nb_lock, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_INIT_LOCK_POLL_SECONDS)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_INIT_LOCK_POLL_SECONDS)
        if not acquired:
            raise TimeoutError(
                f"kanban init lock for {lock_path} was not acquired within "
                f"{_INIT_LOCK_TIMEOUT_SECONDS:.0f}s; refusing to run schema "
                "migration without the cross-process lock"
            )
        yield
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    locking = getattr(msvcrt, "locking")
                    unlock_mode = getattr(msvcrt, "LK_UNLCK")
                    locking(handle.fileno(), unlock_mode, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextlib.contextmanager
def _dispatch_tick_lock(db_path: Path):
    """Non-blocking single-writer guard around one dispatcher tick.

    Yields ``True`` when this process holds the board's dispatch lock and
    may proceed with the tick, or ``False`` when another process already
    holds it (the caller should skip the tick this round).

    Motivation (issue #35240): a ``hermes gateway run --replace`` /
    ``gateway restart`` invoked from a shell on a systemd/launchd host can
    leave an orphan gateway whose dispatcher escapes the service cgroup,
    survives ``systemctl restart``, and becomes a *second* long-lived
    writer on the same ``kanban.db``. Two dispatchers that each believe
    they own the file both pass SQLite ``busy_timeout`` and then race on
    WAL frames — the documented root cause of multi-writer corruption.
    The startup guard (``_guard_supervised_gateway_conflict``) blocks the
    common way an orphan is born, but this lock is the defense-in-depth
    that prevents two dispatchers from ever writing concurrently
    *regardless of how the second one got there*.

    The lock is **non-blocking** on purpose: the gateway's async watcher
    must never stall on a held lock. A losing dispatcher simply skips its
    tick (the winner is making progress on the same board), and tries
    again next interval.

    Board-scoped: the lock file is a ``.dispatch.lock`` sibling of the
    board's ``kanban.db``, so unrelated boards tick independently. On
    platforms without ``fcntl``/``msvcrt`` the guard degrades to a no-op
    (yields ``True``) — single-writer enforcement is best-effort and the
    orphan-dispatcher scenario is specific to POSIX service managers.
    """
    lock_path = db_path.with_name(db_path.name + ".dispatch.lock")
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        if _IS_WINDOWS:
            try:
                import msvcrt

                handle.seek(0)
                locking = getattr(msvcrt, "locking")
                # LK_NBLCK = non-blocking exclusive byte-range lock.
                nb_lock = getattr(msvcrt, "LK_NBLCK")
                locking(handle.fileno(), nb_lock, 1)
                acquired = True
            except (OSError, AttributeError):
                acquired = False
        else:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError):
                acquired = False
    except OSError:
        # Could not even open the lock file (permissions, read-only FS).
        # Degrade to a no-op so a probe failure never blocks dispatch.
        acquired = True
        handle = None
    try:
        yield acquired
    finally:
        if handle is not None:
            try:
                if acquired:
                    if _IS_WINDOWS:
                        import msvcrt

                        handle.seek(0)
                        locking = getattr(msvcrt, "locking")
                        unlock_mode = getattr(msvcrt, "LK_UNLCK")
                        locking(handle.fileno(), unlock_mode, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, AttributeError):
                pass
            finally:
                handle.close()


def _looks_like_tls_record_at(data: bytes, offset: int) -> bool:
    """Return True for a TLS record header at ``data[offset:]``."""
    if len(data) < offset + 5:
        return False
    content_type = data[offset]
    major = data[offset + 1]
    minor = data[offset + 2]
    length = int.from_bytes(data[offset + 3:offset + 5], "big")
    return (
        content_type in {0x14, 0x15, 0x16, 0x17}
        and major == 0x03
        and minor in {0x00, 0x01, 0x02, 0x03, 0x04}
        and 0 < length <= 18432
    )


def _validate_sqlite_header(path: Path) -> None:
    """Fail early with an actionable error for non-SQLite Kanban DB files.

    ``sqlite3.connect()`` creates missing files, so missing paths are allowed.
    An already-existing zero-byte file is treated as truncation and rejected;
    otherwise a damaged board could be silently reinitialized. Existing files
    must have the SQLite header before we hand them to SQLite/WAL setup. This
    keeps corrupted page-0 failures from
    being collapsed into a generic PRAGMA error and lets the gateway's corrupt
    board handling identify the board by fingerprint.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.st_size == 0:
        raise sqlite3.DatabaseError(
            f"refusing to initialize existing zero-byte kanban DB at {path}; "
            "remove it explicitly only when creating a new board"
        )
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
    except OSError:
        return
    if head.startswith(_SQLITE_HEADER):
        return
    signature = ""
    if head.startswith(b"SQLit") and _looks_like_tls_record_at(head, 5):
        signature = " (TLS record header detected at byte offset 5)"
    elif _looks_like_tls_record_at(head, 0):
        signature = " (TLS record header detected at byte offset 0)"
    raise sqlite3.DatabaseError(
        "file is not a database: invalid SQLite header for "
        f"{path}{signature}; first_32={head[:32].hex(' ')}"
    )


class KanbanDbCorruptError(RuntimeError):
    """Raised when an existing kanban DB file fails integrity checks.

    Fail-closed guard against silent recreation of a corrupt board file,
    which would otherwise destroy the user's tasks. Carries both the
    original path and the timestamped backup we made before refusing.
    """

    def __init__(self, db_path: Path, backup_path: Optional[Path], reason: str):
        self.db_path = db_path
        self.backup_path = backup_path
        self.reason = reason
        backup_str = str(backup_path) if backup_path is not None else "<backup failed>"
        super().__init__(
            f"Refusing to open corrupt kanban DB at {db_path}: {reason}. "
            f"Original preserved; backup at {backup_str}."
        )


class PostCommitIntegrityError(RuntimeError):
    """Integrity guard failed after COMMIT; the mutation is already durable."""


def _backup_corrupt_db(path: Path) -> Optional[Path]:
    """Atomically preserve a stable corrupt DB/WAL/SHM evidence set.

    Returns ``None`` if any source changes while it is copied. That is safer
    than presenting a mixed-generation main/WAL set as forensic evidence.
    """
    resolved = path.resolve()
    parent = resolved.parent
    base_name = resolved.name
    temp_paths: list[Path] = []
    created_paths: list[Path] = []

    def signature(source: Path) -> tuple[int, int]:
        stat = source.stat()
        return stat.st_size, stat.st_mtime_ns

    def file_digest(source: Path) -> bytes:
        value = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.digest()

    def same_bytes(left: Path, right: Path) -> bool:
        return file_digest(left) == file_digest(right)

    def copy_stable(source: Path, target: Path) -> None:
        before = signature(source)
        temp = parent / f".{target.name}.{secrets.token_hex(6)}.tmp"
        temp_paths.append(temp)
        with source.open("rb") as reader, temp.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if signature(source) != before:
            raise OSError(f"source changed while copying: {source}")
        if target.exists():
            if not same_bytes(temp, target):
                raise OSError(f"existing forensic backup differs: {target}")
            temp.unlink()
            temp_paths.remove(temp)
        else:
            os.replace(temp, target)
            temp_paths.remove(temp)
            created_paths.append(target)

    try:
        before_sidecars = {
            suffix for suffix in ("-wal", "-shm")
            if (parent / (base_name + suffix)).exists()
        }
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        candidate = parent / f"{base_name}.corrupt.{digest.hexdigest()[:16]}.bak"
        if candidate.parent != parent:
            return None
        copy_stable(resolved, candidate)
        if file_digest(candidate) != digest.digest():
            raise OSError("main DB changed between hashing and stable copy")
        for suffix in sorted(before_sidecars):
            copy_stable(
                parent / (base_name + suffix),
                parent / (candidate.name + suffix),
            )
        after_sidecars = {
            suffix for suffix in ("-wal", "-shm")
            if (parent / (base_name + suffix)).exists()
        }
        if after_sidecars != before_sidecars:
            raise OSError("SQLite sidecar set changed while copying")
        dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return candidate
    except OSError:
        for temp in temp_paths:
            temp.unlink(missing_ok=True)
        for created in reversed(created_paths):
            created.unlink(missing_ok=True)
        return None


def _guard_existing_db_is_healthy(path: Path) -> None:
    """Run ``PRAGMA integrity_check`` on an existing non-empty DB file.

    Opens the probe in read/write mode so SQLite can recover or
    checkpoint a healthy WAL/hot-journal DB before we declare it
    corrupt. If the file is malformed, copy it (and any WAL/SHM
    sidecars) to a timestamped backup and raise
    :class:`KanbanDbCorruptError` so callers cannot silently recreate
    the schema on top of a damaged DB.

    Transient lock/busy errors (``sqlite3.OperationalError``) are NOT
    treated as corruption; they propagate raw so the caller sees a
    normal lock failure and no spurious ``.corrupt`` backup is made.

    No-op for missing files, zero-byte files (treated as fresh), and
    paths already proven healthy this process (cache hit).

    Path-trust note: ``path`` arrives via :func:`connect`, which itself
    resolves it from an explicit ``db_path`` argument, the
    :func:`kanban_db_path` env-var chain, or the kanban-home default —
    all sources Hermes treats as user-controlled-but-trusted on the
    user's own machine. We additionally resolve the path here and
    confine all filesystem writes to its parent directory so any
    accidental ``..`` segments are collapsed before any I/O happens.
    """
    # Resolve before any I/O. ``Path.resolve()`` normalizes ``..`` and
    # symlinks, giving us a canonical path whose parent dir we can pin.
    try:
        resolved = path.resolve()
    except OSError:
        return
    try:
        if not resolved.exists() or resolved.stat().st_size == 0:
            return
    except OSError:
        return
    if str(resolved) in _INITIALIZED_PATHS:
        return
    reason: Optional[str] = None
    try:
        probe = _sqlite_connect(resolved)
        try:
            row = probe.execute("PRAGMA integrity_check").fetchone()
        finally:
            probe.close()
        if not row or (row[0] or "").lower() != "ok":
            reason = f"integrity_check returned {row[0] if row else '<no row>'!r}"
    except sqlite3.OperationalError:
        # Lock contention, busy, transient IO — not corruption. Let it propagate.
        raise
    except sqlite3.DatabaseError as exc:
        reason = f"sqlite refused to open file: {exc}"
    if reason is None:
        return
    backup = _backup_corrupt_db(resolved)
    raise KanbanDbCorruptError(resolved, backup, reason)


class _CompletionArtifactLockError(RuntimeError):
    pass


@contextlib.contextmanager
def _completion_artifact_lock(conn: sqlite3.Connection):
    """Serialize scavenging with artifact promotion through manifest commit."""
    row = conn.execute("PRAGMA database_list").fetchone()
    database_path = Path(row[2]) if row and row[2] else kanban_db_path()
    lock_path = database_path.with_name(database_path.name + ".completion-artifacts.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise _CompletionArtifactLockError(
            f"cannot open completion artifact lock {lock_path}: {exc}"
        ) from exc
    handle = os.fdopen(lock_fd, "a+b")
    acquired = False
    try:
        deadline = time.monotonic() + _ARTIFACT_LOCK_TIMEOUT_SECONDS
        if _IS_WINDOWS:
            import msvcrt

            locking = getattr(msvcrt, "locking")
            nb_lock = getattr(msvcrt, "LK_NBLCK")
            while True:
                try:
                    handle.seek(0)
                    locking(handle.fileno(), nb_lock, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_INIT_LOCK_POLL_SECONDS)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_INIT_LOCK_POLL_SECONDS)
        if not acquired:
            raise _CompletionArtifactLockError(
                f"completion artifact lock for {lock_path} was not acquired within "
                f"{_ARTIFACT_LOCK_TIMEOUT_SECONDS:.0f}s"
            )
        yield
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    locking = getattr(msvcrt, "locking")
                    unlock_mode = getattr(msvcrt, "LK_UNLCK")
                    locking(handle.fileno(), unlock_mode, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _scavenge_completion_artifacts(
    conn: sqlite3.Connection,
    *,
    board: Optional[str],
    grace_seconds: int = 3600,
    now: Optional[float] = None,
) -> int:
    with _completion_artifact_lock(conn):
        return _scavenge_completion_artifacts_locked(
            conn, board=board, grace_seconds=grace_seconds, now=now
        )


def _scavenge_completion_artifacts_locked(
    conn: sqlite3.Connection,
    *,
    board: Optional[str],
    grace_seconds: int = 3600,
    now: Optional[float] = None,
) -> int:
    """Remove stale files that no committed manifest references."""
    root = completion_artifacts_root(board=board).absolute()
    if not root.exists() or not hasattr(os, "fwalk"):
        return 0
    referenced = {
        str(row[0]) for row in conn.execute("SELECT durable_path FROM task_artifacts")
    }
    cutoff = (time.time() if now is None else now) - max(0, grace_seconds)
    removed = 0
    root_fd = _open_directory_path_nofollow(root, create=False)
    try:
        for relative_dir, directory_names, file_names, directory_fd in os.fwalk(
            ".", topdown=False, follow_symlinks=False, dir_fd=root_fd
        ):
            changed = False
            for name in file_names:
                candidate = root / relative_dir / name
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if info.st_mtime > cutoff or str(candidate) in referenced:
                    continue
                if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    os.unlink(name, dir_fd=directory_fd)
                    removed += 1
                    changed = True
            for name in directory_names:
                candidate = root / relative_dir / name
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode):
                    if info.st_mtime <= cutoff and str(candidate) not in referenced:
                        os.unlink(name, dir_fd=directory_fd)
                        removed += 1
                        changed = True
            if changed:
                os.fsync(directory_fd)
    finally:
        os.close(root_fd)
    return removed


def connect(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> sqlite3.Connection:
    """Open (and initialize if needed) the kanban DB.

    WAL mode is enabled on every connection; it's a no-op after the first
    time but keeps the code robust if the DB file is ever re-created.

    The first connection to a given path auto-runs :func:`init_db` so
    fresh installs and test harnesses that construct `connect()`
    directly don't have to remember a separate init step. Subsequent
    connections skip the schema check via a module-level path cache.

    Path resolution:

    * ``db_path`` explicit → used as-is (legacy callers, tests).
    * ``board`` explicit → resolves to that board's DB.
    * Neither → :func:`kanban_db_path` resolves via
      ``HERMES_KANBAN_DB`` env → ``HERMES_KANBAN_BOARD`` env →
      ``<root>/kanban/current`` → ``default``.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: once THIS process has initialized this path, the expensive
    # first-open work (header validation, integrity probe, schema + additive
    # migrations) is already done and cached in _INITIALIZED_PATHS. Acquiring
    # the cross-process init lock on every connect is what let a single stalled
    # holder (e.g. an external `hermes kanban list` mid-integrity-probe) block
    # the long-lived gateway dispatcher's next-tick connect() forever — an
    # unbounded flock with no timeout, no LOCK_NB, no recovery (#36644). On the
    # steady-state path there is nothing for the cross-process lock to protect
    # (no schema/migration writes run), so skip it entirely and just open the
    # connection with WAL/pragmas under the cheap in-process _INIT_LOCK.
    resolved = str(path.resolve())
    if resolved in _INITIALIZED_PATHS:
        conn = _sqlite_connect(path)
        try:
            conn.row_factory = sqlite3.Row
            with _INIT_LOCK:
                from hermes_state import apply_wal_with_fallback
                apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
                conn.execute("PRAGMA synchronous=FULL")
                # Keep the steady-state path identical to first initialization.
                # Reapplying the old 100-page value here silently undid the safer
                # default on every subsequent connection in long-lived processes.
                conn.execute("PRAGMA wal_autocheckpoint=1000")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA secure_delete=ON")
                conn.execute("PRAGMA cell_size_check=ON")
        except Exception:
            conn.close()
            raise
        return conn

    with _cross_process_init_lock(path):
        # Cheap byte-level check first — catches the #29507 TLS-overwrite shape
        # and other invalid-header cases without opening a sqlite connection.
        _validate_sqlite_header(path)
        # Full integrity probe — catches corruption past the header (malformed
        # pages, broken internal metadata). Cached per-path after first success
        # via _INITIALIZED_PATHS so it only runs once per process per path.
        _guard_existing_db_is_healthy(path)
        resolved = str(path.resolve())
        conn = _sqlite_connect(path)
        try:
            conn.row_factory = sqlite3.Row
            with _INIT_LOCK:
                # WAL activation can take an exclusive lock while SQLite creates the
                # sidecar files for a fresh database. Keep it in the same process-local
                # critical section as schema initialization so concurrent gateway
                # startup threads do not race before _INITIALIZED_PATHS is populated.
                # WAL doesn't work on network filesystems (NFS/SMB/FUSE). Shared helper
                # falls back to DELETE with one WARNING so kanban stays usable there.
                # See hermes_state._WAL_INCOMPAT_MARKERS for detection logic.
                from hermes_state import apply_wal_with_fallback
                apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
                # FULL (was NORMAL): fsync before each checkpoint to narrow the
                # crash window that can leave a b-tree page header torn.
                conn.execute("PRAGMA synchronous=FULL")
                # SQLite's default 1000 pages keeps checkpoint churn low under
                # worker bursts while synchronous=FULL still fsyncs committed WAL
                # frames.  A lower value forces frequent checkpoints and expands
                # the window for integrity probes/invariants to observe hot main
                # DB state mid-checkpoint.
                conn.execute("PRAGMA wal_autocheckpoint=1000")
                conn.execute("PRAGMA foreign_keys=ON")
                # Zero freed pages so a later torn write cannot expose stale
                # cell content; persisted in the DB header for new DBs.
                conn.execute("PRAGMA secure_delete=ON")
                # Surface corrupt cells as read errors instead of silent
                # wrong-data returns.
                conn.execute("PRAGMA cell_size_check=ON")
                needs_init = resolved not in _INITIALIZED_PATHS
                if needs_init:
                    # Idempotent: runs CREATE TABLE IF NOT EXISTS + the additive
                    # migrations. Cached so subsequent connect() calls in the same
                    # process are cheap. The lock prevents same-process dispatcher
                    # threads from racing through the additive ALTER TABLE pass with
                    # stale PRAGMA snapshots during gateway startup.
                    # Reject ambiguous legacy attention bindings before SCHEMA_SQL
                    # or an additive migration can persist any partial schema.
                    _preflight_task_attention_migration(conn)
                    conn.executescript(SCHEMA_SQL)
                    # Individual migrations use their own write transactions
                    # where they need a lock; keep this entry point compatible
                    # with those nested lifecycle migrations.
                    _migrate_add_optional_columns(conn)
                    canonical_board_path = str(kanban_db_path(board=board).resolve())
                    if db_path is None or resolved == canonical_board_path:
                        try:
                            _scavenge_completion_artifacts(conn, board=board)
                        except Exception as exc:
                            _log.warning("kanban artifact scavenger failed: %s", exc)
                    _INITIALIZED_PATHS.add(resolved)
        except Exception:
            conn.close()
            raise
    return conn


@contextlib.contextmanager
def connect_closing(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
):
    """Open a kanban DB connection and guarantee it is closed on exit.

    Use this instead of ``with kb.connect() as conn:`` — sqlite3's
    built-in connection context manager only commits/rollbacks the
    transaction; it does NOT close the file descriptor. In long-lived
    processes (gateway, dashboard) that route every kanban operation
    through ``connect()`` (e.g. ``run_slash`` dispatching ``/kanban …``
    commands, ``decompose_task_endpoint`` calling
    ``kanban_decompose.decompose_task``), the unclosed connections
    accumulate as open FDs to ``kanban.db`` and ``kanban.db-wal``. After
    enough operations the process hits the kernel FD limit and dies
    with ``[Errno 24] Too many open files``.

    See #33159 for the production incident.

    The ``connect()`` function itself remains unchanged so callers that
    intentionally manage the connection lifetime (tests, long-lived
    callers) continue to work.
    """
    conn = connect(db_path=db_path, board=board)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> Path:
    """Create the schema if it doesn't exist; return the path used.

    Kept as a public entry point so CLI ``hermes kanban init`` and the
    daemon have something explicit to call. Unlike :func:`connect`'s
    first-time auto-init (which caches by path), ``init_db`` always
    re-runs the migration pass. Callers that know the on-disk schema
    may have drifted — tests that write legacy event kinds directly,
    external tools that upgrade an old DB file — can call this to
    force re-migration.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    # Clear the cache entry so the underlying connect() re-runs the
    # schema + migration pass unconditionally.
    with _INIT_LOCK:
        _INITIALIZED_PATHS.discard(resolved)
    with contextlib.closing(connect(path, board=board)):
        pass
    return path


def _preflight_task_attention_migration(conn: sqlite3.Connection) -> None:
    """Reject ambiguous legacy attention data before any initializer mutation.

    ``executescript`` commits any open transaction before its DDL, so these
    fail-closed checks must run before it, every ALTER, index, rebuild, or
    backfill.  Do not select an arbitrary duplicate during migration.
    """
    existing = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('task_attentions', 'task_pending_actions')"
        )
    }
    if "task_attentions" not in existing:
        return
    attention_cols = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(task_attentions)")
    }
    if "action_id" not in attention_cols:
        return
    duplicate_action = conn.execute(
        "SELECT action_id FROM task_attentions WHERE action_id IS NOT NULL "
        "GROUP BY action_id HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if duplicate_action is not None:
        raise RuntimeError("cannot rebuild task attentions: duplicate non-null action_id")
    if "task_pending_actions" not in existing:
        return
    action_cols = {
        str(row["name"]) for row in conn.execute("PRAGMA table_info(task_pending_actions)")
    }
    if "attention_id" in action_cols:
        duplicate_attention = conn.execute(
            "SELECT attention_id FROM task_pending_actions WHERE attention_id IS NOT NULL "
            "GROUP BY attention_id HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate_attention is not None:
            raise RuntimeError("cannot migrate pending actions: duplicate attention_id binding")
        unbound = "a.attention_id IS NULL"
    else:
        unbound = "1=1"
    if {"id", "task_id"} <= action_cols and {"task_id", "type"} <= attention_cols:
        ambiguous = conn.execute(
            "SELECT a.id FROM task_pending_actions a JOIN task_attentions x "
            "ON x.action_id=a.id AND x.task_id=a.task_id AND x.type='exact_action' "
            f"WHERE {unbound} GROUP BY a.id HAVING COUNT(*) != 1 LIMIT 1"
        ).fetchone()
        if ambiguous is not None:
            raise RuntimeError("cannot migrate pending actions: ambiguous exact attention backfill")


def _migrate_task_attention_shape(conn: sqlite3.Connection) -> None:
    """Rebuild the old action-only projection table transactionally.

    SQLite cannot relax the historical ``action_id NOT NULL UNIQUE`` in place.
    Keep explicit ids so public opaque identities survive the migration.
    """
    columns = {row["name"]: row for row in conn.execute("PRAGMA table_info(task_attentions)")}
    if not columns:
        return
    action = columns.get("action_id")
    needs_rebuild = (action is not None and int(action["notnull"]) == 1) or "version" not in columns or "origin_run_id" not in columns
    if not needs_rebuild:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_task_attentions_action_live "
                     "ON task_attentions(action_id) WHERE action_id IS NOT NULL")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_task_attentions_task_cause "
                     "ON task_attentions(task_id, cause_fingerprint)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_attentions_task_created "
                     "ON task_attentions(task_id, created_at DESC)")
        return
    duplicates = conn.execute("SELECT action_id FROM task_attentions WHERE action_id IS NOT NULL GROUP BY action_id HAVING COUNT(*) > 1 LIMIT 1").fetchone()
    if duplicates is not None:
        raise RuntimeError("cannot rebuild task attentions: duplicate non-null action_id")
    conn.execute("DROP TABLE IF EXISTS task_attentions_rebuild")
    conn.execute("CREATE TABLE task_attentions_rebuild ("
                 "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, action_id INTEGER, "
                 "type TEXT NOT NULL CHECK(type IN ('exact_action','decision','capability','transient','loop_triage','protocol','review')), "
                 "cause_fingerprint TEXT NOT NULL, summary TEXT NOT NULL, created_at INTEGER NOT NULL, "
                 "version INTEGER NOT NULL DEFAULT 1, origin_run_id INTEGER, UNIQUE(task_id, cause_fingerprint))")
    version_expr = "COALESCE(version, 1)" if "version" in columns else "1"
    origin_expr = "origin_run_id" if "origin_run_id" in columns else "NULL"
    conn.execute("INSERT INTO task_attentions_rebuild (id, task_id, action_id, type, cause_fingerprint, summary, created_at, version, origin_run_id) "
                 f"SELECT id, task_id, action_id, type, cause_fingerprint, summary, created_at, {version_expr}, {origin_expr} FROM task_attentions")
    conn.execute("DROP TABLE task_attentions")
    conn.execute("ALTER TABLE task_attentions_rebuild RENAME TO task_attentions")
    conn.execute("CREATE UNIQUE INDEX uq_task_attentions_action_live ON task_attentions(action_id) WHERE action_id IS NOT NULL")
    conn.execute("CREATE UNIQUE INDEX uq_task_attentions_task_cause ON task_attentions(task_id, cause_fingerprint)")
    conn.execute("CREATE INDEX idx_task_attentions_task_created ON task_attentions(task_id, created_at DESC)")


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after v1 release to legacy DBs.

    Called by ``init_db`` so opening an old DB is always safe.
    """
    delivery_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(kanban_attention_deliveries)")
    }
    if delivery_cols and "lease_version" not in delivery_cols:
        _add_column_if_missing(
            conn,
            "kanban_attention_deliveries",
            "lease_version",
            "lease_version INTEGER NOT NULL DEFAULT 0",
        )
    if delivery_cols and "subscription_generation" not in delivery_cols:
        _add_column_if_missing(
            conn, "kanban_attention_deliveries", "subscription_generation",
            "subscription_generation INTEGER NOT NULL DEFAULT 1",
        )

    action_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(task_pending_actions)")
    }
    if action_cols and "fingerprint" not in action_cols:
        _add_column_if_missing(
            conn, "task_pending_actions", "fingerprint", "fingerprint TEXT"
        )
    if action_cols and "mutation_kind" not in action_cols:
        _add_column_if_missing(
            conn, "task_pending_actions", "mutation_kind", "mutation_kind TEXT"
        )
    if action_cols and "cancelled_at" not in action_cols:
        _add_column_if_missing(
            conn, "task_pending_actions", "cancelled_at", "cancelled_at INTEGER"
        )
    if action_cols and "attention_id" not in action_cols:
        _add_column_if_missing(
            conn, "task_pending_actions", "attention_id", "attention_id INTEGER"
        )
    if action_cols and "retry_origin_run_id" not in action_cols:
        _add_column_if_missing(
            conn, "task_pending_actions", "retry_origin_run_id", "retry_origin_run_id INTEGER"
        )
    if action_cols:
        _migrate_task_attention_shape(conn)
        duplicate_attention = conn.execute(
            "SELECT attention_id FROM task_pending_actions WHERE attention_id IS NOT NULL "
            "GROUP BY attention_id HAVING COUNT(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate_attention is not None:
            raise RuntimeError("cannot migrate pending actions: duplicate attention_id binding")
        # Only one exact, task-bound row is safe to backfill.  A correlated
        # scalar subquery would silently select an arbitrary duplicate in a
        # tampered legacy DB, so reject before writing anything.
        ambiguous = conn.execute(
            "SELECT a.id FROM task_pending_actions a JOIN task_attentions x "
            "ON x.action_id=a.id AND x.task_id=a.task_id AND x.type='exact_action' "
            "WHERE a.attention_id IS NULL GROUP BY a.id HAVING COUNT(*) != 1 LIMIT 1"
        ).fetchone()
        if ambiguous is not None:
            raise RuntimeError("cannot migrate pending actions: ambiguous exact attention backfill")
        conn.execute(
            "UPDATE task_pending_actions SET attention_id=(SELECT x.id FROM task_attentions x "
            "WHERE x.action_id=task_pending_actions.id AND x.task_id=task_pending_actions.task_id "
            "AND x.type='exact_action') WHERE attention_id IS NULL AND EXISTS "
            "(SELECT 1 FROM task_attentions x WHERE x.action_id=task_pending_actions.id "
            "AND x.task_id=task_pending_actions.task_id AND x.type='exact_action')"
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_actions_attention_id "
                     "ON task_pending_actions(attention_id) WHERE attention_id IS NOT NULL")
    # Exact-action lifecycle is additive and remains authoritative; the linked
    # attention row is deliberately only an operator projection.
    if action_cols:
        for name, definition in (
            ("state", "state TEXT NOT NULL DEFAULT 'pending'"),
            ("version", "version INTEGER NOT NULL DEFAULT 1"),
            ("updated_at", "updated_at INTEGER NOT NULL DEFAULT 0"),
            ("resolved_at", "resolved_at INTEGER"),
        ):
            if name not in action_cols:
                _add_column_if_missing(conn, "task_pending_actions", name, definition)
        _migrate_pending_action_lifecycle(conn)

    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "tenant" not in cols:
        _add_column_if_missing(conn, "tasks", "tenant", "tenant TEXT")
    if "result" not in cols:
        _add_column_if_missing(conn, "tasks", "result", "result TEXT")
    if "branch_name" not in cols:
        _add_column_if_missing(conn, "tasks", "branch_name", "branch_name TEXT")
    if "project_id" not in cols:
        _add_column_if_missing(conn, "tasks", "project_id", "project_id TEXT")
    if "idempotency_key" not in cols:
        _add_column_if_missing(
            conn, "tasks", "idempotency_key", "idempotency_key TEXT"
        )
    # ``idx_tasks_idempotency`` is created unconditionally below alongside
    # the other additive-column indexes — see the block after the
    # legacy-column migration. Creating it here too would be redundant.

    # Refresh after early additive migrations above. Some existing DBs were
    # partially migrated in older releases and can already contain the later
    # columns (for example ``consecutive_failures``) even when this function's
    # initial snapshot did not. Re-snapshot here so the legacy-column migration
    # below is truly idempotent and never re-adds columns that already exist.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}

    # Legacy column migration: ``spawn_failures`` → ``consecutive_failures``
    # and ``last_spawn_error`` → ``last_failure_error``.
    #
    # Avoid ``ALTER TABLE ... RENAME COLUMN`` for two reasons:
    #   1. Primary: very old DBs may never have had ``spawn_failures`` at
    #      all, so RENAME raises OperationalError: no such column (the crash
    #      reported in issue #20842 after the #20410 update).
    #   2. Secondary: SQLite reparses the whole schema on any RENAME, which
    #      fails if related objects (views, triggers) reference the old name.
    #
    # ADD-first-then-copy is tolerant of both shapes and preserves
    # historical counter values when the legacy columns do exist.
    if "consecutive_failures" not in cols:
        added = _add_column_if_missing(
            conn,
            "tasks",
            "consecutive_failures",
            "consecutive_failures INTEGER NOT NULL DEFAULT 0",
        )
        if added and "spawn_failures" in cols:
            conn.execute(
                "UPDATE tasks SET consecutive_failures = COALESCE(spawn_failures, 0)"
            )
    if "worker_pid" not in cols:
        _add_column_if_missing(conn, "tasks", "worker_pid", "worker_pid INTEGER")
    if "last_failure_error" not in cols:
        added = _add_column_if_missing(
            conn, "tasks", "last_failure_error", "last_failure_error TEXT"
        )
        if added and "last_spawn_error" in cols:
            conn.execute(
                "UPDATE tasks SET last_failure_error = last_spawn_error"
            )
    if "max_runtime_seconds" not in cols:
        _add_column_if_missing(
            conn, "tasks", "max_runtime_seconds", "max_runtime_seconds INTEGER"
        )
    if "last_heartbeat_at" not in cols:
        _add_column_if_missing(
            conn, "tasks", "last_heartbeat_at", "last_heartbeat_at INTEGER"
        )
    if "last_activity_at" not in cols:
        _add_column_if_missing(
            conn, "tasks", "last_activity_at", "last_activity_at INTEGER"
        )
    if "last_semantic_progress_at" not in cols:
        _add_column_if_missing(
            conn, "tasks", "last_semantic_progress_at", "last_semantic_progress_at INTEGER"
        )
    if "current_run_id" not in cols:
        _add_column_if_missing(
            conn, "tasks", "current_run_id", "current_run_id INTEGER"
        )
    if "workflow_template_id" not in cols:
        _add_column_if_missing(
            conn, "tasks", "workflow_template_id", "workflow_template_id TEXT"
        )
    if "current_step_key" not in cols:
        _add_column_if_missing(
            conn, "tasks", "current_step_key", "current_step_key TEXT"
        )
    if "skills" not in cols:
        # JSON array of skill names the dispatcher force-loads into the
        # worker via --skills. NULL is fine for existing rows.
        _add_column_if_missing(conn, "tasks", "skills", "skills TEXT")

    if "max_retries" not in cols:
        # Per-task override for the consecutive-failure circuit breaker.
        # NULL = fall through to the dispatcher-level ``kanban.failure_limit``
        # config, then ``DEFAULT_FAILURE_LIMIT``. Existing rows get NULL,
        # which is the correct default (they keep the global behaviour
        # they were getting before the column existed).
        _add_column_if_missing(conn, "tasks", "max_retries", "max_retries INTEGER")

    if "model_override" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN model_override TEXT")

    if "effort" not in cols:
        # Per-task reasoning-effort override written by the tars-workflow plugin
        # (chained goal_mode cards). The plugin issues UPDATE tasks SET effort=?;
        # existing boards happened to carry the column, but a freshly `kb init`'d
        # board lacked it → create_chain with an effort override raised
        # OperationalError: no such column: effort. NULL = engine default.
        _add_column_if_missing(conn, "tasks", "effort", "effort TEXT")

    if "task_class" not in cols:
        # Optional quality class / task grade (e.g. "hard"). Set manually via
        # ``--class`` or by a plugin; read by the decomposer to pick a
        # class-specific planner auxiliary role. NULL for existing rows =
        # unclassified = today's behaviour.
        _add_column_if_missing(conn, "tasks", "task_class", "task_class TEXT")

    if "goal_mode" not in cols:
        # Ralph-style goal loop toggle for the dispatched worker. 0 (the
        # default) = classic single-shot worker, preserving the behaviour
        # existing rows had before the column existed.
        _add_column_if_missing(
            conn, "tasks", "goal_mode", "goal_mode INTEGER NOT NULL DEFAULT 0"
        )

    if "goal_max_turns" not in cols:
        # Per-task goal-loop turn budget. NULL = goals-engine default.
        _add_column_if_missing(
            conn, "tasks", "goal_max_turns", "goal_max_turns INTEGER"
        )

    if "session_id" not in cols:
        # Originating agent/chat session id, populated when the task is
        # created from within an agent loop that propagated
        # ``HERMES_SESSION_ID`` (e.g. ACP). NULL on legacy rows and on any
        # creation path that doesn't set the env var (CLI, dashboard).
        _add_column_if_missing(
            conn, "tasks", "session_id", "session_id TEXT"
        )

    if "block_kind" not in cols:
        # Typed block reason (VALID_BLOCK_KINDS) or NULL for legacy/un-typed
        # blocks. Existing blocked rows get NULL, which is treated as a
        # generic human blocker — same behaviour they had before the column.
        _add_column_if_missing(conn, "tasks", "block_kind", "block_kind TEXT")

    if "block_recurrences" not in cols:
        # Unblock-loop counter. Existing rows start at 0, so the loop breaker
        # only begins counting from the first re-block after this migration.
        _add_column_if_missing(
            conn,
            "tasks",
            "block_recurrences",
            "block_recurrences INTEGER NOT NULL DEFAULT 0",
        )

    # Legacy block rows have no auditable typed cause. Keep these NULL rather
    # than guessing from event prose, so their old counter cannot trip a new
    # cause's loop threshold after upgrade.
    for name, definition in (
        ("block_cause_fingerprint", "block_cause_fingerprint TEXT"),
        ("block_reason_code", "block_reason_code TEXT"),
        ("block_cause_version", "block_cause_version INTEGER"),
    ):
        if name not in cols:
            _add_column_if_missing(conn, "tasks", name, definition)

    if "completion_contract" not in cols:
        _add_column_if_missing(
            conn, "tasks", "completion_contract", "completion_contract TEXT"
        )

    if "human_gate" not in cols:
        # Human-Gate v1. 0 (the default) = today's behaviour for every
        # existing row; only a card explicitly marked via the CLI opts in.
        _add_column_if_missing(
            conn, "tasks", "human_gate", "human_gate INTEGER NOT NULL DEFAULT 0"
        )
    if "gate_token_hash" not in cols:
        _add_column_if_missing(conn, "tasks", "gate_token_hash", "gate_token_hash TEXT")
    if "gate_token_issued_at" not in cols:
        _add_column_if_missing(
            conn, "tasks", "gate_token_issued_at", "gate_token_issued_at INTEGER"
        )
    for name, definition in (
        ("gate_token_board", "gate_token_board TEXT"),
        ("gate_token_task_id", "gate_token_task_id TEXT"),
        ("gate_token_action", "gate_token_action TEXT"),
        ("gate_scope_hash", "gate_scope_hash TEXT"),
        ("gate_scope_version", "gate_scope_version INTEGER"),
        ("governance_target_ref", "governance_target_ref TEXT"),
        ("governance_mutation_class", "governance_mutation_class TEXT"),
        ("gate_failed_attempts", "gate_failed_attempts INTEGER NOT NULL DEFAULT 0"),
        ("gate_failure_window_started_at", "gate_failure_window_started_at INTEGER"),
        ("gate_locked_until", "gate_locked_until INTEGER"),
    ):
        if name not in cols:
            _add_column_if_missing(conn, "tasks", name, definition)

    # Indexes over additive ``tasks`` columns must be created after the
    # columns exist. Keeping them in SCHEMA_SQL breaks legacy boards: SQLite
    # parses each statement in ``executescript`` against the live schema, so a
    # ``CREATE INDEX`` over a missing column aborts initialization before the
    # additive ``ALTER TABLE`` migrations below can run. Re-running them here
    # is cheap thanks to ``IF NOT EXISTS`` and stays correct on fresh DBs
    # (where the columns already exist from SCHEMA_SQL).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)"
    )

    # task_events gained a run_id column; back-fill it as NULL for
    # historical events (they predate runs and can't be attributed).
    ev_cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_events)")}
    if "run_id" not in ev_cols:
        _add_column_if_missing(conn, "task_events", "run_id", "run_id INTEGER")

    # Same ordering rule as the additive ``tasks`` indexes above: create the
    # index after the additive column migration so legacy ``task_events``
    # tables don't fail during SCHEMA_SQL execution before ``run_id`` exists.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_run "
        "ON task_events(run_id, id)"
    )

    # task_runs gained a session_id column (event-sourcing resume, #2). It pins
    # the worker's resumable session id for the run so a re-claim after a crash
    # can continue the same agent conversation instead of starting fresh. NULL on
    # legacy runs and whenever kanban.resume_on_reclaim is off. Additive only —
    # mirrors the tasks.session_id migration above; an older binary ignores it.
    runs_table_present = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs'"
    ).fetchone() is not None
    if runs_table_present:
        run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")}
        if "session_id" not in run_cols:
            _add_column_if_missing(conn, "task_runs", "session_id", "session_id TEXT")
        for name, definition in (
            ("last_activity_at", "last_activity_at INTEGER"),
            ("last_semantic_progress_at", "last_semantic_progress_at INTEGER"),
            ("worker_start_ticks", "worker_start_ticks INTEGER"),
            ("d_state_since", "d_state_since INTEGER"),
            ("resource_sample", "resource_sample TEXT"),
            ("termination_pending_since", "termination_pending_since INTEGER"),
        ):
            if name not in run_cols:
                _add_column_if_missing(conn, "task_runs", name, definition)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_session ON task_runs(session_id)"
        )

    notify_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kanban_notify_subs'"
    ).fetchone() is not None
    if notify_table_exists:
        notify_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(kanban_notify_subs)")
        }
        if "notifier_profile" not in notify_cols:
            _add_column_if_missing(
                conn, "kanban_notify_subs", "notifier_profile", "notifier_profile TEXT"
            )
        if "active" not in notify_cols:
            _add_column_if_missing(
                conn, "kanban_notify_subs", "active", "active INTEGER NOT NULL DEFAULT 1"
            )
        if "generation" not in notify_cols:
            _add_column_if_missing(
                conn, "kanban_notify_subs", "generation", "generation INTEGER NOT NULL DEFAULT 1"
            )

    # One-shot backfill: any task that is 'running' before runs existed
    # had its claim_lock / claim_expires / worker_pid on the task row.
    # Synthesize a matching task_runs row so subsequent end-run / heartbeat
    # calls have something to write to. Wrapped in write_txn to serialize
    # against any concurrent dispatcher, and the per-row UPDATE uses
    # ``current_run_id IS NULL`` as a CAS guard so a racing claim can't
    # produce an orphaned row if it interleaves with the backfill pass.
    runs_exist = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs'"
    ).fetchone() is not None
    if runs_exist:
        with write_txn(conn):
            inflight = conn.execute(
                "SELECT id, assignee, claim_lock, claim_expires, worker_pid, "
                "       max_runtime_seconds, last_heartbeat_at, started_at "
                "FROM tasks "
                "WHERE status = 'running' AND current_run_id IS NULL"
            ).fetchall()
            for row in inflight:
                started = row["started_at"] or int(time.time())
                cur = conn.execute(
                    """
                    INSERT INTO task_runs (
                        task_id, profile, status,
                        claim_lock, claim_expires, worker_pid,
                        max_runtime_seconds, last_heartbeat_at,
                        started_at
                    ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"], row["assignee"], row["claim_lock"],
                        row["claim_expires"], row["worker_pid"],
                        row["max_runtime_seconds"], row["last_heartbeat_at"],
                        started,
                    ),
                )
                # CAS: only install the pointer if nothing else claimed
                # the task between our SELECT and here (shouldn't happen
                # under the write_txn, but belt-and-suspenders). If the
                # CAS fails we've got an orphan run_row — mark it
                # reclaimed so it doesn't look in-flight.
                upd = conn.execute(
                    "UPDATE tasks SET current_run_id = ? "
                    "WHERE id = ? AND current_run_id IS NULL",
                    (cur.lastrowid, row["id"]),
                )
                if upd.rowcount != 1:
                    conn.execute(
                        "UPDATE task_runs SET status = 'reclaimed', "
                        "    outcome = 'reclaimed', ended_at = ? "
                        "WHERE id = ?",
                        (int(time.time()), cur.lastrowid),
                    )

    # One-shot event-kind rename pass. The old names ("ready", "priority",
    # "spawn_auto_blocked") still worked but were awkward on the wire;
    # rename them in-place so existing DBs migrate cleanly. Fires once
    # per DB because after the UPDATE no rows match the old kinds.
    _EVENT_RENAMES = (
        # (old, new)
        ("ready",              "promoted"),
        ("priority",           "reprioritized"),
        ("spawn_auto_blocked", "gave_up"),
    )
    for old, new in _EVENT_RENAMES:
        conn.execute(
            "UPDATE task_events SET kind = ? WHERE kind = ?",
            (new, old),
        )

    _rebuild_drifted_tables(conn)


# Legacy DBs defined these tables with a ``TEXT PRIMARY KEY`` id (or, for
# ``kanban_notify_subs``, a nullable ``TEXT last_event_id``). The current
# schema uses ``INTEGER PRIMARY KEY AUTOINCREMENT`` / ``INTEGER NOT NULL
# DEFAULT 0``. ``CREATE TABLE IF NOT EXISTS`` skips existing tables
# regardless of schema and ``_add_column_if_missing`` only adds columns, so
# neither can fix a drifted column type — the table must be rebuilt. See
# #35096.
#
# Each entry pairs the canonical CREATE TABLE with the CREATE INDEX
# statements that DROP TABLE would otherwise take down with it (including
# ``idx_events_run``, added by the additive pass above). To guard against
# this list drifting from SCHEMA_SQL, ``test_rebuilt_schema_matches_fresh``
# asserts a rebuilt legacy DB is byte-identical to a fresh one.
_REBUILD_SPECS = {
    "task_events": (
        "CREATE TABLE task_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, run_id INTEGER, kind TEXT NOT NULL,"
        " payload TEXT, created_at INTEGER NOT NULL)",
        (
            "CREATE INDEX idx_events_task ON task_events(task_id, created_at)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ),
    ),
    "task_comments": (
        "CREATE TABLE task_comments ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL,"
        " created_at INTEGER NOT NULL)",
        ("CREATE INDEX idx_comments_task ON task_comments(task_id, created_at)",),
    ),
    "task_runs": (
        "CREATE TABLE task_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, profile TEXT, step_key TEXT,"
        " status TEXT NOT NULL, claim_lock TEXT, claim_expires INTEGER,"
        " worker_pid INTEGER, max_runtime_seconds INTEGER,"
        " last_heartbeat_at INTEGER, last_activity_at INTEGER,"
        " last_semantic_progress_at INTEGER, worker_start_ticks INTEGER,"
        " d_state_since INTEGER, resource_sample TEXT,"
        " termination_pending_since INTEGER, started_at INTEGER NOT NULL,"
        " ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT,"
        " error TEXT, session_id TEXT)",
        (
            "CREATE INDEX idx_runs_task ON task_runs(task_id, started_at)",
            "CREATE INDEX idx_runs_status ON task_runs(status)",
            "CREATE INDEX idx_runs_session ON task_runs(session_id)",
        ),
    ),
    "kanban_notify_subs": (
        "CREATE TABLE kanban_notify_subs ("
        " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,"
        " notifier_profile TEXT, created_at INTEGER NOT NULL,"
        " last_event_id INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,"
        " generation INTEGER NOT NULL DEFAULT 1,"
        " PRIMARY KEY (task_id, platform, chat_id, thread_id))",
        ("CREATE INDEX idx_notify_task ON kanban_notify_subs(task_id)",),
    ),
}


def _table_has_drifted(conn: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` still carries the legacy (pre-AUTOINCREMENT) shape."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not info:
        return False  # table absent — nothing to rebuild
    if table == "kanban_notify_subs":
        lei = next((c for c in info if c["name"] == "last_event_id"), None)
        return lei is not None and (lei["type"] or "").upper() != "INTEGER"
    # task_events / task_comments / task_runs: id must be INTEGER and a PK.
    id_col = next((c for c in info if c["name"] == "id"), None)
    if id_col is None:
        return False
    return not ((id_col["type"] or "").upper() == "INTEGER" and id_col["pk"])


def _rebuild_drifted_tables(conn: sqlite3.Connection) -> None:
    """Rebuild any kanban table whose column types drifted from SCHEMA_SQL.

    Old boards crash the gateway notifier (``int(None)`` on a NULL id in
    ``unseen_events_for_sub``) and never match the ``id > cursor`` filter, so
    every kanban notification is silently lost (#35096). Each affected table is
    rebuilt with the standard SQLite pattern — CREATE new → INSERT shared
    columns → DROP old → RENAME — recreating its indexes too (DROP TABLE takes
    them down). The legacy TEXT ids are dropped (they aren't valid integers);
    AUTOINCREMENT assigns fresh ones and ``last_event_id`` cursors reset to 0,
    so the first post-migration tick replays a task's event history once —
    the safe failure mode for a feature that was already fully broken.

    The whole pass runs in one transaction so an interruption can't leave a
    table half-renamed, and under ``connect()``'s init locks so nothing races
    it. Idempotent: a correctly-typed DB skips every table and returns without
    opening a transaction.
    """
    drifted = [t for t in _REBUILD_SPECS if _table_has_drifted(conn, t)]
    if not drifted:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in drifted:
            create_sql, index_sqls = _REBUILD_SPECS[table]
            old_cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})")]
            _log.info("kanban migration: rebuilding %s to match current schema", table)
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            conn.execute(create_sql)
            new_cols = {c["name"] for c in conn.execute(f"PRAGMA table_info({table})")}
            if table == "kanban_notify_subs":
                # Cast the legacy TEXT cursor to INTEGER; NULL / non-numeric → 0.
                shared = [c for c in old_cols if c in new_cols and c != "last_event_id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}, last_event_id) "
                    f"SELECT {cols_csv}, COALESCE(CAST(last_event_id AS INTEGER), 0) "
                    f"FROM {table}_legacy"
                )
            else:
                # Drop the legacy TEXT id; AUTOINCREMENT reassigns it.
                shared = [c for c in old_cols if c in new_cols and c != "id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}) "
                    f"SELECT {cols_csv} FROM {table}_legacy"
                )
            conn.execute(f"DROP TABLE {table}_legacy")
            for index_sql in index_sqls:
                conn.execute(index_sql)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise


def _check_file_length_invariant(conn: sqlite3.Connection) -> None:
    """Compare logical page count against file size in rollback-journal mode.

    Raises sqlite3.DatabaseError if the file is shorter than SQLite's logical
    snapshot. This invariant is only meaningful for rollback journal modes. In
    WAL mode the main DB file may legitimately lag while committed frames still
    live in ``-wal``; comparing only the main file there races normal checkpoints
    and false-positives under parallel kanban workers.

    For rollback journals, hold a SQLite read transaction while observing both
    ``page_count`` and the main-file size. The previous implementation read the
    header and ``stat()`` independently after COMMIT, so another process could
    commit between those reads and manufacture a false mismatch even without
    WAL. A read snapshot keeps valid SQLite writers out until the check ends.
    """
    started_read_txn = False
    try:
        journal_mode_row = conn.execute("PRAGMA journal_mode").fetchone()
        if journal_mode_row and str(journal_mode_row[0]).lower() == "wal":
            return
        if not conn.in_transaction:
            conn.execute("BEGIN")
            started_read_txn = True
        # BEGIN DEFERRED alone does not acquire a shared read lock.
        conn.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone()
        row = conn.execute("PRAGMA database_list").fetchone()
        if row is None:
            return
        path_str = row[2]  # column 2 is the file path; empty for in-memory DBs
        if not path_str:
            return  # in-memory or unnamed DB; skip
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        logical_pages = conn.execute("PRAGMA page_count").fetchone()[0]
        file_size = os.path.getsize(path_str)
        if logical_pages == 0:
            return  # new/empty DB; skip
        actual_pages = file_size // page_size
        if file_size % page_size != 0 or actual_pages < logical_pages:
            raise sqlite3.DatabaseError(
                f"torn-extend detected: page count mismatch on {path_str}: "
                f"SQLite snapshot reports {logical_pages} pages, "
                f"file has {actual_pages} pages "
                f"(missing {max(0, logical_pages - actual_pages)} pages, "
                f"file_size={file_size}, page_size={page_size})"
            )
    except sqlite3.DatabaseError:
        raise
    except Exception:
        pass  # I/O errors during check are non-fatal; let normal ops continue
    finally:
        if started_read_txn:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass


# SQLite's own busy_timeout uses a near-deterministic backoff, so concurrent
# writers re-collide in lockstep under a stampede. A jittered retry on the
# transaction boundary breaks that convoy. Mirrors state.db's _execute_write:
# a fixed 20-150ms jitter band (a 20ms floor prevents a near-zero retry from
# busy-spinning back into the collision). Only BEGIN IMMEDIATE and COMMIT are
# retried -- both are idempotent re-issues that touch no transaction body, so a
# CAS inside write_txn is never replayed. kanban keeps fewer retries than
# state.db (5 vs 15) because its 120s busy_timeout already absorbs most waits;
# the retry is the backstop for the tail SQLite returns BUSY on immediately.
_BUSY_MAX_RETRIES = 5
_BUSY_RETRY_MIN_S = 0.020  # 20ms
_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """Context manager for an IMMEDIATE write transaction.

    Use for any multi-statement write (creating a task + link, claiming a
    task + recording an event, etc.).  A claim CAS inside this context is
    atomic -- at most one concurrent writer can succeed.

    The explicit ROLLBACK on exception is wrapped in try/except so that
    a SQLite auto-rollback (which leaves no active transaction) does not
    shadow the original exception with a spurious rollback error.
    """
    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception as exc:
        # Authentication failures persist only their bounded failure/audit
        # counters. Human-gate guards run immediately before the protected
        # transition, so no status mutation has occurred when this commits.
        if isinstance(exc, GateTokenError) and exc.persist_failure:
            _execute_boundary_with_retry(conn, "COMMIT")
            raise
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            # SQLite has already auto-rolled-back the transaction (typical
            # under EIO, lock contention, or corruption). Nothing to undo;
            # do not let this secondary failure shadow the real one.
            pass
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            # COMMIT exhausted retries with the txn still open; roll back so the
            # connection isn't poisoned for the next BEGIN IMMEDIATE.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        # This guard runs after COMMIT. Surface a distinct exception so callers
        # know the mutation may not be retried as though it had rolled back.
        try:
            _check_file_length_invariant(conn)
        except Exception as exc:
            raise PostCommitIntegrityError(
                "post-commit integrity guard failed; transaction is already "
                "committed and must not be retried blindly"
            ) from exc


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_task_id() -> str:
    """Generate a short, URL-safe task id.

    4 hex bytes = ~4.3B possibilities. At 10k tasks the collision
    probability is ~1.2e-5; at 100k it's ~1.2e-3. Previously we used 2
    hex bytes (65k possibilities) which hit the birthday paradox hard:
    ~5% collision probability at 1k tasks, ~50% at 10k. Callers that
    care about idempotency should pass ``idempotency_key`` to
    :func:`create_task` rather than rely on id uniqueness.
    """
    return "t_" + secrets.token_hex(4)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Task creation / mutation
# ---------------------------------------------------------------------------

def _canonical_assignee(assignee: Optional[str]) -> Optional[str]:
    """Lowercase-assignee normalization for Kanban rows (dashboard/CLI parity)."""
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name

    return normalize_profile_name(assignee)


def _reject_foreign_workspace_refs(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    title: str,
    body: Optional[str],
    board: Optional[str],
) -> None:
    """Reject handoffs that depend on another task's disposable workspace."""
    root = str(workspaces_root(board=board)).rstrip("/\\")
    if not root:
        return
    pattern = re.compile(
        re.escape(root) + r"[/\\](t_[A-Za-z0-9_-]+)(?=$|[/\\\s'\"`),.;:!?])"
    )
    referenced_ids = {
        match.group(1)
        for text in (title, body or "")
        for match in pattern.finditer(text)
        if match.group(1) != task_id
    }
    if not referenced_ids:
        return

    details: list[str] = []
    for referenced_id in sorted(referenced_ids):
        durable_paths = [
            str(row["durable_path"])
            for row in conn.execute(
                "SELECT durable_path FROM task_artifacts "
                "WHERE task_id = ? ORDER BY producer_run_id, id",
                (referenced_id,),
            ).fetchall()
        ]
        separator = "\\" if "\\" in root and "/" not in root else "/"
        detail = f"{root}{separator}{referenced_id}"
        if durable_paths:
            detail += "\n  durable artifact path(s):\n  - " + "\n  - ".join(durable_paths)
        details.append(detail)

    raise ValueError(
        "task title/body references another task's scratch workspace, which may "
        "be deleted after that task completes. Use the referenced task's durable "
        "artifact path instead, or explicitly set allow_workspace_refs=True only "
        "for an intentional live-workspace dependency:\n- "
        + "\n- ".join(details)
    )


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: Optional[str] = None,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None,
    tenant: Optional[str] = None,
    priority: int = 0,
    parents: Iterable[str] = (),
    triage: bool = False,
    idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None,
    skills: Optional[Iterable[str]] = None,
    task_class: Optional[str] = None,
    max_retries: Optional[int] = None,
    goal_mode: bool = False,
    goal_max_turns: Optional[int] = None,
    initial_status: str = "running",
    session_id: Optional[str] = None,
    board: Optional[str] = None,
    project_id: Optional[str] = None,
    completion_contract: Optional[dict] = None,
    allow_workspace_refs: bool = False,
) -> str:
    """Create a new task and optionally link it under parent tasks.

    Returns the new task id.  Status is ``ready`` when there are no
    parents (or all parents already ``done``), otherwise ``todo``.
    If ``triage=True``, status is forced to ``triage`` regardless of
    parents — a specifier/triager is expected to promote the task to
    ``todo`` once the spec is fleshed out.

    If ``idempotency_key`` is provided and a non-archived task with the
    same key already exists, returns the existing task's id instead of
    creating a duplicate. Useful for retried webhooks / automation that
    should not double-write.

    ``max_runtime_seconds`` caps how long a worker may run before the
    dispatcher SIGTERMs (then SIGKILLs after a grace window) and
    re-queues the task. ``None`` means no cap (default).

    ``skills`` is an optional list of skill names to force-load into
    the worker when dispatched. Stored as JSON; the dispatcher passes
    each name to ``hermes --skills ...``. Use this to pin a task to a
    specialist skill (e.g. ``skills=["translation"]`` so the worker loads the
    translation skill regardless of the profile's default config).

    ``task_class`` is an optional free-form quality-class label (e.g.
    ``"hard"``). It is read by the decomposer to select a class-specific
    planner auxiliary role; ``None`` (the common case) preserves today's
    behaviour.
    """
    assignee = _canonical_assignee(assignee)
    if completion_contract is not None:
        if not isinstance(completion_contract, dict):
            raise ValueError("completion_contract must be an object/dict")
        unknown_contract_keys = set(completion_contract) - {
            "tests_or_smokes", "readback", "artifacts"
        }
        if unknown_contract_keys:
            raise ValueError(
                "unknown completion_contract field(s): "
                + ", ".join(sorted(unknown_contract_keys))
            )
        completion_contract = {
            key: True for key, value in completion_contract.items() if value
        } or None
    if task_class is not None:
        task_class = str(task_class).strip() or None
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in VALID_INITIAL_STATUSES:
        raise ValueError(
            f"initial_status must be one of {sorted(VALID_INITIAL_STATUSES)}"
        )
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    # Resolve an optional first-class Project link. A project-linked task is
    # anchored to the project's primary repo as a git worktree, so its branch
    # can be named deterministically (project slug + task id) instead of the
    # random ``wt/<task-id>`` fallback the worker skill applies when no branch
    # is set. Projects live in the creator's per-profile projects.db; the repo
    # path is absolute (profile-independent) and the branch name is pure, so the
    # cross-profile dispatcher needs no projects.db access at dispatch time.
    project_obj = None
    # Primary repo of a project-linked worktree task whose path we still need to
    # derive (a fresh worktree dir under the repo, computed once task_id exists).
    project_repo: Optional[str] = None
    if project_id is not None:
        project_id = str(project_id).strip() or None
    if project_id:
        try:
            from hermes_cli import projects_db as _pdb

            with _pdb.connect_closing() as _pconn:
                project_obj = _pdb.get_project(_pconn, project_id)
        except Exception:
            project_obj = None
        if project_obj is None:
            # A project id/slug that doesn't resolve must not crash task
            # creation or persist a dangling reference — drop the link and
            # create the task as an ordinary (scratch) task.
            project_id = None
        else:
            # Canonicalise (a slug may have been passed) and anchor the
            # worktree under the project's primary repo.
            project_id = project_obj.id
            if workspace_kind == "scratch" and project_obj.primary_path:
                workspace_kind = "worktree"
            if (
                workspace_kind == "worktree"
                and workspace_path is None
                and project_obj.primary_path
            ):
                # Defer the concrete path to the insert loop: it's a fresh
                # ``<repo>/.worktrees/<task-id>`` dir keyed on the new task id.
                project_repo = str(project_obj.primary_path)

    parents = tuple(p for p in parents if p)

    # Normalise + validate skills: strip whitespace, drop empties, dedupe
    # (preserving order). Refuse commas inside a single name so we don't
    # invisibly splatter a comma-joined string into one argv slot — the
    # `hermes --skills X,Y` comma syntax is handled in the dispatcher,
    # not here.
    skills_list: Optional[list[str]] = None
    if skills is not None:
        cleaned: list[str] = []
        seen: set[str] = set()
        # Collect all toolset-name confusions up front so the user sees the
        # whole list at once. Raising on the first hit is friendly when the
        # input has one mistake, but agents that confuse skills with toolsets
        # usually pass several at once (`skills=["web", "browser", "terminal"]`)
        # and serial-correcting one per failure round-trips wastes tokens.
        toolset_typos: list[str] = []
        for s in skills:
            if not s:
                continue
            name = str(s).strip()
            if not name:
                continue
            if "," in name:
                raise ValueError(
                    f"skill name cannot contain comma: {name!r} "
                    f"(pass a list of separate names instead of a comma-joined string)"
                )
            if name.casefold() in KNOWN_TOOLSET_NAMES:
                toolset_typos.append(name)
                continue
            if name in seen:
                continue
            seen.add(name)
            cleaned.append(name)
        if toolset_typos:
            quoted = ", ".join(repr(n) for n in toolset_typos)
            noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
            raise ValueError(
                f"{quoted} {noun}, not skill name(s). "
                "Put toolsets in the assignee profile's `toolsets:` config "
                "instead of per-task skills. Skills are named skill bundles "
                "(e.g. `blogwatcher`, `github-code-review`); toolsets are runtime "
                "capabilities (e.g. `web`, `browser`, `terminal`)."
            )
        skills_list = cleaned

    # Idempotency check — return the existing task instead of creating a
    # duplicate. Done BEFORE entering write_txn to keep the fast path fast
    # and to avoid holding a write lock during the lookup. Race is
    # acceptable: two concurrent creators with the same key might both
    # insert, at which point both rows exist but the next lookup stabilises.
    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        if row:
            return row["id"]

    now = int(time.time())

    # Resolve workspace_path from board-level default_workdir when the
    # caller did not specify one explicitly. Board defaults represent
    # persistent project checkouts, so only persistent workspace kinds may
    # inherit them. Scratch workspaces are auto-deleted on completion and
    # must stay under the per-board scratch root created by
    # ``resolve_workspace``; inheriting ``default_workdir`` for a scratch
    # task would point cleanup at the user's source tree (#28818). The
    # containment guard in ``_cleanup_workspace`` is the safety rail, but
    # we also stop the bad state from being created in the first place.
    if (
        workspace_path is None
        and project_repo is None
        and workspace_kind in {"dir", "worktree"}
    ):
        board_slug = board if board else get_current_board()
        board_meta = read_board_metadata(board_slug)
        board_default = board_meta.get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            with write_txn(conn):
                if not allow_workspace_refs:
                    _reject_foreign_workspace_refs(
                        conn,
                        task_id=task_id,
                        title=title,
                        body=body,
                        board=board,
                    )
                # Determine task status from parent status, unless the caller
                # parks it directly in blocked for human-ops review or in
                # triage for a specifier.
                if initial_status == "blocked":
                    task_status = "blocked"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                elif triage:
                    task_status = "triage"
                else:
                    task_status = "ready"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                        # If any parent is not yet done, we're todo.
                        rows = conn.execute(
                            "SELECT status FROM tasks WHERE id IN "
                            "(" + ",".join("?" * len(parents)) + ")",
                            parents,
                        ).fetchall()
                        if any(r["status"] != "done" for r in rows):
                            task_status = "todo"
                # Even in triage mode we still need to validate parent ids
                # so the eventual link rows don't dangle.
                if triage and parents:
                    missing = _find_missing_parents(conn, parents)
                    if missing:
                        raise ValueError(f"unknown parent task(s): {', '.join(missing)}")

                # Project-linked worktree: a fresh worktree dir under the repo
                # plus a deterministic branch (project slug + task id). Together
                # these kill the random ``wt/<task-id>`` worker fallback and the
                # unanchored ``.worktrees/<id>`` under the dispatcher's cwd.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(
                            project_repo, ".worktrees", task_id
                        )
                    if not branch_name:
                        # _pdb was imported above when project_obj was resolved.
                        try:
                            branch_name = _pdb.branch_name_for(
                                project_obj, task_id, title=title or ""
                            )
                        except Exception:
                            branch_name = None

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, goal_mode, goal_max_turns, session_id,
                        task_class, completion_contract
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        title.strip(),
                        body,
                        assignee,
                        task_status,
                        priority,
                        created_by,
                        now,
                        workspace_kind,
                        workspace_path,
                        branch_name,
                        project_id,
                        tenant,
                        idempotency_key,
                        int(max_runtime_seconds) if max_runtime_seconds is not None else None,
                        json.dumps(skills_list) if skills_list is not None else None,
                        int(max_retries) if max_retries is not None else None,
                        1 if goal_mode else 0,
                        int(goal_max_turns) if goal_max_turns is not None else None,
                        session_id,
                        task_class,
                        json.dumps(completion_contract, sort_keys=True) if completion_contract else None,
                    ),
                )
                for pid in parents:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                        (pid, task_id),
                    )
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "tenant": tenant,
                        "branch_name": branch_name,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "task_class": task_class,
                        "completion_contract": completion_contract,
                    },
                )
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _find_missing_parents(conn: sqlite3.Connection, parents: Iterable[str]) -> list[str]:
    parents = list(parents)
    if not parents:
        return []
    placeholders = ",".join("?" * len(parents))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        parents,
    ).fetchall()
    present = {r["id"] for r in rows}
    return [p for p in parents if p not in present]


def get_task(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return Task.from_row(row) if row else None


# Canonical sort-order mappings for ``hermes kanban list --sort``.
# Each value is a raw SQL fragment appended after ``ORDER BY``.
VALID_SORT_ORDERS: dict[str, str] = {
    "created": "created_at ASC, id ASC",
    "created-desc": "created_at DESC, id DESC",
    "priority": "priority DESC, created_at ASC",
    "priority-desc": "priority ASC, created_at ASC",
    "status": "status ASC, created_at ASC",
    "assignee": "assignee ASC, created_at ASC",
    "title": "title ASC, id ASC",
    "updated": "started_at DESC NULLS LAST, created_at DESC",
}


def list_tasks(
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    session_id: Optional[str] = None,
    include_archived: bool = False,
    limit: Optional[int] = None,
    order_by: Optional[str] = None,
    workflow_template_id: Optional[str] = None,
    current_step_key: Optional[str] = None,
) -> list[Task]:
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if assignee is not None:
        query += " AND assignee = ?"
        params.append(_canonical_assignee(assignee))
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if session_id is not None:
        query += " AND session_id = ?"
        params.append(session_id)
    if workflow_template_id is not None:
        query += " AND workflow_template_id = ?"
        params.append(workflow_template_id)
    if current_step_key is not None:
        query += " AND current_step_key = ?"
        params.append(current_step_key)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    if order_by is not None:
        order_by = order_by.strip().lower()
        if order_by not in VALID_SORT_ORDERS:
            raise ValueError(
                f"order_by must be one of {sorted(VALID_SORT_ORDERS.keys())}"
            )
        query += f" ORDER BY {VALID_SORT_ORDERS[order_by]}"
    else:
        query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [Task.from_row(r) for r in rows]


def assign_task(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assign or reassign a task.  Returns True on success.

    Refuses to reassign a task that's currently running (claim_lock set).
    Reassign after the current run completes if needed.
    """
    profile = _canonical_assignee(profile)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return False
        if row["claim_lock"] is not None and row["status"] == "running":
            raise RuntimeError(
                f"cannot reassign {task_id}: currently running (claimed). "
                "Wait for completion or reclaim the stale lock first."
            )
        if row["assignee"] != profile:
            # The retry guard is scoped to the task/profile combination. A
            # human reassigning the task is an explicit recovery action, so the
            # new profile should not inherit the previous profile's streak.
            conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
                "last_failure_error = NULL WHERE id = ?",
                (profile, task_id),
            )
        else:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
        _append_event(conn, task_id, "assigned", {"assignee": profile})
        return True


def set_task_class(
    conn: sqlite3.Connection, task_id: str, task_class: Optional[str]
) -> bool:
    """Set or clear a task's quality class. Returns True on success.

    ``task_class=None`` (or an empty/whitespace string, or one of the clear
    sentinels ``none``/``-``/``null``) clears the class, reverting the task to
    unclassified/default routing. The sentinel normalisation lives here rather
    than only in the CLI wrapper so a plugin calling the setter directly can't
    leak a literal ``"none"`` string into task_class.
    """
    if task_class is not None:
        normalized = str(task_class).strip()
        task_class = None if normalized.lower() in {"", "none", "-", "null"} else normalized
    with write_txn(conn):
        row = conn.execute(
            "SELECT id FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE tasks SET task_class = ? WHERE id = ?", (task_class, task_id)
        )
        _append_event(conn, task_id, "reclassified", {"task_class": task_class})
        return True


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    with write_txn(conn):
        missing = _find_missing_parents(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _would_cycle(conn, parent_id, child_id):
            raise ValueError(
                f"linking {parent_id} -> {child_id} would create a cycle"
            )
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        # If child was ready but the new parent is not terminal, demote it.
        # Archived parents satisfy dependencies just like done parents; this
        # must match recompute_ready() or linking can strand a child in todo.
        parent_status = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (parent_id,)
        ).fetchone()["status"]
        if parent_status not in ("done", "archived"):
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
                (child_id,),
            )
        _append_event(
            conn, child_id, "linked",
            {"parent": parent_id, "child": child_id},
        )


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Return True if adding parent->child creates a cycle.

    A cycle exists iff ``parent_id`` is already a descendant of
    ``child_id`` via existing parent->child links.  We walk downward
    from ``child_id`` and check whether we reach ``parent_id``.
    """
    seen = set()
    stack = [child_id]
    while stack:
        node = stack.pop()
        if node == parent_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (node,)
        ).fetchall()
        stack.extend(r["child_id"] for r in rows)
    return False


def unlink_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        if cur.rowcount:
            _append_event(
                conn, child_id, "unlinked",
                {"parent": parent_id, "child": child_id},
            )
        removed = cur.rowcount > 0
    if removed:
        # Dependency edge removed — re-evaluate promotion eligibility for the
        # child immediately.  Matches the contract of complete_task and
        # unblock_task; without this the child stays stuck in todo until the
        # next dispatcher tick or a manual `hermes kanban recompute` (issue #22459).
        recompute_ready(conn)
    return removed


def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    return [r["parent_id"] for r in rows]


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
        (task_id,),
    ).fetchall()
    return [r["child_id"] for r in rows]


def parent_results(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, Optional[str]]]:
    """Return ``(parent_id, result)`` for every done parent of ``task_id``."""
    rows = conn.execute(
        """
        SELECT t.id AS id, t.result AS result
        FROM tasks t
        JOIN task_links l ON l.parent_id = t.id
        WHERE l.child_id = ? AND t.status = 'done'
        ORDER BY t.completed_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [(r["id"], r["result"]) for r in rows]


# ---------------------------------------------------------------------------
# Comments & events
# ---------------------------------------------------------------------------

def add_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str
) -> int:
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, author.strip(), body.strip(), now),
        )
        _append_event(conn, task_id, "commented", {"author": author, "len": len(body)})
        return int(cur.lastrowid or 0)


def list_comments(conn: sqlite3.Connection, task_id: str) -> list[Comment]:
    rows = conn.execute(
        "SELECT * FROM task_comments WHERE task_id = ? ORDER BY created_at ASC",
        (task_id,),
    ).fetchall()
    return [
        Comment(
            id=r["id"],
            task_id=r["task_id"],
            author=r["author"],
            body=r["body"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

def add_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size: int = 0,
    uploaded_by: Optional[str] = None,
) -> int:
    """Record a file attachment for a task. Returns the new attachment id.

    The caller is responsible for writing the blob to ``stored_path``
    first (under :func:`task_attachments_dir`); this only persists the
    metadata row and appends an ``attached`` event.
    """
    if not filename or not filename.strip():
        raise ValueError("attachment filename is required")
    if not stored_path or not stored_path.strip():
        raise ValueError("attachment stored_path is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                filename.strip(),
                stored_path,
                content_type,
                int(size),
                uploaded_by,
                now,
            ),
        )
        _append_event(
            conn,
            task_id,
            "attached",
            {"filename": filename.strip(), "size": int(size), "by": uploaded_by},
        )
        return int(cur.lastrowid or 0)


def list_attachments(conn: sqlite3.Connection, task_id: str) -> list[Attachment]:
    rows = conn.execute(
        "SELECT * FROM task_attachments WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [
        Attachment(
            id=r["id"],
            task_id=r["task_id"],
            filename=r["filename"],
            stored_path=r["stored_path"],
            content_type=r["content_type"],
            size=r["size"] or 0,
            uploaded_by=r["uploaded_by"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def get_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    r = conn.execute(
        "SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)
    ).fetchone()
    if r is None:
        return None
    return Attachment(
        id=r["id"],
        task_id=r["task_id"],
        filename=r["filename"],
        stored_path=r["stored_path"],
        content_type=r["content_type"],
        size=r["size"] or 0,
        uploaded_by=r["uploaded_by"],
        created_at=r["created_at"],
    )


def delete_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    """Delete an attachment row and its on-disk blob. Returns the removed row.

    Returns ``None`` when no row matched. The blob is removed best-effort
    (a missing file is not an error); the metadata row is the source of
    truth for whether an attachment "exists".
    """
    with write_txn(conn):
        att = get_attachment(conn, attachment_id)
        if att is None:
            return None
        conn.execute("DELETE FROM task_attachments WHERE id = ?", (attachment_id,))
        _append_event(
            conn, att.task_id, "attachment_removed", {"filename": att.filename}
        )
    try:
        p = Path(att.stored_path)
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    return att


def list_events(conn: sqlite3.Connection, task_id: str) -> list[Event]:
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(
            Event(
                id=r["id"],
                task_id=r["task_id"],
                kind=r["kind"],
                payload=payload,
                created_at=r["created_at"],
                run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
            )
        )
    return out


# Bound how much a single goal-loop progress event can carry. The payload
# is already response-hash-only (see ``goals.run_kanban_goal_loop``'s
# ``_emit``), but this is a second, independent backstop against an
# unbounded/secret-carrying payload ever reaching durable storage.
_GOAL_PROGRESS_EVENT_PAYLOAD_LIMIT = 4000


def record_goal_progress_event(
    conn: sqlite3.Connection,
    task_id: str,
    payload: dict,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Persist one ``goals.run_kanban_goal_loop`` budget/progress snapshot.

    This is the dispatcher-visible sink for the goal loop's per-turn
    telemetry (turns used/max, judge verdict history, no-progress count,
    closeout-corridor state, ...) — a human or the dashboard can tail
    ``list_events(conn, task_id)`` for ``kind == "goal_progress"`` rows to
    see WHY a goal-mode card is burning budget without having to attach a
    debugger. The goal loop itself has zero hard dependency on this module
    (see ``goals.py``'s module docstring) — callers wire this in as the
    ``emit_progress`` callback (see ``cli._run_kanban_goal_loop_q``).

    Best-effort by design: a caller that can't persist this (DB busy, task
    already archived, ...) should log and move on rather than let telemetry
    failures interrupt the goal loop — see the try/except around the
    ``emit_progress`` call site.
    """
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) > _GOAL_PROGRESS_EVENT_PAYLOAD_LIMIT:
        # Never silently drop the event over a payload-size surprise — trim
        # it to a short, clearly-marked summary instead so the telemetry
        # stream stays informative (and bounded).
        payload = {
            "task_id": payload.get("task_id"),
            "phase": payload.get("phase"),
            "turns_used": payload.get("turns_used"),
            "max_turns": payload.get("max_turns"),
            "truncated": True,
            "original_size": len(encoded),
        }
    with write_txn(conn):
        _append_event(conn, task_id, "goal_progress", payload, run_id=run_id)


def _append_event(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    payload: Optional[dict] = None,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Record an event row.  Called from within an already-open txn.

    ``run_id`` is optional: pass the current run id so UIs can group
    events by attempt. For events that aren't scoped to a single run
    (task created/edited/archived, dependency promotion) leave it None
    and the row carries NULL.
    """
    now = int(time.time())
    pl = json.dumps(payload, ensure_ascii=False) if payload else None
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, pl, now),
    )


def _end_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
    status: Optional[str] = None,
) -> Optional[int]:
    """Close the currently-active run for ``task_id`` and clear the pointer.

    ``outcome`` is the semantic result (completed / blocked / crashed /
    timed_out / spawn_failed / gave_up / reclaimed). ``status`` is the
    run-row status (usually just ``outcome``, but callers can pass it
    explicitly). Returns the closed run_id or ``None`` if no active run
    existed (e.g. a CLI user calling ``hermes kanban complete`` on a
    task that was never claimed).
    """
    now = int(time.time())
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if not row or not row["current_run_id"]:
        return None
    run_id = int(row["current_run_id"])
    conn.execute(
        """
        UPDATE task_runs
           SET status        = ?,
               outcome       = ?,
               summary       = ?,
               error         = ?,
               metadata      = ?,
               ended_at      = ?,
               claim_lock    = NULL,
               claim_expires = NULL,
               worker_pid    = NULL
         WHERE id = ?
           AND ended_at IS NULL
        """,
        (
            status or outcome,
            outcome,
            summary,
            error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now,
            run_id,
        ),
    )
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,),
    )
    return run_id


def _current_run_id(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    return int(row["current_run_id"]) if row and row["current_run_id"] else None


def _synthesize_ended_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Insert a zero-duration, already-closed run row.

    Used when a terminal transition happens on a task that was never
    claimed (CLI user calling ``hermes kanban complete <ready-task>
    --summary X``, or dashboard "mark done" on a ready task). Without
    this, the handoff fields (summary / metadata / error) would be
    silently dropped: ``_end_run`` is a no-op because there's no
    current run.

    The synthetic run has ``started_at == ended_at == now`` so it
    shows up in attempt history as "instant" and doesn't skew elapsed
    stats. Caller is responsible for leaving ``current_run_id`` NULL
    (or for clearing it elsewhere in the same txn) since this
    function does NOT touch the tasks row.
    """
    now = int(time.time())
    trow = conn.execute(
        "SELECT assignee, current_step_key FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    profile = trow["assignee"] if trow else None
    step_key = trow["current_step_key"] if trow else None
    cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key,
            status, outcome,
            summary, error, metadata,
            started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, profile, step_key,
            outcome, outcome,
            summary, error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now, now,
        ),
    )
    return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------
# Dependency resolution (todo -> ready)
# ---------------------------------------------------------------------------

def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return True when ``task_id`` is sticky-blocked by an explicit
    worker/operator ``kanban_block`` call (#28712).

    A ``blocked`` status can come from two very different sources:

    * **Worker- or operator-initiated** — a worker called
      ``kanban_block(reason="review-required: ...")`` (or somebody ran
      ``hermes kanban block <id>``).  This is a deliberate handoff that
      should stay blocked until an operator unblocks it.  The block tool
      emits a ``"blocked"`` event row in ``task_events``.

    * **Circuit-breaker** — ``_record_task_failure`` tripped after
      repeated crashes / spawn failures / timeouts.  This emits
      ``"gave_up"``, *not* ``"blocked"``, and is meant to recover
      automatically once the underlying conditions change (e.g. parents
      finish, transient infra error clears).

    The cheapest signal that distinguishes the two is the most recent
    ``"blocked"`` / ``"unblocked"`` event for the task.  If the most
    recent one is ``"blocked"`` (or there is a ``"blocked"`` event and
    no ``"unblocked"`` event has fired since), the task is sticky and
    ``recompute_ready`` must *not* auto-promote it.

    Returns ``False`` when there is no such event at all (e.g. the task
    was set to ``status='blocked'`` by the circuit breaker or by direct
    DB manipulation) — preserves the pre-#28712 auto-recover semantics
    for that path.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return bool(row) and row["kind"] == "blocked"


def recompute_ready(
    conn: sqlite3.Connection, failure_limit: int = None,
) -> int:
    """Promote ``todo`` tasks to ``ready`` when all parents are ``done`` or ``archived``.

    Returns the number of tasks promoted.  Safe to call inside or outside
    an existing transaction; it opens its own IMMEDIATE txn.

    ``blocked`` tasks are also considered for promotion (so a task
    blocked purely by a parent dependency unblocks itself when the
    parent completes), *except* in two cases:

    1. The most recent block event was a worker-initiated
       ``kanban_block`` — those stay blocked until an explicit
       ``kanban_unblock`` (#28712).

    2. The task's ``consecutive_failures`` has reached the effective
       failure limit.  This prevents infinite retry loops when a task
       repeatedly exhausts its iteration budget: without this guard the
       counter would reset on every recovery cycle and the circuit
       breaker could never trip (#35072).

    The effective failure limit resolves in the same order as the
    circuit breaker in ``_record_task_failure`` so the two never
    disagree about when a task is permanently blocked:

      1. per-task ``max_retries`` if set
      2. caller-supplied ``failure_limit`` (the dispatcher passes the
         ``kanban.failure_limit`` config value through ``dispatch_once``)
      3. ``DEFAULT_FAILURE_LIMIT``

    Human-Gate v1 (2026-07-11 repair review self-audit): a
    ``human_gate=1`` card is skipped here unconditionally, even one whose
    block isn't "sticky" (e.g. gated via ``kanban gate <id> on`` after a
    legacy/un-typed block) — a bulk auto-promotion pass must never raise
    (it would abort promoting every OTHER row in this tick), so this is a
    silent ``continue``, not a call into ``_assert_human_gate_open``.
    ``unblock_task`` (with a valid token) remains the only exit.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries, human_gate "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and (
                row["human_gate"] or _has_sticky_block(conn, task_id)
                or conn.execute("SELECT 1 FROM task_attentions WHERE task_id=? LIMIT 1", (task_id,)).fetchone() is not None
            ):
                # Worker / operator asked for human review, OR the card is
                # hard-gated — do not silently auto-recover.
                # ``unblock_task`` (with a valid token, for gated cards)
                # is the only legitimate exit; it emits ``"unblocked"``
                # which flips the sticky-block predicate back.
                continue
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?",
                (task_id,),
            ).fetchall()
            if all(p["status"] in ("done", "archived") for p in parents):
                if cur_status == "blocked":
                    # Don't auto-recover tasks that have hit the
                    # circuit-breaker failure limit.  Without this
                    # guard, a task that repeatedly exhausts its
                    # iteration budget would cycle forever:
                    # block → auto-recover → respawn → budget
                    # exhausted → block → …  The counter must also
                    # be preserved so the breaker can accumulate
                    # across recovery cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = 'ready' "
                        "WHERE id = ? AND status = 'blocked'",
                        (task_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'todo'",
                        (task_id,),
                    )
                _append_event(conn, task_id, "promoted", None)
                promoted += 1
    return promoted


# ---------------------------------------------------------------------------
# Claim / complete / block
# ---------------------------------------------------------------------------

def _resolve_worker_session(
    conn: sqlite3.Connection, task_id: str, run_id: int, *, allow_resume: bool = True
) -> tuple[Optional[str], bool]:
    """Decide the resumable worker session id for a freshly-created run (#2).

    Returns ``(session_id, is_resume)``:
      * ``(None, False)`` when resume-on-reclaim is OFF -> caller writes no
        session_id and the worker keeps its random session (unchanged behaviour).
      * ``(<prev>, True)`` to RESUME: the MOST RECENT prior run of this task
        pinned a session_id, ended abnormally (``crashed``/``timed_out``), and
        that session's crash budget isn't spent -> reuse it so the re-claimed
        worker continues the same conversation from the last flushed message.
      * ``(<new>, False)`` otherwise: first claim, previous run ended any other
        way (completed/blocked/reclaimed/...), goal_mode task, poisoned session,
        or ``allow_resume=False`` -> a fresh, globally unique id.

    Dev-chain-audit fixes (2026-07-02, DEVCHAIN-AUDIT F1/F5):
      * OUTCOME GATE: resume ONLY when the newest prior run actually crashed or
        timed out. Without it, review-lane claims resumed the WORKER's session
        (reviewer independence destroyed), blocked->unblock re-claims resumed
        silently, and TTL-stale reclaims (outcome='reclaimed') resumed a stuck
        conversation forever WITHOUT ever burning the poison budget.
      * ``allow_resume=False`` lets claim_review_task pin a fresh, traceable
        session while structurally never resuming.
      * goal_mode tasks never resume (interim, F5): a restored goal-loop history
        plus a re-fired first turn duplicates the goal framing and resets the
        turn budget per spawn — excluded until the goal loop is resume-aware.

    Must be called INSIDE claim_task's write_txn so the decision is atomic against
    parallel dispatchers (the ready->running CAS already serializes the claim).
    """
    if not _resolve_resume_on_reclaim():
        return None, False
    max_attempts = _resolve_resume_max_attempts()
    if allow_resume and max_attempts >= 1:
        goal_row = conn.execute(
            "SELECT goal_mode FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if goal_row and goal_row["goal_mode"]:
            return f"kbwrk_{task_id}_{run_id}_{secrets.token_hex(3)}", False
        prev = conn.execute(
            "SELECT session_id, outcome FROM task_runs "
            "WHERE task_id = ? AND session_id IS NOT NULL AND id <> ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id, run_id),
        ).fetchone()
        if (
            prev
            and prev["session_id"]
            and prev["outcome"] in ("crashed", "timed_out")
        ):
            sess = prev["session_id"]
            crashes = conn.execute(
                "SELECT COUNT(*) AS c FROM task_runs "
                "WHERE task_id = ? AND session_id = ? "
                "AND outcome IN ('crashed', 'timed_out')",
                (task_id, sess),
            ).fetchone()["c"]
            # ``crashes`` past failures of THIS session == the resume attempt we
            # are about to make. Allow up to ``max_attempts`` resumes, then treat
            # the session as poisoned. (max_attempts=1 -> resume once after the
            # first crash; the second crash falls through to a fresh session.)
            if crashes <= max_attempts:
                return sess, True
            # Poisoned session (crash budget spent) -> fall through to a fresh id.
    return f"kbwrk_{task_id}_{run_id}_{secrets.token_hex(3)}", False


def claim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        # A raw status write, stale client, or old automation must not bypass an
        # unresolved exact-action gate. Only the combined approve+unblock path
        # can leave an approved grant on a ready task.
        unresolved_action = conn.execute(
            "SELECT id FROM task_pending_actions WHERE task_id = ? AND state = 'pending' "
            "AND approved_at IS NULL AND consumed_at IS NULL AND cancelled_at IS NULL AND expires_at > ? "
            "ORDER BY id DESC LIMIT 1",
            (task_id, now),
        ).fetchone()
        if unresolved_action is not None:
            moved = conn.execute(
                "UPDATE tasks SET status='blocked', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL "
                "WHERE id=? AND status='ready'",
                (task_id,),
            )
            if moved.rowcount:
                _append_event(
                    conn, task_id, "claim_rejected",
                    {"reason": "terminal_approval_pending",
                     "action_id": int(unresolved_action["id"])},
                )
            return None
        # Structural invariant: never transition ready -> running while any
        # parent is not yet 'done'. This is the single enforcement point
        # regardless of which writer (create_task, link_tasks, unblock_task,
        # release_stale_claims, manual SQL) set status='ready'. If a racy
        # writer promoted a task with undone parents, demote it back to
        # 'todo' here — recompute_ready will re-promote when the parents
        # actually finish. See RCA at
        # kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md.
        undone = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? AND p.status NOT IN ('done', 'archived') LIMIT 1",
            (task_id,),
        ).fetchone()
        if undone:
            # Honor an explicit operator override exactly once: when the most
            # recent promote/claim-arbitration event is a forced manual
            # promote, the operator has consciously bypassed the parent gate
            # (e.g. a review card that must run WHILE its review subject is
            # blocked). Without this, promote --force was silently reverted
            # here on the next tick — two enforcement points for the same
            # invariant, only one of which knew about the override
            # (Audit 2026-07-10, live beobachtet an t_a8d9d631).
            last_arbitration = conn.execute(
                "SELECT kind, payload FROM task_events "
                "WHERE task_id = ? AND kind IN "
                "('promoted_manual', 'claim_rejected', 'promoted') "
                "ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            forced_override = False
            if last_arbitration and last_arbitration["kind"] == "promoted_manual":
                try:
                    forced_override = bool(
                        json.loads(last_arbitration["payload"] or "{}").get("forced")
                    )
                except (TypeError, ValueError):
                    forced_override = False
            if not forced_override:
                conn.execute(
                    "UPDATE tasks SET status = 'todo' "
                    "WHERE id = ? AND status = 'ready'",
                    (task_id,),
                )
                _append_event(
                    conn, task_id, "claim_rejected",
                    {"reason": "parents_not_done"},
                )
                return None
        # Defensive: if a prior run somehow leaked (invariant violation from
        # an unknown code path), close it as 'reclaimed' so we don't strand
        # it when the CAS resets the pointer below. No-op when the invariant
        # holds (the common case).
        stale = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ? AND status = 'ready'",
            (task_id,),
        ).fetchone()
        if stale and stale["current_run_id"]:
            conn.execute(
                """
                UPDATE task_runs
                   SET status = 'reclaimed', outcome = 'reclaimed',
                       summary = COALESCE(summary, 'invariant recovery on re-claim'),
                       ended_at = ?,
                       claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
                 WHERE id = ? AND ended_at IS NULL
                """,
                (now, int(stale["current_run_id"])),
            )
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'ready'
               AND claim_lock IS NULL
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        # Look up the current task row so we can populate the run with
        # its assignee / step / runtime cap.
        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_cur = conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, step_key, status,
                claim_lock, claim_expires, max_runtime_seconds,
                started_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        # Event-sourcing resume (#2): pin this run's resumable worker session id.
        # On a re-claim of a crashed task the same id is reused and a ``resumed``
        # marker is written to the run metadata so detect_crashed_workers can tell
        # a resumed clean-exit apart from a genuine protocol violation. The run row
        # was just inserted with NULL metadata, so writing it here is race-free.
        worker_session, resume_requested = _resolve_worker_session(conn, task_id, run_id)
        if worker_session:
            if resume_requested:
                conn.execute(
                    "UPDATE task_runs SET session_id = ?, metadata = ? WHERE id = ?",
                    (worker_session, '{"resumed": true}', run_id),
                )
            else:
                conn.execute(
                    "UPDATE task_runs SET session_id = ? WHERE id = ?",
                    (worker_session, run_id),
                )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id},
            run_id=run_id,
        )
        claimed = get_task(conn, task_id)
        if claimed is not None:
            claimed.worker_session_id = worker_session
            claimed.resume_requested = resume_requested
    _fire_kanban_lifecycle_hook(
        "kanban_task_claimed",
        task_id,
        board=get_current_board(),
        assignee=claimed.assignee if claimed else None,
        run_id=run_id,
    )
    return claimed


def claim_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``review -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``review`` status).

    Unlike ``claim_task`` (which handles ``ready -> running``), this
    does NOT check parent dependencies — the task already passed that
    gate on its original ``todo -> ready -> running`` transition.

    Creates a new run entry so the review agent's lifecycle is tracked
    independently from the original worker run.
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'review'
               AND claim_lock IS NULL
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_cur = conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, step_key, status,
                claim_lock, claim_expires, max_runtime_seconds,
                started_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        # Event-sourcing resume (#2): review agents get a stable, traceable
        # session id but NEVER resume (DEVCHAIN-AUDIT F1) — resuming here would
        # hand the reviewer the WORKER's conversation, destroying reviewer
        # independence (the worker would effectively review its own work).
        worker_session, resume_requested = _resolve_worker_session(
            conn, task_id, run_id, allow_resume=False
        )
        if worker_session:
            if resume_requested:
                conn.execute(
                    "UPDATE task_runs SET session_id = ?, metadata = ? WHERE id = ?",
                    (worker_session, '{"resumed": true}', run_id),
                )
            else:
                conn.execute(
                    "UPDATE task_runs SET session_id = ? WHERE id = ?",
                    (worker_session, run_id),
                )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id,
             "source_status": "review"},
            run_id=run_id,
        )
        claimed = get_task(conn, task_id)
        if claimed is not None:
            claimed.worker_session_id = worker_session
            claimed.resume_requested = resume_requested
        return claimed


VALID_REVIEW_DECISIONS = frozenset({"ACCEPT", "NEEDS_REPAIR", "BLOCK"})


def _pending_review_request(
    conn: sqlite3.Connection, task_id: str
) -> Optional[dict]:
    """Return the latest unmatched ``review_requested`` payload, if any."""
    row = conn.execute(
        """
        SELECT kind, payload
          FROM task_events
         WHERE task_id = ?
           AND kind IN ('review_requested', 'review_decided')
         ORDER BY id DESC
         LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    if row is None or row["kind"] != "review_requested":
        return None
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def request_task_review(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reviewer: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Atomically hand an implementation run to the first-class review lane.

    The same task is reassigned to ``reviewer`` and moved to ``review``; no
    dependency child is created, so a review-waiting implementation cannot gate
    its own reviewer. Repeating the same request while it is pending is a no-op.
    """
    reviewer_name = _canonical_assignee(reviewer)
    if not reviewer_name:
        raise ValueError("reviewer is required")
    reviewer = reviewer_name
    with write_txn(conn):
        pending = _pending_review_request(conn, task_id)
        row = conn.execute(
            "SELECT status, assignee, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return False
        if pending is not None:
            return row["status"] in {"review", "running"}
        if row["status"] != "running":
            return False
        run_id = row["current_run_id"]
        if expected_run_id is not None and run_id != int(expected_run_id):
            return False
        implementation_assignee = row["assignee"]
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = 'review', assignee = ?,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
             WHERE id = ? AND status = 'running'
            """ + ("" if expected_run_id is None else " AND current_run_id = ?"),
            (reviewer, task_id)
            if expected_run_id is None
            else (reviewer, task_id, int(expected_run_id)),
        )
        if cur.rowcount != 1:
            return False
        closed_run_id = _end_run(
            conn,
            task_id,
            outcome="review_requested",
            status="review",
            summary=summary,
            metadata=metadata,
        )
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "reviewer": reviewer,
                "implementation_assignee": implementation_assignee,
                "summary": (summary or "").strip().splitlines()[0][:400] or None,
            },
            run_id=closed_run_id,
        )
    _fire_kanban_lifecycle_hook(
        "kanban_review_requested",
        task_id,
        board=get_current_board(),
        assignee=reviewer,
        run_id=closed_run_id,
        summary=summary,
    )
    return True


def decide_task_review(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    decision: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Atomically apply ``ACCEPT|NEEDS_REPAIR|BLOCK`` to a review run."""
    decision = str(decision).strip().upper()
    if decision not in VALID_REVIEW_DECISIONS:
        raise ValueError(
            f"review decision must be one of {sorted(VALID_REVIEW_DECISIONS)}"
        )
    if decision == "ACCEPT":
        return _accept_task_review(
            conn,
            task_id,
            summary=summary,
            metadata=metadata,
            expected_run_id=expected_run_id,
        )
    now = int(time.time())
    with write_txn(conn):
        request = _pending_review_request(conn, task_id)
        if request is None:
            return False
        row = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None or row["status"] != "running":
            return False
        run_id = row["current_run_id"]
        if run_id is None:
            return False
        if expected_run_id is not None and run_id != int(expected_run_id):
            return False
        implementation_assignee = request.get("implementation_assignee")
        if not implementation_assignee:
            return False
        target_status = {
            "ACCEPT": "done",
            "NEEDS_REPAIR": "ready",
            "BLOCK": "blocked",
        }[decision]
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?, assignee = ?,
                   completed_at = CASE WHEN ? = 'done' THEN ? ELSE NULL END,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL,
                   block_kind = CASE WHEN ? = 'blocked' THEN 'needs_input' ELSE NULL END,
                   block_recurrences = CASE
                       WHEN ? != 'blocked' THEN block_recurrences
                       WHEN block_kind = 'needs_input' THEN block_recurrences + 1
                       ELSE 1
                   END
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (
                target_status,
                implementation_assignee,
                target_status,
                now,
                target_status,
                target_status,
                task_id,
                int(run_id),
            ),
        )
        if cur.rowcount != 1:
            return False
        _end_run(
            conn,
            task_id,
            outcome=decision.lower(),
            status=target_status,
            summary=summary,
            metadata=metadata,
        )
        _append_event(
            conn,
            task_id,
            "review_decided",
            {
                "decision": decision,
                "summary": (summary or "").strip().splitlines()[0][:400] or None,
            },
            run_id=run_id,
        )
        if decision == "BLOCK":
            # A review BLOCK is a deliberate human-attention handoff. Without
            # this event, _has_sticky_block() (which only looks at
            # 'blocked'/'unblocked' events) classifies the task as
            # circuit-breaker-blocked and recompute_ready() silently reopens
            # it on the next dispatcher tick — the needs_input gate never
            # actually held (Audit 2026-07-10, reproduziert).
            _append_event(
                conn,
                task_id,
                "blocked",
                {
                    "reason": (summary or "").strip().splitlines()[0][:400]
                    or "review decision BLOCK",
                    "kind": "needs_input",
                },
                run_id=run_id,
            )
        if decision == "ACCEPT":
            _append_event(
                conn,
                task_id,
                "completed",
                {"summary": (summary or "").strip().splitlines()[0][:400] or None},
                run_id=run_id,
            )
    if decision == "ACCEPT":
        _clear_failure_counter(conn, task_id)
        recompute_ready(conn)
        _cleanup_workspace(conn, task_id)
        hook_name = "kanban_task_completed"
    elif decision == "NEEDS_REPAIR":
        recompute_ready(conn)
        hook_name = "kanban_review_needs_repair"
    else:
        hook_name = "kanban_task_blocked"
    task = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        hook_name,
        task_id,
        board=get_current_board(),
        assignee=task.assignee if task else None,
        run_id=run_id,
        summary=summary,
        decision=decision,
    )
    return True


def _accept_task_review(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    summary: Optional[str],
    metadata: Optional[dict],
    expected_run_id: Optional[int],
) -> bool:
    """Accept review only after the completion contract and artifacts commit."""
    try:
        with _completion_artifact_lock(conn):
            request = _pending_review_request(conn, task_id)
            task = get_task(conn, task_id)
            if request is None or task is None or task.status != "running":
                return False
            run_id = task.current_run_id
            if run_id is None or (expected_run_id is not None and run_id != int(expected_run_id)):
                return False
            implementation_assignee = request.get("implementation_assignee")
            if not implementation_assignee:
                return False
            event = conn.execute(
                "SELECT run_id FROM task_events WHERE task_id=? AND kind='review_requested' "
                "ORDER BY id DESC LIMIT 1", (task_id,),
            ).fetchone()
            implementation_metadata: dict = {}
            if event and event["run_id"] is not None:
                run = conn.execute(
                    "SELECT metadata FROM task_runs WHERE id=?", (event["run_id"],)
                ).fetchone()
                try:
                    parsed = json.loads(run["metadata"] or "{}") if run else {}
                except (TypeError, json.JSONDecodeError):
                    parsed = {}
                if isinstance(parsed, dict):
                    implementation_metadata = parsed
            completion_metadata = dict(implementation_metadata)
            if isinstance(metadata, dict):
                completion_metadata.update(metadata)
            created_cards = completion_metadata.get("created_cards") or []
            verified_cards, phantom_cards = _verify_created_cards(conn, task_id, created_cards)
            if phantom_cards:
                raise HallucinatedCardsError(phantom_cards, task_id)
            created_artifact_paths: list[Path] = []
            try:
                _validate_completion_contract(task, completion_metadata)
                artifact_manifest, created_artifact_paths = _promote_completion_artifacts(
                    task, completion_metadata, board=get_current_board()
                )
            except CompletionEvidenceError as exc:
                _record_completion_rejection(conn, task_id, exc, run_id=run_id)
                raise
            if artifact_manifest:
                completion_metadata["artifact_manifest"] = artifact_manifest
                completion_metadata["artifacts"] = [item["durable_path"] for item in artifact_manifest]
            with _completion_evidence_error_boundary(
                conn, task_id, run_id=run_id
            ), _completion_artifact_txn(conn, created_artifact_paths) as promotion_state:
                if _pending_review_request(conn, task_id) is None:
                    return False
                cur = conn.execute(
                    "UPDATE tasks SET status='done', assignee=?, completed_at=?, "
                    "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
                    "block_kind=NULL, block_recurrences=0 "
                    "WHERE id=? AND status='running' AND current_run_id=?",
                    (implementation_assignee, int(time.time()), task_id, int(run_id)),
                )
                if cur.rowcount != 1:
                    return False
                for manifest in artifact_manifest:
                    _persist_completion_artifact_manifest(conn, manifest)
                _end_run(
                    conn, task_id, outcome="accept", status="done",
                    summary=summary, metadata=completion_metadata,
                )
                summary_lines = (summary or "").strip().splitlines()
                preview = summary_lines[0][:400] if summary_lines else None
                _append_event(
                    conn, task_id, "review_decided",
                    {"decision": "ACCEPT", "summary": preview}, run_id=run_id,
                )
                payload = {"summary": preview, "artifact_manifest": artifact_manifest}
                if verified_cards:
                    payload["created_cards"] = verified_cards
                _append_event(conn, task_id, "completed", payload, run_id=run_id)
                promotion_state["accepted"] = True
    except _CompletionArtifactLockError as exc:
        error = CompletionEvidenceError(
            "artifact_promotion_failed", f"completion artifact lock failed: {exc}"
        )
        current = get_task(conn, task_id)
        _record_completion_rejection(
            conn, task_id, error,
            run_id=current.current_run_id if current is not None else None,
        )
        raise error from exc
    _clear_failure_counter(conn, task_id)
    recompute_ready(conn)
    _cleanup_workspace(conn, task_id)
    completed = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_completed", task_id, board=get_current_board(),
        assignee=completed.assignee if completed else None, run_id=run_id,
        summary=summary, decision="ACCEPT",
    )
    return True


def heartbeat_claim(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim.  Returns True if we still own it.

    Workers that know they'll exceed 15 minutes should call this every
    few minutes to keep ownership.
    """
    expires = int(time.time()) + _resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ?",
            (expires, task_id, lock),
        )
        if cur.rowcount == 1:
            run_id = _current_run_id(conn, task_id)
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                    (expires, run_id),
                )
            return True
        return False


def release_stale_claims(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> int:
    """Reset any ``running`` task whose claim has expired.

    A stale-by-TTL claim whose host-local worker PID is still alive is
    *extended* (with a ``claim_extended`` event) instead of being
    reclaimed. Reclaiming a live worker mid-flight produces the spawn-
    then-immediately-reclaim loop seen on slow models that spend longer
    than ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM
    call (#23025): no tool calls means no ``kanban_heartbeat``, even
    though the subprocess is healthy.

    Backstop (#29747 gap 3): if the worker's PID is still alive but its
    ``last_heartbeat_at`` is stale by more than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h), the worker has
    been making no observable progress and we reclaim anyway — even if
    ``_pid_alive`` is still true. This catches the wedged-in-a-logic-loop
    case where the process is technically running but accomplishing
    nothing. ``_touch_activity`` (run_agent.py) bridges chunk-level
    liveness into ``last_heartbeat_at`` via #31752, so any genuinely
    active worker keeps its heartbeat fresh as a side effect of normal
    API traffic. ``enforce_max_runtime`` and ``detect_crashed_workers``
    remain the upper bounds for genuinely wedged or dead workers.

    Returns the number of stale claims actually reclaimed (live-pid
    extensions don't count). Safe to call often.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    stale = conn.execute(
        "SELECT id, claim_lock, worker_pid, claim_expires, last_heartbeat_at "
        "FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ?",
        (now,),
    ).fetchall()
    for row in stale:
        lock = row["claim_lock"] or ""
        host_local = lock.startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Heartbeat staleness backstop: if we have a heartbeat at all
        # and it's older than the max-stale threshold, the worker is
        # not making observable progress.  Reclaim instead of extending,
        # even if the PID is still alive (it's likely in a logic loop).
        heartbeat_stale = (
            hb is not None
            and (now - int(hb)) > DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        )
        if (
            host_local
            and row["worker_pid"]
            and _pid_alive(row["worker_pid"])
            and not heartbeat_stale
        ):
            new_expires = now + _resolve_claim_ttl_seconds()
            with write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET claim_expires = ? "
                    "WHERE id = ? AND status = 'running' "
                    "  AND claim_lock IS ? "
                    "  AND claim_expires IS NOT NULL "
                    "  AND claim_expires < ?",
                    (new_expires, row["id"], row["claim_lock"], now),
                )
                if cur.rowcount != 1:
                    continue
                run_id = _current_run_id(conn, row["id"])
                if run_id is not None:
                    conn.execute(
                        "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                        (new_expires, run_id),
                    )
                _append_event(
                    conn, row["id"], "claim_extended",
                    {
                        "reason": "pid_alive",
                        "worker_pid": int(row["worker_pid"]),
                        "claim_lock": row["claim_lock"],
                        "claim_expires_was": int(row["claim_expires"]),
                        "claim_expires_now": new_expires,
                        "last_heartbeat_at": (
                            int(row["last_heartbeat_at"])
                            if row["last_heartbeat_at"] is not None
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            continue

        termination = _terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn,
        )
        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
            )
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS NOT NULL AND claim_expires < ?",
                (row["id"], row["claim_lock"], now),
            )
            if cur.rowcount != 1:
                continue
            run_id = _end_run(
                conn, row["id"],
                outcome="reclaimed", status="reclaimed",
                error=f"stale_lock={row['claim_lock']}",
                metadata=termination,
            )
            payload = {
                "stale_lock": row["claim_lock"],
                "worker_pid": (
                    int(row["worker_pid"])
                    if row["worker_pid"] is not None else None
                ),
                "claim_expires": int(row["claim_expires"]),
                "last_heartbeat_at": (
                    int(row["last_heartbeat_at"])
                    if row["last_heartbeat_at"] is not None else None
                ),
                "now": now,
                "host_local": host_local,
                "heartbeat_stale": bool(heartbeat_stale),
            }
            payload.update(termination)
            _append_event(
                conn, row["id"], "reclaimed",
                payload,
                run_id=run_id,
            )
            reclaimed += 1
        if heartbeat_stale:
            # H1 (Audit 2026-07-11): a wedged-but-alive worker (PID up,
            # heartbeat stale >1h) used to be reclaimed WITHOUT failure
            # accounting — the same structural wedge respawned forever and
            # the circuit breaker never tripped (livelock without backoff).
            # Count it like a crash/timeout so consecutive_failures reaches
            # failure_limit and the card auto-blocks for a human.
            _record_task_failure(
                conn, row["id"],
                "wedged worker reclaimed: pid alive but heartbeat stale "
                f">{DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS}s "
                f"(lock={row['claim_lock']})",
                outcome="reclaimed",
                event_payload_extra={
                    "wedged": True,
                    "worker_pid": (
                        int(row["worker_pid"])
                        if row["worker_pid"] is not None else None
                    ),
                    "last_heartbeat_at": (
                        int(row["last_heartbeat_at"])
                        if row["last_heartbeat_at"] is not None else None
                    ),
                },
            )
    return reclaimed


def reclaim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    signal_fn=None,
) -> bool:
    """Operator-driven reclaim: release the claim and reset to ``ready``.

    Unlike :func:`release_stale_claims` which only acts on tasks whose
    ``claim_expires`` has passed, this function reclaims immediately
    regardless of TTL. Intended for the dashboard/CLI recovery flow
    when an operator wants to abort a running worker without waiting
    for the TTL to expire (e.g. after seeing a hallucination warning).

    Returns True if a reclaim happened, False if the task isn't in a
    reclaimable state (not running, or doesn't exist).

    Human-Gate v1, refuse-only: a card that is ``blocked``/``scheduled``
    with ``human_gate=1`` always refuses (raises :class:`GateTokenError`),
    regardless of ``reason`` — reclaim has no natural token parameter to
    accept (per the 2026-07-11 repair review), so there is no positive
    path here at all; use ``unblock_task`` with the token instead, or
    ``kanban gate <id> off`` first. 2026-07-11 review finding: this
    function's early-return only checked ``status != 'running'``, which
    let a card S4c's resource-stall detector had blocked WITHOUT clearing
    ``claim_lock`` slip through (``status == 'blocked'`` but
    ``claim_lock`` still set) straight back to ``ready`` with no gate
    check at all.
    """
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        return False
    if row["status"] != "running" and row["claim_lock"] is None:
        # Nothing to reclaim — already ready / blocked / done.
        return False
    prev_lock = row["claim_lock"]
    termination = _terminate_reclaimed_worker(
        row["worker_pid"], prev_lock, signal_fn=signal_fn,
    )
    with write_txn(conn):
        # Authoritative (atomic-with-the-UPDATE) gate check. Deliberately
        # after the worker-termination call above rather than gating that
        # too: terminating a lingering process tied to a card that turns
        # out to be gate-refused is harmless cleanup, not a state change,
        # and keeping a single check point here (vs. duplicating it in an
        # earlier non-atomic pre-check) avoids the two checks drifting.
        _assert_human_gate_open(conn, task_id, token=None, action="reclaim")
        cur = conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status IN ('running', 'ready', 'blocked') "
            "AND claim_lock IS ?",
            (task_id, prev_lock),
        )
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            error=(
                f"manual_reclaim: {reason}" if reason
                else f"manual_reclaim lock={prev_lock}"
            ),
            metadata=termination,
        )
        payload = {
            "manual": True,
            "reason": reason,
            "prev_lock": prev_lock,
        }
        payload.update(termination)
        _append_event(
            conn, task_id, "reclaimed",
            payload,
            run_id=run_id,
        )
    # Operator intervention — they've looked at the task, so the
    # consecutive-failures counter is now stale. Give the next retry
    # a fresh budget. (_clear_failure_counter opens its own write_txn,
    # so it runs after the enclosing one commits.)
    _clear_failure_counter(conn, task_id)
    return True


def reassign_task(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Reassign a task, optionally reclaiming a stuck running worker first.

    This is the recovery path for "this profile's model is broken, try
    a different one". If ``reclaim_first`` is True, any active claim is
    released (via :func:`reclaim_task`) before the reassign happens;
    otherwise the function refuses to reassign a currently-running task
    and returns False (caller can retry with ``reclaim_first=True``).

    Returns True if the reassign landed. ``profile`` may be ``None`` to
    unassign entirely.
    """
    if reclaim_first:
        # Safe to call even if nothing to reclaim.
        reclaim_task(conn, task_id, reason=reason or "reassign")
    # assign_task handles its own txn + the still-running guard.
    try:
        return assign_task(conn, task_id, profile)
    except RuntimeError:
        # Task is still running and reclaim_first was False; caller
        # needs to decide whether to retry with reclaim.
        return False


def _verify_created_cards(
    conn: sqlite3.Connection,
    completing_task_id: str,
    claimed_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Partition ``claimed_ids`` into (verified, phantom).

    A card is "verified" iff a row exists in ``tasks`` AND at least one
    of the following holds:

    * ``created_by`` matches the completing task's ``assignee`` profile
      (the common case: worker A spawns a card via ``kanban_create``,
      which stamps ``created_by=A``).
    * ``created_by`` matches the completing task's id (edge case where
      a worker passed its own task id as the ``created_by`` value).
    * The card is linked as a ``task_links.child`` of the completing
      task — i.e. the worker explicitly called ``kanban_create`` with
      ``parents=[<current_task>]``. This accepts cards created through
      the dashboard/CLI by a different principal but then attached to
      the completing task by the worker.

    ``phantom`` returns ids that either don't exist at all, or exist
    but don't satisfy any of the three trust conditions. The caller
    decides what to do with each bucket; this helper never mutates.
    """
    claimed = [str(x).strip() for x in (claimed_ids or []) if str(x).strip()]
    if not claimed:
        return [], []
    # Dedupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in claimed:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    row = conn.execute(
        "SELECT assignee FROM tasks WHERE id = ?", (completing_task_id,),
    ).fetchone()
    if row is None:
        # Completing task not found — nothing resolves.
        return [], ordered
    completing_assignee = row["assignee"]

    # Batch-fetch existence + created_by in one query.
    placeholders = ",".join(["?"] * len(ordered))
    rows = conn.execute(
        f"SELECT id, created_by FROM tasks WHERE id IN ({placeholders})",
        tuple(ordered),
    ).fetchall()
    found = {r["id"]: r["created_by"] for r in rows}

    # Pull the set of cards linked as children of the completing task.
    # Cheap: one query, indexed on parent_id.
    linked_children: set[str] = set(child_ids(conn, completing_task_id))

    verified: list[str] = []
    phantom: list[str] = []
    for cid in ordered:
        created_by = found.get(cid)
        if created_by is None:
            phantom.append(cid)
            continue
        # Accept if any of the three trust conditions holds.
        if completing_assignee and created_by == completing_assignee:
            verified.append(cid)
        elif created_by == completing_task_id:
            verified.append(cid)
        elif cid in linked_children:
            verified.append(cid)
        else:
            phantom.append(cid)
    return verified, phantom


# Task-id pattern used both by ``kanban_create`` (``t_<12 hex>``) and
# ``_new_task_id`` below. Kept permissive on length for forward compat:
# accept 8+ hex chars after the ``t_`` prefix.
_TASK_ID_PROSE_RE = re.compile(r"\bt_[a-f0-9]{8,}\b")


def _scan_prose_for_phantom_ids(
    conn: sqlite3.Connection,
    text: str,
) -> list[str]:
    """Regex-scan free-form text for ``t_<hex>`` references; return the
    ones that don't exist in ``tasks``.

    Used as a non-blocking advisory check on completion summaries. An
    empty return means "no suspicious references found" — either the
    text had no IDs at all, or every ID it mentioned resolves to a real
    task. Duplicates are deduped.
    """
    if not text:
        return []
    matches = _TASK_ID_PROSE_RE.findall(text)
    if not matches:
        return []
    # Dedupe preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    placeholders = ",".join(["?"] * len(unique))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        tuple(unique),
    ).fetchall()
    existing = {r["id"] for r in rows}
    return [m for m in unique if m not in existing]


class HallucinatedCardsError(ValueError):
    """Raised by ``complete_task`` when ``created_cards`` contains ids
    that don't exist or weren't created by the completing worker.

    The phantom list is attached as ``.phantom`` for callers that want
    structured access. Kept as ``ValueError`` subclass so existing
    tool-error handlers treat it as a recoverable user error.
    """

    def __init__(self, phantom: list[str], completing_task_id: str):
        self.phantom = list(phantom)
        self.completing_task_id = completing_task_id
        super().__init__(
            f"completion blocked: claimed created_cards that do not exist "
            f"or were not created by this worker: {', '.join(phantom)}"
        )


class CompletionEvidenceError(ValueError):
    """Typed fail-closed rejection raised by the completion evidence gate."""

    def __init__(self, kind: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = details


class WorkerGateError(ValueError):
    """Raised when a dispatcher-spawned worker process attempts a lifecycle
    transition the worker gates refuse.

    Audit 2026-07-11 C1: ownership, needs_input, the goal-mode judge and the
    goal-mode block-kind restriction lived only in the ``kanban_*`` tool
    handlers, so ``hermes kanban complete/block`` from a worker's own
    terminal bypassed every one of them. These gates now live in the kernel
    (``complete_task``/``block_task``) and key on the PROCESS being scoped to
    a task via ``HERMES_KANBAN_TASK`` — CLI, tool and any future surface in a
    worker process share one gate. Operator CLI sessions, the gateway
    process (dispatcher/dashboard/Telegram) and orchestrator profiles carry
    no such scope and are unaffected.
    """

    def __init__(self, gate: str, message: str) -> None:
        super().__init__(message)
        self.gate = gate


def _worker_scope() -> tuple[Optional[str], Optional[int]]:
    """(task_id, run_id) this PROCESS is scoped to, or (None, None).

    A dispatcher-spawned worker carries both ``HERMES_KANBAN_TASK`` and
    ``HERMES_KANBAN_RUN_ID``. A delegate_task subagent's terminal subprocess
    keeps TASK but has RUN_ID deliberately stripped (see
    ``tools/environments/local.py:_strip_delegated_subagent_kanban_env``), so
    "scoped task without a run identity" identifies exactly the delegated
    path the gates must refuse.
    """
    tid = (os.environ.get("HERMES_KANBAN_TASK") or "").strip() or None
    raw_rid = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    rid: Optional[int] = None
    if raw_rid:
        try:
            rid = int(raw_rid)
        except ValueError:
            rid = None
    return tid, rid


def _goal_judge_available() -> bool:
    """True when an auxiliary client is configured for the goal judge.

    ``judge_goal`` is fail-open at the source: with no reachable auxiliary
    model it returns a ``"continue"`` verdict indistinguishable from a real
    "not done yet". Enforcing the gate then would wedge every goal_mode
    worker, so we probe availability first (mirrors judge_goal's own lookup).
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return False
    return client is not None and bool(model)


def _record_worker_gate_refusal(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    error: "WorkerGateError",
    *,
    run_id: Optional[int] = None,
) -> None:
    """Best-effort audit event for a refused worker lifecycle transition."""
    try:
        with write_txn(conn):
            _append_event(
                conn, task_id, kind,
                {"gate": error.gate, "error": str(error)},
                run_id=run_id,
            )
    except Exception as audit_error:
        _log.warning(
            "worker gate refusal audit failed for task %s (%s): %s",
            task_id, error.gate, audit_error,
        )


def _enforce_worker_complete_gates(
    conn: sqlite3.Connection,
    task,
    task_id: str,
    summary: Optional[str],
    result: Optional[str],
) -> None:
    """Kernel worker gates for ``complete_task`` (audit 2026-07-11 C1).

    No-op unless the calling PROCESS is scoped to a task (see
    ``_worker_scope``). Raises :class:`WorkerGateError` (and records an
    audit event) instead of returning False so every surface can show the
    worker an actionable message.
    """
    scope_tid, scope_rid = _worker_scope()
    if scope_tid is None:
        return

    def _refuse(gate: str, message: str) -> None:
        err = WorkerGateError(gate, message)
        _record_worker_gate_refusal(
            conn, task_id, "completion_blocked_gate", err,
            run_id=getattr(task, "current_run_id", None),
        )
        raise err

    if task_id != scope_tid:
        _refuse(
            "ownership",
            f"worker is scoped to task {scope_tid}; refusing to complete "
            f"{task_id}. Use kanban_comment to hand off information to "
            f"other tasks, or kanban_create to spawn follow-up work.",
        )
    if scope_rid is None or (
        task.current_run_id is not None
        and int(task.current_run_id) != scope_rid
    ):
        _refuse(
            "run_identity",
            f"this process carries no valid run identity for {task_id} "
            "(HERMES_KANBAN_RUN_ID missing or stale). Delegated subagents "
            "never complete their parent's card; a reclaimed worker must "
            "not complete a run it no longer owns.",
        )
    if task.status == "blocked" and (task.block_kind or "") == "needs_input":
        _refuse(
            "needs_input",
            f"{task_id} is blocked with kind=needs_input — a human decision "
            "gate. Completing it from a worker context would dissolve the "
            "gate without any human in the loop: post your evidence as a "
            "comment and leave the card for the operator.",
        )
    if getattr(task, "goal_mode", 0) and _goal_judge_available():
        verdict, reason = "done", ""
        try:
            from hermes_cli.goals import judge_goal
            verdict, reason, _ = judge_goal(
                goal=f"{task.title}\n\n{task.body or ''}".strip(),
                last_response=(summary or result or "").strip(),
            )
        except Exception as judge_exc:
            # Defensive: judge_goal swallows its own errors, but if it ever
            # raises, fail open rather than wedge the worker.
            _log.warning(
                "goal judge check failed, allowing completion: %s",
                judge_exc, exc_info=True,
            )
        if verdict != "done":
            _refuse(
                "goal_judge",
                f"Goal completion rejected by judge: {reason}. To proceed, "
                "either: (1) provide explicit acceptance evidence in your "
                "summary matching the task's criteria, or (2) create "
                f"continuation tasks with parents=[{task_id}] and keep this "
                "task alive.",
            )


def _enforce_worker_block_gates(
    conn: sqlite3.Connection,
    task_id: str,
    kind: Optional[str],
) -> None:
    """Kernel worker gates for ``block_task`` (audit 2026-07-11 C1)."""
    scope_tid, scope_rid = _worker_scope()
    if scope_tid is None:
        return

    def _refuse(gate: str, message: str) -> None:
        err = WorkerGateError(gate, message)
        _record_worker_gate_refusal(conn, task_id, "block_refused_gate", err)
        raise err

    if task_id != scope_tid:
        _refuse(
            "ownership",
            f"worker is scoped to task {scope_tid}; refusing to block "
            f"{task_id}. Use kanban_comment to hand off information to "
            f"other tasks.",
        )
    if scope_rid is None:
        _refuse(
            "run_identity",
            f"this process carries no valid run identity for {task_id} "
            "(HERMES_KANBAN_RUN_ID missing). Delegated subagents never "
            "drive their parent card's lifecycle.",
        )
    task = get_task(conn, task_id)
    if (
        task is not None
        and getattr(task, "goal_mode", 0)
        and kind not in GOAL_MODE_BLOCK_ALLOWED_KINDS
    ):
        _refuse(
            "goal_mode_block",
            f"goal_mode tasks can only block with kind in "
            f"{sorted(GOAL_MODE_BLOCK_ALLOWED_KINDS)} (got {kind!r}). If "
            "the task is actually finished or cannot proceed for another "
            "reason, call kanban_complete instead — the completion judge "
            "will evaluate it.",
        )


def _record_completion_rejection(
    conn: sqlite3.Connection,
    task_id: str,
    error: CompletionEvidenceError,
    *,
    run_id: Optional[int],
) -> bool:
    try:
        with write_txn(conn):
            _append_event(
                conn,
                task_id,
                error.kind,
                {"error": str(error), **error.details},
                run_id=run_id,
            )
    except Exception as audit_error:
        _log.warning(
            "completion rejection audit failed for task %s (%s): %s",
            task_id,
            error.kind,
            audit_error,
        )
        return False
    return True


def _validate_completion_contract(task: Task, metadata: dict) -> None:
    contract = task.completion_contract or {}
    missing = [
        key for key in ("tests_or_smokes", "readback", "artifacts")
        if contract.get(key) and not metadata.get(key)
    ]
    if missing:
        raise CompletionEvidenceError(
            "evidence_missing",
            "completion contract is missing required evidence: " + ", ".join(missing),
            missing=missing,
        )


def _artifact_source_root(
    source: Path, task: Task, board: Optional[str]
) -> Optional[Path]:
    roots = [
        workspaces_root(board=board),
        attachments_root(board=board),
        completion_artifacts_root(board=board),
    ]
    if task.workspace_path:
        roots.append(Path(task.workspace_path).expanduser())
    resolved_roots = [root.resolve(strict=False) for root in roots]
    matches = [root for root in resolved_roots if source.is_relative_to(root)]
    return max(matches, key=lambda root: len(root.parts), default=None)


def _artifact_path_looks_secret(source: Path) -> bool:
    lowered_parts = {part.casefold() for part in source.parts}
    if lowered_parts & {
        ".git",
        ".ssh",
        ".gnupg",
        ".aws",
        ".kube",
        ".docker",
        ".config",
        ".azure",
        ".gcloud",
        "pairing",
    }:
        return True
    name = source.name.casefold()
    if name.startswith(".env") or name in {
        ".npmrc",
        ".pypirc",
        ".netrc",
        "credentials.json",
        "auth.json",
        "google_token.json",
        ".anthropic_oauth.json",
    }:
        return True
    if source.suffix.casefold() in {".pem", ".key", ".p12", ".pfx"}:
        return True
    return any(marker in name for marker in ("credential", "private-key", "secret", "token"))


_OS_OPEN_SUPPORTS_DIR_FD = os.open in getattr(os, "supports_dir_fd", set())
_OS_MKDIR_SUPPORTS_DIR_FD = os.mkdir in getattr(os, "supports_dir_fd", set())


def _open_directory_path_nofollow(path: Path, *, create: bool) -> int:
    """Open an absolute directory by walking every component from its anchor."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not (
        path.is_absolute()
        and nofollow
        and directory
        and _OS_OPEN_SUPPORTS_DIR_FD
        and (not create or _OS_MKDIR_SUPPORTS_DIR_FD)
    ):
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_RDONLY | directory | nofollow | cloexec)

    flags = os.O_RDONLY | directory | nofollow | cloexec
    current_fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            if create:
                created = False
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(current_fd)
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_artifact_source(source: Path, root: Path) -> int:
    """Open *source* beneath *root* without following path-component symlinks."""
    relative = source.relative_to(root)
    if not relative.parts:
        raise CompletionEvidenceError(
            "artifact_not_durable",
            f"artifact path names a directory root: {source}",
            path=str(source),
        )

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    supports_secure_walk = bool(nofollow and directory and _OS_OPEN_SUPPORTS_DIR_FD)
    if not supports_secure_walk:
        before = source.stat(follow_symlinks=False)
        fd = os.open(source, os.O_RDONLY | cloexec | getattr(os, "O_BINARY", 0))
        after = os.fstat(fd)
        if not os.path.samestat(before, after) or not stat.S_ISREG(after.st_mode):
            os.close(fd)
            raise CompletionEvidenceError(
                "artifact_promotion_failed",
                f"artifact changed while being opened: {source}",
                path=str(source),
            )
        return fd

    current_fd = _open_directory_path_nofollow(root, create=False)
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | directory | nofollow | cloexec,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        fd = os.open(
            relative.parts[-1],
            os.O_RDONLY | nofollow | cloexec | getattr(os, "O_BINARY", 0),
            dir_fd=current_fd,
        )
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise CompletionEvidenceError(
                "artifact_not_durable",
                f"artifact is not a regular file: {source}",
                path=str(source),
            )
        return fd
    finally:
        os.close(current_fd)


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update on platforms that support it."""
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not directory_flag:
        return
    fd = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_artifact_destination(root: Path, components: Iterable[str]) -> tuple[Path, int]:
    """Create and open a board-owned artifact directory without symlink traversal."""
    destination = root.absolute()
    for component in components:
        if not component or component in {".", ".."} or "/" in component or "\\" in component:
            raise CompletionEvidenceError(
                "artifact_promotion_failed",
                f"invalid artifact destination component: {component!r}",
            )
        destination /= component
    try:
        destination_fd = _open_directory_path_nofollow(destination, create=True)
    except OSError as exc:
        raise CompletionEvidenceError(
            "artifact_promotion_failed",
            f"artifact destination changed while being opened: {destination}",
            path=str(destination),
        ) from exc
    return destination, destination_fd


def _open_artifact_destination_child(
    parent: Path, parent_fd: int, component: str
) -> tuple[Path, int]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    created = False
    try:
        os.mkdir(component, mode=0o700, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    if created:
        os.fsync(parent_fd)
    try:
        child_fd = os.open(component, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise CompletionEvidenceError(
            "artifact_promotion_failed",
            f"artifact destination changed while being opened: {parent / component}",
            path=str(parent / component),
        ) from exc
    return parent / component, child_fd


def _copy_artifact_atomically(
    source: Path, source_root: Path, destination_dir: Path, destination_fd: int
) -> tuple[Path, str, int, bool]:
    temporary_name = f".promoting-{secrets.token_hex(8)}"
    digest = hashlib.sha256()
    size = 0
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        source_fd = _open_artifact_source(source, source_root)
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=destination_fd,
        )
        with os.fdopen(source_fd, "rb") as src, os.fdopen(temporary_fd, "wb") as dst:
            if not stat.S_ISREG(os.fstat(src.fileno()).st_mode):
                raise CompletionEvidenceError(
                    "artifact_not_durable",
                    f"artifact is not a regular file: {source}",
                    path=str(source),
                )
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        sha256 = digest.hexdigest()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name).strip("._") or "artifact"
        destination_name = f"{sha256}-{safe_name}"
        destination = destination_dir / destination_name
        try:
            existing_fd = os.open(destination_name, source_flags, dir_fd=destination_fd)
        except FileNotFoundError:
            existing_fd = None
        if existing_fd is not None:
            existing_digest = hashlib.sha256()
            with os.fdopen(existing_fd, "rb") as existing:
                if not stat.S_ISREG(os.fstat(existing.fileno()).st_mode):
                    raise CompletionEvidenceError(
                        "artifact_promotion_failed",
                        f"durable artifact collision for {source}",
                        path=str(source),
                    )
                while True:
                    chunk = existing.read(1024 * 1024)
                    if not chunk:
                        break
                    existing_digest.update(chunk)
            if existing_digest.hexdigest() != sha256:
                raise CompletionEvidenceError(
                    "artifact_promotion_failed",
                    f"durable artifact collision for {source}",
                    path=str(source),
                )
            os.unlink(temporary_name, dir_fd=destination_fd)
            return destination, sha256, size, False
        os.replace(
            temporary_name,
            destination_name,
            src_dir_fd=destination_fd,
            dst_dir_fd=destination_fd,
        )
        try:
            os.fsync(destination_fd)
        except Exception:
            os.unlink(destination_name, dir_fd=destination_fd)
            raise
        return destination, sha256, size, True
    finally:
        try:
            os.unlink(temporary_name, dir_fd=destination_fd)
        except FileNotFoundError:
            pass


def _promote_completion_artifacts(
    task: Task, metadata: dict, *, board: Optional[str]
) -> tuple[list[dict], list[Path]]:
    raw_paths = metadata.get("artifacts") or []
    if not isinstance(raw_paths, (list, tuple)):
        raise CompletionEvidenceError(
            "evidence_missing", "metadata.artifacts must be a list of paths"
        )
    producer_run_id = int(task.current_run_id or 0)
    if not raw_paths:
        return [], []
    destination_dir, destination_fd = _open_artifact_destination(
        completion_artifacts_root(board=board), (task.id, str(producer_run_id))
    )
    manifests: list[dict] = []
    seen: set[str] = set()
    created_paths: list[Path] = []
    try:
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise CompletionEvidenceError(
                    "evidence_missing", "artifact path must be a non-empty string"
                )
            source_input = Path(raw_path).expanduser()
            if not source_input.is_absolute():
                raise CompletionEvidenceError(
                    "artifact_not_durable",
                    f"artifact path must be absolute: {raw_path}",
                    path=raw_path,
                )
            try:
                source = source_input.resolve(strict=True)
            except (FileNotFoundError, RuntimeError):
                raise CompletionEvidenceError(
                    "evidence_missing",
                    f"artifact does not exist: {raw_path}",
                    path=raw_path,
                )
            if str(source) in seen:
                continue
            seen.add(str(source))
            source_root = _artifact_source_root(source, task, board)
            if source_root is None or _artifact_path_looks_secret(source):
                raise CompletionEvidenceError(
                    "artifact_not_durable",
                    f"artifact is outside allowed roots or is secret-like: {raw_path}",
                    path=raw_path,
                )
            if not source.is_file() or not os.access(source, os.R_OK):
                raise CompletionEvidenceError(
                    "evidence_missing",
                    f"artifact is not a readable regular file: {raw_path}",
                    path=raw_path,
                )
            source_id = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
            source_destination, source_destination_fd = _open_artifact_destination_child(
                destination_dir, destination_fd, source_id
            )
            try:
                durable, sha256, size, created = _copy_artifact_atomically(
                    source, source_root, source_destination, source_destination_fd
                )
            except CompletionEvidenceError:
                raise
            except Exception as exc:
                raise CompletionEvidenceError(
                    "artifact_promotion_failed",
                    f"failed to promote artifact {source}: {exc}",
                    path=str(source),
                ) from exc
            finally:
                os.close(source_destination_fd)
            if created:
                created_paths.append(durable)
            manifests.append(
                {
                    "task_id": task.id,
                    "producer_run_id": producer_run_id,
                    "original_path": raw_path,
                    "durable_path": str(durable),
                    "sha256": sha256,
                    "size": size,
                    "content_type": mimetypes.guess_type(source.name)[0] or "application/octet-stream",
                    "validated_at": int(time.time()),
                    "retention_class": "task_completion",
                }
            )
        return manifests, created_paths
    except Exception:
        for created_path in created_paths:
            _unlink_artifact_durably(created_path)
        raise
    finally:
        os.close(destination_fd)


def _unlink_artifact_durably(path: Path) -> None:
    parent = path.parent.absolute()
    try:
        parent_fd = _open_directory_path_nofollow(parent, create=False)
    except OSError as exc:
        _log.warning("refusing unsafe artifact cleanup for %s: %s", path, exc)
        return
    try:
        try:
            os.unlink(path.name, dir_fd=parent_fd)
        except FileNotFoundError:
            return
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)

    try:
        grandparent_fd = _open_directory_path_nofollow(parent.parent, create=False)
    except OSError:
        return
    try:
        try:
            os.rmdir(parent.name, dir_fd=grandparent_fd)
        except OSError:
            return
        os.fsync(grandparent_fd)
    finally:
        os.close(grandparent_fd)


def _remove_unreferenced_artifacts(
    conn: sqlite3.Connection, created_paths: Iterable[Path]
) -> None:
    for created_path in created_paths:
        try:
            referenced = conn.execute(
                "SELECT 1 FROM task_artifacts WHERE durable_path = ? LIMIT 1",
                (str(created_path),),
            ).fetchone()
        except sqlite3.Error:
            # Preserve the file when the DB cannot prove it is unreferenced.
            # A harmless orphan is safer than deleting another concurrent
            # completion's committed artifact.
            continue
        if referenced is None:
            _unlink_artifact_durably(created_path)


@contextlib.contextmanager
def _completion_artifact_txn(
    conn: sqlite3.Connection, created_paths: Iterable[Path]
):
    """Roll back newly promoted files unless their manifest transaction commits."""
    state = {"accepted": False}
    try:
        with write_txn(conn):
            yield state
    except Exception:
        _remove_unreferenced_artifacts(conn, created_paths)
        raise
    if not state["accepted"]:
        _remove_unreferenced_artifacts(conn, created_paths)


def _persist_completion_artifact_manifest(
    conn: sqlite3.Connection, manifest: dict
) -> None:
    """Insert one manifest row and preserve a typed failure boundary."""
    try:
        conn.execute(
            """
            INSERT INTO task_artifacts (
                task_id, producer_run_id, original_path, durable_path,
                sha256, size, content_type, validated_at, retention_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest["task_id"],
                manifest["producer_run_id"],
                manifest["original_path"],
                manifest["durable_path"],
                manifest["sha256"],
                manifest["size"],
                manifest["content_type"],
                manifest["validated_at"],
                manifest["retention_class"],
            ),
        )
    except sqlite3.Error as exc:
        raise CompletionEvidenceError(
            "artifact_promotion_failed",
            f"failed to persist completion artifact manifest: {exc}",
            path=manifest.get("original_path"),
        ) from exc


@contextlib.contextmanager
def _completion_evidence_error_boundary(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int],
):
    """Type and audit manifest persistence failures after rollback/cleanup.

    This boundary must wrap :func:`_completion_artifact_txn`, rather than run
    inside it: only after that inner context exits has SQLite rolled back the
    task transition and removed newly promoted, unreferenced files. The
    rejection event is then written in its own transaction.
    """
    try:
        yield
    except CompletionEvidenceError as exc:
        _record_completion_rejection(conn, task_id, exc, run_id=run_id)
        raise


def complete_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None,
    expected_run_id: Optional[int] = None,
    board: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Complete a task while excluding concurrent artifact scavenging.

    ``token``: Human-Gate v1 one-time token, required (via
    :func:`_assert_human_gate_open`) to complete a ``blocked`` card
    marked ``human_gate=1`` — see ``_complete_task_locked`` for where the
    check runs. ``None`` for every other card (the overwhelming common
    case: completing from ``running``/``ready``) is a no-op.

    Raises :class:`WorkerGateError` when the calling process is a
    dispatcher-spawned worker and the transition fails the worker gates
    (ownership / run identity / needs_input / goal judge) — audit
    2026-07-11 C1. Enforced HERE (not in the tool layer) so CLI, tool and
    every other surface inside a worker process share one gate. Runs
    before the artifact lock so the judge's LLM call never holds it.
    """
    if _worker_scope()[0] is not None:
        _gate_task = get_task(conn, task_id)
        if _gate_task is not None:
            _enforce_worker_complete_gates(
                conn, _gate_task, task_id, summary, result
            )
    try:
        with _completion_artifact_lock(conn):
            return _complete_task_locked(
                conn,
                task_id,
                result=result,
                summary=summary,
                metadata=metadata,
                created_cards=created_cards,
                expected_run_id=expected_run_id,
                board=board,
                token=token,
            )
    except _CompletionArtifactLockError as exc:
        error = CompletionEvidenceError(
            "artifact_promotion_failed",
            f"completion artifact lock failed: {exc}",
        )
        task = get_task(conn, task_id)
        _record_completion_rejection(
            conn,
            task_id,
            error,
            run_id=task.current_run_id if task is not None else None,
        )
        raise error from exc


def _complete_task_locked(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None,
    expected_run_id: Optional[int] = None,
    board: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Transition ``running|ready -> done`` and record ``result``.

    Accepts a task that is merely ``ready`` too, so a manual CLI
    completion (``hermes kanban complete <id>``) works without requiring
    a claim/start/complete sequence.

    Also accepts (and this is the whole reason it needs ``token``) a
    task that is ``blocked`` — an operator affordance for closing out a
    card without a formal unblock first. 2026-07-11 repair review: this
    is precisely why a ``human_gate=1`` blocked card could previously be
    completed straight past the gate (Dashboard PATCH status=done, bulk
    update, and bare ``hermes kanban complete`` outside a worker context
    all reach here). ``_assert_human_gate_open`` closes that — see the
    call right before the ``done`` CAS below.

    ``summary`` and ``metadata`` are stored on the closing run (if any)
    and surfaced to downstream children via :func:`build_worker_context`.
    When ``summary`` is omitted we fall back to ``result`` so single-run
    callers do not have to pass both. ``metadata`` is a free-form dict
    (e.g. ``{"changed_files": [...], "tests_run": [...]}``) — workers
    are encouraged to use it for structured handoff facts.

    ``created_cards`` is an optional list of task ids the completing
    worker claims to have created. Each id is verified against
    ``tasks.created_by``. If any id is phantom (does not exist or was
    not created by this worker's assignee profile), completion is blocked
    with a ``HallucinatedCardsError`` and a
    ``completion_blocked_hallucination`` event is emitted so the rejected
    attempt is auditable. When all ids verify, they are recorded on the
    ``completed`` event payload.

    After a successful completion, ``summary`` and ``result`` are scanned
    for prose references like ``t_deadbeefcafe`` that do not resolve.
    Any suspected phantom references are recorded as a
    ``suspected_hallucinated_references`` event. This pass is advisory
    and never blocks.
    """
    now = int(time.time())

    # A pending first-class review can only reach ``done`` through an explicit
    # ACCEPT decision. This prevents a reviewer run from bypassing the handshake
    # by calling the generic completion path.
    if _pending_review_request(conn, task_id) is not None:
        return False

    # Validate and promote evidence before the done CAS and scratch cleanup.
    task = get_task(conn, task_id)
    if task is None or task.status not in {"running", "ready", "blocked"}:
        return False
    if expected_run_id is not None and task.current_run_id != int(expected_run_id):
        return False
    # Gate: verify created_cards BEFORE the main write txn. A rejected
    # completion still needs an auditable event, so we emit it in a
    # tiny dedicated txn, then raise. The caller is responsible for
    # surfacing HallucinatedCardsError to the worker; this function
    # never mutates task state on a phantom-card rejection.
    if created_cards:
        verified_cards, phantom_cards = _verify_created_cards(
            conn, task_id, created_cards
        )
        if phantom_cards:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "completion_blocked_hallucination",
                    {
                        "phantom_cards": phantom_cards,
                        "verified_cards": verified_cards,
                        "summary_preview": (
                            (summary or result or "").strip().splitlines()[0][:200]
                            if (summary or result)
                            else None
                        ),
                    },
                )
            raise HallucinatedCardsError(phantom_cards, task_id)
    else:
        verified_cards = []

    completion_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    created_artifact_paths: list[Path] = []
    try:
        _validate_completion_contract(task, completion_metadata)
        artifact_manifest, created_artifact_paths = _promote_completion_artifacts(
            task, completion_metadata, board=board
        )
    except CompletionEvidenceError as exc:
        _record_completion_rejection(
            conn, task_id, exc, run_id=task.current_run_id
        )
        raise
    if artifact_manifest:
        completion_metadata["artifact_manifest"] = artifact_manifest
        completion_metadata["artifacts"] = [
            item["durable_path"] for item in artifact_manifest
        ]
    metadata = completion_metadata

    with _completion_evidence_error_boundary(
        conn,
        task_id,
        run_id=task.current_run_id,
    ), _completion_artifact_txn(conn, created_artifact_paths) as promotion_state:
        # Human-Gate v1: no-op unless task_id is currently blocked/scheduled
        # AND human_gate=1, in which case a missing/wrong token raises
        # GateTokenError here — inside the same write_txn as (and strictly
        # before) the 'done' CAS below, so a rejection never touches state
        # and a valid token is consumed atomically with the transition.
        _assert_human_gate_open(conn, task_id, token=token, action="complete")
        if expected_run_id is None:
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked')
                """,
                (result, now, task_id),
            )
        else:
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked')
                   AND current_run_id = ?
                """,
                (result, now, task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        _cancel_active_actions_for_terminal_task(
            conn, task_id, now=now, reason="task_completed",
        )
        for manifest in artifact_manifest:
            _persist_completion_artifact_manifest(conn, manifest)
        run_id = _end_run(
            conn, task_id,
            outcome="completed", status="done",
            summary=summary if summary is not None else result,
            metadata=metadata,
        )
        # If complete_task was called on a never-claimed task (ready or
        # blocked → done with no run in flight), synthesize a
        # zero-duration run so the handoff fields are persisted in
        # attempt history instead of silently lost.
        if run_id is None and (summary or metadata or result):
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=summary if summary is not None else result,
                metadata=metadata,
            )
        # Carry the handoff summary in the event payload so gateway
        # notifiers and dashboard WS consumers can render it without a
        # second SQL round-trip. First line only, 400 char cap — the
        # full summary stays on the run row.
        ev_summary = (summary if summary is not None else result) or ""
        ev_summary = ev_summary.strip().splitlines()[0][:400] if ev_summary else ""
        completed_payload: dict = {
            "result_len": len(result) if result else 0,
            "summary": ev_summary or None,
        }
        if verified_cards:
            completed_payload["verified_cards"] = verified_cards
        # Carry artifact paths in the event payload so the gateway
        # notifier can upload them as native attachments alongside the
        # completion message. Workers pass these via
        # ``kanban_complete(artifacts=[...])`` which stashes the list in
        # ``metadata["artifacts"]`` — we promote it onto the event so
        # consumers don't have to fetch the run row to find it.
        if isinstance(metadata, dict):
            md_artifacts = metadata.get("artifacts")
            if isinstance(md_artifacts, (list, tuple)):
                cleaned_artifacts = [
                    str(p).strip() for p in md_artifacts if isinstance(p, str) and str(p).strip()
                ]
                if cleaned_artifacts:
                    completed_payload["artifacts"] = cleaned_artifacts
        if artifact_manifest:
            completed_payload["artifact_manifest"] = artifact_manifest
        _append_event(
            conn, task_id, "completed",
            completed_payload,
            run_id=run_id,
        )
        promotion_state["accepted"] = True
    # Prose-scan the summary + result for t_<hex> references that do
    # not resolve. Advisory — does not block the completion. Runs in
    # its own txn so the completion itself is already durable by the
    # time we emit the warning.
    scan_text = " ".join(filter(None, [summary, result]))
    if scan_text:
        phantom_refs = _scan_prose_for_phantom_ids(conn, scan_text)
        # Drop any phantom refs that were already flagged as verified
        # above (shouldn't happen — verified means they exist — but
        # belt-and-suspenders).
        phantom_refs = [p for p in phantom_refs if p not in set(verified_cards)]
        if phantom_refs:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "suspected_hallucinated_references",
                    {
                        "phantom_refs": phantom_refs,
                        "source": "completion_summary",
                    },
                    run_id=run_id,
                )
    # Successful completion — wipe the consecutive-failures counter.
    # Failure history stays on the event log for audit; the counter
    # just tracks "is there a current pathology the breaker should
    # care about", and a success resets that question.
    _clear_failure_counter(conn, task_id)
    # Recompute ready status for dependents (separate txn so children see done).
    recompute_ready(conn)
    # Clean up the scratch workspace and any stale tmux session for the worker.
    _cleanup_workspace(conn, task_id)
    _done_task = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_completed",
        task_id,
        board=board if board is not None else get_current_board(),
        assignee=_done_task.assignee if _done_task else None,
        run_id=run_id,
        summary=(summary if summary is not None else result),
    )
    return True


# ---------------------------------------------------------------------------
# Workspace / tmux cleanup
# ---------------------------------------------------------------------------

def _is_managed_scratch_path(p: Path) -> bool:
    """Return True iff *p* is a strict descendant of a kanban-managed scratch root.

    A managed root is exclusively a ``workspaces/`` directory — never the
    broader kanban home, a board root, or sibling subtrees like ``logs/`` or
    ``boards/<slug>/`` itself. Allowed roots:

    * ``HERMES_KANBAN_WORKSPACES_ROOT`` when set (worker-side override
      injected by the dispatcher).
    * ``<kanban_home>/kanban/workspaces`` — legacy default-board scratch root.
    * ``<kanban_home>/kanban/boards/<slug>/workspaces`` for each board slug
      that currently exists on disk.

    The check requires strict descendancy: a path equal to one of these
    roots is NOT managed (deleting the workspaces root would wipe every
    task's scratch dir at once), and a path that resolves to ``<kanban_home>
    /kanban`` itself, ``<kanban_home>/kanban/logs``, or
    ``<kanban_home>/kanban/boards/<slug>`` is rejected because those
    subtrees hold Hermes' own DB, metadata, and logs, not task workspaces.

    Used by :func:`_cleanup_workspace` to refuse to ``shutil.rmtree`` paths
    outside Hermes-managed storage. A board ``default_workdir`` pointing at a
    real source tree can otherwise pair with ``workspace_kind='scratch'`` and
    cause task completion to delete user data (#28818).
    """
    try:
        p_abs = p.resolve(strict=False)
    except OSError:
        return False
    roots: list[Path] = []
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        try:
            roots.append(Path(override).expanduser().resolve(strict=False))
        except OSError:
            pass
    try:
        home = kanban_home()
    except OSError:
        home = None
    if home is not None:
        try:
            roots.append((home / "kanban" / "workspaces").resolve(strict=False))
        except OSError:
            pass
        try:
            boards_parent = (home / "kanban" / "boards").resolve(strict=False)
        except OSError:
            boards_parent = None
        if boards_parent is not None:
            try:
                entries = list(boards_parent.iterdir())
            except OSError:
                entries = []
            for entry in entries:
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                try:
                    roots.append((entry / "workspaces").resolve(strict=False))
                except OSError:
                    continue
    for root in roots:
        if p_abs == root:
            continue
        try:
            if p_abs.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


def _worktree_has_unpushed_commits(worktree_path: str, timeout: int = 10) -> bool:
    """Whether a worktree has commits not reachable from any remote branch.

    Replicated from ``cli.py`` (kept self-contained to avoid a cli<->kanban_db
    circular import). Fails SAFE: on any error returns True so we never remove a
    worktree whose push-state we cannot determine. A repo with NO remote-tracking
    refs has no baseline to prove the commits are pushed anywhere -> we CANNOT show
    the work is safe, so treat it as unpushed (return True = keep). Returning False
    here (the earlier behaviour) let the reaper force-delete committed-but-unpushed
    local-only work in a remote-less repo — a data-loss vector (re-audit 2026-07-07).
    """
    try:
        remote_refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", "refs/remotes"],
            capture_output=True, text=True, timeout=timeout, cwd=worktree_path,
        )
        if remote_refs.returncode != 0:
            return True
        if not remote_refs.stdout.strip():
            # No remotes at all -> unverifiable -> fail SAFE (keep the worktree).
            return True
        result = subprocess.run(
            ["git", "log", "--oneline", "HEAD", "--not", "--remotes"],
            capture_output=True, text=True, timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _worktree_is_dirty(worktree_path: str, timeout: int = 10) -> bool:
    """Whether a worktree has uncommitted changes (staged/unstaged/untracked).

    Replicated from ``cli.py``. Fails SAFE: on any error returns True.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _is_protected_branch(repo_root: Path, branch: str, timeout: int = 10) -> bool:
    """Whether ``branch`` is a shared/long-lived branch that must never be reaped.

    The worktree cleanup deletes the task's branch after removing its worktree.
    A worker-skill default branch (``wt/<task-id>``) is ephemeral and safe by
    construction, but ``tasks.branch_name`` is free-form — a task created with
    ``--branch main`` (or pointed at a shared integration branch) must not get
    that branch ``git branch -D``'d. Protects: the repo's default branch (via
    ``origin/HEAD``, falling back to the main checkout's current HEAD) plus the
    conventional ``main``/``master`` names. Fails SAFE: on any ambiguity or
    error, treats the branch as protected (skip deletion).
    """
    b = (branch or "").strip()
    if not b:
        return True
    if b in {"main", "master", "trunk", "develop"}:
        return True
    try:
        # Repo default branch, e.g. "origin/main" -> "main".
        res = subprocess.run(
            ["git", "-C", str(repo_root), "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if res.returncode == 0:
            default = res.stdout.strip().split("/", 1)[-1]
            if default and b == default:
                return True
        # The branch currently checked out in the main worktree.
        res2 = subprocess.run(
            ["git", "-C", str(repo_root), "symbolic-ref", "--short", "HEAD"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if res2.returncode == 0 and res2.stdout.strip() == b:
            return True
    except Exception:
        return True  # fail safe — don't delete when we can't verify
    return False


def _maybe_cleanup_worktree(conn: sqlite3.Connection, task_id: str, path: str) -> None:
    """Best-effort teardown of a completed task's git worktree (landscape-research #4).

    Opt-in via :func:`_resolve_worktree_auto_cleanup` (default OFF). Conservative
    safety gates: reap ONLY a linked worktree that is clean AND provably fully
    pushed — unpushed/unverifiable commits OR uncommitted changes keep it (the cron
    repo-hygiene reaper remains the backstop for stale leftovers). Deferred while
    child tasks may still read the tree. Removes the worktree and its actual branch
    (``tasks.branch_name`` if set, else the worker-skill default ``wt/<task-id>``).
    """
    try:
        if not _resolve_worktree_auto_cleanup():
            return
        # Defer while any child task may still need the shared tree.
        active = conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? AND t.status NOT IN "
            "('done', 'archived', 'failed', 'cancelled') LIMIT 1",
            (task_id,),
        ).fetchone()
        if active:
            return
        wt = Path(path)
        if not wt.is_dir():
            return
        # The anchor repo (where `git worktree remove` must run) is the parent of
        # the shared git common-dir, NOT the worktree's own toplevel (which is the
        # worktree itself for a linked worktree). Deriving it via the common-dir
        # avoids self-referential removal and finds the real main checkout.
        common = _git_common_dir(wt)
        if common is None:
            return  # not a git worktree we can reason about — leave it
        repo_root = common.parent  # <main>/.git -> <main>
        if repo_root.resolve() == wt.resolve():
            return  # path is the main checkout, not a linked worktree — leave it
        if _worktree_has_unpushed_commits(str(wt)) or _worktree_is_dirty(str(wt)):
            _log.info(
                "Keeping worktree for task %s (unpushed commits or uncommitted "
                "changes): %s", task_id, wt,
            )
            return
        # Reap the task's ACTUAL branch. A project-linked / custom worktree uses
        # ``tasks.branch_name``; only the worker-skill default is ``wt/<task-id>``.
        # Deleting the hardcoded ``wt/<task-id>`` for a custom-branch worktree left
        # the real branch orphaned (re-audit 2026-07-07). Safe to delete: we only
        # reach here once the worktree is clean AND its commits are on a remote.
        branch = None
        try:
            row = conn.execute(
                "SELECT branch_name FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row is not None:
                branch = (row["branch_name"] if "branch_name" in row.keys() else None)
        except Exception:
            branch = None
        branch = (branch or f"wt/{task_id}").strip() or f"wt/{task_id}"
        # Guard the branch reap: a custom ``tasks.branch_name`` could name a
        # shared/long-lived branch (e.g. a task created with ``--branch main``).
        # Deleting the worktree is always fine, but ``branch -D`` on a protected
        # branch would destroy shared history — so drop the branch step for
        # those. ``wt/<task-id>`` and other task-scoped branches still get reaped.
        steps = [
            ("unlock", ["git", "-C", str(repo_root), "worktree", "unlock", str(wt)]),
            ("remove", ["git", "-C", str(repo_root), "worktree", "remove", str(wt), "--force"]),
        ]
        if _is_protected_branch(repo_root, branch):
            _log.info(
                "worktree cleanup: not deleting protected branch %r for task %s "
                "(worktree removed, branch kept)", branch, task_id,
            )
        else:
            # branch -D is harmless if the branch is already gone.
            steps.append(("branch", ["git", "-C", str(repo_root), "branch", "-D", branch]))
        for label, cmd in steps:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
                if res.returncode != 0 and label == "remove":
                    # A failed removal leaves a stale worktree the cron reaper must
                    # mop up — surface it instead of silently swallowing (re-audit).
                    _log.warning(
                        "worktree cleanup: 'git worktree remove' failed for task %s "
                        "(%s): %s", task_id, wt, (res.stderr or res.stdout or "").strip()[:200],
                    )
            except Exception as exc:
                if label == "remove":
                    _log.warning(
                        "worktree cleanup: 'git worktree remove' errored for task %s "
                        "(%s): %s", task_id, wt, exc,
                    )
        _log.debug("Auto-cleaned worktree for task %s (branch %s): %s", task_id, branch, wt)
    except Exception:
        pass  # best-effort — never block completion


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.

    Called from :func:`complete_task` after the DB transaction commits.
    Best-effort — any error is swallowed so cleanup never blocks task completion.
    Only ``scratch`` workspaces are removed unconditionally; ``worktree`` cleanup
    is opt-in + safety-gated (see :func:`_maybe_cleanup_worktree`); ``dir``
    workspaces are always preserved.
    """
    try:
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        kind: Optional[str] = row["workspace_kind"]
        path: Optional[str] = row["workspace_path"]
        if kind == "worktree" and path:
            # Opt-in, safety-gated worktree teardown (landscape-research #4).
            _maybe_cleanup_worktree(conn, task_id, path)
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        if kind != "scratch" or not path:
            # This task's own workspace isn't a removable scratch dir, but its
            # completion may still unblock a deferred parent scratch cleanup
            # (e.g. a 'dir' child whose scratch parent was waiting on it). #33774
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Check if this task has children that still need the workspace.
        # If any child is not yet done/archived, defer cleanup so the
        # child can read handoff artifacts from the scratch dir (#33774).
        _active_children = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
            "LIMIT 1",
            (task_id,),
        ).fetchone()
        if _active_children:
            _log.debug(
                "Deferring scratch workspace cleanup for task %s: "
                "active children still need workspace at %s",
                task_id, path,
            )
            return
        import shutil
        wp = Path(path)
        if wp.is_dir():
            # Containment guard (#28818): a board's ``default_workdir`` can
            # pair ``workspace_kind='scratch'`` with a user-supplied path
            # pointing at a real source tree. Without this check, task
            # completion would unconditionally ``shutil.rmtree`` that path
            # and silently delete the user's source data.
            if _is_managed_scratch_path(wp):
                shutil.rmtree(wp, ignore_errors=True)
                _log.debug("Removed scratch workspace: %s", wp)
            else:
                _log.warning(
                    "Refusing to remove out-of-scratch workspace for task %s: %s "
                    "(workspace_kind='scratch' but path is outside any "
                    "kanban-managed workspaces root)",
                    task_id, wp,
                )
        # Also kill the tmux session for the worker that owned this task,
        # if the tmux session is now dead (worker process exited).
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent
        # tasks now have all children done — their deferred cleanup can
        # proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _try_cleanup_parent_workspaces(conn: sqlite3.Connection, task_id: str) -> None:
    """Clean up parent scratch workspaces now that *task_id* completed.

    When a parent task's cleanup was deferred because it had active children,
    this function is called after each child completes.  If all children of a
    parent are now done/archived/failed/cancelled, the parent's scratch
    workspace is removed (#33774).
    """
    try:
        parents = conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        for (parent_id,) in parents:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
                continue
            # Check if ALL children of this parent are terminal
            active = conn.execute(
                "SELECT 1 FROM task_links l "
                "JOIN tasks t ON t.id = l.child_id "
                "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
                "LIMIT 1",
                (parent_id,),
            ).fetchone()
            if active:
                continue  # still has active children
            # All children done — safe to clean up parent workspace
            import shutil
            wp = Path(row["workspace_path"])
            if wp.is_dir() and _is_managed_scratch_path(wp):
                shutil.rmtree(wp, ignore_errors=True)
                _log.debug("Deferred cleanup: removed parent %s scratch workspace: %s", parent_id, wp)
    except Exception:
        pass  # best-effort


def _cleanup_worker_tmux(conn: sqlite3.Connection, task_id: str) -> None:
    """Kill the tmux session associated with a task's assignee, if dead."""
    try:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row or not row["assignee"]:
            return
        assignee: str = row["assignee"]
        # Workers named swarm1-12 use tmux sessions named swarm-swarm1 etc.
        session = f"swarm-{assignee}"
        # Check if session exists and pane is dead before killing
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip() == "1":
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                capture_output=True, timeout=5,
            )
            _log.debug("Killed stale tmux session: %s", session)
    except Exception:
        pass  # best-effort — never block completion


# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------
#
# Scratch workspaces are intentionally ephemeral — ``_cleanup_workspace``
# removes them as soon as ``complete_task`` runs.  New users often don't
# realize that and lose worker output (community report, May 2026).  The
# behavior is right; the lack of warning is the bug.
#
# On the FIRST scratch workspace materialization across the whole install
# we:
#   1. Log a warning line on the dispatcher logger.
#   2. Append a ``tip_scratch_workspace`` event on the task so it's visible
#      via ``hermes kanban show <id>`` and the dashboard.
#   3. Touch a sentinel file under ``kanban_home() / '.scratch_tip_shown'``
#      so we don't repeat the tip — once you know, you know.
#
# Scope is per-install, not per-board: a user creating a second board
# already learned the lesson on board #1.

_SCRATCH_TIP_SENTINEL_NAME = ".scratch_tip_shown"

_SCRATCH_TIP_MESSAGE = (
    "scratch workspaces are ephemeral — they're deleted when the task "
    "completes. Use --workspace worktree: (git worktree) or "
    "--workspace dir:/abs/path (existing dir) to preserve worker output."
)


def _scratch_tip_sentinel_path() -> Path:
    """Path to the per-install scratch-workspace-tip sentinel file."""
    return kanban_home() / _SCRATCH_TIP_SENTINEL_NAME


def _scratch_tip_shown() -> bool:
    """True iff the scratch-workspace tip has already been emitted on this
    install. Best-effort — any error means we re-emit, which is the safer
    failure mode for a help message."""
    try:
        return _scratch_tip_sentinel_path().exists()
    except OSError:
        return False


def _mark_scratch_tip_shown() -> None:
    """Touch the sentinel so future scratch workspaces stay silent.

    Best-effort: a failure here just means the tip might appear once more,
    which is preferable to crashing dispatch over a help message.
    """
    try:
        path = _scratch_tip_sentinel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _maybe_emit_scratch_tip(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_kind: Optional[str],
) -> None:
    """Emit the first-use scratch-workspace tip exactly once per install.

    Called from the dispatcher right after a scratch workspace is
    materialized. No-op for ``worktree`` / ``dir`` workspaces (they're
    preserved by design) and no-op after the sentinel exists.
    """
    if (workspace_kind or "scratch") != "scratch":
        return
    if _scratch_tip_shown():
        return
    try:
        _log.warning("kanban: %s (task %s)", _SCRATCH_TIP_MESSAGE, task_id)
        with write_txn(conn):
            _append_event(
                conn, task_id, "tip_scratch_workspace",
                {"message": _SCRATCH_TIP_MESSAGE},
            )
    except Exception:
        # Best-effort — never block the spawn loop over a help message.
        pass
    finally:
        _mark_scratch_tip_shown()


def edit_completed_task_result(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Backfill the user-visible result for an already completed task."""
    handoff_summary = summary if summary is not None else result
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if not row or row["status"] != "done":
            return False
        conn.execute(
            "UPDATE tasks SET result = ? WHERE id = ?",
            (result, task_id),
        )
        run = conn.execute(
            """
            SELECT id FROM task_runs
             WHERE task_id = ?
               AND outcome = 'completed'
             ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        run_id = int(run["id"]) if run else None
        if run_id is None:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=handoff_summary,
                metadata=metadata,
            )
        else:
            conn.execute(
                "UPDATE task_runs SET summary = ? WHERE id = ?",
                (handoff_summary, run_id),
            )
            if metadata is not None:
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), run_id),
                )
        ev_summary = (
            handoff_summary.strip().splitlines()[0][:400]
            if handoff_summary else ""
        )
        _append_event(
            conn, task_id, "edited",
            {
                "fields": (
                    ["result", "summary"]
                    + (["metadata"] if metadata is not None else [])
                ),
                "result_len": len(result) if result else 0,
                "summary": ev_summary or None,
            },
            run_id=run_id,
        )
    return True


class GateTokenError(ValueError):
    """A human-gate grant was absent, invalid, expired, or locked out."""

    def __init__(self, message: str, *, persist_failure: bool = False):
        super().__init__(message)
        self.persist_failure = persist_failure


def _human_gate_config() -> dict[str, int]:
    defaults = {
        "token_ttl_seconds": 600,
        "max_failed_attempts": 5,
        "failure_window_seconds": 300,
        "lockout_seconds": 300,
    }
    try:
        from hermes_cli.config import load_config
        raw = (load_config().get("kanban") or {}).get("human_gate") or {}
    except Exception:
        raw = {}
    limits = {
        "token_ttl_seconds": (30, 3600),
        "max_failed_attempts": (1, 20),
        "failure_window_seconds": (30, 3600),
        "lockout_seconds": (30, 3600),
    }
    for key, (low, high) in limits.items():
        try:
            defaults[key] = max(low, min(int(raw.get(key, defaults[key])), high))
        except (TypeError, ValueError, AttributeError):
            pass
    return defaults


def hash_gate_token(token: str) -> str:
    """sha256 hex digest of a plaintext gate token. The DB only ever
    stores this — the plaintext exists only in memory and in exactly one
    ntfy push."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


GOVERNANCE_SCOPE_VERSION = 1

# Internal control-plane declarations.  They deliberately never appear in Task
# projections or public create/tool/dashboard arguments.
VALID_GOVERNANCE_MUTATION_CLASSES = frozenset({
    "live-config", "worker-profile", "systemd-unit", "agent-runtime",
    "runtime-plugin", "runtime-skill", "runtime-containment", "custom-pattern",
})
_GOVERNANCE_TARGET_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@+:-]{0,254}$")


def _validate_governance_declarations(*, target_ref: str, mutation_class: str) -> tuple[str, str]:
    """Validate dispatcher-only governance declarations fail-closed."""
    target = str(target_ref or "").strip()
    klass = str(mutation_class or "").strip()
    if not _GOVERNANCE_TARGET_REF_RE.fullmatch(target):
        raise ValueError("invalid internal governance target ref")
    if klass not in VALID_GOVERNANCE_MUTATION_CLASSES:
        raise ValueError("invalid internal governance mutation class")
    return target, klass


def _set_internal_governance_declarations(
    conn: sqlite3.Connection, task_id: str, *, target_ref: str, mutation_class: str
) -> None:
    """Private dispatcher writer; caller must own the surrounding transaction."""
    target, klass = _validate_governance_declarations(
        target_ref=target_ref, mutation_class=mutation_class
    )
    cur = conn.execute(
        "UPDATE tasks SET governance_target_ref = ?, governance_mutation_class = ? "
        "WHERE id = ?",
        (target, klass, task_id),
    )
    if cur.rowcount != 1:
        raise ValueError("unknown task for internal governance declaration")


def _normalise_governance_text(value: Optional[str]) -> str:
    """Apply the *only* formatting-insensitive governance normalization."""
    return (value or "").replace("\r\n", "\n").rstrip("\n")


def governance_scope_hash(
    conn: sqlite3.Connection, task_id: str, *, board: Optional[str] = None
) -> str:
    """Return the private SHA-256 binding for a card's governance scope.

    Callers must be inside their write transaction when using this for an
    authorization decision. The manifest is intentionally never returned,
    emitted, or logged.
    """
    row = conn.execute(
        "SELECT id, title, body, workspace_kind, workspace_path, project_id, "
        "branch_name, governance_target_ref, governance_mutation_class "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        raise GateTokenError(f"unknown human-gated task {task_id}")
    board_slug = _normalize_board_slug(board) or get_current_board()
    # Bind to the connection's actual main DB as well as its routing slug.
    # Environment/context overrides cannot replay a grant across DB boards.
    db_rows = conn.execute("PRAGMA database_list").fetchall()
    db_path = next((str(r[2]) for r in db_rows if r[1] == "main"), "")
    manifest = {
        "board": {"slug": board_slug, "db_path": os.path.realpath(db_path) if db_path else ""},
        "body": _normalise_governance_text(row["body"]),
        "mutation_class": row["governance_mutation_class"] or "",
        "project_anchor": row["project_id"] or "",
        "scope_version": GOVERNANCE_SCOPE_VERSION,
        "target_ref": row["governance_target_ref"] or row["branch_name"] or "",
        "task_id": row["id"],
        "title": _normalise_governance_text(row["title"]),
        "workspace_anchor": row["workspace_path"] or "",
        "workspace_kind": row["workspace_kind"] or "",
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def issue_gate_token(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    action: str = "unblock",
    board: Optional[str] = None,
) -> Optional[str]:
    """Generate + persist (hashed) a fresh one-time unblock token.

    Only meaningful for a card that is currently ``blocked`` with
    ``human_gate=1`` — returns ``None`` (no-op) for anything else, so
    callers can call this speculatively without checking state first.
    Returns the PLAINTEXT token for exactly one delivery by the caller
    (ntfy push); the plaintext is never written to the DB, an event
    payload, or a log line.
    """
    token = secrets.token_urlsafe(6)
    now = int(time.time())
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, human_gate FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None or row["status"] != "blocked" or not row["human_gate"]:
            return None
        board_slug = _normalize_board_slug(board) or get_current_board()
        scope_hash = governance_scope_hash(conn, task_id, board=board_slug)
        conn.execute(
            "UPDATE tasks SET gate_token_hash = ?, gate_token_issued_at = ?, "
            "gate_token_board = ?, gate_token_task_id = ?, gate_token_action = ?, "
            "gate_scope_hash = ?, gate_scope_version = ?, "
            "gate_failed_attempts = 0, gate_failure_window_started_at = NULL, "
            "gate_locked_until = NULL WHERE id = ?",
            (hash_gate_token(token), now, board_slug, task_id, action,
             scope_hash, GOVERNANCE_SCOPE_VERSION, task_id),
        )
        # Audit-visible grant context, never the token itself.
        _append_event(conn, task_id, "gate_token_issued", {
            "issued_at": now, "board": board_slug, "action": action,
        })
    return token


def _load_ntfy_env() -> dict[str, str]:
    """Read ntfy credentials the same way ``scripts/tars_cron_ntfy_relay.py``
    does: parse ``~/.hermes/.env`` (or ``$HERMES_HOME/.env``) on top of the
    real process environment, without pulling in python-dotenv."""
    env = os.environ.copy()
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    env_path = Path(home) / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def send_gate_token_ntfy(
    task_id: str,
    token: str,
    *,
    board: Optional[str] = None,
    action: str = "unblock",
) -> bool:
    """Push an action-bound human-gate token to the operator's ntfy channel.

    Best-effort, short timeout, stdlib-only urllib POST — mirrors
    ``send_ntfy`` in ``scripts/tars_cron_ntfy_relay.py`` (same env keys:
    ``NTFY_BASE_URL``/``NTFY_TOPIC``/``NTFY_TOKEN`` or
    ``NTFY_USER``/``NTFY_PASSWORD``). Returns False on any failure
    (missing config, network error, non-2xx response); the caller must
    treat that as fail-closed. Never logs the token itself.
    """
    env = _load_ntfy_env()
    base_url = env.get("NTFY_BASE_URL", "https://ntfy.stardock.cloud").rstrip("/")
    topic = env.get("NTFY_TOPIC", "").strip("/")
    if not topic:
        _log.warning(
            "human_gate: NTFY_TOPIC not configured; token for %s was NOT "
            "delivered (card stays hard-blocked, fail closed)", task_id,
        )
        return False
    board_flag = f" --board {board}" if board else ""
    commands = {
        "unblock": f"hermes kanban unblock {task_id} --reason '...' --token {token}{board_flag}",
        "complete": f"hermes kanban complete {task_id} --summary '...' --token {token}{board_flag}",
        "promote": f"hermes kanban promote {task_id} --token {token}{board_flag}",
    }
    command = commands.get(action)
    if command is None:
        raise ValueError(f"unsupported human-gate action: {action}")
    message = (
        f"Karte {task_id} wartet auf deine Freigabe für {action}:\n"
        f"{command}"
    )
    req = urllib.request.Request(
        f"{base_url}/{urllib.parse.quote(topic)}",
        method="POST",
        data=message.encode("utf-8"),
        headers={
            "Title": f"Human Gate: {task_id}"[:120],
            "Priority": "high",
            "Tags": "robot,lock",
            "User-Agent": "hermes-kanban-human-gate",
        },
    )
    ntfy_token = env.get("NTFY_TOKEN", "")
    user = env.get("NTFY_USER", "")
    password = env.get("NTFY_PASSWORD", "")
    if ntfy_token:
        req.add_header("Authorization", f"Bearer {ntfy_token}")
    elif user and password:
        raw = f"{user}:{password}".encode("utf-8")
        req.add_header("Authorization", "Basic " + base64.b64encode(raw).decode("ascii"))
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            ok = 200 <= resp.status < 300
    except Exception as exc:
        _log.warning(
            "human_gate: ntfy push failed for %s: %s: %s",
            task_id, type(exc).__name__, exc,
        )
        return False
    if not ok:
        _log.warning("human_gate: ntfy push for %s returned a non-2xx status", task_id)
    return ok


def issue_and_notify_gate_token(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    action: str = "unblock",
    board: Optional[str] = None,
) -> bool:
    """Issue a fresh gate token for a blocked ``human_gate=1`` card and
    push it via ntfy in one step. Returns True only if a token was both
    issued and delivered.

    On any failure the card stays hard-blocked (fail closed): either no
    token was issued at all (nothing to gate), or a token hash is now
    persisted with nobody holding the plaintext. Rescue path in both
    cases: ``hermes kanban gate <id> off`` (interactive CLI only).
    """
    if action not in {"unblock", "complete", "promote"}:
        raise ValueError(f"unsupported human-gate action: {action}")
    token = issue_gate_token(conn, task_id, action=action, board=board)
    if token is None:
        return False
    delivered = send_gate_token_ntfy(task_id, token, board=board, action=action)
    if not delivered:
        _log.warning(
            "human_gate: %s is now hard-locked (token issued, ntfy push "
            "failed) — rescue with `hermes kanban gate %s off`",
            task_id, task_id,
        )
    return delivered


def set_human_gate(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    on: bool,
    actor: Optional[str] = None,
) -> bool:
    """CLI-only: mark/unmark a card as human-gated (``kanban gate <id>
    on|off``). No tool surface exposes this — see kanban_tools.py.

    Turning the gate off also clears any pending token hash. This is the
    documented rescue path when ntfy delivery fails or was never
    configured: since the plaintext token never existed anywhere in that
    case, ``gate off`` is the only way back to ``ready``.
    """
    with write_txn(conn):
        row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return False
        if on:
            conn.execute("UPDATE tasks SET human_gate = 1 WHERE id = ?", (task_id,))
        else:
            conn.execute(
                "UPDATE tasks SET human_gate = 0, gate_token_hash = NULL, "
                "gate_token_issued_at = NULL, gate_token_board = NULL, "
                "gate_token_task_id = NULL, gate_token_action = NULL, "
                "gate_scope_hash = NULL, gate_scope_version = NULL, "
                "gate_failed_attempts = 0, gate_failure_window_started_at = NULL, "
                "gate_locked_until = NULL WHERE id = ?",
                (task_id,),
            )
        _append_event(conn, task_id, "gate_set", {"on": bool(on), "actor": actor})
    return True


def _assert_human_gate_open(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    token: Optional[str] = None,
    action: str = "unblock",
    persist_failures: bool = True,
    consume: bool = True,
) -> bool:
    """Shared Human-Gate v1 enforcement point for every exit out of
    ``blocked`` (unblock/complete/reclaim/promote/schedule/archive/direct
    status writes — see human-gate-design.md and the 2026-07-11 repair
    review that found ``complete_task``/``reclaim_task``/``promote_task``
    bypassing the original unblock-only check).

    MUST be called from inside the SAME write transaction as the state
    transition that leaves ``blocked``, immediately before the mutating
    UPDATE — the check and the token's single-use consumption must be
    atomic with the transition, or a second caller could race past a
    stale check or replay an already-spent token.

    Returns ``False`` (a pure no-op, nothing enforced) when the task is
    not currently ``blocked``/``scheduled`` with ``human_gate=1`` — every
    ungated task, and every task in any other status, is unaffected,
    preserving behavioural neutrality. ``scheduled`` is included for the
    same reason ``unblock_task`` always covered it: a gated card that
    reaches ``scheduled`` (only possible pre-repair, or via a future bug)
    must stay just as locked as one still sitting in ``blocked``.
    Returns ``True`` when a valid token was supplied and consumed
    (single-use). Raises :class:`GateTokenError` when the task IS gated
    and blocked/scheduled but ``token`` is missing, wrong, or none has
    been issued yet. A raise NEVER touches ``gate_token_hash`` — a bad
    guess (or a caller that never had a token to offer, e.g. an
    automated/refuse-only caller passing ``token=None``) must never
    invalidate a token a legitimate holder still has.
    """
    row = conn.execute(
        "SELECT human_gate, gate_token_hash, gate_token_issued_at, "
        "gate_token_board, gate_token_task_id, gate_token_action, "
        "gate_scope_hash, gate_scope_version, "
        "gate_failed_attempts, gate_failure_window_started_at, gate_locked_until "
        "FROM tasks WHERE id = ? AND status IN ('blocked', 'scheduled')",
        (task_id,),
    ).fetchone()
    if row is None or not row["human_gate"]:
        return False

    now = int(time.time())
    cfg = _human_gate_config()
    locked_until = int(row["gate_locked_until"] or 0)
    if locked_until > now:
        raise GateTokenError(f"{task_id} human-gate is temporarily locked")
    if locked_until:
        if persist_failures:
            # A completed lockout opens a fresh attempt window rather than
            # immediately re-locking on the next typo from the old window.
            conn.execute(
                "UPDATE tasks SET gate_failed_attempts = 0, "
                "gate_failure_window_started_at = NULL, gate_locked_until = NULL "
                "WHERE id = ?",
                (task_id,),
            )
        row = dict(row)
        row["gate_failed_attempts"] = 0
        row["gate_failure_window_started_at"] = None
        row["gate_locked_until"] = None

    stored_hash = row["gate_token_hash"]
    if not stored_hash:
        raise GateTokenError(
            f"{task_id} is human-gated but no token has been issued yet "
            "(or the ntfy push failed) — wait for the push, or run "
            f"`hermes kanban gate {task_id} off` to release it"
        )

    # A scoped board is the caller's explicit routing context.  Use it even
    # when the corresponding board directory is not visible through this
    # connection: falling back to ``default`` here would turn a malformed or
    # stale cross-board call into a valid grant replay.
    scoped_board = (_CURRENT_BOARD_OVERRIDE.get() or "").strip()
    current_board = _normalize_board_slug(scoped_board) if scoped_board else get_current_board()
    failure_reason = None
    issued_at = row["gate_token_issued_at"]
    if not issued_at or now - int(issued_at) > cfg["token_ttl_seconds"]:
        failure_reason = "token has expired"
    elif row["gate_token_board"] != current_board:
        failure_reason = "token was issued for a different board"
    elif row["gate_token_task_id"] != task_id:
        failure_reason = "token was issued for a different task"
    elif row["gate_token_action"] != action:
        failure_reason = "token was issued for a different action"
    elif (
        row["gate_scope_version"] != GOVERNANCE_SCOPE_VERSION
        or not row["gate_scope_hash"]
        or not hmac.compare_digest(
            row["gate_scope_hash"], governance_scope_hash(conn, task_id, board=current_board)
        )
    ):
        # Safe, enumerable reason only: never expose the manifest or its hash.
        failure_reason = "governance_scope_changed"
    elif not token or not hmac.compare_digest(hash_gate_token(token), stored_hash):
        failure_reason = "token was rejected"

    if failure_reason:
        # Scope drift is not a bad human attempt: durable state, including the
        # still-unconsumed token, remains byte-for-byte unchanged.
        if failure_reason == "governance_scope_changed" or not persist_failures:
            raise GateTokenError(f"{task_id} is human-gated: " + (
                "governance scope changed" if failure_reason == "governance_scope_changed" else failure_reason
            ))
        window_start = int(row["gate_failure_window_started_at"] or 0)
        attempts = int(row["gate_failed_attempts"] or 0)
        if not window_start or now - window_start > cfg["failure_window_seconds"]:
            window_start, attempts = now, 0
        attempts += 1
        new_locked_until = (
            now + cfg["lockout_seconds"]
            if attempts >= cfg["max_failed_attempts"] else None
        )
        conn.execute(
            "UPDATE tasks SET gate_failed_attempts = ?, "
            "gate_failure_window_started_at = ?, gate_locked_until = ? WHERE id = ?",
            (attempts, window_start, new_locked_until, task_id),
        )
        _append_event(conn, task_id, "gate_token_rejected", {
            "action": action, "board": current_board,
            "reason": failure_reason, "locked_until": new_locked_until,
        })
        _log.warning(
            "human_gate grant rejected task=%s action=%s board=%s reason=%s",
            task_id, action, current_board, failure_reason,
        )
        raise GateTokenError(
            f"{task_id} is human-gated: "
            f"{'governance scope changed' if failure_reason == 'governance_scope_changed' else failure_reason}",
            persist_failure=True,
        )

    if not consume:
        return True
    conn.execute(
        "UPDATE tasks SET gate_token_hash = NULL, gate_token_issued_at = NULL, "
        "gate_token_board = NULL, gate_token_task_id = NULL, gate_token_action = NULL, "
        "gate_scope_hash = NULL, gate_scope_version = NULL, "
        "gate_failed_attempts = 0, gate_failure_window_started_at = NULL, "
        "gate_locked_until = NULL WHERE id = ?",
        (task_id,),
    )
    return True

def _normalise_pending_action_command(command: str) -> str:
    """Return the byte-preserving command representation used for approval.

    Exact-action approval must distinguish every UTF-8 byte sequence that the
    target shell or filesystem can distinguish. In particular, Unicode NFC/NFD
    path spellings and CRLF/LF are not interchangeable on Linux. Shell quoting,
    whitespace, wrappers, environment prefixes, refs, leases, and line endings
    therefore all remain untouched.
    """
    return str(command)


def _pending_action_hash(command: str) -> str:
    canonical = _normalise_pending_action_command(command)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pending_action_mutation_kind(command: str, explicit_kind: Optional[str] = None) -> str:
    """Classify terminal payloads, or accept the one internal code payload kind.

    ``execute-code-arbitrary`` is intentionally not inferred from source: it is
    a private, whole-payload, one-shot approval boundary supplied by the code
    guard. Keeping that explicit prevents Python source from being treated as a
    shell command or as a bounded mutation.
    """
    if explicit_kind is not None:
        if explicit_kind != "execute-code-arbitrary":
            raise ValueError("unsupported explicit pending-action mutation kind")
        return explicit_kind
    canonical = _normalise_pending_action_command(command)
    lower = canonical.lower()
    if re.search(r"\bgit\s+push\b", lower):
        plain_force = bool(re.search(r"(?:^|\s)(?:--force|-f)(?:\s|$)", lower))
        leased = "--force-with-lease" in lower
        if leased and not re.search(
            r"--force-with-lease=[^\s:]+:[0-9a-f]{7,64}(?:\s|$)", lower,
        ):
            raise ValueError(
                "force-with-lease approval requires exact --force-with-lease=<ref>:<old-sha>"
            )
        if plain_force:
            raise ValueError("plain git push --force is not approvable; use exact --force-with-lease=<ref>:<old-sha>")
        return "git-push-force-with-lease" if leased else "git-push"
    return "terminal-command"


def _pending_action_board_identity(conn: sqlite3.Connection) -> str:
    row = next(
        (r for r in conn.execute("PRAGMA database_list") if r["name"] == "main"),
        None,
    )
    raw = str(row["file"] if row is not None else "")
    return str(Path(raw).resolve()) if raw else "memory"


def _pending_action_fingerprint(
    *,
    board_identity: str,
    task_id: str,
    run_id: Optional[int],
    command_hash: str,
    mutation_kind: str,
    profile: str,
    workspace: str,
    expires_at: Optional[int] = None,
    version: int = 2,
) -> str:
    """Hash immutable exact-action bindings; expiry is mutable metadata."""
    manifest = {
        "version": version,
        "board": board_identity,
        "task_id": task_id,
        "origin_run_id": run_id,
        "command_hash": command_hash,
        "mutation_kind": mutation_kind,
        "profile": profile,
        "workspace": workspace,
    }
    if version == 1:
        if expires_at is None:
            raise ValueError("v1 fingerprint requires expires_at")
        manifest["expires_at"] = int(expires_at)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pending_action_from_row(row: sqlite3.Row) -> PendingAction:
    return PendingAction(
        id=int(row["id"]), task_id=row["task_id"], run_id=row["run_id"],
        command_hash=row["command_hash"],
        fingerprint=row["fingerprint"] or "",
        mutation_kind=row["mutation_kind"] or "terminal-command",
        summary=row["summary"], profile=row["profile"], workspace=row["workspace"],
        expires_at=int(row["expires_at"]),
        approved_at=row["approved_at"], consumed_at=row["consumed_at"],
        cancelled_at=row["cancelled_at"], state=row["state"] if "state" in row.keys() else "pending",
        version=int(row["version"] or 1) if "version" in row.keys() else 1,
        updated_at=int(row["updated_at"] or row["created_at"]) if "updated_at" in row.keys() else int(row["created_at"]),
        resolved_at=row["resolved_at"] if "resolved_at" in row.keys() else None,
    )


def _pending_action_fingerprint_valid(
    conn: sqlite3.Connection, row: sqlite3.Row,
) -> bool:
    # Security bindings are mandatory; legacy incomplete rows are historical,
    # never active authorization candidates.
    required = ("run_id", "command_hash", "mutation_kind", "profile", "workspace", "fingerprint")
    if any(row[name] is None or str(row[name]) == "" for name in required):
        return False
    try:
        bindings = dict(
            board_identity=_pending_action_board_identity(conn), task_id=row["task_id"],
            run_id=int(row["run_id"]), command_hash=row["command_hash"],
            mutation_kind=row["mutation_kind"], profile=row["profile"], workspace=row["workspace"],
        )
        v2 = _pending_action_fingerprint(**bindings)
        v1 = _pending_action_fingerprint(**bindings, expires_at=int(row["expires_at"]), version=1)
    except (TypeError, ValueError):
        return False
    return secrets.compare_digest(str(row["fingerprint"]), v2) or secrets.compare_digest(str(row["fingerprint"]), v1)


def _materialize_expired_actions(conn: sqlite3.Connection, now: int) -> None:
    # ``task_attentions`` is a current projection, not action history. Remove
    # every projection before its action becomes terminal: stable exact-action
    # identity applies only while the action is live, never after expiry.
    conn.execute(
        "DELETE FROM task_attentions WHERE action_id IN ("
        "SELECT id FROM task_pending_actions "
        "WHERE state IN ('pending','approved') AND expires_at <= ?)",
        (now,),
    )
    conn.execute(
        "UPDATE task_pending_actions SET state='expired', version=version+1, updated_at=? "
        "WHERE state IN ('pending','approved') AND expires_at <= ?", (now, now),
    )


def _materialize_expired_action(conn: sqlite3.Connection, task_id: str, action_id: int, now: int) -> None:
    """Materialize only a known target action, never global expiry housekeeping."""
    conn.execute(
        "DELETE FROM task_attentions WHERE task_id=? AND action_id=? AND EXISTS ("
        "SELECT 1 FROM task_pending_actions WHERE id=? AND task_id=? "
        "AND state IN ('pending','approved') AND expires_at<=?)",
        (task_id, int(action_id), int(action_id), task_id, now),
    )
    conn.execute(
        "UPDATE task_pending_actions SET state='expired', version=version+1, updated_at=? "
        "WHERE id=? AND task_id=? AND state IN ('pending','approved') AND expires_at<=?",
        (now, int(action_id), task_id, now),
    )


def _approve_after_action_cas_hook() -> None:
    """Private fault seam after action CAS and before task transition."""


def _approve_after_task_transition_hook() -> None:
    """Private fault seam after task transition and before event/commit."""


def _delete_attention_for_action(conn: sqlite3.Connection, task_id: str, action_id: int) -> None:
    """Delete every current projection for a terminal action; history stays on the action."""
    # Stable exact-action identity is only a live-projection rebind property.
    # Once an action settles, no projection may survive its terminal transition.
    conn.execute(
        "DELETE FROM task_attentions WHERE task_id=? AND action_id=?",
        (task_id, int(action_id)),
    )


def _attention_cleanup_hook() -> None:
    """Private in-transaction fault seam for projection-cleanup rollback tests."""


def _upsert_exact_action_attention(conn: sqlite3.Connection, action_id: int, task_id: str, fingerprint: str, now: int) -> None:
    """Create/recover the one live projection without reusing terminal IDs."""
    action = conn.execute(
        "SELECT attention_id FROM task_pending_actions WHERE id=? AND task_id=?",
        (int(action_id), task_id),
    ).fetchone()
    if action is None:
        return
    historical_id = action["attention_id"]
    if historical_id is not None:
        # A live recovery must either restore precisely the historical opaque ID
        # or fail closed.  Never create a second public identity for one action.
        existing = conn.execute(
            "SELECT task_id, action_id FROM task_attentions WHERE id=?",
            (int(historical_id),),
        ).fetchone()
        if existing is not None:
            if str(existing["task_id"]) == task_id and existing["action_id"] == int(action_id):
                return
            raise RuntimeError("attention identity is already bound to another projection")
        conn.execute("DELETE FROM task_attentions WHERE task_id=? AND cause_fingerprint=?", (task_id, fingerprint))
        conn.execute(
            "INSERT INTO task_attentions (id, task_id, action_id, type, cause_fingerprint, summary, created_at) "
            "VALUES (?, ?, ?, 'exact_action', ?, ?, ?)",
            (int(historical_id), task_id, int(action_id), fingerprint, PENDING_ACTION_OPERATOR_SUMMARY, now),
        )
        return
    # A terminal row cannot be a source of authority for a fresh request.
    conn.execute("DELETE FROM task_attentions WHERE task_id=? AND cause_fingerprint=?", (task_id, fingerprint))
    cur = conn.execute(
        "INSERT INTO task_attentions (task_id, action_id, type, cause_fingerprint, summary, created_at) "
        "VALUES (?, ?, 'exact_action', ?, ?, ?)",
        (task_id, int(action_id), fingerprint, PENDING_ACTION_OPERATOR_SUMMARY, now),
    )
    if conn.execute(
        "UPDATE task_pending_actions SET attention_id=? WHERE id=? AND task_id=? AND attention_id IS NULL",
        (int(cur.lastrowid), int(action_id), task_id),
    ).rowcount != 1:
        raise RuntimeError("attention identity binding conflict")


def _migrate_pending_action_lifecycle(conn: sqlite3.Connection) -> None:
    now = int(time.time())
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        rows = conn.execute("SELECT * FROM task_pending_actions").fetchall()
        for row in rows:
            valid = _pending_action_fingerprint_valid(conn, row)
            state = row["state"] or "pending"
            # Once a v2 lifecycle reached a terminal state, re-initialization
            # must never infer a new pending grant from its old timestamps.
            # This is essential because dashboard connections initialize on
            # every request; resurrecting a resolved row would recreate a
            # supposedly terminal opaque attention identity.
            if state in {"consumed", "cancelled", "expired", "resolved"}:
                pass
            elif row["consumed_at"] is not None:
                state = "consumed"
            elif row["cancelled_at"] is not None:
                state = "cancelled"
            elif not valid:
                state = "resolved"
            elif row["approved_at"] is not None and int(row["expires_at"]) > now:
                state = "approved"
            elif int(row["expires_at"]) <= now:
                state = "expired"
            else:
                state = "pending"
            fingerprint = row["fingerprint"]
            if valid:
                bindings = dict(board_identity=_pending_action_board_identity(conn), task_id=row["task_id"], run_id=int(row["run_id"]), command_hash=row["command_hash"], mutation_kind=row["mutation_kind"], profile=row["profile"], workspace=row["workspace"])
                fingerprint = _pending_action_fingerprint(**bindings)
            conn.execute(
                "UPDATE task_pending_actions SET state=?, version=COALESCE(version, 1), "
                "updated_at=CASE WHEN COALESCE(updated_at, 0)=0 THEN COALESCE(cancelled_at, consumed_at, approved_at, created_at) ELSE updated_at END, "
                "resolved_at=CASE WHEN ?='resolved' THEN COALESCE(resolved_at, ?) ELSE resolved_at END, fingerprint=? WHERE id=?",
                (state, state, now, fingerprint, row["id"]),
            )
        # Pre-index legacy schemas could contain several active rows.  Keep one
        # deterministic survivor per task/origin (approved before pending, then
        # newest/largest id) and retain every loser as cancelled history.
        active = conn.execute(
            "SELECT * FROM task_pending_actions WHERE state IN ('pending','approved') "
            "ORDER BY task_id, run_id, CASE state WHEN 'approved' THEN 0 ELSE 1 END, id DESC"
        ).fetchall()
        survivors: list[sqlite3.Row] = []
        seen_origins: set[tuple[str, int]] = set()
        for row in active:
            key = (str(row["task_id"]), int(row["run_id"]))
            if key in seen_origins:
                conn.execute(
                    "UPDATE task_pending_actions SET state='cancelled', cancelled_at=COALESCE(cancelled_at, ?), "
                    "updated_at=?, version=version+1 WHERE id=? AND state IN ('pending','approved')",
                    (now, now, row["id"]),
                )
            else:
                seen_origins.add(key)
                survivors.append(row)
        # Project each active valid survivor.  The UPSERT deliberately retains
        # an existing stable attention id while rebinding its action id.
        for row in survivors:
            current = conn.execute("SELECT * FROM task_pending_actions WHERE id=?", (row["id"],)).fetchone()
            if current is not None and current["state"] in ("pending", "approved"):
                _upsert_exact_action_attention(conn, int(current["id"]), str(current["task_id"]), str(current["fingerprint"]), now)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_action_active_identity ON task_pending_actions(task_id, run_id, command_hash, mutation_kind, profile, workspace) WHERE state IN ('pending','approved')")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_action_active_origin ON task_pending_actions(task_id, run_id) WHERE state IN ('pending','approved')")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_action_current ON task_pending_actions(task_id, state, expires_at, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_action_fingerprint ON task_pending_actions(fingerprint, state)")


def _pending_action_after_persist_hook() -> None:
    """Private no-op fault-injection seam for transaction rollback tests."""


def record_pending_action_and_block(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: Optional[int],
    command: str,
    summary: str,
    profile: str,
    workspace: str,
    expires_at: int,
    mutation_kind: Optional[str] = None,
) -> dict[str, Any]:
    """Atomically persist one exact action and park its running origin."""
    now = int(time.time())
    profile = (profile or "").strip()
    workspace = str(Path(workspace).resolve()) if workspace else ""
    if run_id is None or not profile or not workspace:
        raise ValueError("exact action requires run_id, profile, and workspace")
    if int(expires_at) <= now:
        raise ValueError("pending action expiry must be in the future")
    command_hash = _pending_action_hash(command)
    mutation_kind = _pending_action_mutation_kind(command, mutation_kind)
    fingerprint = _pending_action_fingerprint(
        board_identity=_pending_action_board_identity(conn), task_id=task_id, run_id=run_id,
        command_hash=command_hash, mutation_kind=mutation_kind, profile=profile, workspace=workspace,
    )
    result: dict[str, Any]
    hook_assignee: Optional[str] = None
    with write_txn(conn):
        # Read-only classification comes first: rejected stale origins must not
        # even materialize another task's expiries. A settled retry is likewise
        # returned without projection/version/event drift.
        task = conn.execute("SELECT status, current_run_id, assignee FROM tasks WHERE id=?", (task_id,)).fetchone()
        run = conn.execute("SELECT task_id, status, ended_at, profile FROM task_runs WHERE id=?", (int(run_id),)).fetchone()
        existing = conn.execute(
            "SELECT * FROM task_pending_actions WHERE task_id=? AND run_id=? AND state IN ('pending','approved') ORDER BY id DESC LIMIT 1",
            (task_id, int(run_id)),
        ).fetchone()
        reused = existing is not None and str(existing["fingerprint"] or "") == fingerprint
        settled_retry = (reused and task is not None and task["status"] == "blocked"
                         and task["current_run_id"] is None and run is not None
                         and run["task_id"] == task_id and run["status"] == "blocked"
                         and run["ended_at"] is not None and run["profile"] == profile)
        fresh_origin = (task is not None and task["status"] == "running"
                        and task["current_run_id"] == int(run_id) and task["assignee"] == profile
                        and run is not None and run["task_id"] == task_id and run["status"] == "running"
                        and run["ended_at"] is None and run["profile"] == profile)
        if settled_retry:
            action = _pending_action_from_row(existing)
            attention = conn.execute("SELECT id FROM task_attentions WHERE task_id=? AND action_id=? AND type='exact_action' ORDER BY id DESC LIMIT 1", (task_id, action.id)).fetchone()
            result = {"action_id": action.id, "attention_id": int(attention["id"]), "attention_status": action.state, "reused": True}
        else:
            if not fresh_origin:
                raise ValueError("pending approval origin run is stale or mismatched")
            _materialize_expired_actions(conn, now)
            existing = conn.execute(
                "SELECT * FROM task_pending_actions WHERE task_id=? AND run_id=? AND state IN ('pending','approved') ORDER BY id DESC LIMIT 1",
                (task_id, int(run_id)),
            ).fetchone()
            reused = existing is not None and str(existing["fingerprint"] or "") == fingerprint
            if existing is not None and not reused:
                raise ValueError("origin run already has a different active exact action")
            if reused:
                action = _pending_action_from_row(existing)
            else:
                cur = conn.execute(
                    "INSERT INTO task_pending_actions (task_id, run_id, command_hash, fingerprint, mutation_kind, summary, profile, workspace, created_at, expires_at, state, version, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 1, ?)",
                    (task_id, int(run_id), command_hash, fingerprint, mutation_kind, PENDING_ACTION_OPERATOR_SUMMARY, profile, workspace, now, int(expires_at), now),
                )
                action = _pending_action_from_row(conn.execute("SELECT * FROM task_pending_actions WHERE id=?", (int(cur.lastrowid),)).fetchone())
                _append_event(conn, task_id, "terminal_approval_pending", {"action_id": action.id, "mutation_kind": mutation_kind, "summary": PENDING_ACTION_OPERATOR_SUMMARY, "expires_at": int(expires_at)}, run_id=int(run_id))
            _upsert_exact_action_attention(conn, action.id, task_id, fingerprint, now)
            _pending_action_after_persist_hook()
            if conn.execute(
                "UPDATE tasks SET status='blocked', current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, block_kind='needs_input', block_recurrences=CASE WHEN block_kind='needs_input' THEN block_recurrences+1 ELSE 1 END WHERE id=? AND status='running' AND current_run_id=?",
                (task_id, int(run_id)),
            ).rowcount != 1:
                raise RuntimeError("approval task transition lost ownership")
            if conn.execute(
                "UPDATE task_runs SET status='blocked', outcome='blocked', ended_at=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=? AND task_id=? AND status='running' AND ended_at IS NULL",
                (now, int(run_id), task_id),
            ).rowcount != 1:
                raise RuntimeError("approval origin run transition lost ownership")
            if not reused:
                _append_event(conn, task_id, "blocked", {"kind": "needs_input", "reason": "terminal_approval_required", "action_id": action.id}, run_id=int(run_id))
            attention = conn.execute("SELECT id FROM task_attentions WHERE task_id=? AND action_id=? AND type='exact_action' ORDER BY id DESC LIMIT 1", (task_id, action.id)).fetchone()
            result = {"action_id": action.id, "attention_id": int(attention["id"]), "attention_status": action.state, "reused": reused}
            hook_assignee = ""
    if hook_assignee is not None:
        _fire_kanban_lifecycle_hook(
            "kanban_task_blocked", task_id, board=get_current_board(),
            run_id=int(run_id), reason="terminal_approval_required",
        )
    return result


def record_pending_action(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: Optional[int],
    command: str,
    summary: str,
    profile: str,
    workspace: str,
    expires_at: int,
) -> PendingAction:
    """Persist an exact action fingerprint without storing raw command text."""
    now = int(time.time())
    command_hash = _pending_action_hash(command)
    mutation_kind = _pending_action_mutation_kind(command)
    # Approval descriptions originate at the command boundary. Persist only a
    # fixed operator prompt so redaction configuration or failure can never
    # turn the durable DB/event log into a command or credential side channel.
    summary = PENDING_ACTION_OPERATOR_SUMMARY
    profile = profile or "default"
    workspace = str(Path(workspace).resolve())
    expires_at = int(expires_at)
    if expires_at <= now:
        raise ValueError("pending action expiry must be in the future")
    fingerprint = _pending_action_fingerprint(
        board_identity=_pending_action_board_identity(conn), task_id=task_id, run_id=run_id,
        command_hash=command_hash, mutation_kind=mutation_kind, profile=profile, workspace=workspace,
    )
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        task = conn.execute("SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None:
            raise ValueError(f"task {task_id!r} does not exist")
        if run_id is None or not profile or not workspace:
            raise ValueError("exact action requires run_id, profile, and workspace")
        existing = conn.execute(
            "SELECT * FROM task_pending_actions WHERE task_id=? AND run_id=? AND state IN ('pending','approved') ORDER BY id DESC LIMIT 1",
            (task_id, run_id),
        ).fetchone()
        if existing is not None and str(existing["fingerprint"] or "") == fingerprint:
            # A grant is immutable: retrying the exact identity must not extend
            # TTL/version or manufacture another pending event.
            if existing["state"] == "approved":
                _upsert_exact_action_attention(conn, int(existing["id"]), task_id, fingerprint, now)
                return _pending_action_from_row(existing)
            cur = conn.execute(
                "UPDATE task_pending_actions SET expires_at=MAX(expires_at, ?), updated_at=?, version=version+1 "
                "WHERE id=? AND version=? AND state='pending'",
                (expires_at, now, existing["id"], existing["version"]),
            )
            if cur.rowcount == 1:
                refreshed = conn.execute("SELECT * FROM task_pending_actions WHERE id=?", (existing["id"],)).fetchone()
                _upsert_exact_action_attention(conn, int(existing["id"]), task_id, fingerprint, now)
                return _pending_action_from_row(refreshed)
            # BEGIN IMMEDIATE normally serializes this path, but return a
            # settled matching action on a genuine CAS race rather than raising
            # a lifecycle exception or creating a duplicate row.
            settled = conn.execute("SELECT * FROM task_pending_actions WHERE id=?", (existing["id"],)).fetchone()
            if settled is not None and settled["state"] in ("pending", "approved") and str(settled["fingerprint"] or "") == fingerprint:
                _upsert_exact_action_attention(conn, int(settled["id"]), task_id, fingerprint, now)
                return _pending_action_from_row(settled)
            raise RuntimeError("exact action refresh conflict")
        if task["current_run_id"] != run_id or task["status"] != "running":
            raise ValueError("pending action must be recorded by the task's current run")
        if existing is not None:
            cur = conn.execute(
                "UPDATE task_pending_actions SET state='cancelled', cancelled_at=?, updated_at=?, version=version+1 "
                "WHERE id=? AND version=? AND state IN ('pending','approved')",
                (now, now, existing["id"], existing["version"]),
            )
            if cur.rowcount == 1:
                _delete_attention_for_action(conn, task_id, int(existing["id"]))
                _attention_cleanup_hook()
                _append_event(conn, task_id, "terminal_approval_cancelled", {"action_id": int(existing["id"]), "reason": "superseded"}, run_id=run_id)
        try:
            cur = conn.execute(
                "INSERT INTO task_pending_actions (task_id, run_id, command_hash, fingerprint, mutation_kind, summary, profile, workspace, created_at, expires_at, state, version, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 1, ?)",
                (task_id, run_id, command_hash, fingerprint, mutation_kind, summary, profile, workspace, now, expires_at, now),
            )
        except sqlite3.IntegrityError as exc:
            settled = conn.execute(
                "SELECT * FROM task_pending_actions WHERE task_id=? AND run_id=? AND fingerprint=? "
                "AND state IN ('pending','approved') ORDER BY id DESC LIMIT 1",
                (task_id, run_id, fingerprint),
            ).fetchone()
            if settled is not None:
                _upsert_exact_action_attention(conn, int(settled["id"]), task_id, fingerprint, now)
                return _pending_action_from_row(settled)
            raise RuntimeError("exact action creation conflict") from exc
        action_id = int(cur.lastrowid)
        _upsert_exact_action_attention(conn, action_id, task_id, fingerprint, now)
        _append_event(conn, task_id, "terminal_approval_pending", {"action_id": action_id, "mutation_kind": mutation_kind, "summary": summary, "expires_at": expires_at}, run_id=run_id)
        row = conn.execute("SELECT * FROM task_pending_actions WHERE id=?", (action_id,)).fetchone()
        return _pending_action_from_row(row)


def get_pending_action(
    conn: sqlite3.Connection, task_id: str, *, now: Optional[int] = None,
) -> Optional[PendingAction]:
    now = int(time.time()) if now is None else int(now)
    row = conn.execute(
        "SELECT * FROM task_pending_actions WHERE task_id = ? "
        "AND state IN ('pending','approved') AND expires_at > ? ORDER BY id DESC LIMIT 1",
        (task_id, now),
    ).fetchone()
    if row is None:
        return None
    return _pending_action_from_row(row)


def get_pending_action_by_id(
    conn: sqlite3.Connection, task_id: str, action_id: int,
) -> Optional[PendingAction]:
    """Return an action for API conflict/gone classification, including history."""
    row = conn.execute(
        "SELECT * FROM task_pending_actions WHERE task_id = ? AND id = ?",
        (task_id, int(action_id)),
    ).fetchone()
    return _pending_action_from_row(row) if row is not None else None


def approve_pending_action(
    conn: sqlite3.Connection, task_id: str, action_id: int,
    *, now: Optional[int] = None, actor: str = "operator",
) -> bool:
    """Grant one exact pending action; replay and expired grants fail closed."""
    now = int(time.time()) if now is None else int(now)
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        row = conn.execute(
            "SELECT a.* FROM task_pending_actions a "
            "JOIN tasks t ON t.id = a.task_id "
            "JOIN task_runs r ON r.id = a.run_id AND r.task_id = a.task_id "
            "WHERE a.id = ? AND a.task_id = ? AND a.state = 'pending' AND a.expires_at > ? "
            "AND t.status = 'blocked' AND r.outcome = 'blocked'",
            (int(action_id), task_id, now),
        ).fetchone()
        if row is None or not _pending_action_fingerprint_valid(conn, row):
            return False
        cur = conn.execute(
            "UPDATE task_pending_actions SET approved_at = ?, state='approved', updated_at=?, version=version+1 "
            "WHERE id = ? AND task_id = ? AND fingerprint = ? AND version=? "
            "AND state='pending' AND expires_at > ?",
            (now, now, int(action_id), task_id, row["fingerprint"], row["version"], now),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "terminal_approval_granted",
            {"action_id": int(action_id), "actor": actor},
        )
        return True


def approve_pending_action_and_unblock_versioned(
    conn: sqlite3.Connection, task_id: str, action_id: int, *,
    expected_version: int, actor: str = "operator", token: Optional[str] = None,
    now: Optional[int] = None,
) -> ApprovePendingActionResult:
    """CAS-approve one exact action and unblock its card in one transaction.

    ``expected_version`` is mandatory at this public seam.  The matching
    exact-action projection is part of the same compare-and-set, preventing a
    stale dashboard card from authorizing a replacement action.
    """
    now = int(time.time()) if now is None else int(now)
    hook: Optional[tuple[Optional[Task], Optional[int]]] = None
    with write_txn(conn):
        # Foreign/missing requests are no-ops: do not sweep unrelated expiry.
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            return ApprovePendingActionResult("not_found", task_id)
        action = conn.execute(
            "SELECT * FROM task_pending_actions WHERE id=? AND task_id=?",
            (int(action_id), task_id),
        ).fetchone()
        if action is None:
            return ApprovePendingActionResult("not_found", task_id)
        _materialize_expired_action(conn, task_id, int(action_id), now)
        action = conn.execute(
            "SELECT * FROM task_pending_actions WHERE id=? AND task_id=?",
            (int(action_id), task_id),
        ).fetchone()
        assert action is not None
        attention = conn.execute(
            "SELECT id, type FROM task_attentions WHERE task_id=? AND action_id=? "
            "ORDER BY id DESC LIMIT 1", (task_id, int(action_id)),
        ).fetchone()
        if attention is None:
            return ApprovePendingActionResult("gone", task_id, int(action_id))
        # A version mismatch is a stale active CAS request even if another
        # contender already approved the action. A caller that observes the
        # current approved/terminal version reaches the following gone branch.
        if int(action["version"]) != int(expected_version):
            return ApprovePendingActionResult("conflict", task_id, int(action_id), int(attention["id"]))
        if action["state"] != "pending" or int(action["expires_at"]) <= now:
            return ApprovePendingActionResult("gone", task_id, int(action_id))
        if (attention["type"] != "exact_action"
                or task["status"] not in ("blocked", "triage", "todo", "ready")
                or not _pending_action_fingerprint_valid(conn, action)):
            return ApprovePendingActionResult("conflict", task_id, int(action_id), int(attention["id"]))
        origin = conn.execute(
            "SELECT outcome FROM task_runs WHERE id=? AND task_id=?", (action["run_id"], task_id),
        ).fetchone()
        if origin is None or origin["outcome"] != "blocked":
            return ApprovePendingActionResult("conflict", task_id, int(action_id), int(attention["id"]))
        undone = conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
            "WHERE l.child_id=? AND p.status NOT IN ('done','archived') LIMIT 1", (task_id,),
        ).fetchone()
        target: Literal["ready", "todo"] = "todo" if undone else "ready"
        # Governance is independent of exact-action approval.  Validate only
        # after all non-mutating CAS preconditions so a failed request cannot
        # consume a one-shot token.
        try:
            _assert_human_gate_open(conn, task_id, token=token, action="unblock", persist_failures=False, consume=False)
        except GateTokenError:
            return ApprovePendingActionResult("conflict", task_id, int(action_id), int(attention["id"]))
        # Consume only after the non-mutating validation passed; any following
        # CAS/fault exception rolls this one-shot grant back with the txn.
        _assert_human_gate_open(conn, task_id, token=token, action="unblock")
        cur = conn.execute(
            "UPDATE task_pending_actions SET approved_at=?, state='approved', updated_at=?, version=version+1 "
            "WHERE id=? AND task_id=? AND version=? AND state='pending' AND expires_at>?",
            (now, now, int(action_id), task_id, int(expected_version), now),
        )
        if cur.rowcount != 1:
            return ApprovePendingActionResult("conflict", task_id, int(action_id), int(attention["id"]))
        _approve_after_action_cas_hook()
        if conn.execute(
            "UPDATE tasks SET status=?, current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "consecutive_failures=0, last_failure_error=NULL WHERE id=? AND status IN ('blocked','triage','todo','ready')",
            (target, task_id),
        ).rowcount != 1:
            raise RuntimeError("approval task CAS failed")
        _approve_after_task_transition_hook()
        _append_event(conn, task_id, "terminal_approval_granted", {"action_id": int(action_id), "actor": str(actor)[:120]})
        _append_event(conn, task_id, "unblocked", {"status": target, "terminal_action_id": int(action_id), "actor": str(actor)[:120]})
        hook = (get_task(conn, task_id), action["run_id"])
        result = ApprovePendingActionResult("approved", task_id, int(action_id), int(attention["id"]), int(expected_version) + 1, target)
    if hook is not None:
        approved_task, run_id = hook
        _fire_kanban_lifecycle_hook("kanban_task_unblocked", task_id, board=get_current_board(),
            assignee=approved_task.assignee if approved_task else None, run_id=run_id,
            reason="terminal_approval_granted")
    return result


def approve_pending_action_and_unblock(
    conn: sqlite3.Connection, task_id: str, action_id: int,
    *, now: Optional[int] = None, actor: str = "operator", token: Optional[str] = None,
) -> bool:
    """Legacy bool wrapper over the versioned approval seam."""
    now = int(time.time()) if now is None else int(now)
    row = conn.execute("SELECT version FROM task_pending_actions WHERE id=? AND task_id=?", (int(action_id), task_id)).fetchone()
    if row is None:
        return False
    return bool(approve_pending_action_and_unblock_versioned(
        conn, task_id, int(action_id), expected_version=int(row["version"]), actor=actor, token=token, now=now,
    ))


def consume_approved_action(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: int,
    command: str,
    profile: str,
    workspace: str,
    now: Optional[int] = None,
    mutation_kind: Optional[str] = None,
) -> bool:
    """Atomically consume a grant from the currently running resumed attempt."""
    now = int(time.time()) if now is None else int(now)
    command_hash = _pending_action_hash(command)
    mutation_kind = _pending_action_mutation_kind(command, mutation_kind)
    workspace = str(Path(workspace).resolve())
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        row = conn.execute(
            "SELECT a.* FROM task_pending_actions a "
            "JOIN tasks t ON t.id = a.task_id "
            "JOIN task_runs origin ON origin.id = a.run_id AND origin.task_id = a.task_id "
            "JOIN task_runs active ON active.id = ? AND active.task_id = a.task_id "
            "WHERE a.task_id = ? AND a.command_hash = ? AND a.mutation_kind = ? "
            "AND a.profile = ? AND a.workspace = ? AND a.state='approved' AND a.expires_at > ? "
            "AND t.status = 'running' AND t.current_run_id = active.id "
            "AND active.status = 'running' AND active.ended_at IS NULL "
            "AND active.profile = a.profile AND origin.outcome = 'blocked' "
            "ORDER BY a.id DESC LIMIT 1",
            (int(run_id), task_id, command_hash, mutation_kind,
             profile or "default", workspace, now),
        ).fetchone()
        if row is None:
            return False
        expected_fingerprint = _pending_action_fingerprint(
            board_identity=_pending_action_board_identity(conn),
            task_id=task_id,
            run_id=row["run_id"],
            command_hash=command_hash,
            mutation_kind=mutation_kind,
            profile=profile or "default",
            workspace=workspace,
            expires_at=int(row["expires_at"]),
        )
        if not row["fingerprint"] or not secrets.compare_digest(
            str(row["fingerprint"]), expected_fingerprint
        ):
            return False
        cur = conn.execute(
            "UPDATE task_pending_actions SET consumed_at=?, state='consumed', updated_at=?, version=version+1 "
            "WHERE id=? AND fingerprint=? AND version=? AND state='approved' AND expires_at > ?",
            (now, now, int(row["id"]), expected_fingerprint, row["version"], now),
        )
        if cur.rowcount != 1:
            return False
        _delete_attention_for_action(conn, task_id, int(row["id"]))
        _attention_cleanup_hook()
        _append_event(
            conn, task_id, "terminal_approval_consumed",
            {"action_id": int(row["id"]), "run_id": int(run_id)},
            run_id=int(run_id),
        )
        return True


def _upsert_current_typed_attention_in_txn(
    conn: sqlite3.Connection, *, task_id: str, attention_type: str,
    reason_code: str, cause_scope: Optional[dict[str, str]] = None,
    summary: Optional[str] = None, origin_run_id: Optional[int] = None,
    now: Optional[int] = None,
) -> Attention:
    """In-transaction implementation; callers already own ``write_txn``."""
    if attention_type not in {"decision", "protocol", "review", "loop_triage", "capability", "transient"}:
        raise ValueError("unsupported typed attention")
    now = int(time.time()) if now is None else int(now)
    fingerprint = _block_cause_fingerprint(conn, task_id=task_id, attention_type=attention_type,
        reason_code=reason_code, scope=cause_scope)
    assert fingerprint is not None
    task = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task is None or task["status"] in ("done", "archived"):
        raise ValueError("typed attention requires a live task")
    current = conn.execute(
        "SELECT * FROM task_attentions WHERE task_id=? AND action_id IS NULL "
        "AND cause_fingerprint=?", (task_id, fingerprint),
    ).fetchone()
    text = (summary or reason_code)[:400]
    if current is None:
        conn.execute("DELETE FROM task_attentions WHERE task_id=? AND action_id IS NULL", (task_id,))
        cur = conn.execute(
            "INSERT INTO task_attentions (task_id, action_id, type, cause_fingerprint, summary, created_at, version, origin_run_id) "
            "VALUES (?, NULL, ?, ?, ?, ?, 1, ?)",
            (task_id, attention_type, fingerprint, text, now, origin_run_id),
        )
        attention_id, version = int(cur.lastrowid), 1
    else:
        conn.execute(
            "UPDATE task_attentions SET type=?, summary=?, origin_run_id=?, version=version+1 WHERE id=?",
            (attention_type, text, origin_run_id, int(current["id"])),
        )
        attention_id, version = int(current["id"]), int(current["version"]) + 1
    return Attention(attention_id, task_id, None, attention_type, text, now, "pending", version, None,
        attention_type in {"decision", "protocol", "review", "capability"}, False)


def upsert_current_typed_attention(
    conn: sqlite3.Connection, *, task_id: str, attention_type: str,
    reason_code: str, cause_scope: Optional[dict[str, str]] = None,
    summary: Optional[str] = None, origin_run_id: Optional[int] = None,
    now: Optional[int] = None,
) -> Attention:
    """Persist a typed blocker projection outside an existing transition."""
    with write_txn(conn):
        return _upsert_current_typed_attention_in_txn(
            conn, task_id=task_id, attention_type=attention_type, reason_code=reason_code,
            cause_scope=cause_scope, summary=summary, origin_run_id=origin_run_id, now=now,
        )


def get_current_attentions(
    conn: sqlite3.Connection, task_ids: Sequence[str], *, now: Optional[int] = None,
) -> dict[str, Attention]:
    """Read only current projections in bounded chunks; never materializes history."""
    now = int(time.time()) if now is None else int(now)
    ids = list(dict.fromkeys(str(task_id) for task_id in task_ids))
    result: dict[str, Attention] = {}
    # SQLite's default variable limit is 999; leave room for time predicates.
    for start in range(0, len(ids), 900):
        chunk = ids[start:start + 900]
        if not chunk:
            continue
        marks = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT x.id AS attention_id, x.task_id, x.action_id, x.type, x.summary, "
            "x.created_at AS attention_created_at, x.version AS projection_version, a.state AS action_state, "
            "a.version AS action_version, a.expires_at "
            "FROM task_attentions x JOIN tasks t ON t.id=x.task_id "
            "LEFT JOIN task_pending_actions a ON a.id=x.action_id "
            f"WHERE x.task_id IN ({marks}) AND t.status NOT IN ('done','archived') "
            "AND ((x.type='exact_action' AND a.state IN ('pending','approved') AND a.expires_at>?) "
            "OR (x.type IN ('capability','transient') AND x.action_id IS NOT NULL AND a.state='approved' AND a.expires_at>?) "
            "OR (x.type IN ('decision','protocol','review','loop_triage','capability','transient') AND x.action_id IS NULL)) "
            "ORDER BY x.id DESC",
            (*chunk, now, now),
        ).fetchall()
        for row in rows:
            task_id = str(row["task_id"])
            if task_id in result:
                continue
            exact = row["type"] == "exact_action"
            action_id = int(row["action_id"]) if row["action_id"] is not None else None
            result[task_id] = Attention(
                id=int(row["attention_id"]), task_id=task_id, action_id=action_id,
                type=row["type"], summary=row["summary"], created_at=int(row["attention_created_at"]),
                state=row["action_state"] if exact else "pending",
                # Exact-action approval is versioned on the durable action;
                # technical retries are versioned on their replacement projection.
                version=int(row["action_version"] if exact else row["projection_version"]),
                expires_at=(int(row["expires_at"]) if row["expires_at"] is not None else None),
                requires_human_action=(row["action_state"] == "pending") if exact else row["type"] in {"decision", "protocol", "review", "capability"},
                approvable=exact and row["action_state"] == "pending",
            )
    return result


def get_current_attention(conn: sqlite3.Connection, task_id: str, *, now: Optional[int] = None) -> Optional[Attention]:
    """Return the one live operator projection, never reconstructed from events."""
    return get_current_attentions(conn, [task_id], now=now).get(task_id)


def sync_attention_deliveries(
    conn: sqlite3.Connection, *, task_ids: Sequence[str], now: Optional[int] = None,
) -> dict[str, int]:
    """Project current attention x *active* subscription into durable deliveries.

    Subscription generation is an ABA fence: resubscribing the same channel
    re-arms exactly its old row while an old sender can no longer finish it.
    """
    now = int(time.time()) if now is None else int(now)
    ids = list(dict.fromkeys(str(x) for x in task_ids))
    current = get_current_attentions(conn, ids, now=now)
    created = cancelled = suppressed = 0
    with write_txn(conn):
        for task_id in ids:
            attention = current.get(task_id)
            if attention is None:
                cur = conn.execute("UPDATE kanban_attention_deliveries SET state='cancelled', lease_until=NULL, updated_at=? WHERE task_id=? AND state IN ('pending','sending')", (now, task_id))
                cancelled += cur.rowcount
                continue
            cur = conn.execute("UPDATE kanban_attention_deliveries SET state='cancelled', lease_until=NULL, updated_at=? WHERE task_id=? AND state IN ('pending','sending') AND (attention_id<>? OR attention_version<>?)", (now, task_id, attention.id, attention.version))
            cancelled += cur.rowcount
            subs = conn.execute("SELECT * FROM kanban_notify_subs WHERE task_id=? AND active=1", (task_id,)).fetchall()
            for sub in subs:
                key = (task_id, attention.id, attention.version, sub['platform'], sub['chat_id'], sub['thread_id'] or '')
                row = conn.execute("SELECT subscription_generation FROM kanban_attention_deliveries WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=?", key).fetchone()
                if row is None:
                    conn.execute("INSERT INTO kanban_attention_deliveries (task_id,attention_id,attention_version,platform,chat_id,thread_id,notifier_profile,subscription_generation,updated_at) VALUES (?,?,?,?,?,?,?,?,?)", (*key, sub['notifier_profile'], int(sub['generation']), now))
                    created += 1
                elif int(row['subscription_generation']) != int(sub['generation']):
                    conn.execute("UPDATE kanban_attention_deliveries SET notifier_profile=?, subscription_generation=?, state='pending', attempts=0, lease_version=lease_version+1, lease_until=NULL, delivered_at=NULL, last_error=NULL, updated_at=? WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=?", (sub['notifier_profile'], int(sub['generation']), now, *key))
                    created += 1
                else:
                    suppressed += 1
    return {'created': created, 'cancelled_resolved': cancelled, 'suppressed_duplicate': suppressed}


def has_current_attention_delivery(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str, thread_id: str = '',
    now: Optional[int] = None,
) -> bool:
    """Whether this channel has a durable row for exactly the live human attention."""
    current = get_current_attention(conn, task_id, now=now)
    if current is None or not current.requires_human_action:
        return False
    row = conn.execute(
        "SELECT 1 FROM kanban_attention_deliveries WHERE task_id=? AND attention_id=? "
        "AND attention_version=? AND platform=? AND chat_id=? AND thread_id=? "
        "AND state IN ('pending','sending','delivered')",
        (task_id, current.id, current.version, platform, chat_id, thread_id or ''),
    ).fetchone()
    return row is not None


def claim_attention_delivery(
    conn: sqlite3.Connection, *, task_id: str, attention_id: int, attention_version: int,
    platform: str, chat_id: str, thread_id: str = '', now: Optional[int] = None,
    lease_seconds: int = 60,
) -> Optional[dict[str, Any]]:
    """Atomically lease one current attention/channel delivery, or return None."""
    now = int(time.time()) if now is None else int(now)
    key = (task_id, int(attention_id), int(attention_version), platform, chat_id, thread_id or '')
    with write_txn(conn):
        live = get_current_attention(conn, task_id, now=now)
        if live is None or live.id != int(attention_id) or live.version != int(attention_version):
            conn.execute("UPDATE kanban_attention_deliveries SET state='cancelled', lease_until=NULL, updated_at=? "
                         "WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=? "
                         "AND state IN ('pending','sending')", (now, *key))
            return None
        row = conn.execute(
            "SELECT * FROM kanban_attention_deliveries WHERE task_id=? AND attention_id=? AND attention_version=? "
            "AND platform=? AND chat_id=? AND thread_id=?", key,
        ).fetchone()
        if row is None or row['state'] in ('delivered', 'cancelled'):
            return None
        sub = conn.execute(
            "SELECT 1 FROM kanban_notify_subs WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=? "
            "AND active=1 AND generation=?",
            (task_id, platform, chat_id, thread_id or '', int(row['subscription_generation'])),
        ).fetchone()
        if sub is None:
            conn.execute("UPDATE kanban_attention_deliveries SET state='cancelled', lease_until=NULL, updated_at=? WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=? AND state IN ('pending','sending')", (now, *key))
            return None
        if row['state'] == 'sending' and (row['lease_until'] or 0) > now:
            return None
        cur = conn.execute(
            "UPDATE kanban_attention_deliveries SET state='sending', attempts=attempts+1, lease_version=lease_version+1, lease_until=?, updated_at=? "
            "WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=? "
            "AND (state='pending' OR (state='sending' AND COALESCE(lease_until,0)<=?))",
            (now + int(lease_seconds), now, *key, now),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute("SELECT * FROM kanban_attention_deliveries WHERE task_id=? AND attention_id=? "
                               "AND attention_version=? AND platform=? AND chat_id=? AND thread_id=?", key).fetchone()
        return dict(claimed) if claimed else None


def finish_attention_delivery(conn: sqlite3.Connection, delivery: Mapping[str, Any], *, success: bool,
                              now: Optional[int] = None) -> bool:
    """CAS a lease to delivered or retry-pending, fenced by subscription generation."""
    now = int(time.time()) if now is None else int(now)
    key = tuple(delivery[x] for x in ('task_id','attention_id','attention_version','platform','chat_id','thread_id'))
    try:
        lease_version = int(delivery['lease_version'])
        subscription_generation = int(delivery['subscription_generation'])
    except (KeyError, TypeError, ValueError):
        return False
    with write_txn(conn):
        if success:
            cur = conn.execute("UPDATE kanban_attention_deliveries SET state='delivered', delivered_at=?, lease_until=NULL, last_error=NULL, updated_at=? WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=? AND state='sending' AND lease_version=? AND subscription_generation=?", (now, now, *key, lease_version, subscription_generation))
        else:
            cur = conn.execute("UPDATE kanban_attention_deliveries SET state='pending', lease_until=NULL, last_error='send_failed', updated_at=? WHERE task_id=? AND attention_id=? AND attention_version=? AND platform=? AND chat_id=? AND thread_id=? AND state='sending' AND lease_version=? AND subscription_generation=?", (now, *key, lease_version, subscription_generation))
        return cur.rowcount == 1


def get_action_by_attention_id(
    conn: sqlite3.Connection, task_id: str, attention_id: int,
) -> Optional[PendingAction]:
    """Task-bound opaque-ID history resolver for terminal 410 mapping.

    Legacy rows without an immutable binding intentionally return ``None``;
    events are never used as authority or reconstruction input.
    """
    row = conn.execute(
        "SELECT * FROM task_pending_actions WHERE task_id=? AND attention_id=?",
        (task_id, int(attention_id)),
    ).fetchone()
    return _pending_action_from_row(row) if row is not None else None


def finalize_goal_block_or_reuse_current_attention(
    conn: sqlite3.Connection, task_id: str, *, reason: str,
) -> bool:
    """Atomically choose exact-action precedence or the typed goal fallback.

    The choice and fallback transition share one IMMEDIATE transaction: a
    concurrent action request cannot appear in the gap between a read and a
    generic block.  Hooks deliberately run only after commit.
    """
    now = int(time.time())
    hook: Optional[tuple[Optional[Task], Optional[int]]] = None
    with write_txn(conn):
        task = conn.execute("SELECT status, current_run_id, block_recurrences, block_cause_fingerprint FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None or task["status"] in ("done", "archived"):
            return False
        action = conn.execute(
            "SELECT a.*, x.id AS attention_id FROM task_pending_actions a "
            "JOIN task_attentions x ON x.action_id=a.id AND x.type='exact_action' "
            "WHERE a.task_id=? AND a.state IN ('pending','approved') AND a.expires_at>? "
            "ORDER BY a.id DESC LIMIT 1", (task_id, now),
        ).fetchone()
        if action is not None:
            if task["status"] == "blocked" and task["current_run_id"] is None:
                return True
            if task["status"] != "running" or task["current_run_id"] != action["run_id"]:
                return False
            run = conn.execute("SELECT status, ended_at FROM task_runs WHERE id=? AND task_id=?", (action["run_id"], task_id)).fetchone()
            if run is None or run["status"] != "running" or run["ended_at"] is not None:
                return False
            conn.execute("UPDATE tasks SET status='blocked', current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, block_kind='needs_input' WHERE id=? AND status='running' AND current_run_id=?", (task_id, action["run_id"]))
            conn.execute("UPDATE task_runs SET status='blocked', outcome='blocked', ended_at=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=? AND task_id=? AND status='running' AND ended_at IS NULL", (now, action["run_id"], task_id))
            hook = (get_task(conn, task_id), int(action["run_id"]))
        else:
            # Typed protocol fallback, fully in this transaction.  Free worker
            # prose never becomes a task_runs summary (which is public via
            # latest_run/latest_summary); only this enumerable code persists.
            fingerprint = _block_cause_fingerprint(conn, task_id=task_id,
                attention_type="protocol", reason_code="goal_closeout_missing",
                scope={"protocol": "goal_closeout"})
            assert fingerprint is not None
            recurrences = int(task["block_recurrences"] or 0) + 1 if task["block_cause_fingerprint"] == fingerprint else 1
            routed_to = "triage" if recurrences >= BLOCK_RECURRENCE_LIMIT else "blocked"
            cur = conn.execute(
                "UPDATE tasks SET status=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
                "block_kind='needs_input', block_recurrences=?, block_cause_fingerprint=?, "
                "block_reason_code=?, block_cause_version=? WHERE id=? AND status IN ('running','ready')",
                (routed_to, recurrences, fingerprint, "goal_closeout_missing", BLOCK_CAUSE_VERSION, task_id),
            )
            if cur.rowcount != 1:
                return False
            _goal_finalizer_after_task_update_hook()
            run_id = _end_run(conn, task_id, outcome="blocked", status="blocked", summary="goal_closeout_missing")
            if run_id is None:
                run_id = _synthesize_ended_run(conn, task_id, outcome="blocked", summary="goal_closeout_missing")
            _append_event(conn, task_id, "block_loop_detected" if routed_to == "triage" else "blocked", {
                "kind": "needs_input", "attention_type": "protocol",
                "reason_code": "goal_closeout_missing", "recurrences": recurrences,
                **({"limit": BLOCK_RECURRENCE_LIMIT} if routed_to == "triage" else {}),
            }, run_id=run_id)
            hook = (get_task(conn, task_id), run_id)
    if hook is not None:
        blocked_task, run_id = hook
        _fire_kanban_lifecycle_hook("kanban_task_blocked", task_id,
            board=get_current_board(), assignee=blocked_task.assignee if blocked_task else None,
            run_id=run_id,
            reason=("terminal_approval_required" if action is not None else "goal_closeout_missing"))
    return True


def _goal_finalizer_after_task_update_hook() -> None:
    """Private in-transaction fault seam for finalizer rollback tests."""


def _technical_attention_after_replace_hook() -> None:
    """Private in-transaction fault seam for technical replacement rollback tests."""


def _approved_action_retry_after_projection_restore_hook() -> None:
    """Private in-transaction fault seam for opaque retry rollback tests."""


def block_approved_action_for_technical_failure(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    action_id: int,
    expected_run_id: int,
    attention_type: str,
    reason_code: str,
    cause_scope: Optional[dict[str, str]] = None,
    now: Optional[int] = None,
) -> bool:
    """Atomically replace a waiting grant's projection with one typed failure.

    The approved exact action remains internal and valid for an explicit retry;
    the technical projection is the sole current operator-facing attention.
    """
    if attention_type not in {"capability", "transient"}:
        raise ValueError("technical replacement requires capability or transient attention")
    now = int(time.time()) if now is None else int(now)
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        fingerprint = _block_cause_fingerprint(
            conn, task_id=task_id, attention_type=attention_type,
            reason_code=reason_code, scope=cause_scope,
        )
        assert fingerprint is not None
        action = conn.execute(
            "SELECT * FROM task_pending_actions WHERE id=? AND task_id=? "
            "AND state='approved' AND expires_at>?",
            (int(action_id), task_id, now),
        ).fetchone()
        task = conn.execute(
            "SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if action is None or task is None or task["status"] != "running" or task["current_run_id"] != int(expected_run_id):
            return False
        run = conn.execute(
            "SELECT status, ended_at FROM task_runs WHERE id=? AND task_id=?",
            (int(expected_run_id), task_id),
        ).fetchone()
        if run is None or run["status"] != "running" or run["ended_at"] is not None:
            return False
        if conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, block_kind=? WHERE id=? "
            "AND status='running' AND current_run_id=?",
            (attention_type, task_id, int(expected_run_id)),
        ).rowcount != 1:
            return False
        if conn.execute(
            "UPDATE task_runs SET status='blocked', outcome='blocked', ended_at=?, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=? "
            "AND task_id=? AND status='running' AND ended_at IS NULL",
            (now, int(expected_run_id), task_id),
        ).rowcount != 1:
            return False
        # Bind retry authority to this exact resumed run before exposing the
        # technical projection.  A prior blocked run cannot authorize a retry.
        if conn.execute(
            "UPDATE task_pending_actions SET retry_origin_run_id=? WHERE id=? AND task_id=? "
            "AND state='approved' AND expires_at>?",
            (int(expected_run_id), int(action_id), task_id, now),
        ).rowcount != 1 or action["attention_id"] is None:
            raise RuntimeError("technical retry provenance binding lost")
        # Preserve the opaque attention identity across technical replacement.
        conn.execute("DELETE FROM task_attentions WHERE task_id=?", (task_id,))
        conn.execute(
            "INSERT INTO task_attentions (id, task_id, action_id, type, cause_fingerprint, summary, created_at, origin_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (int(action["attention_id"]), task_id, int(action_id), attention_type, fingerprint,
             "A technical worker failure requires attention.", now, int(expected_run_id)),
        )
        _technical_attention_after_replace_hook()
        _append_event(conn, task_id, "blocked", {
            "kind": attention_type, "attention_type": attention_type,
            "reason_code": reason_code, "recurrences": 1,
        }, run_id=int(expected_run_id))
    _fire_kanban_lifecycle_hook(
        "kanban_task_blocked", task_id, board=get_current_board(),
        run_id=int(expected_run_id), reason=reason_code,
    )
    return True


def _park_approved_action_on_technical_failure_if_current(
    conn: sqlite3.Connection, *, task_id: str, expected_run_id: int,
    attention_type: Literal["capability", "transient"], reason_code: str,
    now: Optional[int] = None,
) -> bool:
    """Dispatcher bridge for a running approved-action attempt.

    It must run before a normal crash/timeout requeue: after the task has
    lost its running origin, a technical retry is no longer authoritative.
    """
    now = int(time.time()) if now is None else int(now)
    row = conn.execute(
        "SELECT a.id FROM task_pending_actions a JOIN tasks t ON t.id=a.task_id "
        "WHERE a.task_id=? AND a.state='approved' AND a.expires_at>? "
        "AND t.status='running' AND t.current_run_id=? ORDER BY a.id DESC LIMIT 1",
        (task_id, now, int(expected_run_id)),
    ).fetchone()
    if row is None:
        return False
    return block_approved_action_for_technical_failure(
        conn, task_id=task_id, action_id=int(row["id"]),
        expected_run_id=int(expected_run_id), attention_type=attention_type,
        reason_code=reason_code, now=now,
    )


def restore_approved_action_attention(
    conn: sqlite3.Connection, task_id: str, action_id: int, *, now: Optional[int] = None,
) -> bool:
    """Explicit retry lifecycle step that re-projects a still-valid approved grant."""
    now = int(time.time()) if now is None else int(now)
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        action = conn.execute(
            "SELECT a.* FROM task_pending_actions a JOIN tasks t ON t.id=a.task_id "
            "JOIN task_runs r ON r.id=a.run_id AND r.task_id=a.task_id "
            "WHERE a.id=? AND a.task_id=? AND a.state='approved' AND a.expires_at>? "
            "AND t.status='blocked' AND t.current_run_id IS NULL "
            "AND t.claim_lock IS NULL AND t.claim_expires IS NULL AND t.worker_pid IS NULL "
            "AND r.status='blocked' AND r.outcome='blocked' AND r.ended_at IS NOT NULL",
            (int(action_id), task_id, now),
        ).fetchone()
        if action is None:
            return False
        technical = conn.execute(
            "SELECT x.origin_run_id FROM task_attentions x JOIN task_runs r ON r.id=x.origin_run_id "
            "WHERE x.task_id=? AND x.action_id=? AND x.type IN ('capability','transient') "
            "AND r.task_id=x.task_id AND r.status='blocked' AND r.outcome='blocked' "
            "AND r.ended_at IS NOT NULL",
            (task_id, int(action_id)),
        ).fetchone()
        if technical is None or technical["origin_run_id"] != action["retry_origin_run_id"]:
            return False
        conn.execute("DELETE FROM task_attentions WHERE task_id=?", (task_id,))
        _upsert_exact_action_attention(conn, int(action_id), task_id, str(action["fingerprint"]), now)
        return True


def resume_approved_action_retry(
    conn: sqlite3.Connection, *, task_id: str, expected_attention_id: int,
    expected_attention_version: int, expected_origin_run_id: int,
    actor: str = "operator", now: Optional[int] = None,
) -> ResumeApprovedActionRetryResult:
    """Resume the one approved grant bound to this opaque retry attention.

    ``attention_id`` is the public authority; the action is resolved only
    task-bound inside this transaction.  ``expected_origin_run_id`` pins the
    technical failure that is being retried, so an old blocked run cannot
    satisfy the lifecycle predicate.
    """
    now = int(time.time()) if now is None else int(now)
    with write_txn(conn):
        task = conn.execute(
            "SELECT status, current_run_id, claim_lock, claim_expires, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        action = conn.execute(
            "SELECT * FROM task_pending_actions WHERE task_id=? AND attention_id=?",
            (task_id, int(expected_attention_id)),
        ).fetchone()
        if task is None or action is None:
            return ResumeApprovedActionRetryResult("not_found", task_id)
        if action["state"] != "approved" or int(action["expires_at"]) <= now:
            return ResumeApprovedActionRetryResult("gone", task_id)
        # The current technical projection is the sole public retry authority.
        # Resolve its private action binding only inside this transaction.
        attention = conn.execute(
            "SELECT x.id, x.version, x.origin_run_id FROM task_attentions x "
            "WHERE x.task_id=? AND x.id=? AND x.action_id=? "
            "AND x.type IN ('capability','transient')",
            (task_id, int(expected_attention_id), int(action["id"])),
        ).fetchone()
        if attention is None:
            return ResumeApprovedActionRetryResult("gone", task_id)
        if int(attention["version"]) != int(expected_attention_version):
            return ResumeApprovedActionRetryResult("conflict", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]))
        if attention["origin_run_id"] != int(expected_origin_run_id):
            return ResumeApprovedActionRetryResult("conflict", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]))
        origin = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id=? AND task_id=?",
            (int(expected_origin_run_id), task_id),
        ).fetchone()
        if (origin is None or origin["status"] != "blocked" or origin["outcome"] != "blocked"
                or origin["ended_at"] is None or action["retry_origin_run_id"] != int(expected_origin_run_id)):
            return ResumeApprovedActionRetryResult("conflict", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]))
        if (task["status"] != "blocked" or task["current_run_id"] is not None
                or task["claim_lock"] is not None or task["claim_expires"] is not None or task["worker_pid"] is not None):
            return ResumeApprovedActionRetryResult("conflict", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]))
        # Refuse an overlapping active lifecycle even if a damaged database has
        # several action rows; this is a fail-closed authority boundary.
        conflict = conn.execute(
            "SELECT 1 FROM task_pending_actions WHERE task_id=? AND id<>? "
            "AND state IN ('pending','approved') LIMIT 1",
            (task_id, int(action["id"])),
        ).fetchone()
        if conflict is not None:
            return ResumeApprovedActionRetryResult("conflict", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]))
        target: Literal["ready", "todo"] = "todo" if conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id WHERE l.child_id=? AND p.status NOT IN ('done','archived') LIMIT 1", (task_id,)
        ).fetchone() else "ready"
        # Rebind the stable opaque projection in-place before releasing the
        # task.  The worker will see the same public ID as an approved exact
        # action until it consumes the one-time grant.
        if conn.execute(
            "UPDATE task_attentions SET type='exact_action', origin_run_id=NULL, summary=? "
            "WHERE id=? AND task_id=? AND action_id=? AND type IN ('capability','transient') "
            "AND version=? AND origin_run_id=?",
            (PENDING_ACTION_OPERATOR_SUMMARY, int(attention["id"]), task_id, int(action["id"]),
             int(expected_attention_version), int(expected_origin_run_id)),
        ).rowcount != 1:
            return ResumeApprovedActionRetryResult("conflict", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]))
        _approved_action_retry_after_projection_restore_hook()
        if conn.execute(
            "UPDATE tasks SET status=?, current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL "
            "WHERE id=? AND status='blocked' AND current_run_id IS NULL AND claim_lock IS NULL "
            "AND claim_expires IS NULL AND worker_pid IS NULL",
            (target, task_id),
        ).rowcount != 1:
            raise RuntimeError("approved action retry task CAS lost")
        _append_event(conn, task_id, "unblocked", {"status": target, "actor": str(actor)[:120]})
        _append_event(conn, task_id, "approved_action_retry_resumed", {"actor": str(actor)[:120]}, run_id=int(expected_origin_run_id))
        return ResumeApprovedActionRetryResult("resumed", task_id, attention_id=int(attention["id"]), attention_version=int(attention["version"]), task_status=target)


def resolve_pending_action(
    conn: sqlite3.Connection, task_id: str, action_id: int, *,
    expected_version: Optional[int] = None, now: Optional[int] = None,
) -> ResolvePendingActionResult:
    """Atomically resolve a pending/approved action with safe HTTP semantics.

    ``not_found`` maps to 404, an active version mismatch to 409, and an
    expired/already-terminal action to 410.  Omitting ``expected_version`` is
    retained only for legacy internal callers; public callers must provide it.
    """
    now = int(time.time()) if now is None else int(now)
    with write_txn(conn):
        _materialize_expired_actions(conn, now)
        row = conn.execute(
            "SELECT * FROM task_pending_actions WHERE id=? AND task_id=?",
            (int(action_id), task_id),
        ).fetchone()
        if row is None:
            return ResolvePendingActionResult("not_found")
        if row["state"] not in ("pending", "approved"):
            return ResolvePendingActionResult("gone")
        if expected_version is not None and int(expected_version) != int(row["version"]):
            return ResolvePendingActionResult("conflict")
        version = int(row["version"])
        cur = conn.execute(
            "UPDATE task_pending_actions SET state='resolved', resolved_at=?, "
            "updated_at=?, version=version+1 WHERE id=? AND task_id=? AND "
            "version=? AND state IN ('pending','approved')",
            (now, now, int(action_id), task_id, version),
        )
        if cur.rowcount == 1:
            _delete_attention_for_action(conn, task_id, int(action_id))
            _attention_cleanup_hook()
            return ResolvePendingActionResult("resolved")
        settled = conn.execute(
            "SELECT state, version FROM task_pending_actions WHERE id=? AND task_id=?",
            (int(action_id), task_id),
        ).fetchone()
        if settled is None:
            return ResolvePendingActionResult("not_found")
        return ResolvePendingActionResult(
            "gone" if settled["state"] not in ("pending", "approved") else "conflict"
        )


def _cancel_active_actions_for_terminal_task(
    conn: sqlite3.Connection, task_id: str, *, now: int, reason: str,
) -> None:
    """Terminal tasks retain action history but never retain active projections."""
    rows = conn.execute(
        "SELECT id FROM task_pending_actions WHERE task_id=? AND state IN ('pending','approved')",
        (task_id,),
    ).fetchall()
    for row in rows:
        action_id = int(row["id"])
        if conn.execute(
            "UPDATE task_pending_actions SET state='cancelled', cancelled_at=?, updated_at=?, version=version+1 "
            "WHERE id=? AND state IN ('pending','approved')",
            (now, now, action_id),
        ).rowcount == 1:
            _delete_attention_for_action(conn, task_id, action_id)
            _attention_cleanup_hook()
            _append_event(conn, task_id, "terminal_approval_cancelled", {
                "action_id": action_id, "reason": reason,
            })


def _cancel_approved_pending_action(
    conn: sqlite3.Connection, task_id: str, *, now: int,
    reason: str, run_id: Optional[int],
) -> None:
    """Cancel granted-but-unexecuted actions when their resumed run re-blocks."""
    rows = conn.execute(
        "SELECT id FROM task_pending_actions WHERE task_id = ? "
        "AND approved_at IS NOT NULL AND consumed_at IS NULL "
        "AND cancelled_at IS NULL AND expires_at > ?",
        (task_id, int(now)),
    ).fetchall()
    for row in rows:
        cur = conn.execute(
            "UPDATE task_pending_actions SET cancelled_at = ?, state='cancelled', updated_at=?, version=version+1 "
            "WHERE id = ? AND state='approved'",
            (int(now), int(now), int(row["id"])),
        )
        if cur.rowcount == 1:
            _delete_attention_for_action(conn, task_id, int(row["id"]))
            _attention_cleanup_hook()
            _append_event(
                conn, task_id, "terminal_approval_cancelled",
                {"action_id": int(row["id"]), "reason": reason},
                run_id=run_id,
            )


def _block_cause_fingerprint(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    attention_type: Optional[str],
    reason_code: Optional[str],
    scope: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Return a safe typed cause identity, or None for legacy callers.

    Compatibility is intentionally fail-closed: omitted typed inputs do not
    hash free-form reason text and therefore never continue an old chain.
    """
    if attention_type is None and reason_code is None:
        return None
    if attention_type not in VALID_BLOCK_ATTENTION_TYPES:
        raise ValueError("unsupported block attention type")
    if reason_code not in VALID_BLOCK_REASON_CODES:
        raise ValueError("unsupported block reason code")
    allowed_scope = BLOCK_CAUSE_SCOPE_ENUMS.get((attention_type, reason_code))
    if allowed_scope is None:
        raise ValueError("unsupported block attention type/reason code pair")
    clean_scope: dict[str, str] = {}
    for key, value in (scope or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("block cause scope must use string enums")
        permitted = allowed_scope.get(key)
        if permitted is None or value not in permitted:
            raise ValueError("unsupported block cause scope")
        clean_scope[key] = value
    manifest = {"v": BLOCK_CAUSE_VERSION, "board": _pending_action_board_identity(conn),
                "task_id": task_id, "attention_type": attention_type,
                "reason_code": reason_code, "scope": clean_scope}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _block_cause_after_task_update_hook() -> None:
    """Private fault-injection seam; runs inside the block transaction."""


def block_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    kind: Optional[str] = None,
    expected_run_id: Optional[int] = None,
    require_unclaimed: bool = False,
    trusted_internal: bool = False,
    human_gate: Optional[bool] = None,
    human_summary: Optional[str] = None,
    human_action: Optional[str] = None,
    attention_type: Optional[str] = None,
    reason_code: Optional[str] = None,
    cause_scope: Optional[dict[str, str]] = None,
    _governance_target_ref: Optional[str] = None,
    _governance_mutation_class: Optional[str] = None,
) -> bool:
    """Transition ``running``/``ready`` → ``blocked`` (or route elsewhere).

    ``kind`` (one of :data:`VALID_BLOCK_KINDS`, or ``None`` for a legacy
    un-typed block) drives routing instead of every block landing in one
    undifferentiated ``blocked`` bucket:

    * ``dependency`` — the task is only waiting on another task. It does NOT
      sit in ``blocked`` (where a cron would keep "unblocking" it); it goes to
      ``todo`` so the existing parent-gating / ``recompute_ready`` machinery
      promotes it automatically once its parents finish. No human, no cron, no
      retry storm. This is Dale's "Type 2 — dependency blocked".

    * ``needs_input`` / ``capability`` / ``None`` — "truly blocked" (Dale's
      "Type 1"). Lands in ``blocked`` for a human. BUT: each time such a task
      is re-blocked for the SAME kind after having been unblocked, the
      unblock-loop counter (``block_recurrences``) increments. When it reaches
      :data:`BLOCK_RECURRENCE_LIMIT`, the task is routed to ``triage`` instead
      of ``blocked`` — breaking the cron-unblock ↔ worker-re-block loop and
      forcing a human-in-the-loop triage decision.

    * ``transient`` — treated like a generic block for routing, but a worker
      can use it to signal "this might clear on its own"; it still participates
      in the loop breaker so a forever-flaky task eventually escalates.

    ``human_gate=True`` marks the card as hard-gated (Human-Gate v1) the
    moment it lands in ``blocked`` — CLI-only (``kanban block
    --human-gate``); no tool surface can set this. ``None`` (the default,
    used by every automated block/re-block) leaves the existing flag
    untouched, so a gate set once persists across re-block cycles.
    Meaningless (ignored) for the ``dependency`` and loop-breaker
    ``triage`` routes, which never sit in ``blocked`` for a human.

    ``human_summary`` / ``human_action`` (optional) are the layman-facing
    counterpart to the technical ``reason``: 1–3 plain-language sentences on
    what went wrong and what the operator should do. They are stored verbatim
    in the block event payload so notification relays (Telegram/ntfy) can
    lead with them instead of the technical reason. Enforcement that workers
    supply them lives in the ``kanban_block`` tool handler, not here — the
    kernel accepts blocks without them (CLI/legacy callers).

    ``require_unclaimed=True`` restricts the transition to a card that is
    still unclaimed (``claim_lock IS NULL``). This is the CAS guard for
    dispatcher-initiated blocks of *ready* cards (self-modify gate,
    active_pr escalation): a never-run ready card has ``current_run_id``
    NULL, so ``expected_run_id`` cannot express "nobody claimed it since my
    snapshot" — but any claim always sets ``claim_lock``, so this guard
    atomically refuses to de-claim a card that was claimed between snapshot
    and block.

    ``trusted_internal=True`` skips the kernel worker gates (audit
    2026-07-11 C1) for harness code that legitimately blocks in-process
    inside a worker (e.g. the goal-loop escape in cli.py). Model-driven
    surfaces (tool handler, CLI) must never set it.

    Raises :class:`WorkerGateError` when the calling process is a
    dispatcher-spawned worker and the transition fails the worker gates
    (ownership / run identity / goal-mode block kinds).

    Returns True on any successful transition (to ``blocked``, ``todo``, or
    ``triage``), False when the task wasn't in a blockable state.
    """
    if kind is not None and kind not in VALID_BLOCK_KINDS:
        raise ValueError(
            f"block kind must be one of {sorted(VALID_BLOCK_KINDS)} or None"
        )
    if (_governance_target_ref is None) != (_governance_mutation_class is None):
        raise ValueError("internal governance declarations require target and class")
    if _governance_target_ref is not None:
        _validate_governance_declarations(
            target_ref=_governance_target_ref,
            mutation_class=_governance_mutation_class or "",
        )
    if not trusted_internal:
        _enforce_worker_block_gates(conn, task_id, kind)
    unclaimed_guard = " AND claim_lock IS NULL" if require_unclaimed else ""

    def _with_human_fields(payload: dict) -> dict:
        if human_summary and str(human_summary).strip():
            payload["human_summary"] = str(human_summary).strip()
        if human_action and str(human_action).strip():
            payload["human_action"] = str(human_action).strip()
        return payload

    def _typed_event_payload(*, recurrences: int, loop: bool = False) -> dict[str, object]:
        """Only enumerable typed cause data may enter durable operator surfaces."""
        payload: dict[str, object] = {
            "kind": kind, "attention_type": attention_type,
            "reason_code": reason_code, "recurrences": recurrences,
        }
        if loop:
            payload["limit"] = BLOCK_RECURRENCE_LIMIT
        return payload

    typed_cause = attention_type is not None or reason_code is not None
    # Run summaries are public through latest_run/latest_summary.  Typed causes
    # are protocol data, so retain only their allowlisted reason code there.
    persisted_summary = reason_code if typed_cause else reason

    # Dependency waits are a complete transition of their own. Keep the hook
    # outside ``write_txn`` so subscribers can only observe committed state.
    if kind == "dependency":
        # Dependencies deliberately have no attention projection or recurrence
        # fingerprint, but callers may still supply a typed cause for audit
        # classification. Validate it before treating its code as safe output.
        if typed_cause:
            _block_cause_fingerprint(
                conn, task_id=task_id, attention_type=attention_type,
                reason_code=reason_code, scope=cause_scope,
            )
        with write_txn(conn):
            if expected_run_id is None:
                params = (kind, task_id)
                run_guard = ""
            else:
                params = (kind, task_id, int(expected_run_id))
                run_guard = " AND current_run_id = ?"
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status = 'todo', claim_lock = NULL,
                       claim_expires = NULL, worker_pid = NULL, block_kind = ?
                 WHERE id = ? AND status IN ('running', 'ready')
                """ + run_guard + unclaimed_guard,
                params,
            )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn,
                task_id,
                outcome="blocked",
                status="blocked",
                summary=persisted_summary,
            )
            _cancel_approved_pending_action(
                conn,
                task_id,
                now=int(time.time()),
                reason="resumed_run_reblocked",
                run_id=run_id,
            )
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id, outcome="blocked", summary=persisted_summary,
                )
            _append_event(
                conn,
                task_id,
                "dependency_wait",
                _typed_event_payload(recurrences=0) if typed_cause else _with_human_fields({"reason": reason, "kind": kind}),
                run_id=run_id,
            )
            blocked_task = get_task(conn, task_id)
        _fire_kanban_lifecycle_hook(
            "kanban_task_blocked",
            task_id,
            board=get_current_board(),
            assignee=blocked_task.assignee if blocked_task else None,
            run_id=run_id,
            reason=reason_code if typed_cause else reason,
        )
        return True

    routed_to = "blocked"
    recurrences = 0
    now = int(time.time())
    with write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences, block_cause_fingerprint FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        prev_kind = cur_row["block_kind"] if "block_kind" in cur_row.keys() else None
        prev_recurrences = (
            int(cur_row["block_recurrences"])
            if "block_recurrences" in cur_row.keys()
            and cur_row["block_recurrences"] is not None
            else 0
        )

        # Only an exact persisted typed fingerprint can continue a chain.
        # Legacy callers intentionally receive a fresh recurrence every time.
        cause_fingerprint = _block_cause_fingerprint(
            conn, task_id=task_id, attention_type=attention_type,
            reason_code=reason_code, scope=cause_scope,
        )
        same_cause = (cause_fingerprint is not None
                      and str(cur_row["block_cause_fingerprint"] or "") == cause_fingerprint)
        recurrences = prev_recurrences + 1 if same_cause else 1

        pending_terminal_action = (
            kind == "needs_input"
            and get_pending_action(conn, task_id, now=now) is not None
        )
        if recurrences >= BLOCK_RECURRENCE_LIMIT and not pending_terminal_action:
            # Loop detected — stop letting the unblocker spin this task. Route
            # to triage for a human-in-the-loop decision instead of blocked.
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'triage',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       block_kind    = ?,
                       block_recurrences = ?
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + ("" if expected_run_id is None else " AND current_run_id = ?")
                + unclaimed_guard,
                (kind, recurrences, task_id) if expected_run_id is None
                else (kind, recurrences, task_id, int(expected_run_id)),
            )
            if cur.rowcount != 1:
                return False
            conn.execute(
                "UPDATE tasks SET block_cause_fingerprint=?, block_reason_code=?, block_cause_version=? WHERE id=?",
                (cause_fingerprint, reason_code, BLOCK_CAUSE_VERSION if cause_fingerprint else None, task_id),
            )
            _block_cause_after_task_update_hook()
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=persisted_summary,
            )
            _cancel_approved_pending_action(
                conn, task_id, now=now,
                reason="resumed_run_reblocked", run_id=run_id,
            )
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id, outcome="blocked", summary=persisted_summary,
                )
            _append_event(
                conn, task_id, "block_loop_detected",
                _typed_event_payload(recurrences=recurrences, loop=True) if typed_cause else _with_human_fields({
                    "reason": reason,
                    "kind": kind,
                    "recurrences": recurrences,
                    "limit": BLOCK_RECURRENCE_LIMIT,
                }),
                run_id=run_id,
            )
            routed_to = "triage"
        else:
            # COALESCE(?, human_gate): passing None leaves the existing flag
            # untouched (the common case — every automated block/re-block),
            # passing 1 sets it (only ``kanban block --human-gate``).
            gate_param = 1 if human_gate else None
            if expected_run_id is None:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?,
                           human_gate    = COALESCE(?, human_gate)
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                    """ + unclaimed_guard,
                    (kind, recurrences, gate_param, task_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?,
                           human_gate    = COALESCE(?, human_gate)
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                       AND current_run_id = ?
                    """ + unclaimed_guard,
                    (kind, recurrences, gate_param, task_id, int(expected_run_id)),
                )
            if cur.rowcount != 1:
                return False
            if _governance_target_ref is not None:
                _set_internal_governance_declarations(
                    conn,
                    task_id,
                    target_ref=_governance_target_ref,
                    mutation_class=_governance_mutation_class or "",
                )
            conn.execute(
                "UPDATE tasks SET block_cause_fingerprint=?, block_reason_code=?, block_cause_version=? WHERE id=?",
                (cause_fingerprint, reason_code, BLOCK_CAUSE_VERSION if cause_fingerprint else None, task_id),
            )
            _block_cause_after_task_update_hook()
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=persisted_summary,
            )
            _cancel_approved_pending_action(
                conn, task_id, now=now,
                reason="resumed_run_reblocked", run_id=run_id,
            )
            # Synthesize a run when blocking a never-claimed task so the
            # reason is preserved in attempt history.
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id,
                    outcome="blocked",
                    summary=persisted_summary,
                )
            if typed_cause and attention_type is not None and reason_code is not None:
                _upsert_current_typed_attention_in_txn(
                    conn, task_id=task_id,
                    attention_type="loop_triage" if routed_to == "triage" else attention_type,
                    reason_code=reason_code, cause_scope=cause_scope,
                    summary=reason_code, origin_run_id=run_id, now=now,
                )
            _append_event(
                conn, task_id, "blocked",
                _typed_event_payload(recurrences=recurrences) if typed_cause else _with_human_fields(
                    {"reason": reason, "kind": kind, "recurrences": recurrences}
                ),
                run_id=run_id,
            )
        _blocked_task = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_blocked",
        task_id,
        board=get_current_board(),
        assignee=_blocked_task.assignee if _blocked_task else None,
        run_id=run_id,
        reason=reason_code if typed_cause else reason,
    )
    return True



def promote_task(
    conn: sqlite3.Connection, task_id: str, *, actor: str, reason: Optional[str] = None,
    force: bool = False, dry_run: bool = False, token: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Promote atomically; dry runs are read-only observations only."""
    now = int(time.time())

    def validate() -> tuple[Optional[str], Optional[str]]:
        row = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return None, f"task {task_id} not found"
        status = row["status"]
        if status not in ("todo", "blocked"):
            return None, f"task {task_id} is {status!r}; promote only applies to 'todo' or 'blocked'"
        if conn.execute("SELECT 1 FROM task_pending_actions WHERE task_id=? AND state IN ('pending','approved') AND expires_at>? LIMIT 1", (task_id, now)).fetchone():
            return None, "exact terminal action approval is still pending"
        if conn.execute("SELECT 1 FROM task_attentions WHERE task_id=? LIMIT 1", (task_id,)).fetchone():
            return None, "current attention requires its typed lifecycle operation"
        if not force and conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
            "WHERE l.child_id=? AND p.status NOT IN ('done','archived') LIMIT 1", (task_id,)
        ).fetchone():
            return None, "unsatisfied parent dependencies (use --force to override)"
        return status, None

    if dry_run:
        _, error = validate()
        return error is None, error
    with write_txn(conn):
        status, error = validate()
        if error:
            return False, error
        try:
            _assert_human_gate_open(conn, task_id, token=token, action="promote")
        except GateTokenError as exc:
            return False, str(exc)
        if conn.execute("UPDATE tasks SET status='ready' WHERE id=? AND status=?", (task_id, status)).rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(conn, task_id, "promoted_manual", {"actor": actor, "reason": reason, "forced": force})
    return True, None


def transition_task_status_with_attention(
    conn: sqlite3.Connection, *, task_id: str,
    status: Literal["ready", "todo", "triage"], actor: str = "webui",
    expected_attention_id: Optional[int] = None,
    expected_attention_version: Optional[int] = None, token: Optional[str] = None,
    now: Optional[int] = None,
) -> AttentionStatusTransitionResult:
    """Only safe generic blocked exit; expiry maintenance is target-scoped."""
    if status not in {"ready", "todo", "triage"}:
        raise ValueError("status must be ready, todo, or triage")
    now = int(time.time()) if now is None else int(now)
    hook: Optional[tuple[Optional[Task], Optional[int], str]] = None
    with write_txn(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None:
            return AttentionStatusTransitionResult("not_found", task_id)
        if task["status"] != "blocked":
            return AttentionStatusTransitionResult("conflict", task_id)
        projection = conn.execute(
            "SELECT x.id, x.action_id, x.type, x.version AS projection_version, a.state, a.version, a.expires_at "
            "FROM task_attentions x LEFT JOIN task_pending_actions a ON a.id=x.action_id "
            "WHERE x.task_id=? ORDER BY x.id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if projection is not None and projection["action_id"] is not None and (
            projection["state"] not in ("pending", "approved") or projection["expires_at"] is None
            or int(projection["expires_at"]) <= now
        ):
            _materialize_expired_action(conn, task_id, int(projection["action_id"]), now)
            return AttentionStatusTransitionResult("gone", task_id, attention_id=int(projection["id"]))
        if projection is not None:
            # Non-exact typed blockers are resolved only by their opaque current
            # projection + observed version; generic unblock stays fail-closed.
            if projection["action_id"] is None:
                actual_version = int(projection["projection_version"])
                if expected_attention_id != int(projection["id"]) or expected_attention_version != actual_version:
                    return AttentionStatusTransitionResult("conflict", task_id, attention_id=int(projection["id"]), attention_version=actual_version)
                if conn.execute("DELETE FROM task_attentions WHERE id=? AND task_id=? AND version=? AND action_id IS NULL",
                                (int(projection["id"]), task_id, actual_version)).rowcount != 1:
                    return AttentionStatusTransitionResult("conflict", task_id, attention_id=int(projection["id"]), attention_version=actual_version)
            else:
                return AttentionStatusTransitionResult(
                    "conflict", task_id, attention_id=int(projection["id"]),
                    attention_version=(int(projection["version"]) if projection["version"] is not None else None),
                )
        if projection is None and (expected_attention_id is not None or expected_attention_version is not None):
            return AttentionStatusTransitionResult("conflict", task_id)
        target: Literal["ready", "todo", "triage"] = status
        if target == "ready" and conn.execute(
            "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
            "WHERE l.child_id=? AND p.status NOT IN ('done','archived') LIMIT 1", (task_id,)
        ).fetchone():
            target = "todo"
        try:
            _assert_human_gate_open(conn, task_id, token=token, action="change_status")
        except GateTokenError:
            return AttentionStatusTransitionResult("conflict", task_id)
        run_id = task["current_run_id"]
        if run_id is not None:
            conn.execute("UPDATE task_runs SET status='reclaimed', outcome='reclaimed', ended_at=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=? AND task_id=? AND ended_at IS NULL", (now, int(run_id), task_id))
        if conn.execute(
            "UPDATE tasks SET status=?, current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, consecutive_failures=0, last_failure_error=NULL WHERE id=? AND status='blocked'",
            (target, task_id),
        ).rowcount != 1:
            return AttentionStatusTransitionResult("conflict", task_id)
        _append_event(conn, task_id, "unblocked" if target in ("ready", "todo") else "status", {"status": target, "actor": str(actor)[:120]})
        transitioned = get_task(conn, task_id)
        hook = (transitioned, run_id, "unblock" if target in ("ready", "todo") else "status_change")
        result = AttentionStatusTransitionResult("transitioned", task_id, target)
    if hook is not None:
        transitioned, run_id, reason = hook
        _fire_kanban_lifecycle_hook("kanban_task_unblocked", task_id, board=get_current_board(), assignee=transitioned.assignee if transitioned else None, run_id=run_id, reason=reason)
    return result


def unblock_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Transition ``blocked``/``scheduled`` -> ready or todo.

    ``actor``/``reason`` are recorded in the ``unblocked`` event payload so
    an unblock is attributable after the fact — the audit trail previously
    showed only an empty event, making it impossible to distinguish a human
    approval from an autonomous agent lifting its own gate
    (Audit 2026-07-10: 22 nicht attribuierbare Unblocks an einem Tag).

    Human-Gate v1: if the card has ``human_gate=1``, ``token`` must match
    the currently-issued one-time token (delivered out-of-band via ntfy —
    see ``issue_and_notify_gate_token``) or this raises
    :class:`GateTokenError` and the card is left untouched. The check is
    delegated to the shared :func:`_assert_human_gate_open` guard — the
    2026-07-11 repair review found ``complete_task``/``reclaim_task``/
    ``promote_task`` reaching ``ready``/``done``/``archived`` from
    ``blocked`` WITHOUT this check because it originally lived only here,
    inline. Every one of those functions (plus ``schedule_task``,
    ``archive_task``, ``recompute_ready``, and the dashboard's
    ``_set_status_direct``) now calls the same guard, so no exit out of
    ``blocked`` — CLI, tool, dashboard, or automatic — can bypass it.
    A matching token is single-use: consumed atomically inside the guard,
    in the same transaction as the status transition below.

    Defensively closes any stale ``current_run_id`` pointer before flipping
    status. In the common path (``block_task`` closed the run already) this
    is a no-op. If a future or external write left the pointer dangling,
    the leaked run is closed as ``reclaimed`` inside the same txn so the
    runs invariant (``current_run_id IS NULL`` ⇔ run row in terminal
    state) holds for the rest of this function's lifetime.
    """
    now = int(time.time())
    with write_txn(conn):
        # Read all independent business preconditions before consuming a
        # single-use human-gate token.  A refused unblock must be retryable by
        # the same authorized operator.
        # Legacy unblock owns only ordinary blocked cards.  Any live action or
        # any current projection belongs to its typed lifecycle seam; missing or
        # tampered projections are not an escape hatch.
        if conn.execute(
            "SELECT 1 FROM task_pending_actions WHERE task_id=? AND state IN ('pending','approved') LIMIT 1",
            (task_id,),
        ).fetchone() is not None or conn.execute(
            "SELECT 1 FROM task_attentions WHERE task_id=? LIMIT 1", (task_id,)
        ).fetchone() is not None:
            return False
        stale = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ? AND status IN ('blocked', 'scheduled')",
            (task_id,),
        ).fetchone()
        if stale and stale["current_run_id"]:
            conn.execute(
                """
                UPDATE task_runs
                   SET status = 'reclaimed', outcome = 'reclaimed',
                       summary = COALESCE(summary, 'invariant recovery on unblock'),
                       ended_at = ?,
                       claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
                 WHERE id = ? AND ended_at IS NULL
                """,
                (now, int(stale["current_run_id"])),
            )
        # Re-gate on parent completion before flipping 'blocked' back to
        # 'ready'. Unconditionally setting status='ready' here bypasses the
        # parent-completion invariant (the dispatcher trusts that column);
        # if parents are still in progress the task must wait in 'todo'
        # until recompute_ready picks it up. RCA: Bug 2 at
        # kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md.
        undone_parents = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? AND p.status != 'done' LIMIT 1",
            (task_id,),
        ).fetchone()
        new_status = "todo" if undone_parents else "ready"
        # Token validation/consumption is the final authorization operation in
        # this transaction, immediately before the status CAS below.
        gated = _assert_human_gate_open(conn, task_id, token=token, action="unblock")
        # NOTE: deliberately does NOT touch ``block_recurrences`` or
        # ``block_kind``. Resetting the recurrence counter on unblock is exactly
        # the amnesia that let a cron unblock → worker re-block loop run
        # unbounded (Dale's report). The counter survives the unblock so that a
        # subsequent same-cause ``block_task`` can detect the loop and route to
        # triage at ``BLOCK_RECURRENCE_LIMIT``. It is reset to 0 only on a
        # successful completion (see ``complete_task``). ``consecutive_failures``
        # (the *dispatcher* spawn/crash/timeout counter — a different signal) is
        # still reset here, which is correct: a deliberate unblock is a fresh
        # start for the dispatcher's retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')",
            (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        payload: dict = {}
        if new_status != "ready":
            payload["status"] = new_status
        if actor:
            payload["actor"] = str(actor)[:120]
        if reason:
            payload["reason"] = str(reason)[:400]
        if gated:
            payload["human_gate"] = True
        _append_event(
            conn, task_id, "unblocked",
            payload or None,
        )
        return True


def specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """Flesh out a triage task and promote it to ``todo``.

    Atomically updates ``title`` / ``body`` / ``assignee`` (when provided)
    and transitions ``status: triage -> todo`` in a single write txn. Returns
    False when the task is missing or not in the ``triage`` column — callers
    should surface that as "nothing to specify" rather than an error.

    ``todo`` (not ``ready``) is the correct landing column: ``recompute_ready``
    promotes parent-free / parent-done todos to ``ready`` on the next
    dispatcher tick, which keeps the normal parent-gating behaviour intact
    for specified tasks that happen to have open parents.

    ``author`` is recorded on an audit comment only when at least one of
    ``title`` / ``body`` / ``assignee`` actually changed — avoids noisy
    comment spam for status-only promotions.
    """
    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = _canonical_assignee(assignee)
    with write_txn(conn):
        if get_pending_action(conn, task_id) is not None:
            # Stale status projection must not erase exact-action attention.
            return False
        existing = conn.execute(
            "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        sets: list[str] = ["status = 'todo'"]
        params: list[Any] = []
        changed_fields: list[str] = []
        if title is not None and title.strip() != (existing["title"] or ""):
            sets.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")
        if body is not None and (body or "") != (existing["body"] or ""):
            sets.append("body = ?")
            params.append(body)
            changed_fields.append("body")
        if assignee is not None and assignee != (existing["assignee"] or None):
            sets.append("assignee = ?")
            params.append(assignee)
            changed_fields.append("assignee")
        params.append(task_id)
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} "
            f"WHERE id = ? AND status = 'triage'",
            tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            # Inline INSERT (rather than ``add_comment``) because we're
            # already inside this function's write_txn — nested BEGIN
            # IMMEDIATE would raise OperationalError. We also skip the
            # 'commented' event that ``add_comment`` emits, since the
            # 'specified' event below already records the change.
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    author.strip(),
                    "Specified — updated "
                    + ", ".join(changed_fields)
                    + " and promoted to todo.",
                    int(time.time()),
                ),
            )
        _append_event(
            conn,
            task_id,
            "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
    # Outside the write_txn above, so we don't nest BEGIN IMMEDIATE — the
    # ready-promotion pass opens its own IMMEDIATE txn. This runs the same
    # logic the dispatcher would on its next tick, so a specified task
    # with no open parents flips straight to 'ready' here instead of
    # idling in 'todo' until the next sweep.
    recompute_ready(conn)
    return True


def decompose_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    root_assignee: Optional[str],
    children: list[dict],
    author: Optional[str] = None,
    auto_promote: bool = True,
    max_fanout: Optional[int] = None,
    max_tasks_per_root: Optional[int] = None,
) -> Optional[list[str]]:
    """Fan a triage task out into child tasks and promote the root to ``todo``.

    The root task stays alive and becomes the parent of every child —
    when all children reach ``done``, the root promotes to ``ready`` and
    its assignee (typically the orchestrator profile) wakes back up to
    judge completion or spawn more work.

    ``children`` is a list of dicts, each shaped like::

        {
            "title": "...",
            "body": "...",                     # optional
            "assignee": "profile-name",        # optional, None -> default fallback
            "parents": [0, 2],                 # indices into this same children list
        }

    Returns the list of created child task ids (in input order) on
    success. Returns ``None`` when:
      - The root task does not exist
      - The root task is not in ``triage``
      - A cycle would result (caller built a bad graph)

    Validation of titles/assignees happens inside the same write_txn as
    the inserts so a malformed entry aborts the whole decomposition
    cleanly (no orphan children).
    """
    if not children:
        return None
    # Guardrail (2026-07-02, landscape-research #3): bound the fan-out WIDTH so a
    # single decompose can never spawn an unbounded wave of sibling tasks. This is
    # the primary Kanban-graph runaway guard; it always applies. ``None`` disables.
    if max_fanout and len(children) > max_fanout:
        raise ValueError(
            f"decompose fanout {len(children)} exceeds kanban.max_fanout={max_fanout}"
        )
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)

    # Pre-validate the children list shape outside the txn. Cheap checks
    # that don't need DB access. Bad input aborts before we touch the DB.
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(
                    f"child[{idx}].parents[{p}] is not a valid index into children"
                )
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    # Detect cycles in the sibling parent graph (Kahn's topological sort).
    # link_tasks() calls _would_cycle() for every new edge; here we check
    # the entire sibling graph before touching the DB.  A cycle silently
    # deadlocks every involved child in 'todo' because recompute_ready()
    # can never promote them.
    _in_deg = [0] * len(children)
    _adj: list[list[int]] = [[] for _ in range(len(children))]
    for _i, _c in enumerate(children):
        for _p in (_c.get("parents") or []):
            _adj[_p].append(_i)
            _in_deg[_i] += 1
    _queue = [_i for _i in range(len(children)) if _in_deg[_i] == 0]
    _seen = 0
    while _queue:
        _node = _queue.pop()
        _seen += 1
        for _nb in _adj[_node]:
            _in_deg[_nb] -= 1
            if _in_deg[_nb] == 0:
                _queue.append(_nb)
    if _seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")

    # We do the full decomposition in a SINGLE write_txn so it's
    # atomic: either every child is created AND the root flips to
    # ``todo``, or nothing changes. We deliberately do NOT call any
    # kb helper that opens its own write_txn (create_task, link_tasks,
    # add_comment) from inside this block — see architecture.md
    # write_txn pitfalls. Instead we inline the INSERTs and
    # _append_event calls.
    now = int(time.time())
    child_ids: list[str] = []
    with write_txn(conn):
        if get_pending_action(conn, task_id, now=now) is not None:
            return None
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if root_row is None:
            return None
        if root_row["status"] != "triage":
            return None
        # Guardrail (2026-07-02, landscape-research #3): bound total graph SIZE.
        # task_links are dependency edges and the root is linked as a *child* of
        # every leaf (it waits for the whole graph), so a plain "descendants of
        # root" walk is misleading. Instead measure the connected component the
        # task belongs to (bidirectional walk) — robust regardless of edge
        # direction. A fresh standalone triage task has no links -> component of
        # 1, so this only bites tasks already embedded in a larger graph, as
        # defense-in-depth against accumulation. ``None`` disables.
        if max_tasks_per_root:
            comp_size = conn.execute(
                "WITH RECURSIVE comp(node) AS ("
                "  SELECT ? "
                "  UNION "
                "  SELECT l.child_id FROM comp c JOIN task_links l ON l.parent_id = c.node "
                "  UNION "
                "  SELECT l.parent_id FROM comp c JOIN task_links l ON l.child_id = c.node"
                ") SELECT COUNT(*) FROM comp",
                (task_id,),
            ).fetchone()[0]
            if comp_size + len(children) > max_tasks_per_root:
                raise ValueError(
                    f"task graph size {comp_size + len(children)} exceeds "
                    f"kanban.max_tasks_per_root={max_tasks_per_root}"
                )
        tenant = root_row["tenant"]
        # Children inherit the root's workspace by default so a fan-out
        # of a code-gen task lands in the parent's project dir/worktree
        # rather than throwaway scratch tmp dirs. A child dict can still
        # override with its own 'workspace_kind' / 'workspace_path'.
        root_ws_kind = root_row["workspace_kind"] or "scratch"
        root_ws_path = root_row["workspace_path"]

        # Create children. Status is 'todo' regardless of parents — we
        # link them under the root AFTER creation so the dispatcher
        # sees a coherent state, and recompute_ready() at the end
        # promotes parent-free children to 'ready'.
        for idx, child in enumerate(children):
            new_id = _new_task_id()
            title = child["title"].strip()
            body = child.get("body")
            assignee = _canonical_assignee(child.get("assignee"))
            # Per-child override wins; otherwise inherit the root's
            # workspace. A child that sets workspace_kind without a path
            # falls back to the root path only when kinds match (so a
            # child can't accidentally point a 'dir' at the root's
            # worktree path or vice versa).
            child_ws_kind = child.get("workspace_kind") or root_ws_kind
            if child.get("workspace_path"):
                child_ws_path = child.get("workspace_path")
            elif child_ws_kind == root_ws_kind:
                child_ws_path = root_ws_path
            else:
                child_ws_path = None
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, created_at, created_by) "
                "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?)",
                (
                    new_id,
                    title,
                    body if isinstance(body, str) else None,
                    assignee,
                    child_ws_kind,
                    child_ws_path,
                    tenant,
                    now,
                    (author or "decomposer"),
                ),
            )
            _append_event(
                conn, new_id, "created",
                {"by": author or "decomposer", "from_decompose_of": task_id},
            )
            child_ids.append(new_id)

        # Link children to their sibling parents (within the decomposed graph).
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id = child_ids[p_idx]
                child_id = child_ids[idx]
                conn.execute(
                    "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                    "VALUES (?, ?)",
                    (parent_id, child_id),
                )
                _append_event(
                    conn, child_id, "linked",
                    {"parent": parent_id, "child": child_id},
                )

        # Link the ROOT task as a child of every leaf child — i.e. the
        # root waits for the whole graph. Simpler than computing leaves:
        # link root under every child. Cycle-free because the root is
        # only ever a child here, never a parent of children.
        for cid in child_ids:
            conn.execute(
                "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                "VALUES (?, ?)",
                (cid, task_id),
            )

        # Flip the root: triage -> todo, set assignee to the orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

        # Audit comment + event on the root so the timeline shows the fan-out.
        if author and author.strip():
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    author.strip(),
                    "Decomposed into "
                    + ", ".join(child_ids)
                    + ". Root will wake when all children complete.",
                    now,
                ),
            )
        _append_event(
            conn, task_id, "decomposed",
            {
                "child_ids": child_ids,
                "root_assignee": root_assignee,
            },
        )

    # Outside the write_txn: promote parent-free children to 'ready'
    # so the dispatcher picks them up on its next tick. Same pattern
    # specify_triage_task uses.  When auto_promote is False children
    # stay in 'todo' until the user manually promotes them — useful
    # for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Archive a task from any non-archived status.

    Human-Gate v1, refuse-only (2026-07-11 repair review self-audit): a
    ``blocked``/``scheduled``+``human_gate=1`` card always refuses —
    archiving disposes of a card without ever requiring the human
    decision the gate exists to force. No token parameter; run
    ``kanban gate <id> off`` first if archiving a gated card is
    genuinely needed.
    """
    with write_txn(conn):
        _assert_human_gate_open(conn, task_id, token=None, action="archive")
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status != 'archived'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        now = int(time.time())
        _cancel_active_actions_for_terminal_task(
            conn, task_id, now=now, reason="task_archived",
        )
        # If archive happened while a run was still in flight (e.g. user
        # archived a running task from the dashboard), close that run with
        # outcome='reclaimed' so attempt history isn't orphaned.
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    # ``archived`` parents no longer block children, same as ``done``.
    # Promote newly-unblocked dependents immediately instead of waiting
    # for a later dispatcher tick.
    recompute_ready(conn)
    return True


def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Permanently remove an already-archived task and its related rows.

    Safety guard: only archived tasks can be deleted. Active / blocked / done
    tasks must be explicitly archived first so accidental data loss requires a
    second deliberate action.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row or row["status"] != "archived":
            return False
        conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? OR child_id = ?",
            (task_id, task_id),
        )
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount == 1


def delete_task(
    conn: sqlite3.Connection, task_id: str, *, operator_reason: Optional[str] = None
) -> bool:
    """Retention hard-delete for archived, non-gated tasks only.

    Because the schema does not use ``ON DELETE CASCADE`` foreign keys,
    we explicitly delete from child tables first, then the task row.
    This keeps the operation atomic (single ``write_txn``).

    Returns ``True`` if the task existed and was deleted, ``False``
    if the task was not found.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, human_gate FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return False
        if row["human_gate"]:
            raise GateTokenError("human-gated tasks cannot be hard-deleted")
        if row["status"] != "archived":
            raise ValueError("hard-delete requires an archived task; archive it first")
        if not str(operator_reason or "").strip():
            raise ValueError("hard-delete requires an operator retention reason")
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.execute("DELETE FROM task_links WHERE parent_id = ? OR child_id = ?", (task_id, task_id))
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
    recompute_ready(conn)
    return True


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    try:
        return Path(out).expanduser().resolve()
    except Exception:
        return Path(out).expanduser()


def _git_branch_exists(repo_root: Path, branch_name: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show-ref", "--verify", f"refs/heads/{branch_name}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _git_common_dir(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    return Path(out).expanduser().resolve(strict=False)


def _git_dir(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    return Path(out).expanduser().resolve(strict=False)


def _git_current_branch(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def _is_linked_worktree_checkout(path: Path) -> bool:
    git_dir = _git_dir(path)
    common_dir = _git_common_dir(path)
    if git_dir is None or common_dir is None:
        return False
    return git_dir != common_dir


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    current = _nearest_existing_path(path).resolve(strict=False)
    while True:
        repo_root = _git_toplevel(current)
        if repo_root is not None:
            return repo_root
        if current == current.parent:
            return None
        current = current.parent


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists() and repo_common is not None:
        target_common = _git_common_dir(target)
        if target_common == repo_common:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    if _git_branch_exists(repo_root, branch_name):
        cmd = ["git", "-C", str(repo_root), "worktree", "add", str(target), branch_name]
    else:
        cmd = [
            "git", "-C", str(repo_root), "worktree", "add", "-b", branch_name,
            str(target), "HEAD",
        ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
        )


def _resolve_worktree_workspace(
    task: Task, *, board: Optional[str] = None
) -> tuple[Path, str]:
    """Resolve + materialize a linked git worktree for ``task``.

    When ``task.workspace_path`` is unset, the anchor is the board's
    ``default_workdir`` (a persistent project checkout). This keeps every
    worktree task under a meaningful, board-owned repo — ``<repo>/.worktrees/
    <task-id>`` — instead of silently landing under the dispatcher's current
    working directory (which is whatever directory the gateway happened to be
    launched from, e.g. the Hermes checkout). If no anchor is configured
    anywhere, we fail loudly rather than guess.
    """
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    if not task.workspace_path:
        # Anchor on the board's configured default_workdir, not Path.cwd().
        # The dispatcher's CWD is incidental (gateway launch dir) and using it
        # scatters worktrees under whatever repo the gateway started in.
        board_slug = board if board else get_current_board()
        board_default = (read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                f"and board {board_slug!r} has no default_workdir set. Set a board "
                "default workdir (a git repo) or create the task with "
                "--workspace worktree:<absolute-repo-path>."
            )
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"board {board_slug!r} default_workdir {board_default!r} is not "
                "absolute; use an absolute path to a git repo"
            )
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but board "
                f"{board_slug!r} default_workdir {board_default!r} is not inside a git repo"
            )
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name)
        return target, branch_name

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path "
            f"{task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)

    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        return requested_resolved, actual_branch or branch_name

    repo_root = _git_toplevel(requested)
    if repo_root is not None and requested_resolved == repo_root:
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name)
        return target, branch_name

    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo "
            "and does not point at a git repo root"
        )
    _ensure_git_worktree(repo_root, requested, branch_name)
    return requested, branch_name


def resolve_workspace(task: Task, *, board: Optional[str] = None) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    - ``scratch``: a fresh dir under ``<board-root>/workspaces/<id>/``,
      where ``<board-root>`` is the active board's root. The path is the
      same for the dispatcher and every profile worker, so handoff is
      path-stable.
    - ``dir:<path>``: the path stored in ``workspace_path``.  Created
      if missing.  MUST be absolute — relative paths are rejected to
      prevent confused-deputy traversal where ``../../../tmp/attacker``
      resolves against the dispatcher's CWD instead of a meaningful
      root.  Users who want a kanban-root-relative workspace should
      compute the absolute path themselves.
    - ``worktree``: a real linked git worktree. If ``workspace_path`` names
      a repo root, Hermes treats it as an anchor and materializes a linked
      worktree at ``<repo>/.worktrees/<task-id>``. If ``workspace_path`` names
      a concrete target path, Hermes creates/reuses that linked worktree. With
      no ``workspace_path``, Hermes anchors on the board's ``default_workdir``
      and materializes ``<repo>/.worktrees/<task-id>`` per task; if no
      ``default_workdir`` is configured it raises rather than guessing from the
      dispatcher's CWD. When ``branch_name`` is empty, Hermes uses
      ``wt/<task-id>``.

    Persist the resolved path back to the task row via ``set_workspace_path``
    so subsequent runs reuse the same directory.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "scratch":
        if task.workspace_path:
            # Legacy scratch tasks that were set to an explicit path get the
            # same absolute-path guard as dir: — consistent with the
            # threat model.
            p = Path(task.workspace_path).expanduser()
            if not p.is_absolute():
                raise ValueError(
                    f"task {task.id} has non-absolute workspace_path "
                    f"{task.workspace_path!r}; workspace paths must be absolute"
                )
        else:
            p = workspaces_root(board=board) / task.id
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "dir":
        if not task.workspace_path:
            raise ValueError(
                f"task {task.id} has workspace_kind=dir but no workspace_path"
            )
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; use an absolute path "
                f"(relative paths are ambiguous against the dispatcher's CWD)"
            )
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "worktree":
        p, _branch_name = _resolve_worktree_workspace(task, board=board)
        return p
    raise ValueError(f"unknown workspace_kind: {kind}")


def set_workspace_path(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(path), task_id),
        )


def set_branch_name(
    conn: sqlite3.Connection, task_id: str, branch_name: str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET branch_name = ? WHERE id = ?",
            (str(branch_name), task_id),
        )


# ---------------------------------------------------------------------------
def schedule_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Park a task in ``scheduled`` so it is waiting on time, not human input.

    ``scheduled`` tasks are intentionally not dispatchable; an external cron,
    human action, or automation can later call ``unblock_task`` to re-gate them
    to ``ready`` (or ``todo`` if parents are still incomplete).

    Human-Gate v1, refuse-only: parking a ``blocked``+``human_gate=1``
    card in ``scheduled`` is refused outright (2026-07-11 repair review
    self-audit) — even though ``unblock_task`` still gates the eventual
    return to ``ready``/``todo``, silently letting a gated card be
    "laundered" out of the ``blocked`` column into ``scheduled`` without
    a token would hide it from operator attention with no authorization
    at all. No token parameter here; use ``kanban gate <id> off`` first
    if scheduling a gated card is genuinely needed.
    """
    with write_txn(conn):
        _assert_human_gate_open(conn, task_id, token=None, action="schedule")
        params: list[Any] = [task_id]
        sql = """
            UPDATE tasks
               SET status       = 'scheduled',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked')
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="scheduled", status="scheduled",
            summary=reason,
        )
        if run_id is None and reason:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="scheduled",
                summary=reason,
            )
        _append_event(conn, task_id, "scheduled", {"reason": reason}, run_id=run_id)
        return True


# Dispatcher (one-shot pass)
# ---------------------------------------------------------------------------

# After this many consecutive non-success attempts on a task/profile, the
# dispatcher stops retrying and parks the task in ``blocked`` with a reason so
# a human can investigate. Prevents retry storms when a worker repeatedly times
# out, crashes, or cannot spawn.
DEFAULT_FAILURE_LIMIT = 2
# Legacy alias — callers / tests still reference the old name.
DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT

# Max bytes to keep in a single worker log file. The dispatcher truncates
# and rotates on spawn if the file is larger than this at spawn time.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1

# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and call kanban_block/kanban_complete before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|auth\w*|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Cooldown after a rate-limited (quota-wall) requeue before the dispatcher
# re-spawns the worker. Without this, a task released by the rate-limit path
# would be re-spawned on the very next tick and immediately bounce off the
# same quota wall, burning a worker slot every tick for hours. The cooldown
# spaces retries out so the board keeps cheaply probing whether quota is back
# without thrashing. Overridable via ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS``
# for operators who want a tighter/looser probe cadence.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 300  # 5 minutes

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours
# How long an ``active_pr`` respawn guard may continuously defer a ready task
# before the dispatcher escalates it to an explicit ``blocked``/``needs_input``
# card. The guard exists to prevent duplicate PRs, but for reviewer/gate tasks
# the PR link in their OWN verdict comment re-triggers it every tick — without
# escalation such a task livelocks invisibly in ``ready`` for the full
# 24h window while the board shows nothing wrong.
_RESPAWN_GUARD_PR_ESCALATE_SECONDS = 1800  # 30 minutes

# Pattern matching a GitHub PR URL in task comments.
_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass."""

    reclaimed: int = 0
    promoted: int = 0
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """List of ``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)
    """Ready task ids skipped because they have no assignee at all.
    Operator-actionable — usually a misfiled task waiting for routing."""
    auto_assigned_default: list[str] = field(default_factory=list)
    """Task ids that were unassigned in the DB and had
    ``kanban.default_assignee`` applied this tick before spawning (#27145).
    Surfaces the auto-assignment to telemetry / CLI / dashboard so the
    operator can see when the dispatcher is acting on the fallback rule
    rather than on explicit per-task assignments."""
    skipped_nonspawnable: list[str] = field(default_factory=list)
    """Ready task ids skipped because their assignee names a control-plane
    lane (a Claude Code terminal like ``orion-cc``) rather than a Hermes
    profile. Expected steady-state on multi-lane setups; NOT an
    operator-actionable failure. Tracked separately so health telemetry
    can distinguish "real stuck" (nothing spawned but spawnable work
    available) from "correctly idle" (nothing spawnable in the queue)."""
    skipped_per_profile_capped: list[tuple[str, str, int]] = field(default_factory=list)
    """Tasks deferred this tick because their assignee is already at
    ``kanban.max_in_progress_per_profile`` (#21582). Each entry is
    ``(task_id, assignee, current_running_count)``. NOT an
    operator-actionable failure — the task will be picked up on a
    subsequent tick when the assignee has capacity. Separate bucket so
    telemetry / dashboards can show "this profile is busy" vs
    "task is genuinely stuck"."""
    crashed: list[str] = field(default_factory=list)
    """Task ids reclaimed because their worker PID disappeared."""
    auto_blocked: list[str] = field(default_factory=list)
    """Task ids auto-blocked by the spawn-failure circuit breaker."""
    self_modify_gated: list[tuple[str, str]] = field(default_factory=list)
    """Ready tasks routed to blocked+human_gate by the self-modification
    governance gate, as ``(task_id, reason)`` pairs. The card would change
    how Hermes itself runs (config/profiles/systemd/agent code/plugins/skills)
    and must be human-approved before it can be dispatched. See
    ``classify_self_modification`` / ``kanban.self_modify_gate``."""
    timed_out: list[str] = field(default_factory=list)
    """Task ids whose workers exceeded ``max_runtime_seconds``."""
    stale: list[str] = field(default_factory=list)
    """Task ids reclaimed because no progress (heartbeat) was seen
    within ``dispatch_stale_timeout_seconds``."""
    resource_stalled: list[str] = field(default_factory=list)
    """Task ids blocked after sustained kernel/cgroup resource-stall evidence."""
    respawn_guarded: list[tuple[str, str]] = field(default_factory=list)
    """Tasks skipped by the respawn guard, as ``(task_id, reason)`` pairs.

    Reasons: ``"blocker_auth"`` (quota/auth error — also auto-blocked),
    ``"recent_success"`` (completed run within guard window),
    ``"active_pr"`` (GitHub PR URL in a recent comment)."""
    rate_limited: list[str] = field(default_factory=list)
    """Task ids whose workers bailed on a provider rate-limit / quota wall
    (EX_TEMPFAIL sentinel exit) and were released back to ``ready`` WITHOUT
    counting a failure. These never trip the circuit breaker — a long quota
    window just makes the task bounce cheaply until the window clears."""
    skipped_locked: bool = False
    """True when this tick was skipped because another process already held
    the board's dispatch lock (issue #35240). A losing dispatcher does no
    DB writes this tick — the lock holder is making progress on the same
    board. This is the steady-state signal that a single-writer guard is
    actively preventing two dispatchers from racing on ``kanban.db``."""


# Bounded registry of recently-reaped worker child exits, populated by the
# reap loop at the top of ``dispatch_once`` and consulted by
# ``detect_crashed_workers`` to classify a dead-pid task.
#
# Entry: ``pid -> (raw_wait_status, reaped_at_epoch)``. We keep raw status
# so both ``os.WIFEXITED`` / ``os.WEXITSTATUS`` and ``os.WIFSIGNALED`` can
# be consulted. Entries are trimmed by age (and total size cap as a
# belt-and-braces against unbounded growth on exotic platforms).
_RECENT_WORKER_EXIT_TTL_SECONDS = 600
_RECENT_WORKER_EXITS_MAX = 4096
_recent_worker_exits: "dict[int, tuple[int, float]]" = {}


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped child's exit status for later classification.

    Called from the reap loop in ``dispatch_once``. Safe to call many
    times; duplicate pids overwrite (pids can cycle, latest wins).
    """
    if not pid or pid <= 0:
        return
    now = time.time()
    _recent_worker_exits[int(pid)] = (int(raw_status), now)
    # Age-based trim: drop entries older than the TTL.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX // 2:
        cutoff = now - _RECENT_WORKER_EXIT_TTL_SECONDS
        for _pid in [p for p, (_s, t) in _recent_worker_exits.items() if t < cutoff]:
            _recent_worker_exits.pop(_pid, None)
    # Size cap as a final guard.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX:
        # Drop oldest half.
        ordered = sorted(_recent_worker_exits.items(), key=lambda kv: kv[1][1])
        for _pid, _ in ordered[: len(ordered) // 2]:
            _recent_worker_exits.pop(_pid, None)


def _classify_worker_exit(pid: int) -> "tuple[str, Optional[int]]":
    """Classify a recently-reaped worker by pid.

    Returns ``(kind, code)`` where ``kind`` is one of:

    * ``"clean_exit"`` — ``WIFEXITED`` with ``WEXITSTATUS == 0``. When the
      task is still ``running`` in the DB, this is a protocol violation
      (worker exited without calling ``kanban_complete`` / ``kanban_block``)
      and should be auto-blocked immediately — retrying will just loop.
    * ``"rate_limited"`` — ``WIFEXITED`` with status
      ``KANBAN_RATE_LIMIT_EXIT_CODE``. The worker bailed because the
      provider rate-limited / exhausted quota, NOT because the task failed.
      ``detect_crashed_workers`` releases the task back to ``ready`` without
      counting a failure, so a long quota window can't trip the breaker.
    * ``"nonzero_exit"`` — ``WIFEXITED`` with non-zero status. Real error.
    * ``"signaled"`` — ``WIFSIGNALED`` (OOM killer, SIGKILL, etc). Real crash.
    * ``"unknown"`` — pid was not in the reap registry (either reaped by
      something else, or died between reap tick and liveness check). Fall
      back to existing crashed-counter behavior.

    ``code`` is the exit status (for ``clean_exit`` / ``rate_limited`` /
    ``nonzero_exit``) or the signal number (for ``signaled``), or ``None``
    for ``unknown``.
    """
    entry = _recent_worker_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    try:
        if os.WIFEXITED(raw):
            code = os.WEXITSTATUS(raw)
            if code == 0:
                return ("clean_exit", 0)
            if code == KANBAN_RATE_LIMIT_EXIT_CODE:
                return ("rate_limited", code)
            return ("nonzero_exit", code)
        if os.WIFSIGNALED(raw):
            return ("signaled", os.WTERMSIG(raw))
    except Exception:
        pass
    return ("unknown", None)


def reap_worker_zombies() -> "list[int]":
    """Reap all zombie children of this process without blocking.

    Returns the list of reaped PIDs. Safe to call when there are no
    children (returns []). No-op on Windows.
    """
    reaped: "list[int]" = []
    if os.name != "nt":
        try:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                _record_worker_exit(pid, status)
                reaped.append(pid)
        except Exception:
            pass
    return reaped


def _pid_alive(pid: Optional[int]) -> bool:
    """Return True if ``pid`` is still running on this host.

    Cross-platform: uses ``OpenProcess`` + ``WaitForSingleObject`` on
    Windows (via ``gateway.status._pid_exists``) and ``os.kill(pid, 0)``
    on POSIX. Returns False for falsy PIDs or on any OS error.

    **DO NOT** use ``os.kill(pid, 0)`` directly on Windows — Python's
    Windows ``os.kill`` treats ``sig=0`` as ``CTRL_C_EVENT`` (bpo-14484)
    and will broadcast it to the target's console group, potentially
    killing unrelated processes.

    **Zombie handling:** the existence check succeeds against zombie
    processes (post-exit, pre-reap) because the process table entry
    still exists. A worker that exits without being reaped by its
    parent would stay "alive" to the dispatcher forever. Dispatcher
    workers are started via ``start_new_session=True`` + intentional
    Popen handle abandonment, so init reaps them quickly — but during
    the window between exit and reap, we'd otherwise see stale "alive"
    signals. On Linux we peek at ``/proc/<pid>/status`` and treat
    ``State: Z`` as dead. On macOS we ask ``ps`` for the BSD ``stat``
    field and treat values containing ``Z`` as dead.
    """
    if not pid or pid <= 0:
        return False
    from gateway.status import _pid_exists
    if not _pid_exists(int(pid)):
        return False
    # Still here → process exists. Check for zombie on platforms
    # where we have a cheap, deterministic process-state probe.
    if sys.platform == "linux":
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        # "State:\tZ (zombie)" → dead
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            # proc entry gone → already reaped; treat as dead.
            # PermissionError shouldn't happen for our own children but
            # be defensive.
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
                check=False,
            )
            if proc.returncode != 0:
                return False
            if "Z" in (proc.stdout or "").strip():
                return False
        except (OSError, subprocess.SubprocessError, TimeoutError):
            # If the secondary probe fails, keep the kill(0) answer.
            pass
    return True


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths."""
    import signal

    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info

    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    if not str(claim_lock).startswith(host_prefix):
        return info
    info["host_local"] = True

    kill = signal_fn if signal_fn is not None else (
        os.kill if hasattr(os, "kill") else None
    )
    if kill is None:
        return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process is already gone — that's a successful termination, not a
        # survival. Leaving terminated=False here would make the reclaim guard
        # misread a dead worker as still-alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    for _ in range(10):
        if not _pid_alive(pid):
            info["terminated"] = True
            return info
        time.sleep(0.5)

    if _pid_alive(pid):
        try:
            # signal.SIGKILL doesn't exist on Windows; fall back to SIGTERM
            # (which maps to TerminateProcess via the stdlib shim).
            _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            kill(int(pid), _sigkill)
            info["sigkill"] = True
        except (ProcessLookupError, OSError):
            return info

    info["terminated"] = not _pid_alive(pid)
    return info


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming in this state would release the claim and let the dispatcher
    spawn a second worker while the first is still running — the duplication
    loop. Only host-local workers we actually signalled count: a non-local
    claim lock or a no-op attempt (no ``os.kill`` available) must fall through
    to the normal release path, since we cannot manage that worker anyway.
    """
    return bool(
        termination.get("termination_attempted")
        and termination.get("host_local")
        and not termination.get("terminated")
    )


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records a ``reclaim_deferred``
    event so the hold is visible in ``hermes kanban tail``. The next dispatch
    tick retries the kill; this is self-correcting because not spawning a
    duplicate is what lets the throttled worker finally die.
    """
    grace = now + RECLAIM_DEFER_GRACE_SECONDS
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if cur.rowcount != 1:
            return
        run_id = _current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                (grace, run_id),
            )
        payload = {
            "reason": reason,
            "claim_lock": claim_lock,
            "claim_expires_now": grace,
        }
        payload.update(termination)
        _append_event(conn, task_id, "reclaim_deferred", payload, run_id=run_id)


def heartbeat_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    note: Optional[str] = None,
    expected_run_id: Optional[int] = None,
    semantic: bool = True,
) -> bool:
    """Persist liveness/activity separately from semantic progress.

    Explicit worker heartbeats are semantic by default. Runtime-generated
    activity heartbeats pass ``semantic=False`` so streaming/tool chatter
    cannot conceal a worker that has stopped making useful progress.
    """
    now = int(time.time())
    task_set = "last_heartbeat_at = ?, last_activity_at = ?"
    run_set = "last_heartbeat_at = ?, last_activity_at = ?"
    values: list[Any] = [now, now]
    if semantic:
        task_set += ", last_semantic_progress_at = ?"
        run_set += ", last_semantic_progress_at = ?"
        values.append(now)
    with write_txn(conn):
        if expected_run_id is None:
            cur = conn.execute(
                f"UPDATE tasks SET {task_set} WHERE id = ? AND status = 'running'",
                (*values, task_id),
            )
        else:
            cur = conn.execute(
                f"UPDATE tasks SET {task_set} "
                "WHERE id = ? AND status = 'running' AND current_run_id = ?",
                (*values, task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _current_run_id(conn, task_id)
        )
        if run_id is not None:
            conn.execute(
                f"UPDATE task_runs SET {run_set} WHERE id = ?",
                (*values, run_id),
            )
        _append_event(
            conn, task_id, "heartbeat",
            (
                {"note": note, "semantic": False}
                if not semantic
                else ({"note": note} if note else None)
            ),
            run_id=run_id,
        )
    return True


def detect_resource_stalls(
    conn: sqlite3.Connection,
    *,
    cgroup_path: Optional[str | Path] = None,
    d_state_seconds: int = 120,
    memory_high_ratio: float = 0.98,
    psi_some_avg10: float = 1.0,
    high_event_delta: int = 1,
    signal_fn=None,
) -> list[str]:
    """Fail closed when an identity-bound worker is resource-stalled.

    CROSS-TASK AGGREGATE CGROUP (see also ``kanban_resource_monitor
    .probe_cgroup``): spawned workers share the gateway's cgroup
    (``Popen(start_new_session=True)`` inherits it rather than getting its
    own), so the cgroup reading below reflects memory pressure for ALL
    workers plus the gateway process itself -- not just the one task being
    evaluated. A memory-hungry neighbour can push another, healthy task's
    cgroup reading past threshold. This is only safe to act on because the
    block decision requires a SUSTAINED ``D``-state (uninterruptible sleep)
    observed on THAT task's own worker PID via ``probe_process`` in
    addition to the aggregate cgroup signal -- cgroup pressure alone never
    blocks a task. Real per-task isolation needs per-worker systemd scopes
    (one cgroup per spawned worker); that is future work, not implemented
    here.

    IDENTITY: a run row with no recorded ``worker_start_ticks`` (spawn
    raced, or predates this column) cannot be identity-verified against its
    PID -- a reused PID could belong to an unrelated process. Such tasks
    are skipped and never auto-blocked; a sampled ``monitoring_error``
    event/log records the gap instead.
    """
    from hermes_cli import kanban_resource_monitor as monitor

    if sys.platform != "linux":
        return []
    resolved_cgroup = Path(cgroup_path) if cgroup_path else monitor.current_cgroup_path()
    if resolved_cgroup is None:
        return []
    now = int(time.time())
    blocked: list[str] = []
    rows = conn.execute(
        "SELECT t.id, t.claim_lock, t.worker_pid, t.current_run_id, r.worker_start_ticks, "
        "r.d_state_since, r.resource_sample FROM tasks t "
        "JOIN task_runs r ON r.id=t.current_run_id "
        "WHERE t.status='running' AND t.worker_pid IS NOT NULL"
    ).fetchall()
    # The cgroup counters are the same filesystem read for every row this
    # tick (one shared cgroup, see docstring above) -- read once rather than
    # once per row, then fold in each row's own previous-sample baseline.
    try:
        cgroup_snapshot = monitor.read_cgroup_snapshot(resolved_cgroup)
    except Exception as exc:
        cgroup_snapshot = {"supported": False, "errors": [f"cgroup:{type(exc).__name__}"]}
    for row in rows:
        try:
            previous = json.loads(row["resource_sample"] or "{}")
        except (TypeError, ValueError):
            previous = {}

        if row["worker_start_ticks"] is None:
            error_count = int(previous.get("monitoring_error_count", 0)) + 1
            sample = {"supported": False, "monitoring_error_count": error_count}
            if error_count & (error_count - 1) == 0:
                error_payload = {"count": error_count, "errors": ["worker_start_ticks:missing"]}
                _log.warning(
                    "kanban resource monitor: task %s has no worker_start_ticks; "
                    "skipping resource-stall check, PID identity unverifiable (sample %d)",
                    row["id"], error_count,
                )
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "monitoring_error", error_payload,
                        run_id=int(row["current_run_id"]),
                    )
            encoded = json.dumps(sample, sort_keys=True, separators=(",", ":"))[:8192]
            with write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET resource_sample=? WHERE id=?",
                    (encoded, int(row["current_run_id"])),
                )
            continue

        try:
            process = monitor.probe_process(
                int(row["worker_pid"]), expected_start_ticks=row["worker_start_ticks"]
            )
        except Exception as exc:
            process = {"supported": True, "errors": [f"probe:{type(exc).__name__}"]}
        sample = monitor.cgroup_deltas_since(cgroup_snapshot, previous)
        errors = list(process.get("errors") or []) + list(sample.get("errors") or [])
        error_count = int(previous.get("monitoring_error_count", 0))
        if errors:
            error_count += 1
            sample["monitoring_error_count"] = error_count
            if error_count & (error_count - 1) == 0:
                error_payload = {"count": error_count, "errors": errors}
                _log.warning(
                    "kanban resource monitor error for task %s (sample %d): %s",
                    row["id"], error_count, ", ".join(errors),
                )
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "monitoring_error", error_payload,
                        run_id=int(row["current_run_id"]),
                    )
        if not process.get("identity_matches", True):
            _log.warning("kanban resource monitor: task %s PID identity mismatch", row["id"])
            continue
        is_d = process.get("state") == "D"
        d_since = int(row["d_state_since"]) if row["d_state_since"] is not None else None
        if is_d and d_since is None:
            d_since = now
        elif not is_d:
            d_since = None
        high_ratio = sample.get("high_ratio")
        high_delta = int((sample.get("events_delta") or {}).get("high", 0))
        psi = float((sample.get("pressure") or {}).get("some_avg10", 0.0))
        pressure_rising = high_delta >= max(1, high_event_delta) or psi >= psi_some_avg10
        at_high = high_ratio is not None and float(high_ratio) >= memory_high_ratio
        sustained_d = is_d and d_since is not None and now - d_since >= max(1, d_state_seconds)
        resource_stalled = sustained_d and (at_high or pressure_rising)
        encoded = json.dumps(sample, sort_keys=True, separators=(",", ":"))[:8192]
        with write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET d_state_since=?, resource_sample=? WHERE id=?",
                (d_since, encoded, int(row["current_run_id"])),
            )
        if not resource_stalled:
            continue
        payload = {
            "kind": "resource_stalled", "process": process, "cgroup": sample,
            "d_state_seconds": now - int(d_since),
        }
        termination = _terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn
        )
        payload["termination"] = termination
        termination_pending = _worker_survived_termination(termination)
        termination_pending_since = now if termination_pending else None
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status='blocked', "
                "block_kind='capability', claim_expires=NULL "
                "WHERE id=? AND status='running' AND current_run_id=?",
                (row["id"], row["current_run_id"]),
            )
            if cur.rowcount != 1:
                continue
            conn.execute(
                "UPDATE task_runs SET termination_pending_since=? WHERE id=?",
                (termination_pending_since, int(row["current_run_id"])),
            )
            run_outcome = "termination_pending" if termination_pending else "resource_stalled"
            run_metadata = (
                {"resource_stall": payload, "termination": termination}
                if termination_pending else payload
            )
            run_id = _end_run(
                conn, row["id"], outcome=run_outcome, status="blocked",
                error=run_outcome, metadata=run_metadata,
            )
            _append_event(conn, row["id"], "resource_stalled", payload, run_id=run_id)
            _append_event(
                conn, row["id"], "blocked",
                {"reason": "resource_stalled", "kind": "capability"}, run_id=run_id,
            )
            if termination_pending:
                _append_event(
                    conn, row["id"], "termination_pending",
                    {"pid": int(row["worker_pid"]), "since": now, **termination},
                    run_id=run_id,
                )
            blocked.append(row["id"])
    return blocked


def enforce_max_runtime(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> list[str]:
    """Terminate workers whose per-task ``max_runtime_seconds`` has elapsed.

    Sends SIGTERM, waits a short grace window, then SIGKILL. Emits a
    ``timed_out`` event and drops the task back to ``ready`` so the next
    dispatcher tick re-spawns it — unless the spawn-failure circuit
    breaker has already given up, in which case the task stays blocked
    where ``_record_spawn_failure`` parked it.

    Runs host-local: only tasks claimed by this host are candidates
    (same reasoning as ``detect_crashed_workers``). ``signal_fn`` is a
    test hook; defaults to ``os.kill`` on POSIX.
    """
    import signal
    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.current_run_id, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL "
        "  AND COALESCE(r.started_at, t.started_at) IS NOT NULL "
        "  AND t.worker_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Runtime is per attempt, not lifetime-of-task. ``tasks.started_at``
        # intentionally records the first time a task ever started, so retries
        # must be measured from the active task_runs row when present.
        elapsed = now - int(row["active_started_at"])
        if elapsed < int(row["max_runtime_seconds"]):
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        # SIGTERM then SIGKILL. Keep it simple: 5 s grace. Workers that
        # want a cleaner shutdown can install their own SIGTERM handler
        # before the grace expires.
        killed = False
        kill = signal_fn if signal_fn is not None else (
            os.kill if hasattr(os, "kill") else None
        )
        if kill is not None:
            try:
                kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            # Short polling wait — no time.sleep on the write txn.
            for _ in range(10):
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
            if _pid_alive(pid):
                try:
                    # signal.SIGKILL doesn't exist on Windows.
                    _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    kill(pid, _sigkill)
                    killed = True
                except (ProcessLookupError, OSError):
                    pass

        # An approved exact action is a separate, explicit retry lifecycle.
        # Park it before the legacy ready/requeue mutation loses the origin run.
        if row["current_run_id"] is not None and _park_approved_action_on_technical_failure_if_current(
            conn, task_id=tid, expected_run_id=int(row["current_run_id"]),
            attention_type="transient", reason_code="runtime_timeout", now=now,
        ):
            timed_out.append(tid)
            continue

        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": int(row["max_runtime_seconds"]),
                    "sigkill": killed,
                }
                run_id = _end_run(
                    conn, tid,
                    outcome="timed_out", status="timed_out",
                    error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                    metadata=payload,
                )
                _append_event(
                    conn, tid, "timed_out", payload, run_id=run_id,
                )
                timed_out.append(tid)
        # Increment the unified failure counter. Outside the write_txn
        # above because ``_record_task_failure`` opens its own. If the
        # breaker trips, this flips the task ``ready → blocked`` and
        # emits a ``gave_up`` event on top of the ``timed_out`` we
        # already emitted.
        if cur.rowcount == 1:
            _record_task_failure(
                conn, tid,
                error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "sigkill": killed},
            )
    return timed_out


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks that show no liveness within the staleness
    window.

    A task is considered stale when BOTH of these hold:

    1. It has been running for longer than ``stale_timeout_seconds``
       (measured from the active run's ``started_at``, falling back to
       ``tasks.started_at`` on older runs).
    2. Its most recent LIVENESS signal -- ``max(last_heartbeat_at,
       last_activity_at, last_semantic_progress_at)`` -- is older than the
       same window.

    Liveness and progress are deliberately kept separate columns but this
    reclaim decision uses the freshest of all three: a worker that is still
    emitting automatic activity heartbeats (``semantic=False``, sent by the
    runtime on essentially every tool call / stream chunk -- see
    ``tools/kanban_tools.py::heartbeat_current_worker_from_env``) is alive
    and must NOT be reclaimed, even if it never once made explicit semantic
    progress. Reclaiming on ``last_semantic_progress_at`` alone would kill
    a healthy multi-hour worker the instant ``dispatch_stale_timeout_seconds``
    elapses, because almost nothing calls the explicit ``kanban_heartbeat``
    tool. ``last_semantic_progress_at`` remains the signal used by
    ``detect_resource_stalls``' telemetry and by goal-loop no-progress
    detection -- those care about "did real work happen", this cares about
    "is anyone home".

    On reclaim the task is reset to ``ready``, the run is closed with
    ``outcome='stale'``, and the host-local worker (if still running) is
    terminated.

    Only considers ``status='running'`` tasks. Blocked tasks are never
    candidates.  Returns the list of reclaimed task IDs.

    ``stale_timeout_seconds=0`` disables the check entirely (returns ``[]``
    immediately).  ``signal_fn`` is a test hook; defaults to ``os.kill``
    on POSIX.
    """
    if stale_timeout_seconds <= 0:
        return []


    now = int(time.time())
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.last_heartbeat_at, t.last_activity_at, "
        "       t.last_semantic_progress_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()

    for row in rows:
        # Skip if no started_at (shouldn't happen for running, but be safe).
        if row["active_started_at"] is None:
            continue

        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue  # not old enough to check

        last_hb = row["last_heartbeat_at"]
        last_activity = row["last_activity_at"]
        last_progress = row["last_semantic_progress_at"]
        liveness_candidates = [
            v for v in (last_hb, last_activity, last_progress) if v is not None
        ]
        last_liveness = max(liveness_candidates) if liveness_candidates else None
        liveness_age = (
            now - int(last_liveness) if last_liveness is not None else elapsed
        )
        if liveness_age < stale_timeout_seconds:
            continue  # recent heartbeat/activity/progress → still alive

        progress_age = (
            now - int(last_progress) if last_progress is not None else elapsed
        )
        hb_age = (now - int(last_hb)) if last_hb is not None else None

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        # Terminate the worker if it's still host-local.
        termination = _terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn,
        )

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": (
                    int(last_hb) if last_hb is not None else None
                ),
                "heartbeat_age_seconds": (
                    int(hb_age) if hb_age is not None else None
                ),
                "last_activity_at": (
                    int(last_activity) if last_activity is not None else None
                ),
                "last_semantic_progress_at": (
                    int(last_progress) if last_progress is not None else None
                ),
                "semantic_progress_age_seconds": int(progress_age),
                "liveness_age_seconds": int(liveness_age),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
            }
            payload.update(termination)

            run_id = _end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _append_event(
                conn, tid, "stale", payload, run_id=run_id,
            )
            reclaimed.append(tid)

        # Intentionally NOT calling _record_task_failure here. Stale reclaim
        # is dispatcher-side detection of an absent heartbeat; the task is
        # going straight back to ``ready`` for re-dispatch. Counting it as
        # a worker failure would let two legitimately-long-running tasks
        # (>4h without explicit heartbeat) trip the circuit breaker and
        # auto-block, even though no worker actually failed. The 'stale'
        # event already lives in task_events for auditability; that's the
        # right surface for "this happened" without conflating with the
        # spawn_failed / timed_out / crashed counters.

    return reclaimed


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message for grouping identical failures.

    Strips host-specific details (PIDs, timestamps) so that errors
    with the same root cause produce the same fingerprint.
    """
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


def detect_crashed_workers(conn: sqlite3.Connection) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Appends a ``crashed`` event and drops the task back to ``ready``.
    Different from ``release_stale_claims``: this checks liveness
    immediately rather than waiting for the claim TTL.

    Only considers tasks claimed by *this host* — PIDs from other hosts
    are meaningless here. The host-local check is enough because
    ``_default_spawn`` always runs the worker on the same host as the
    dispatcher (the whole design is single-host).

    When the reap registry shows the worker exited cleanly (rc=0) but
    the task was still ``running`` in the DB, treat it as a protocol
    violation (worker answered conversationally without calling
    ``kanban_complete`` / ``kanban_block``) and trip the circuit breaker
    on the first occurrence — retrying a worker whose CLI keeps
    returning 0 without a terminal transition just loops forever.

    When the reap registry shows the worker exited with the rate-limit
    sentinel (``KANBAN_RATE_LIMIT_EXIT_CODE``), the worker bailed on a
    provider quota wall, NOT a task failure. Such tasks are released back
    to ``ready`` WITHOUT counting a failure (so a long quota window can't
    trip the breaker) and stamped with a quota-blocker error so
    ``check_respawn_guard`` defers their respawn until the window clears.
    The ids are returned via the ``_last_rate_limited`` function attribute
    (the public return stays the crashed-only ``list[str]``).
    """
    crashed: list[str] = []
    rate_limited: list[str] = []
    # Per-crash details collected inside the main txn, used after it
    # closes to run ``_record_task_failure`` (which needs its own
    # write_txn so can't nest). ``protocol_violation`` flags the
    # clean-exit-but-still-running case so we can trip the breaker
    # immediately instead of incrementing by 1.
    crash_details: list[tuple[str, int, str, bool, str]] = []
    # (task_id, pid, claimer, protocol_violation, error_text)
    with write_txn(conn):
        rows = conn.execute(
            "SELECT t.id, t.worker_pid, t.claim_lock, t.started_at, "
            "       r.started_at AS active_started_at "
            "FROM tasks t LEFT JOIN task_runs r ON r.id = t.current_run_id "
            "WHERE t.status = 'running' AND t.worker_pid IS NOT NULL"
        ).fetchall()
        host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
        for row in rows:
            # Only check liveness for claims owned by this host.
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Skip liveness check inside the launch-window grace period
            # so a freshly-spawned worker isn't reclaimed before its PID
            # is visible on /proc. H2 (Audit 2026-07-11): measure from the
            # ACTIVE run, not ``tasks.started_at`` — that column freezes at
            # the first-ever claim, so from the first respawn on the grace
            # was a no-op (exactly the respawn case it exists for).
            # ``enforce_max_runtime`` does the same for the same reason.
            started_at = (
                row["active_started_at"]
                if row["active_started_at"] is not None
                else (row["started_at"] if "started_at" in row.keys() else None)
            )
            if started_at is not None:
                grace = _resolve_crash_grace_seconds()
                if time.time() - started_at < grace:
                    continue
            if _pid_alive(row["worker_pid"]):
                continue

            pid = int(row["worker_pid"])
            kind, code = _classify_worker_exit(pid)
            rate_limited_exit = False
            if kind == "clean_exit":
                # Worker subprocess returned 0 but its task is still
                # ``running`` in the DB — it exited without calling
                # ``kanban_complete`` / ``kanban_block``. Normally a protocol
                # violation (retrying won't help) -> trip the breaker at once.
                # BUT if this run was a RESUME (#2), a clean-exit can be the
                # resumed context wrongly concluding "done" — a poisoned-context
                # symptom, not a deterministic task defect. Downgrade to a normal
                # +1 failure so the resume budget burns down and the NEXT claim
                # starts fresh, instead of an immediate block.
                _run_meta_row = conn.execute(
                    "SELECT r.metadata FROM task_runs r "
                    "JOIN tasks t ON t.current_run_id = r.id WHERE t.id = ?",
                    (row["id"],),
                ).fetchone()
                _was_resume = bool(
                    _run_meta_row and _run_meta_row["metadata"]
                    and '"resumed": true' in _run_meta_row["metadata"]
                )
                protocol_violation = not _was_resume
                if _was_resume:
                    error_text = (
                        "resumed worker exited cleanly (rc=0) without calling "
                        "kanban_complete/kanban_block — counted as one failure "
                        "(resume budget); next claim starts fresh"
                    )
                    event_kind = "protocol_violation_after_resume"
                else:
                    error_text = (
                        "worker exited cleanly (rc=0) without calling "
                        "kanban_complete or kanban_block — protocol violation"
                    )
                    event_kind = "protocol_violation"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
                if _was_resume:
                    # Preserve the resume marker in the run history — _end_run
                    # replaces task_runs.metadata with this payload, which would
                    # otherwise erase {"resumed": true} (DEVCHAIN-AUDIT C4).
                    event_payload["resumed"] = True
            elif kind == "rate_limited":
                # Worker bailed because the provider rate-limited / exhausted
                # quota (EX_TEMPFAIL sentinel). This is NOT a task failure —
                # the task is fine, the account just hit a wall. Release it
                # back to ``ready`` so the respawn guard defers it until the
                # quota window clears, and crucially do NOT count a failure
                # (skip ``_record_task_failure``) so a long quota window can't
                # trip the circuit breaker and permanently block the card.
                protocol_violation = False
                rate_limited_exit = True
                error_text = (
                    f"pid {pid} exited rate-limited (quota wall) — "
                    f"requeued without counting a failure"
                )
                event_kind = "rate_limited"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
            else:
                protocol_violation = False
                if kind == "nonzero_exit":
                    error_text = f"pid {pid} exited with code {code}"
                elif kind == "signaled":
                    error_text = f"pid {pid} killed by signal {code}"
                else:
                    error_text = f"pid {pid} not alive"
                event_kind = "crashed"
                event_payload = {"pid": pid, "claimer": row["claim_lock"]}
                if code is not None and kind != "unknown":
                    event_payload["exit_kind"] = kind
                    event_payload["exit_code"] = code

            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (row["id"], pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                # Rate-limited requeues are a clean release, not a crash —
                # record the run outcome as ``rate_limited`` so the board
                # history doesn't show a phantom crash for a quota wall.
                _run_outcome = "rate_limited" if rate_limited_exit else "crashed"
                run_id = _end_run(
                    conn, row["id"],
                    outcome=_run_outcome, status=_run_outcome,
                    error=error_text,
                    metadata=dict(event_payload),
                )
                _append_event(
                    conn, row["id"], event_kind,
                    event_payload,
                    run_id=run_id,
                )
                if rate_limited_exit:
                    # Stamp the failure-error column so ``check_respawn_guard``
                    # recognizes this as a quota blocker and defers the
                    # respawn until the window clears — WITHOUT touching
                    # ``consecutive_failures`` (that's the whole point: no
                    # breaker trip on a throttle).
                    conn.execute(
                        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                        (error_text[:500], row["id"]),
                    )
                    rate_limited.append(row["id"])
                else:
                    crashed.append(row["id"])
                    crash_details.append(
                        (row["id"], pid, row["claim_lock"],
                         protocol_violation, error_text)
                    )
    # Outside the main txn: increment the unified failure counter for
    # each crashed task. If the breaker trips, the task transitions
    # ready → blocked with a ``gave_up`` event on top of the ``crashed``
    # event we already emitted.
    #
    # Protocol-violation crashes force an immediate trip (failure_limit=1)
    # because clean-exit-without-transition is deterministic: the next
    # respawn will do exactly the same thing. Better to surface to a
    # human with a clear reason than to loop ``DEFAULT_FAILURE_LIMIT``
    # times first.
    auto_blocked: list[str] = []
    if crash_details:
        # Fingerprint errors to detect systemic failures.
        _fp_counts: dict[str, int] = {}
        for _, _, _, _, err_text in crash_details:
            fp = _error_fingerprint(err_text)
            _fp_counts[fp] = _fp_counts.get(fp, 0) + 1
        for tid, pid, claimer, protocol_violation, error_text in crash_details:
            fp = _error_fingerprint(error_text)
            is_systemic = (
                not protocol_violation
                and _fp_counts.get(fp, 0) >= 3
            )
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if (protocol_violation or is_systemic) else None,
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "claimer": claimer},
            )
            if tripped:
                auto_blocked.append(tid)
    # Stash auto-blocked ids on the function for the dispatch loop to pick up.
    # Keeps the public return type (``list[str]``) stable for direct callers
    # and tests that destructure the result; ``dispatch_once`` reads this
    # side-channel attribute to populate ``DispatchResult.auto_blocked``.
    detect_crashed_workers._last_auto_blocked = auto_blocked  # type: ignore[attr-defined]
    # Same side-channel for rate-limited requeues — these did NOT count a
    # failure and are NOT crashes, so they stay out of the ``crashed`` return.
    detect_crashed_workers._last_rate_limited = rate_limited  # type: ignore[attr-defined]
    return crashed


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
    release_claim: bool = False,
    end_run: bool = False,
    event_payload_extra: Optional[dict] = None,
) -> bool:
    """Record a non-success outcome (spawn_failed / crashed / timed_out)
    and maybe trip the circuit breaker.

    Unified replacement for the old spawn-only ``_record_spawn_failure``.
    Every path that ends a task with a non-success outcome funnels
    through here so the ``consecutive_failures`` counter and the
    auto-block threshold stay consistent.

    Returns True when the task was auto-blocked (counter reached
    ``failure_limit``), False when it was just updated in place.

    Modes:

    * ``release_claim=True, end_run=True`` — spawn-failure path.
      Caller has a running task with an open run; this transitions
      it back to ``ready`` (or ``blocked`` when the breaker trips),
      releases the claim, and closes the run with ``outcome=<outcome>``.

    * ``release_claim=False, end_run=False`` — timeout/crash path.
      Caller has ALREADY flipped the task to ``ready`` and closed the
      run with the appropriate outcome. This just increments the
      counter; if the breaker trips, the task is re-transitioned
      ``ready → blocked`` and a ``gave_up`` event is emitted.

    ``event_payload_extra`` merges into the ``gave_up`` event payload
    when the breaker trips, so callers can include outcome-specific
    context (e.g. pid on crash, elapsed on timeout).

    Resolution order for the effective threshold:
      1. per-task ``max_retries`` if set (nothing else overrides)
      2. caller-supplied ``failure_limit`` (gateway passes the config
         value from ``kanban.failure_limit``; tests pass fixed values)
      3. ``DEFAULT_FAILURE_LIMIT``
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    blocked = False
    with write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        failures = int(row["consecutive_failures"]) + 1
        cur_status = row["status"]

        # Per-task override wins over both caller-supplied and default
        # thresholds. None (the common case) falls through.
        task_override = (
            row["max_retries"] if "max_retries" in row.keys() else None
        )
        if task_override is not None:
            effective_limit = int(task_override)
            limit_source = "task"
        else:
            effective_limit = int(failure_limit)
            limit_source = "dispatcher"

        if failures >= effective_limit:
            # Trip the breaker.
            if release_claim:
                # Spawn path: still running, also clear claim state.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('running', 'ready')",
                    (failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: task is already at ``ready``
                # with claim cleared; just flip to blocked + update
                # counter fields.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('ready', 'running')",
                    (failures, error[:500], task_id),
                )
            run_id = None
            if end_run:
                # Only the spawn path has an open run to close.
                run_id = _end_run(
                    conn, task_id,
                    outcome="gave_up", status="gave_up",
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "trigger_outcome": outcome,
                        "effective_limit": effective_limit,
                        "limit_source": limit_source,
                    },
                )
            payload = {
                "failures": failures,
                "effective_limit": effective_limit,
                "limit_source": limit_source,
                "error": error[:500],
                "trigger_outcome": outcome,
            }
            if event_payload_extra:
                payload.update(event_payload_extra)
            _append_event(
                conn, task_id, "gave_up", payload, run_id=run_id,
            )
            blocked = True
        else:
            # Below threshold.
            if release_claim:
                # Spawn path: transition running → ready + clear claim.
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status = 'running'",
                    (failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: task is already at ``ready`` via
                # its own UPDATE. Just bookkeep the counter + last error.
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, "
                    "last_failure_error = ? WHERE id = ?",
                    (failures, error[:500], task_id),
                )
            if end_run:
                # Spawn path: close the open run with outcome.
                run_id = _end_run(
                    conn, task_id,
                    outcome=outcome, status=outcome,
                    error=error[:500],
                    metadata={"failures": failures},
                )
                _append_event(
                    conn, task_id, outcome,
                    {"error": error[:500], "failures": failures},
                    run_id=run_id,
                )
            # Timeout/crash path's caller already emitted its own event.
    return blocked


# Backward-compat alias. Old name is referenced from tests and possibly
# third-party callers. New code should call ``_record_task_failure``.
def _record_spawn_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    failure_limit: int = None,
) -> bool:
    return _record_task_failure(
        conn, task_id, error,
        outcome="spawn_failed",
        failure_limit=failure_limit,
        release_claim=True,
        end_run=True,
    )


def _set_worker_pid(
    conn: sqlite3.Connection,
    task_id: str,
    pid: int,
    *,
    start_ticks: Optional[int] = None,
) -> None:
    """Bind the spawned PID to its Linux process start time."""
    if start_ticks is None:
        from hermes_cli.kanban_resource_monitor import process_start_ticks
        start_ticks = process_start_ticks(int(pid))
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (int(pid), task_id),
        )
        run_id = _current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET worker_pid = ?, worker_start_ticks = ? WHERE id = ?",
                (int(pid), start_ticks, run_id),
            )
        payload = {"pid": int(pid)}
        if start_ticks is not None:
            payload["start_ticks"] = start_ticks
        _append_event(conn, task_id, "spawned", payload, run_id=run_id)


def _clear_failure_counter(conn: sqlite3.Connection, task_id: str) -> None:
    """Reset the unified consecutive-failures counter.

    Called from ``complete_task`` on successful completion — a fresh
    success means the task + profile combination is working and any
    past failures are history. NOT called on spawn success anymore:
    a successful spawn proves the worker could start but says nothing
    about whether the run will succeed, so we need to let timeouts and
    crashes accumulate across spawn boundaries.
    """
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?",
            (task_id,),
        )


# Legacy alias for test-code and anything else that still imports it.
_clear_spawn_failures = _clear_failure_counter


# ---------------------------------------------------------------------------
# Self-modification governance gate (bounded autonomy)
# ---------------------------------------------------------------------------
# A card whose work would change how Hermes *itself* runs — its live config,
# worker profiles, systemd units, agent code, or runtime plugins/skills — must
# not be dispatched to an autonomous worker without a human approving that
# specific change first. This is the constitutional limit that keeps the
# board's autonomy from turning on the board itself (the 2026-07-11 nightly
# self-containerization). It reuses Human-Gate v1: a matching ready card is
# routed to blocked+human_gate, surfaces in the cockpit/Telegram with a plain-
# language reason, and dispatches only after a human taps approve. Work on the
# operator's own projects / client repos / docs is NOT gated — even when it
# writes code. The line is "does it change the Hermes runtime?", not "does it
# write code?".

_SELF_MODIFY_DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\.hermes/config\.yaml", "live config.yaml"),
    (r"(^|[\s`'\"])config\.yaml\b.{0,40}(profile|gateway|hermes|toolset)", "config.yaml"),
    (r"\.hermes/\.env\b", "runtime .env"),
    (r"profiles?/[\w.-]+/(config\.yaml|soul)|\bworker[ -]?profiles?\b", "worker profile"),
    (r"\bsystemctl\b|\bsystemd\b|hermes-[\w-]+\.service", "systemd unit"),
    (r"\bhermes-agent\b", "hermes-agent runtime/code"),
    (r"\.hermes/plugins/|\bplugins/tars-|\btars-config-guard\b|\btars-workflow\b", "runtime plugin"),
    (r"\.hermes/skills/", "runtime skill"),
    (r"\bagent-hooks\b", "agent hook"),
    (r"\bdisabled_toolsets\b|\bdocker_image\b|containeriz|docker_mount_|worker[ -]containment", "worker containment/toolset"),
    (r"restart\b.{0,20}\b(gateway|webui)\b|\b(gateway|webui)\b.{0,15}\brestart", "gateway/webui restart"),
    (r"\.hermes/kanban\.db\b|\bkanban_db\b|\bdispatch(er|_once)\b", "kanban runtime"),
    (r"config-guard", "config guard"),
)


def _self_modify_gate_enabled() -> bool:
    """Whether the self-modification human gate is active. ``kanban.self_modify_gate``
    in config, default True. Fails closed (gate on) when config can't be read."""
    try:
        return bool((load_config().get("kanban") or {}).get("self_modify_gate", True))
    except Exception:
        return True


def _self_modify_mutation_class(reason: str) -> str:
    """Map classifier labels to the closed internal governance vocabulary."""
    lowered = reason.casefold()
    if "profile" in lowered:
        return "worker-profile"
    if "systemd" in lowered or "restart" in lowered:
        return "systemd-unit"
    if "plugin" in lowered or "hook" in lowered:
        return "runtime-plugin"
    if "skill" in lowered:
        return "runtime-skill"
    if "containment" in lowered or "toolset" in lowered:
        return "runtime-containment"
    if "config" in lowered or ".env" in lowered:
        return "live-config"
    if "runtime" in lowered or "kanban" in lowered:
        return "agent-runtime"
    return "custom-pattern"


def _self_modify_patterns() -> list[tuple[Any, str]]:
    """Compiled default patterns plus any operator additions from
    ``kanban.self_modify_patterns`` (list of regex strings)."""
    compiled: list[tuple[Any, str]] = [
        (re.compile(rx, re.IGNORECASE), label) for rx, label in _SELF_MODIFY_DEFAULT_PATTERNS
    ]
    try:
        extra = (load_config().get("kanban") or {}).get("self_modify_patterns") or []
        for item in extra:
            if isinstance(item, str) and item.strip():
                try:
                    compiled.append((re.compile(item, re.IGNORECASE), "custom pattern"))
                except re.error:
                    pass
    except Exception:
        pass
    return compiled


def classify_self_modification(
    task, patterns: Optional[list[tuple[Any, str]]] = None
) -> Optional[str]:
    """Return a short reason label if *task* would modify the Hermes runtime,
    else None.

    Scans the card's title + body, and treats a worktree/dir card whose repo
    is the agent runtime as self-modifying regardless of prose. Content-based
    and fail-OPEN on error (returns None): the only consumer adds a human
    approval step, so a miss degrades to the prior behavior, never a crash.
    High-precision by design — patterns target runtime surfaces (``config.yaml``
    under ``~/.hermes``, worker profiles, systemd, ``hermes-agent``, runtime
    plugins/skills, agent hooks, containment), so ordinary project/client work
    does not match. False positives cost one approval tap; false negatives cost
    an unreviewed runtime change, so the bias is deliberately toward gating."""
    try:
        wp = getattr(task, "workspace_path", None)
        if wp and re.search(r"hermes-agent(\b|/)", str(wp), re.IGNORECASE):
            return "hermes-agent worktree"
        parts = []
        if getattr(task, "title", None):
            parts.append(str(task.title))
        if getattr(task, "body", None):
            parts.append(str(task.body))
        haystack = "\n".join(parts)
        if not haystack.strip():
            return None
        for rx, label in (
            patterns if patterns is not None else _self_modify_patterns()
        ):
            if rx.search(haystack):
                return label
    except Exception:
        return None
    return None


def check_respawn_guard(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready task in ``dispatch_once`` before any claim attempt.
    Returning a reason defers the spawn this tick; the task stays in
    ``ready`` and gets another chance on the next dispatcher tick.

    Checks in priority order:

    ``"rate_limit_cooldown"``
        The task's most recent run ended with the ``rate_limited`` outcome
        (a worker bailed on a provider quota wall via the EX_TEMPFAIL
        sentinel) within ``_resolve_rate_limit_cooldown_seconds()``. The
        quota almost certainly hasn't reset yet, so defer the respawn until
        the cooldown elapses — then allow a cheap probe. This is checked
        BEFORE ``blocker_auth`` because the rate-limit requeue stamps a
        quota-flavored ``last_failure_error`` that would otherwise match the
        auth-blocker regex and park the task forever (the rate-limit path
        never increments ``consecutive_failures``, so the breaker can't free
        it). Once the cooldown elapses the task falls through and respawns.

    ``"blocker_auth"``
        The task's last failure error matches a quota / authentication
        pattern. Retrying immediately is unlikely to help (rate limits
        reset on a timer; auth needs human action), so we defer to the
        next tick. The existing ``consecutive_failures`` counter still
        trips the auto-block circuit breaker after ``failure_limit``
        consecutive failures, so a persistent auth error eventually
        blocks via the normal path — but a transient 429 gets a few
        ticks of recovery first.

    ``"recent_success"``
        A completed run exists within ``_RESPAWN_GUARD_SUCCESS_WINDOW``
        seconds.  Useful work already succeeded for this task; wait for
        human review rather than immediately re-spawning. Bypassed when an
        explicit re-queue event (status change, promote, unblock, reclaim)
        arrives AFTER that completion — that's a deliberate re-run request.

    ``"active_pr"``
        A GitHub PR URL appears in a recent task comment (within
        ``_RESPAWN_GUARD_PR_WINDOW`` seconds).  A prior worker already
        opened a PR; re-spawning risks a duplicate PR on the same task.

    Stale / dead claim locks are NOT a guard reason — they are handled
    by ``release_stale_claims`` and ``detect_crashed_workers`` which
    reset the task to ``ready`` only after verifying the lock is
    genuinely dead (no live PID on this host).
    """
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())

    # 1. Rate-limit cooldown. The most recent run ended ``rate_limited``
    #    (quota wall) — defer while inside the cooldown window, then allow a
    #    cheap probe. Must run BEFORE the blocker_auth regex check, because a
    #    rate-limit requeue stamps a quota-flavored last_failure_error that
    #    the regex would otherwise match → defer forever (no failure counter
    #    increment on this path means the breaker can never free it).
    #
    #    We look at the LATEST run only (ORDER BY ended_at DESC LIMIT 1): if a
    #    newer crash/completion superseded the rate-limit run, this guard
    #    no longer applies and the normal paths take over.
    rl_cooldown = _resolve_rate_limit_cooldown_seconds()
    latest_run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if (
        latest_run is not None
        and latest_run["outcome"] == "rate_limited"
    ):
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, and skip the
            # blocker_auth regex so the stamped rate-limit text doesn't
            # re-trap the task.
            return None
        ended_at = latest_run["ended_at"]
        if ended_at is not None and (now - int(ended_at)) < rl_cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — allow the respawn. Return early so the
        # blocker_auth check below doesn't catch the rate-limit text we
        # stamped on the task; this path intentionally retries forever
        # (cheaply, spaced by the cooldown) until quota returns or a real
        # crash/completion supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.
    err = row["last_failure_error"]
    if err and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    # 3. Completed run within guard window — proof of recent success.
    #    Exception: an explicit re-queue AFTER that success (an operator
    #    dragging done→ready, a dependency re-promotion, an unblock, a
    #    reclaim) is a deliberate "run it again" — honor it instead of
    #    deferring. Without this, a manual done→ready just sits there,
    #    silently held by the guard, until the window elapses.
    cutoff = now - _RESPAWN_GUARD_SUCCESS_WINDOW
    recent_completed = conn.execute(
        "SELECT ended_at FROM task_runs "
        "WHERE task_id = ? AND outcome = 'completed' AND ended_at >= ? "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id, cutoff),
    ).fetchone()
    if recent_completed:
        completed_at = int(recent_completed["ended_at"] or 0)
        requeued_after = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND created_at >= ? "
            "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed') "
            "LIMIT 1",
            (task_id, completed_at),
        ).fetchone()
        if not requeued_after:
            return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    pr_cutoff = now - _RESPAWN_GUARD_PR_WINDOW
    for c in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ?",
        (task_id, pr_cutoff),
    ).fetchall():
        if c["body"] and _RESPAWN_GUARD_PR_URL_RE.search(c["body"]):
            return "active_pr"

    return None


def has_spawnable_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one ready+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Used by the gateway- and CLI-embedded dispatchers' health telemetry to
    decide whether ``0 spawned`` is a "stuck" condition (real spawnable
    work waiting) or a "correctly idle" condition (only control-plane
    lanes like ``orion-cc`` / ``orion-research`` waiting on terminals
    that pull tasks via ``claim_task`` directly).

    Falls back to "any ready+assigned" if ``profile_exists`` is not
    importable (e.g. partial install) — preserves the old behavior so
    the warning still fires in degraded environments.
    """
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL"
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        # Can't introspect — assume spawnable, preserve legacy behavior.
        return True
    for row in rows:
        if profile_exists(row["assignee"]):
            return True
    return False


def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one review+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Mirror of :func:`has_spawnable_ready` for the review column —
    used by the health telemetry to decide whether the dispatcher
    should have spawned a review agent.
    """
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = 'review' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL"
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        return True
    for row in rows:
        if profile_exists(row["assignee"]):
            return True
    return False


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    resource_monitor: Optional[dict[str, Any]] = None,
) -> DispatchResult:
    """Run one dispatcher tick under the board's single-writer lock.

    Thin wrapper around :func:`_dispatch_once_locked`. It acquires a
    non-blocking, board-scoped dispatch lock (issue #35240) so that two
    dispatchers pointed at the same ``kanban.db`` — e.g. the service-
    managed gateway and a shell-spawned orphan that escaped the service
    cgroup — can never run a reclaim/spawn/write tick concurrently and
    race on WAL frames. The losing dispatcher returns an empty
    ``DispatchResult`` with ``skipped_locked=True`` and does no DB writes;
    the holder is already making progress on the same board.

    The lock is keyed off the board's resolved DB path, so unrelated
    boards tick in parallel. See :func:`_dispatch_tick_lock` for the
    cross-process / cross-platform mechanics.

    ``resource_monitor`` is opt-in per caller, not a global default: only
    the gateway's own dispatcher tick (``gateway/kanban_watchers.py``)
    reads ``kanban.resource_monitor`` from config and passes it through.
    The CLI dispatch path (``hermes kanban dispatch`` /
    ``hermes_cli/kanban.py``) and the dashboard's quick-dispatch endpoint
    (``plugins/kanban/dashboard/plugin_api.py``) call this with
    ``resource_monitor=None`` and therefore run WITHOUT the resource-stall
    check, by design -- they're short-lived, human-triggered ticks, not the
    long-running loop the D-state/cgroup heuristic is built for.
    """
    try:
        db_path = kanban_db_path(board=board)
    except Exception:
        # Path resolution should never fail, but if it somehow does we
        # must not lose the tick — fall through to an unguarded dispatch
        # rather than dropping work.
        return _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            resource_monitor=resource_monitor,
        )
    with _dispatch_tick_lock(db_path) as held:
        if not held:
            return DispatchResult(skipped_locked=True)
        return _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            resource_monitor=resource_monitor,
        )


def _dispatch_once_locked(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    resource_monitor: Optional[dict[str, Any]] = None,
) -> DispatchResult:
    """Run one dispatcher tick.

    ``resource_monitor`` is opt-in and caller-supplied -- see the note on
    :func:`dispatch_once` (only the gateway's tick passes it; CLI and
    dashboard dispatch run without the resource-stall check).

    Steps:
      1. Reclaim stale running tasks (TTL expired).
      2. Reclaim stale running tasks (no recent heartbeat).
      3. Reclaim crashed running tasks (host-local PID no longer alive).
      3. Promote todo -> ready where all parents are done.
      4. For each ready task with an assignee, atomically claim and call
         ``spawn_fn(task, workspace_path, board) -> Optional[int]``. The
         return value (if any) is recorded as ``worker_pid`` so subsequent
         ticks can detect crashes before the TTL expires.

    Spawn failures are counted per-task. After ``failure_limit`` consecutive
    failures the task is auto-blocked with the last error as its reason —
    prevents the dispatcher from thrashing forever on an unfixable task.

    ``max_spawn`` is a **live concurrency cap**, not a per-tick spawn budget:
    it counts tasks already in ``status='running'`` plus this tick's spawns
    against the limit. So ``max_spawn=4`` means "at most 4 workers running
    at any time across the whole board" — matching the gateway's stated
    intent ("limit concurrent kanban tasks"). With a per-tick interpretation
    a 60-second tick interval could grow concurrency by N every minute on a
    busy board and accumulate without bound.

    ``spawn_fn`` defaults to ``_default_spawn``. Tests pass a stub.
    ``board`` pins workspace/log/db resolution for this tick to a specific
    board. When omitted, the current-board resolution chain is used.
    """
    # Reap zombie children from previously spawned workers. See
    # reap_worker_zombies() for the full rationale.
    reap_worker_zombies()

    result = DispatchResult()
    if resource_monitor and resource_monitor.get("enabled", False):
        monitor_args = {
            key: resource_monitor[key]
            for key in (
                "cgroup_path", "d_state_seconds", "memory_high_ratio",
                "psi_some_avg10", "high_event_delta",
            )
            if key in resource_monitor
        }
        result.resource_stalled = detect_resource_stalls(conn, **monitor_args)
    result.reclaimed = release_stale_claims(conn)
    result.stale = detect_stale_running(
        conn, stale_timeout_seconds=stale_timeout_seconds,
    )
    result.crashed = detect_crashed_workers(conn)
    # detect_crashed_workers stashes protocol-violation auto-blocks on
    # itself so the public list-return stays stable. Pull them into the
    # DispatchResult here so telemetry / tests see the trip.
    _crash_auto_blocked = getattr(
        detect_crashed_workers, "_last_auto_blocked", []
    )
    if _crash_auto_blocked:
        result.auto_blocked.extend(_crash_auto_blocked)
    # Rate-limited requeues (quota wall, no failure counted) — surface for
    # telemetry / tests. These tasks went back to ``ready`` and the respawn
    # guard will defer them until the quota window clears.
    _crash_rate_limited = getattr(
        detect_crashed_workers, "_last_rate_limited", []
    )
    if _crash_rate_limited:
        result.rate_limited.extend(_crash_rate_limited)
    result.timed_out = enforce_max_runtime(conn)
    result.promoted = recompute_ready(conn, failure_limit=failure_limit)

    # Count tasks already running so max_spawn enforces concurrency rather
    # than a per-tick spawn budget. See the docstring above for the full
    # rationale; the short version is that a 60-second tick interval with a
    # per-tick budget of N would grow concurrency by N every tick on a busy
    # board, since "running" tasks aren't reclaimed by completion alone —
    # they sit in status='running' until the worker calls
    # kanban_complete/kanban_block (or the dispatcher TTL-reclaims them).
    running_count = 0
    if max_spawn is not None:
        running_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )

    ready_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    # Honour kanban.max_in_progress: if the board already has enough running
    # tasks, skip spawning this tick so slow workers (local LLMs,
    # resource-constrained hosts) can finish what they have before more tasks
    # pile up and time out.
    if max_in_progress is not None and ready_rows:
        in_progress = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
        if in_progress >= max_in_progress:
            return result
        # Only spawn enough to reach the cap, respecting max_spawn too.
        remaining = max_in_progress - in_progress
        if max_spawn is None or max_spawn > remaining:
            max_spawn = remaining
    spawned = 0
    # Per-profile concurrency cap (#21582): when set, track how many
    # workers each assignee already has in flight, and refuse to spawn
    # when this would push that assignee past the cap. Prevents
    # fan-out workloads from melting a single profile's local model /
    # API quota / browser pool while leaving other profiles idle.
    # Tasks blocked this way go to skipped_per_profile_capped (not
    # skipped_unassigned — the operator-actionable signal is different:
    # "this profile is busy, try again later" not "this needs routing").
    _per_profile_cap = max_in_progress_per_profile if (
        isinstance(max_in_progress_per_profile, int)
        and max_in_progress_per_profile > 0
    ) else None
    _per_profile_running: dict[str, int] = {}
    if _per_profile_cap is not None:
        for prow in conn.execute(
            "SELECT assignee, COUNT(*) AS n FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL "
            "GROUP BY assignee"
        ):
            _per_profile_running[prow["assignee"]] = int(prow["n"])
    # Normalize default_assignee once: empty/whitespace string → None so the
    # rest of the loop can use ``if default_assignee:`` as a single check.
    # We also resolve profile_exists once here for the same reason.
    _default_assignee = (default_assignee or "").strip() or None
    # Self-modify gate: resolve config + compile patterns ONCE per tick, not
    # per ready card (load_config + re.compile in the hot loop was measurable
    # on large ready queues).
    _sm_gate_on = (not dry_run) and _self_modify_gate_enabled()
    _sm_patterns = _self_modify_patterns() if _sm_gate_on else None
    _default_assignee_resolved = False
    if _default_assignee:
        try:
            from hermes_cli.profiles import profile_exists as _pe
            _default_assignee_resolved = bool(_pe(_default_assignee))
        except Exception:
            # Profiles module not importable (test stubs, exotic envs).
            # Trust the operator's config and try the assignment; the
            # downstream profile_exists check on the assigned row will
            # bucket it as nonspawnable if the profile genuinely isn't
            # there, with the existing diagnostic.
            _default_assignee_resolved = True
    for row in ready_rows:
        if max_spawn is not None and running_count + spawned >= max_spawn:
            break
        row_assignee = row["assignee"]
        if not row_assignee:
            # Honour kanban.default_assignee: when the dispatcher hits an
            # unassigned ready task and an operator-configured fallback
            # exists, persist the assignment and proceed. This removes the
            # dashboard footgun where a task created without an assignee
            # parks in 'ready' forever even though the operator's intent
            # ("default") was perfectly clear (#27145). Mutating the row
            # (not just the in-memory view) keeps diagnostics and the
            # board state consistent: the task is now legitimately owned
            # by ``kanban.default_assignee``, not "unassigned but secretly
            # routed".
            if _default_assignee and _default_assignee_resolved:
                # Dry-run: show what WOULD happen (auto-assign + spawn) without
                # mutating the DB. Real run: mutate the row + emit the
                # 'assigned' event so the board state matches what just happened.
                if not dry_run:
                    try:
                        with write_txn(conn):
                            conn.execute(
                                "UPDATE tasks SET assignee = ? WHERE id = ? "
                                "AND (assignee IS NULL OR assignee = '')",
                                (_default_assignee, row["id"]),
                            )
                            _append_event(
                                conn, row["id"], "assigned",
                                {
                                    "assignee": _default_assignee,
                                    "source": "kanban.default_assignee",
                                },
                            )
                    except Exception:
                        _log.debug(
                            "kanban dispatch: failed to apply default_assignee=%r "
                            "to task %s",
                            _default_assignee, row["id"], exc_info=True,
                        )
                        result.skipped_unassigned.append(row["id"])
                        continue
                row_assignee = _default_assignee
                result.auto_assigned_default.append(row["id"])
            else:
                result.skipped_unassigned.append(row["id"])
                continue
        # Skip ready tasks whose assignee is not a real Hermes profile.
        # `_default_spawn` invokes ``hermes -p <assignee>`` which fails
        # with "Profile 'X' does not exist" when the assignee names a
        # control-plane lane (e.g. an interactive Claude Code terminal
        # like ``orion-cc`` / ``orion-research``) rather than a Hermes
        # profile. Those task lanes are pulled by terminals via
        # ``claim_task`` directly and should NEVER auto-spawn — the
        # subprocess would crash on startup, get reaped as a zombie,
        # the task would loop back to ``ready`` on next tick, and we'd
        # burn CPU forever (#kanban-dispatcher-crash-loop 2026-05-05).
        try:
            from hermes_cli.profiles import profile_exists  # local import: avoids cycle
        except Exception:
            profile_exists = None  # type: ignore[assignment]
        if profile_exists is not None and not profile_exists(row_assignee):
            # Bucket separately from skipped_unassigned: the operator
            # cannot fix this by assigning a profile (the assignee IS the
            # intended owner — a terminal lane). Health telemetry uses
            # this distinction to suppress spurious "stuck" warnings on
            # multi-lane setups where the ready queue is steadily full
            # of human-pulled work.
            result.skipped_nonspawnable.append(row["id"])
            continue
        # Per-profile concurrency cap (#21582): even if there's global
        # headroom, refuse to spawn for an assignee that's already at
        # its in-flight cap. Prevents one profile's local model / API
        # quota / browser pool from being overwhelmed by a fan-out
        # while the global max_in_progress / max_spawn caps still allow
        # work on OTHER profiles.
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row_assignee, 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row_assignee, current)
                )
                continue
        # Respawn guard: refuse to re-spawn when useful work is already
        # in-flight/recent, or when the last failure is a deterministic
        # blocker (quota / auth). The guard defers the spawn this tick so
        # the task gets a chance to clear (rate limits often reset in
        # seconds-to-minutes); the existing consecutive_failures counter
        # still trips the auto-block circuit breaker after failure_limit
        # consecutive failures, so a persistent auth error eventually
        # blocks via the normal path rather than on first occurrence.
        guard_reason = check_respawn_guard(conn, row["id"])
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            # Emit an event so operators can see why the task was
            # skipped when reading `hermes kanban tail` — without
            # this the task appears stuck in ready with no diagnosis.
            if not dry_run:
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason},
                    )
                # Escalation: an ``active_pr`` guard cannot clear on its own
                # while the triggering comment stays within the 24h window, so
                # a ready task it defers would livelock invisibly (one guarded
                # tick per minute, board shows a healthy "ready" card). After
                # _RESPAWN_GUARD_PR_ESCALATE_SECONDS of continuous deferral,
                # convert the livelock into an explicit needs_input block so a
                # human (or reconciler) sees it. Any spawn/claim/promote/
                # unblock resets the continuity clock.
                if guard_reason == "active_pr":
                    first_guarded = conn.execute(
                        "SELECT MIN(created_at) FROM task_events "
                        "WHERE task_id = ? AND kind = 'respawn_guarded' "
                        "AND created_at > COALESCE((SELECT MAX(created_at) "
                        "FROM task_events WHERE task_id = ? AND kind IN "
                        "('unblocked', 'promoted', 'spawned', 'claimed')), 0)",
                        (row["id"], row["id"]),
                    ).fetchone()[0]
                    if (
                        first_guarded is not None
                        and int(time.time()) - int(first_guarded)
                        >= _RESPAWN_GUARD_PR_ESCALATE_SECONDS
                    ):
                        if block_task(
                            conn,
                            row["id"],
                            require_unclaimed=True,
                            reason=(
                                "auto-block: respawn guard 'active_pr' has "
                                "deferred this ready task for over "
                                f"{_RESPAWN_GUARD_PR_ESCALATE_SECONDS // 60} "
                                "minutes — a prior worker left a PR/verdict "
                                "comment but never called kanban_complete. "
                                "Verify the durable work, then complete or "
                                "requeue this card."
                            ),
                            kind="needs_input",
                        ):
                            result.auto_blocked.append(row["id"])
            continue
        # Self-modification governance gate (bounded autonomy): a ready card
        # that would change how Hermes itself runs is routed to blocked+
        # human_gate so a human approves before it can be dispatched. A ready
        # card already carrying human_gate=1 has passed the gate — it could
        # only have reached 'ready' via a token-authorized unblock — so it
        # falls through and dispatches normally (approve-once-per-card).
        # Reuses Human-Gate v1: the block surfaces in the cockpit / Telegram
        # with the plain-language reason and a one-tap approve.
        if _sm_gate_on:
            _sm_task = get_task(conn, row["id"])
            if _sm_task is not None and not getattr(_sm_task, "human_gate", 0):
                _sm_reason = classify_self_modification(
                    _sm_task, patterns=_sm_patterns
                )
                if _sm_reason:
                    if block_task(
                        conn,
                        row["id"],
                        require_unclaimed=True,
                        reason="self-modification gate: " + _sm_reason,
                        kind="needs_input",
                        human_gate=True,
                        human_summary=(
                            "Diese Karte will an Hermes selbst etwas ändern ("
                            + _sm_reason + "). Autonome Ausführung ist gesperrt, "
                            "bis du sie freigibst."
                        ),
                        human_action=(
                            "Prüfe die Karte; tippe im Cockpit auf „Gate & weiter“ "
                            "(oder den Telegram-Gate-Button), wenn diese "
                            "Laufzeit-Änderung gewollt ist."
                        ),
                        _governance_target_ref=(
                            (_sm_task.branch_name or "").strip() or f"task/{row['id']}"
                        ),
                        _governance_mutation_class=_self_modify_mutation_class(_sm_reason),
                    ):
                        result.self_modify_gated.append((row["id"], _sm_reason))
                    continue
        if dry_run:
            result.spawned.append((row["id"], row_assignee, ""))
            # Increment per-profile counter even in dry_run so the cap
            # check sees the would-be spawn on subsequent iterations.
            # Without this, dry_run reports every task as spawnable and
            # under-reports the capped subset (#21582).
            if _per_profile_cap is not None and row_assignee:
                _per_profile_running[row_assignee] = (
                    _per_profile_running.get(row_assignee, 0) + 1
                )
            continue
        claimed = claim_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
        _maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            # Back-compat: older spawn_fn signatures accept only
            # (task, workspace). Test stubs in the suite rely on that.
            # Introspect the callable and pass `board` only when supported.
            import inspect
            try:
                sig = inspect.signature(_spawn)
                if "board" in sig.parameters:
                    pid = _spawn(claimed, str(workspace), board=board)
                else:
                    pid = _spawn(claimed, str(workspace))
            except (TypeError, ValueError):
                pid = _spawn(claimed, str(workspace))
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # NOTE: we intentionally do NOT reset consecutive_failures
            # here. A successful spawn proves the worker can start but
            # doesn't prove the run will succeed. Under unified
            # failure counting, resetting on spawn would let a task
            # that keeps timing out after spawn loop forever. The
            # counter is cleared only on successful completion (see
            # complete_task).
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            # Track the new in-flight count for this profile so later
            # iterations in this same tick respect the per-profile cap
            # (#21582). Subsequent ticks re-query from the DB.
            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, str(exc),
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)

    # ---- review column dispatch ----
    # Review tasks are tasks that a worker moved to 'review' after
    # creating a PR.  The dispatcher spawns a review agent (loading
    # sdlc-review skill) that verifies the PR and either merges (→ done)
    # or rejects (→ back to running for the worker to fix).
    #
    # Same concurrency model as ready dispatch: review spawns count
    # against max_spawn alongside ready tasks, so the total number of
    # running workers stays bounded.
    review_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'review' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    for row in review_rows:
        if max_spawn is not None and running_count + spawned >= max_spawn:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        try:
            from hermes_cli.profiles import profile_exists
        except Exception:
            profile_exists = None  # type: ignore[assignment]
        if profile_exists is not None and not profile_exists(row["assignee"]):
            result.skipped_nonspawnable.append(row["id"])
            continue
        if dry_run:
            result.spawned.append((row["id"], row["assignee"], ""))
            continue
        claimed = claim_review_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
        _maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        # Force-load the sdlc-review skill for review agents — it carries
        # the review logic (AC verification, merge, etc.). The mandatory
        # kanban lifecycle is already injected into every worker's system
        # prompt via KANBAN_GUIDANCE, so this is the only extra skill the
        # review agent needs.
        claimed.skills = ["sdlc-review"]
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            import inspect
            try:
                sig = inspect.signature(_spawn)
                if "board" in sig.parameters:
                    pid = _spawn(claimed, str(workspace), board=board)
                else:
                    pid = _spawn(claimed, str(workspace))
            except (TypeError, ValueError):
                pid = _spawn(claimed, str(workspace))
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, str(exc),
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
    return result


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.

    Defaults preserve the historical behavior: rotate at 2 MiB and keep one
    backup generation (``.log.1``). Operators with long-running workers can
    raise either value from ``config.yaml`` without changing dispatcher code.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    max_bytes = _positive_int(
        (kanban_cfg or {}).get("worker_log_rotate_bytes"),
        DEFAULT_LOG_ROTATE_BYTES,
        minimum=1,
    )
    backup_count = _positive_int(
        (kanban_cfg or {}).get("worker_log_backup_count"),
        DEFAULT_LOG_BACKUP_COUNT,
        minimum=0,
    )
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``.

    ``backup_count=1`` preserves the legacy single-generation behavior:
    ``<log>`` moves to ``<log>.1`` and any previous ``.1`` is replaced.
    Higher values shift older generations up to ``backup_count``.
    """
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(
            backup_count,
            DEFAULT_LOG_BACKUP_COUNT,
            minimum=0,
        )
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        try:
            if oldest.exists():
                oldest.unlink()
        except OSError:
            pass
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            try:
                src.rename(_rotated_log_path(log_path, generation + 1))
            except OSError:
                pass
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _module_hermes_argv() -> list[str]:
    """Return the interpreter-bound Hermes CLI invocation."""
    # ``hermes_cli.main`` is the console-script target declared in
    # pyproject.toml, NOT a top-level ``hermes`` package — there is no
    # ``hermes`` package to import.
    return [sys.executable, "-m", "hermes_cli.main"]


def _absolute_hermes_path(path: str) -> str:
    """Return an absolute filesystem path for a resolved Hermes shim."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isabs(expanded) else os.path.abspath(expanded)


def _looks_like_path(value: str) -> bool:
    """Return true when a command override is an explicit path, not a name."""
    expanded = os.path.expanduser(value)
    return (
        expanded.startswith("~")
        or os.path.isabs(expanded)
        or bool(os.path.dirname(expanded))
        or "\\" in expanded
        or bool(re.match(r"^[A-Za-z]:", expanded))
    )


def _is_windows_batch_shim(path: str) -> bool:
    """Return true for Windows shell/batch shims that should not be argv[0]."""
    return path.lower().endswith((".cmd", ".bat"))


def _path_search_names(command: str) -> list[str]:
    """Return executable names to try for an unqualified command."""
    if not _IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    exts = [ext for ext in raw.split(";") if ext]
    return [command + ext for ext in exts]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    ``shutil.which`` follows platform search behavior. On Windows that can
    include the current directory before PATH for bare names, which is not a
    safe dispatcher primitive. This resolver only considers explicit PATH
    entries and skips empty / ``.`` entries.
    """
    path_env = os.environ.get("PATH", "")
    for raw_dir in path_env.split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if not os.path.isfile(candidate):
                continue
            if _IS_WINDOWS or os.access(candidate, os.X_OK):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """Return argv for a resolved Hermes executable path.

    Windows batch shims (`.cmd` / `.bat`) are not safe as argv[0] for
    worker launches because the argument vector includes task-derived
    values. Prefer the interpreter-bound module form whenever the resolved
    executable is only a shell shim.
    """
    if _IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv parts for ``Popen``.

    Tries in order:

    1. ``$HERMES_BIN`` — explicit operator override. Path-like values are
       normalized to absolute paths; bare command names keep normal PATH
       semantics and never prefer a same-directory file before ``PATH``.
    2. ``shutil.which("hermes")`` — the console-script shim, normalized to
       an absolute path. On Windows, ``which`` can return a relative
       ``.\\hermes.CMD`` when the current directory is on ``PATH``; directly
       launching batch shims is also unsafe with task-derived argv. The
       dispatcher therefore falls back to the interpreter-bound module form
       for implicit ``.cmd`` / ``.bat`` shims.
    3. ``sys.executable -m hermes_cli.main`` — fallback for setups where
       Hermes is launched from a venv and the ``hermes`` shim is not on
       the dispatcher's ``$PATH`` (cron, systemd ``User=`` services,
       launchd jobs, detached processes, etc.). Goes through the running
       interpreter so the result is independent of ``$PATH``.

    Mirrors ``gateway.run._resolve_hermes_bin`` for the same reason. Kept
    local (not imported from gateway) because ``hermes_cli`` sits below
    ``gateway`` in the dependency order.
    """
    import shutil

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes") if _IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    Kanban's ``max_runtime_seconds`` bounds the whole worker attempt. The
    terminal tool has its own default timeout via ``TERMINAL_TIMEOUT``; when
    the worker runtime is longer, raise only the child process default so a
    long command is not killed by the generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Dispatcher-spawned workers are launched from a long-lived gateway process,
    then the child re-enters the CLI with ``-p <assignee>``. Resolve the
    assignee profile's CLI tool surface at dispatch time and pass it as an
    explicit ``--toolsets`` pin so worker startup cannot fall back to a stale
    root/active-profile config or a profile whose top-level ``toolsets`` entry
    is only the kanban orchestrator surface. ``model_tools`` still appends the
    task-scoped kanban lifecycle tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_WORKER_ENV_STRIP_DEFAULT = (
    "NTFY_TOKEN",
    "NTFY_ADMIN_TOKEN",
)


def _worker_env_strip_keys() -> tuple[str, ...]:
    """Secret env keys stripped from dispatcher-spawned worker processes.

    Defaults cover the notification/Human-Gate channel (C4, audit
    2026-07-11). Operators can extend via ``kanban.worker_env_strip``
    (list of env var names) without a code change.
    """
    keys = list(_WORKER_ENV_STRIP_DEFAULT)
    try:
        from hermes_cli.config import load_config
        extra = (load_config().get("kanban") or {}).get("worker_env_strip") or []
        for item in extra:
            if isinstance(item, str) and item.strip():
                keys.append(item.strip())
    except Exception:
        pass
    return tuple(keys)


def _resolve_board_bank(board: Optional[str]) -> Optional[str]:
    """TARS carry (P6 Option 2, dispatcher path): map a kanban board to a Hindsight
    memory bank, so the shared background dispatcher (which runs in ONE context) never
    lands professional cards in the private bank or vice versa.

    Reads ``<hermes-root>/ops/kanban-board-banks.json``:
        {"<board-slug>": "<bank>", "_default": "<bank>"}
    Returns the bank to force, or None (no override → the worker keeps the bank it
    inherited from the dispatcher's env). ``_default`` (optional) covers unmapped
    boards — set it to a quarantine bank for strict no-mix. Read per spawn (edit the
    map without restarting). Fail-soft: any error → None (never break dispatch)."""
    try:
        path = kanban_home() / "ops" / "kanban-board-banks.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        # Underscore-prefixed keys are meta (_README, _default) — never board slugs.
        bank = (data.get(board) if (board and not board.startswith("_")) else None) or data.get("_default")
        bank = str(bank).strip() if bank else ""
        if bank and re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", bank):
            return bank
        return None
    except Exception:
        return None


def _default_spawn(
    task: Task,
    workspace: str,
    *,
    board: Optional[str] = None,
) -> Optional[int]:
    """Fire-and-forget ``hermes -p <profile> chat -Q -q ...`` subprocess.

    Returns the spawned child's PID so the dispatcher can detect crashes
    before the claim TTL expires. The child's completion is still observed
    via the ``complete`` / ``block`` transitions the worker writes itself;
    the PID check is a safety net for crashes, OOM kills, and Ctrl+C.

    ``board`` pins the child's kanban context to that board: the child's
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / workspaces_root env
    vars all resolve to the same board the dispatcher claimed the task
    from. Workers cannot accidentally see other boards.
    """
    import subprocess
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    from hermes_cli.profiles import normalize_profile_name

    profile_arg = normalize_profile_name(task.assignee)

    prompt = f"work kanban task {task.id}"
    env = dict(os.environ)

    # A dispatcher may itself be launched from a task-bound worker. Never let
    # ambient ownership or per-task execution settings bleed into the next
    # child; scrub the whole Kanban namespace, then pin the new task/board
    # explicitly below. The few non-prefixed task settings are cleared too.
    for key in tuple(env):
        if key.startswith("HERMES_KANBAN_"):
            env.pop(key, None)
    for key in ("HERMES_REASONING_EFFORT", "HERMES_TENANT", "TERMINAL_CWD"):
        env.pop(key, None)

    # C4 (Audit 2026-07-11): operational secrets a worker has no business
    # holding. NTFY_* serves the Human-Gate/attention escalation channel —
    # a prompt-injected worker holding it could impersonate or drown the
    # very channel that gates it. Extendable via kanban.worker_env_strip.
    # GITEA_TOKEN is deliberately NOT stripped: workers push branches/PRs
    # as part of normal board work (accepted LAN-secret decision).
    for key in _worker_env_strip_keys():
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml
    # (fallback_providers, toolsets, agent settings, etc.) instead of the root
    # config.  Without this, `env = dict(os.environ)` copies only the parent's
    # env, and when the child process starts `hermes -p <name>` the
    # _apply_profile_override() runs *before* hermes_constants is imported.
    # If HERMES_HOME is absent from the child's env, get_hermes_home() falls
    # back to Path.home() / ".hermes" (the DEFAULT profile root), ignoring the
    # profile-specific config entirely.  Fixes profile-scoped fallback_providers
    # being invisible to kanban workers.
    from hermes_cli.profiles import resolve_profile_env
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        # Profile dir doesn't exist — defer resolution to the CLI's
        # _apply_profile_override() via HERMES_PROFILE (set below).
        # This only happens in test fixtures where the isolated
        # HERMES_HOME never had profiles created.
        pass
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    # Pin TERMINAL_CWD to the task's workspace so the worker's file tools and
    # context-file loader anchor on the workspace, not whatever cwd the
    # dispatching gateway happened to export. The worker subprocess is already
    # launched with cwd=workspace, but TERMINAL_CWD takes precedence over the
    # process cwd in both file_tools._resolve_base_dir (#41312 — relative
    # write_file paths were landing in the gateway user's home) and
    # build_context_files_prompt (#34619 — workers loaded the dispatching
    # gateway's AGENTS.md instead of the task's). Setting it to the workspace
    # fixes both: the workspace is where the task's work actually happens.
    # Only pin a real, absolute directory — file_tools rejects relative /
    # sentinel TERMINAL_CWD values, so a non-dir workspace must NOT be set
    # here (leave the inherited value rather than write a meaningless one).
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    # Event-sourcing resume (#2): hand the worker its pinned resumable session id
    # (None when resume-on-reclaim is off) and, on a re-claim of a crashed task,
    # tell it to resume the prior conversation. HERMES_SESSION_ID is deliberately
    # NOT reused for this — it carries the *originating* session and gets
    # overwritten by agent_init with the worker's own id anyway.
    if getattr(task, "worker_session_id", None):
        env["HERMES_KANBAN_WORKER_SESSION"] = task.worker_session_id
        if getattr(task, "resume_requested", False):
            env["HERMES_KANBAN_RESUME"] = "1"
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    # Goal-loop mode: the worker reads these and wraps its run in the
    # Ralph-style /goal judge loop (see cli.py quiet-mode path). Only set
    # when enabled so non-goal tasks keep a clean env.
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    # Per-task reasoning-effort override (tars-workflow chained goal cards).
    # Only set when present so non-overridden tasks keep a clean env and fall
    # through to the profile's agent.reasoning_effort. The worker's config
    # loader gives this env value precedence (see cli.py reasoning_config).
    if task.effort:
        env["HERMES_REASONING_EFFORT"] = str(task.effort)
    terminal_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_TIMEOUT"),
    )
    if terminal_timeout is not None:
        env["TERMINAL_TIMEOUT"] = terminal_timeout
    foreground_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_MAX_FOREGROUND_TIMEOUT"),
    )
    if foreground_timeout is not None:
        env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] = foreground_timeout
    # Pin the shared board + workspaces root the dispatcher resolved, so
    # that even when the worker activates a profile (`hermes -p <name>`
    # rewrites HERMES_HOME), its kanban paths still match the
    # dispatcher's. Belt-and-braces with the `get_default_hermes_root()`
    # resolution in `kanban_home()` — symmetric resolution is the norm,
    # but unusual symlink / Docker layouts are caught here too.
    env["HERMES_KANBAN_DB"] = str(kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(workspaces_root(board=board))
    # Board slug — the final defense-in-depth pin. If the worker ever
    # resolves kanban paths without the DB / workspaces env vars, the
    # board slug still forces it to the right directory.
    resolved_board = _normalize_board_slug(board) or get_current_board()
    env["HERMES_KANBAN_BOARD"] = resolved_board
    # HERMES_PROFILE is the author the kanban_comment tool defaults to.
    # `hermes -p <assignee>` activates the profile, but the env var is
    # what the tool reads — set it explicitly here so comments are
    # attributed correctly regardless of how the child loads config.
    env["HERMES_PROFILE"] = profile_arg

    # A worker must NEVER boot the interactive TUI: an inherited HERMES_TUI=1
    # or a `display.interface: tui` in the profile's config would send the
    # quiet chat run into the Ink TUI, whose no-TTY bail-out exits 0 without
    # doing the task → "protocol violation" on every attempt. `--cli` is the
    # highest-precedence interface override; dropping the env var covers
    # older hermes builds on PATH that predate the flag's precedence.
    env.pop("HERMES_TUI", None)

    # TARS carry (P6 Option 2, dispatcher path): force the worker's memory bank by
    # board context. The shared dispatcher runs in ONE context (private/default), so
    # without this every dispatched worker would inherit the private bank — mixing
    # professional cards into private memory (or vice versa). Set explicitly so it
    # survives the worker's own profile .env (load_hermes_dotenv override=True).
    # Unmapped board → no override (worker keeps the inherited bank). Fail-soft.
    _board_bank = _resolve_board_bank(resolved_board)
    if _board_bank:
        env["HINDSIGHT_BANK_ID"] = _board_bank

    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile_arg,
        "--cli",
        # Worker subprocesses switch to a profile-scoped HERMES_HOME above,
        # so they see that profile's shell-hook allowlist instead of the
        # dispatcher's root allowlist. Pass --accept-hooks explicitly so
        # profile-local worker sessions still register configured hooks.
        "--accept-hooks",
    ]
    # Per-task force-loaded skills. Each name goes in its own
    # `--skills X` pair rather than a single comma-joined arg: the CLI
    # accepts both forms (action='append' + comma-split), but
    # per-name pairs are easier to read in `ps` output and avoid any
    # quoting ambiguity if a skill name ever contains unusual chars.
    if task.skills:
        for sk in task.skills:
            if sk:
                cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
    worker_toolsets = _resolve_worker_cli_toolsets(env.get("HERMES_HOME"))
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend([
        "chat",
        # Kanban workers are automation, not interactive CLI sessions. The
        # quiet one-shot branch retains the structured run result and maps a
        # terminal provider quota/rate-limit to EX_TEMPFAIL (75), which the
        # dispatcher reaper classifies as transient. Plain ``chat -q`` goes
        # through ``HermesCLI.chat`` and discards that failure envelope.
        "-Q",
        "-q", prompt,
    ])
    # Redirect output to a per-task log under <board-root>/logs/.
    # Anchored at the board root (not the shared kanban root), so
    # `hermes kanban log` on a specific board reads its own file and
    # logs don't collide across boards that happen to share task ids.
    log_dir = worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)

    # Use 'a' so a re-run on unblock appends rather than overwrites.
    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )
    except FileNotFoundError:
        log_f.close()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    # NOTE: we intentionally do NOT close log_f here — we want Popen's
    # child process to keep writing after this function returns.  The
    # handle is kept alive by the child's inheritance.  The parent's
    # reference goes out of scope and is GC'd, but the OS-level FD stays
    # open in the child until the child exits.
    return proc.pid


# ---------------------------------------------------------------------------
# Long-lived dispatcher daemon
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    interval: float = 60.0,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stop_event=None,
    on_tick=None,
) -> None:
    """Run the dispatcher in a loop until interrupted.

    Calls :func:`dispatch_once` every ``interval`` seconds. Exits cleanly
    on SIGINT / SIGTERM so ``hermes kanban daemon`` is systemd-friendly.
    ``stop_event`` (a :class:`threading.Event`) and ``on_tick`` (a
    callable receiving the :class:`DispatchResult`) are test hooks.
    """
    import signal
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    # Install handlers only when running on the main thread — tests call
    # this inline from worker threads and signal() would raise there.
    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle)
                except (ValueError, OSError):
                    pass

    while not stop_event.is_set():
        try:
            with contextlib.closing(connect()) as conn:
                res = dispatch_once(
                    conn,
                    max_spawn=max_spawn,
                    failure_limit=failure_limit,
                )
            if on_tick is not None:
                try:
                    on_tick(res)
                except Exception:
                    pass
        except Exception:
            # Don't let any single tick kill the daemon.
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


# ---------------------------------------------------------------------------
# Worker context builder (what a spawned worker sees)
# ---------------------------------------------------------------------------

def build_worker_context(conn: sqlite3.Connection, task_id: str) -> str:
    """Return the full text a worker should read to understand its task.

    Order:
      1. Task title (mandatory).
      2. Task body (optional opening post, capped at 8 KB).
      3. Prior attempts on THIS task (most recent ``_CTX_MAX_PRIOR_ATTEMPTS``
         shown; older attempts collapsed into a one-line summary).
         Each attempt's ``summary`` / ``error`` / ``metadata`` capped at
         ``_CTX_MAX_FIELD_BYTES`` each.
      4. Structured handoff results of every done parent task. Prefers
         ``run.summary`` / ``run.metadata`` when the parent was executed
         via a run; falls back to ``task.result`` for older data. Same
         per-field cap.
      5. Cross-task role history for the assignee (most recent 5
         completed runs on other tasks).
      6. Comment thread (most recent ``_CTX_MAX_COMMENTS`` shown, older
         collapsed).

    All caps exist so worker prompts stay bounded even on pathological
    boards (retry-heavy tasks, comment storms). The per-field char cap
    prevents a single 1 MB summary from dominating context.
    """
    task = get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")

    # Single clock reading shared by every relative-age stamp below, so all
    # ages in one rendering are consistent ("3h ago" / "3h ago", not drifting
    # by the seconds it takes to build the block).
    _now = int(time.time())

    def _cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
        """Truncate a string to `limit` chars with a visible ellipsis."""
        if not s:
            return ""
        s = s.strip()
        if len(s) <= limit:
            return s
        return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"

    lines: list[str] = []
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _worker_terminal_timeout_env(
            task.max_runtime_seconds,
            os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    lines.append("")

    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")

    # Attachments — files uploaded to this task (PDFs, source docs,
    # images). Surface the absolute on-disk path so the worker, which has
    # full file-tool access, can read them directly (read_file, terminal
    # `pdftotext`, etc.). On the local terminal backend the path resolves
    # as-is; remote backends need the kanban attachments dir mounted.
    attachments = list_attachments(conn, task_id)
    if attachments:
        lines.append("## Attachments")
        lines.append(
            "Files attached to this task. Read them with the file/terminal "
            "tools at the absolute paths below:"
        )
        for att in attachments:
            size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
            size_str = f", {size_kb} KB" if size_kb else ""
            ctype = f", {att.content_type}" if att.content_type else ""
            lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
        lines.append("")

    # Prior attempts — show closed runs so a retrying worker sees the
    # history. Skip the currently-active run (that's this worker).
    # Cap at _CTX_MAX_PRIOR_ATTEMPTS most-recent closed runs; older
    # attempts get collapsed into a one-line marker so the worker knows
    # more exist without bloating the prompt.
    all_prior = [r for r in list_runs(conn, task_id) if r.ended_at is not None]
    # list_runs returns ascending by started_at; "most recent" = last N
    if len(all_prior) > _CTX_MAX_PRIOR_ATTEMPTS:
        omitted = len(all_prior) - _CTX_MAX_PRIOR_ATTEMPTS
        shown = all_prior[-_CTX_MAX_PRIOR_ATTEMPTS:]
        first_shown_idx = omitted + 1
    else:
        omitted = 0
        shown = all_prior
        first_shown_idx = 1
    if shown:
        lines.append("## Prior attempts on this task")
        if omitted:
            lines.append(
                f"_({omitted} earlier attempt{'s' if omitted != 1 else ''} "
                f"omitted; showing most recent {len(shown)})_"
            )
        for offset, run in enumerate(shown):
            idx = first_shown_idx + offset
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(run.started_at))
            age = _relative_age(run.started_at, _now)
            ts_disp = f"{ts}, {age}" if age else ts
            profile = run.profile or "(unknown)"
            outcome = run.outcome or run.status
            lines.append(f"### Attempt {idx} — {outcome} ({profile}, {ts_disp})")
            if run.summary and run.summary.strip():
                lines.append(_cap(run.summary))
            if run.error and run.error.strip():
                lines.append(f"_error_: {_cap(run.error)}")
            if run.metadata:
                try:
                    meta_str = json.dumps(run.metadata, ensure_ascii=False, sort_keys=True)
                    lines.append(f"_metadata_: `{_cap(meta_str)}`")
                except Exception:
                    pass
            lines.append("")

    # Parents: prefer the most-recent 'completed' run's summary + metadata,
    # fall back to ``task.result`` when no run rows exist (legacy DBs,
    # or tasks completed before the runs table landed).
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    parent_ids = [r["parent_id"] for r in parent_rows]

    if parent_ids:
        wrote_header = False
        for pid in parent_ids:
            pt = get_task(conn, pid)
            if not pt or pt.status != "done":
                continue
            runs = [r for r in list_runs(conn, pid) if r.outcome == "completed"]
            runs.sort(key=lambda r: r.started_at, reverse=True)
            run = runs[0] if runs else None

            if not wrote_header:
                lines.append("## Parent task results")
                lines.append(
                    "_Handoffs from upstream tasks, captured when each parent "
                    "completed (see age below). These are point-in-time "
                    "snapshots, not live state — if a result drives your "
                    "current work and it's not recent, re-verify against the "
                    "source before acting on it as current._"
                )
                wrote_header = True

            # When did this parent's result get produced? Prefer the
            # completed run's end time; fall back to the task's completed_at.
            done_ts = None
            if run is not None and getattr(run, "ended_at", None):
                done_ts = run.ended_at
            elif pt.completed_at:
                done_ts = pt.completed_at
            age = _relative_age(done_ts, _now)
            lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))

            body_lines: list[str] = []
            if run is not None and run.summary and run.summary.strip():
                body_lines.append(_cap(run.summary))
            elif pt.result:
                body_lines.append(_cap(pt.result))
            else:
                body_lines.append("(no result recorded)")

            if run is not None and run.metadata:
                try:
                    meta_str = json.dumps(run.metadata, ensure_ascii=False, sort_keys=True)
                    body_lines.append(f"_metadata_: `{_cap(meta_str)}`")
                except Exception:
                    pass
            lines.extend(body_lines)
            lines.append("")

    # Cross-task role history: what else has THIS assignee completed
    # recently? Gives the worker implicit continuity — "I'm the reviewer
    # and my last three reviews focused on security" — without forcing
    # the user to wire anything into SOUL.md / MEMORY.md. Bounded to the
    # most recent 5 completed runs, excluding this task so the retry
    # section above isn't duplicated. Safe on assignee=None (skipped).
    if task.assignee:
        role_rows = conn.execute(
            "SELECT t.id, t.title, r.summary, r.ended_at "
            "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
            "WHERE r.profile = ? AND r.task_id != ? "
            "  AND r.outcome = 'completed' "
            "ORDER BY r.ended_at DESC LIMIT 5",
            (task.assignee, task_id),
        ).fetchall()
        if role_rows:
            lines.append(f"## Recent work by @{task.assignee}")
            for row in role_rows:
                ts = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(int(row["ended_at"]))
                )
                age = _relative_age(row["ended_at"], _now)
                ts_disp = f"{ts}, {age}" if age else ts
                s = (row["summary"] or "").strip().splitlines()
                first = s[0][:200] if s else "(no summary)"
                lines.append(f"- {row['id']} — {row['title']} ({ts_disp}): {first}")
            lines.append("")

    # Comments: cap at the most-recent _CTX_MAX_COMMENTS so
    # comment-storm tasks don't blow out the worker's prompt. Older
    # comments summarised in a one-line marker like prior attempts.
    all_comments = list_comments(conn, task_id)
    if len(all_comments) > _CTX_MAX_COMMENTS:
        omitted_c = len(all_comments) - _CTX_MAX_COMMENTS
        shown_c = all_comments[-_CTX_MAX_COMMENTS:]
    else:
        omitted_c = 0
        shown_c = all_comments
    if shown_c:
        lines.append("## Comment thread")
        if omitted_c:
            lines.append(
                f"_({omitted_c} earlier comment{'s' if omitted_c != 1 else ''} "
                f"omitted; showing most recent {len(shown_c)})_"
            )
        for c in shown_c:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.created_at))
            age = _relative_age(c.created_at, _now)
            ts_disp = f"{ts}, {age}" if age else ts
            # Render author with explicit "comment from worker" framing so
            # operator-controlled HERMES_PROFILE values like "hermes-system"
            # or "operator" can't be misread by the next worker as a system
            # directive above the (attacker-influenceable) comment body.
            # Defense-in-depth — the LLM-controlled author-forgery surface
            # was already closed in #22435. See #22452.
            safe_author = (c.author or "").replace("`", "")
            lines.append(f"comment from worker `{safe_author}` at {ts_disp}:")
            lines.append(_cap(c.body, _CTX_MAX_COMMENT_BYTES))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Stats + SLA helpers
# ---------------------------------------------------------------------------

def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts, plus the oldest ``ready`` age in
    seconds (the clearest staleness signal for a router or HUD).
    """
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        by_assignee.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    oldest_row = conn.execute(
        "SELECT MIN(created_at) AS ts FROM tasks WHERE status = 'ready'"
    ).fetchone()
    now = int(time.time())
    oldest_ready_age = (
        (now - int(oldest_row["ts"]))
        if oldest_row and oldest_row["ts"] is not None else None
    )

    return {
        "by_status": by_status,
        "by_assignee": by_assignee,
        "oldest_ready_age_seconds": oldest_ready_age,
        "now": now,
    }


def _to_epoch(val) -> Optional[int]:
    """Normalise a timestamp to unix epoch seconds.

    Accepts ints (pass-through), numeric strings, and ISO-8601 strings.
    Returns ``None`` for ``None`` / empty values.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # ISO-8601 fallback (e.g. '2026-05-10T15:00:00Z')
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, OSError):
        return None


def task_age(task: Task) -> dict:
    """Return age metrics for a single task. All values are seconds or None."""
    now = int(time.time())
    _c = _to_epoch(task.created_at)
    _s = _to_epoch(task.started_at)
    _co = _to_epoch(task.completed_at)
    age_since_created = now - _c if _c is not None else None
    age_since_started = now - _s if _s is not None else None
    time_to_complete = (
        _co - (_s or _c) if _co is not None else None
    )
    return {
        "created_age_seconds": age_since_created,
        "started_age_seconds": age_since_started,
        "time_to_complete_seconds": time_to_complete,
    }


# ---------------------------------------------------------------------------
# Notification subscriptions (used by the gateway kanban-notifier)
# ---------------------------------------------------------------------------

def add_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    notifier_profile: Optional[str] = None,
) -> None:
    """Register a channel, reactivating a tombstone with a new ABA generation."""
    now = int(time.time())
    thread = thread_id or ""
    with write_txn(conn):
        row = conn.execute("SELECT active FROM kanban_notify_subs WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=?", (task_id, platform, chat_id, thread)).fetchone()
        if row is None:
            conn.execute("INSERT INTO kanban_notify_subs (task_id,platform,chat_id,thread_id,user_id,notifier_profile,created_at,active,generation) VALUES (?,?,?,?,?,?,?,?,1)", (task_id, platform, chat_id, thread, user_id, notifier_profile, now, 1))
        elif not int(row['active']):
            conn.execute("UPDATE kanban_notify_subs SET active=1, generation=generation+1, user_id=COALESCE(?, user_id), notifier_profile=COALESCE(?, notifier_profile) WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=?", (user_id, notifier_profile, task_id, platform, chat_id, thread))
        elif notifier_profile:
            conn.execute("UPDATE kanban_notify_subs SET notifier_profile=? WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=? AND (notifier_profile IS NULL OR notifier_profile='')", (notifier_profile, task_id, platform, chat_id, thread))


def list_notify_subs(
    conn: sqlite3.Connection, task_id: Optional[str] = None, *, include_inactive: bool = False,
) -> list[dict]:
    where = "" if include_inactive else " WHERE active=1"
    params: tuple[Any, ...] = ()
    if task_id is not None:
        where = (" WHERE task_id=?" if include_inactive else " WHERE task_id=? AND active=1")
        params = (task_id,)
    return [dict(r) for r in conn.execute("SELECT * FROM kanban_notify_subs" + where, params).fetchall()]


def remove_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> bool:
    """Deactivate (rather than delete) a channel and cancel its current generation."""
    now = int(time.time())
    thread = thread_id or ""
    with write_txn(conn):
        row = conn.execute("SELECT generation FROM kanban_notify_subs WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=? AND active=1", (task_id, platform, chat_id, thread)).fetchone()
        if row is None:
            return False
        generation = int(row['generation'])
        conn.execute("UPDATE kanban_notify_subs SET active=0 WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=? AND active=1", (task_id, platform, chat_id, thread))
        conn.execute("UPDATE kanban_attention_deliveries SET state='cancelled', lease_until=NULL, lease_version=lease_version+1, updated_at=? WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=? AND subscription_generation=? AND state IN ('pending','sending')", (now, task_id, platform, chat_id, thread, generation))
        return True


def unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, list[Event]]:
    """Return ``(new_cursor, events)`` for a given subscription.

    Only events with ``id > last_event_id`` are returned. The subscription's
    cursor is NOT advanced here; call :func:`advance_notify_cursor` after
    the gateway has successfully delivered the notifications.
    """
    row = conn.execute(
        "SELECT last_event_id FROM kanban_notify_subs "
        "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
        (task_id, platform, chat_id, thread_id or ""),
    ).fetchone()
    if row is None:
        return 0, []
    cursor = int(row["last_event_id"])
    kind_list = list(kinds) if kinds else None
    q = (
        "SELECT * FROM task_events WHERE task_id = ? AND id > ? "
        + ("AND kind IN (" + ",".join("?" * len(kind_list)) + ") " if kind_list else "")
        + "ORDER BY id ASC"
    )
    params: list[Any] = [task_id, cursor]
    if kind_list:
        params.extend(kind_list)
    rows = conn.execute(q, params).fetchall()
    out: list[Event] = []
    max_id = cursor
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(Event(
            id=r["id"], task_id=r["task_id"], kind=r["kind"],
            payload=payload, created_at=r["created_at"],
            run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
        ))
        max_id = max(max_id, int(r["id"]))
    return max_id, out


def claim_unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, int, list[Event]]:
    """Atomically claim unseen notification events for one subscription.

    Returns ``(old_cursor, new_cursor, events)``. When events are returned,
    ``kanban_notify_subs.last_event_id`` has already been advanced to
    ``new_cursor`` inside a ``BEGIN IMMEDIATE`` transaction. That makes the
    notifier's read/claim step single-owner across multiple gateway watcher
    processes pointed at the same board DB: concurrent watchers serialize on
    SQLite's writer lock, and only the first process sees and claims a given
    event range.

    Callers should send the claimed events, then either leave the cursor at
    ``new_cursor`` on success or call :func:`rewind_notify_cursor` if delivery
    failed before any terminal unsubscribe removed the row.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        ).fetchone()
        if row is None:
            return 0, 0, []
        old_cursor = int(row["last_event_id"])
        new_cursor, events = unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or "", int(old_cursor)),
        )
        return old_cursor, new_cursor, events


def advance_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    new_cursor: int,
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or ""),
        )


def rewind_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    claimed_cursor: int,
    old_cursor: int,
) -> bool:
    """Undo a notification claim when delivery fails.

    The CAS guard only rewinds if no later notifier advanced the row after our
    claim. This keeps retry behavior for transient send failures without
    clobbering newer progress.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (
                int(old_cursor), task_id, platform, chat_id, thread_id or "",
                int(claimed_cursor),
            ),
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Retention + garbage collection
# ---------------------------------------------------------------------------

def gc_events(
    conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600,
) -> int:
    """Delete task_events rows older than ``older_than_seconds`` for tasks
    in a terminal state (``done`` or ``archived``). Returns the number of
    rows deleted. Running / ready / blocked tasks keep their full event
    history."""
    cutoff = int(time.time()) - int(older_than_seconds)
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND task_id IN "
            "(SELECT id FROM tasks WHERE status IN ('done', 'archived'))",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def gc_worker_logs(
    *, older_than_seconds: int = 30 * 24 * 3600,
    board: Optional[str] = None,
) -> int:
    """Delete worker log files older than ``older_than_seconds``. Returns
    the number of files removed. Kept separate from ``gc_events`` because
    log files live on disk, not in SQLite. Scoped to ``board`` (defaults
    to the active board) — per-board isolation means deleting logs from
    board A cannot touch board B's logs."""
    log_dir = worker_logs_dir(board=board)
    if not log_dir.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    removed = 0
    for p in log_dir.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Worker log accessor
# ---------------------------------------------------------------------------

def worker_log_path(task_id: str, *, board: Optional[str] = None) -> Path:
    """Return the path to a worker's log file. The file may not exist
    (task never spawned, or log already GC'd).

    When ``board`` is None, resolves via the active board (env var →
    current-board file → default). The dispatcher always passes the
    board explicitly to avoid any resolution ambiguity when multiple
    boards exist."""
    return worker_logs_dir(board=board) / f"{task_id}.log"


def read_worker_log(
    task_id: str, *, tail_bytes: Optional[int] = None,
    board: Optional[str] = None,
) -> Optional[str]:
    """Read the worker log for ``task_id``. Returns None if the file
    doesn't exist. If ``tail_bytes`` is set, only the last N bytes are
    returned (useful for the dashboard drawer which shouldn't page megabytes)."""
    path = worker_log_path(task_id, board=board)
    if not path.exists():
        return None
    try:
        if tail_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                # Skip a partial line if we tailed mid-line. But if the
                # window has no newline at all (one giant log line),
                # readline() would eat everything — in that case don't
                # skip and return the raw tail.
                probe = f.tell()
                partial = f.readline()
                if not partial.endswith(b"\n") and f.tell() >= size:
                    f.seek(probe)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Assignee enumeration (known profiles + per-profile board stats)
# ---------------------------------------------------------------------------

def list_profiles_on_disk() -> list[str]:
    """Return the set of assignee/profile names discovered on disk.

    Includes:
    - named profiles under ``<default-root>/profiles/<name>/config.yaml``
    - the implicit ``default`` profile when the default Hermes root exists

    Reads profile paths directly so this module has no import dependency on
    ``hermes_cli.profiles`` (which pulls in a large chunk of the CLI startup
    path).
    """
    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
        profiles_dir = default_root / "profiles"
    except Exception:
        return []

    names: set[str] = set()
    if default_root.exists():
        names.add("default")

    if profiles_dir.is_dir():
        try:
            for entry in sorted(profiles_dir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "config.yaml").is_file():
                    names.add(entry.name)
        except OSError:
            pass

    return sorted(names)


def known_assignees(conn: sqlite3.Connection) -> list[dict]:
    """Return every assignee name known to the board or on disk.

    Each entry is ``{"name": str, "on_disk": bool, "counts": {status: n}}``.
    A name is included when it's a configured profile on disk OR when
    any non-archived task has it as the assignee. Used by:

    - ``hermes kanban assignees`` for the terminal.
    - The dashboard assignee dropdown (so a fresh profile appears in
      the picker even before it's been given any task).
    - Router-profile heuristics ("who's overloaded?") without scanning
      the whole board.
    """
    on_disk = set(list_profiles_on_disk())

    # Count tasks per (assignee, status), excluding archived.
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    names = sorted(on_disk | set(counts.keys()))
    return [
        {
            "name": name,
            "on_disk": name in on_disk,
            "counts": counts.get(name, {}),
        }
        for name in names
    ]


# ---------------------------------------------------------------------------
# Runs (attempt history on a task)
# ---------------------------------------------------------------------------

def list_runs(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    include_active: bool = True,
    state_type: Optional[str] = None,
    state_name: Optional[str] = None,
) -> list[Run]:
    """Return all runs for ``task_id`` in start order.

    ``include_active=True`` (default) includes the currently-running
    attempt if any. Set False to return only closed runs (useful for
    "how many prior attempts have there been?" checks).

    When ``state_type`` and ``state_name`` are set, restrict to rows
    where that column equals ``state_name`` (``state_type`` is
    ``status`` or ``outcome``). Both must be passed together.
    """
    if (state_type is None) ^ (state_name is None):
        raise ValueError("state_type and state_name must both be set or both omitted")
    if state_type is not None:
        if state_type not in ("status", "outcome"):
            raise ValueError("state_type must be 'status' or 'outcome'")
    q = "SELECT * FROM task_runs WHERE task_id = ?"
    params: list[Any] = [task_id]
    if not include_active:
        q += " AND ended_at IS NOT NULL"
    if state_type is not None:
        q += f" AND {state_type} = ?"
        params.append(state_name)
    q += " ORDER BY started_at ASC, id ASC"
    rows = conn.execute(q, params).fetchall()
    return [Run.from_row(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[Run]:
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (int(run_id),),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_run(conn: sqlite3.Connection, task_id: str) -> Optional[Run]:
    """Return the most recent run regardless of outcome (active or closed)."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_summary(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the latest non-null ``task_runs.summary`` for ``task_id``.

    The worker writes its handoff to ``task_runs.summary``
    via ``complete_task(summary=...)``; ``tasks.result`` is left empty
    unless the caller passes ``result=`` explicitly. Dashboards and CLI
    "show" views need this value to surface what a worker actually did
    — without it, ``tasks.result`` is NULL and the task looks like a
    no-op even when the run completed.

    Picks the most recent run by ``ended_at`` (falling back to ``id``
    for ties or unfinished rows). Returns None if no run has a summary.
    """
    row = conn.execute(
        "SELECT summary FROM task_runs "
        "WHERE task_id = ? AND summary IS NOT NULL AND summary != '' "
        "ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row["summary"] if row else None


def latest_summaries(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, str]:
    """Batch-fetch latest non-null summaries for a list of task ids.

    Used by the dashboard board endpoint to attach ``latest_summary`` to
    every card in a single SQL query, avoiding the N+1 pattern of
    calling :func:`latest_summary` per task. Returns a dict mapping
    ``task_id`` → summary string, omitting tasks with no summary.

    Approach: a window function picks the newest non-null-summary row
    per ``task_id``; works against SQLite ≥ 3.25 (default on every
    supported platform).
    """
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT task_id, summary FROM (
            SELECT task_id, summary,
                   ROW_NUMBER() OVER (
                       PARTITION BY task_id
                       ORDER BY COALESCE(ended_at, started_at) DESC, id DESC
                   ) AS rn
              FROM task_runs
             WHERE task_id IN ({placeholders})
               AND summary IS NOT NULL AND summary != ''
        ) WHERE rn = 1
        """,
        ids,
    ).fetchall()
    return {r["task_id"]: r["summary"] for r in rows}

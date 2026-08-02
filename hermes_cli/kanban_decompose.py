"""Kanban decomposer — fan a triage task out into a graph of child tasks.

Invoked by ``hermes kanban decompose [task_id | --all]`` and the
auto-decompose path in the gateway dispatcher loop. Reads the user's
profile roster (with descriptions) and asks the auxiliary LLM to
return a task graph in JSON. Then atomically creates the children,
links them under the root, and flips the root ``triage -> todo``.

The root task stays alive and becomes the parent of every leaf child,
so when the whole graph completes the root wakes back up — its
assignee (the orchestrator profile) gets a chance to judge completion
and add more tasks if the work isn't done yet.

Design notes
------------

* Mirrors the shape of ``hermes_cli/kanban_specify.py``: lazy aux
  client import inside the function, lenient response parse, never
  raises on expected failure modes.

* The system prompt sees the *configured* profile roster — names plus
  descriptions plus the default fallback. Profiles without a
  description are still listed (with a note) so the decomposer can
  match on name as a fallback, but the user has an obvious incentive
  to describe them.

* ``fanout=false`` collapses to the same effect as ``kanban specify``:
  we tighten the body and flip ``triage -> todo`` as a single task,
  no children created. This makes ``decompose`` a strict superset of
  ``specify`` from the user's perspective.

* If the LLM picks an assignee that doesn't exist as a profile, we
  rewrite it to the configured ``default_assignee`` (or the default
  profile if unset). A child task NEVER ends up with ``assignee=None``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import profiles as profiles_mod

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are the Kanban decomposer for the Hermes Agent board.

A user dropped a rough idea into the Triage column. Your job is to break it
into a small graph of concrete child tasks and route each one to the best-
matching profile from the available roster.

You will be given:
  - The original task title and body
  - The list of available profiles (each with name + description)
  - The fallback "default_assignee" used when no profile fits

Output a single JSON object with this exact shape:

  {
    "fanout": true,
    "rationale": "<one sentence on why this decomposition>",
    "tasks": [
      {
        "title": "<concrete task title, imperative voice, <= 80 chars>",
        "body":  "<detailed spec for the worker on this child task>",
        "assignee": "<profile name from the roster, or null for default>",
        "parents": [<int>, ...]
      },
      ...
    ]
  }

Rules:
  - "parents" is a list of INDICES (0-based) into this same "tasks" list,
    expressing actual data dependencies. Tasks with no parents run in
    PARALLEL. Tasks with parents wait until every parent completes.
  - Prefer parallelism. If two tasks can be done independently, give
    them no parents so the dispatcher fans them out at once.
  - Use 2-6 tasks for normal work. Don't create 20 tiny tasks. Don't
    cram everything into 1 task.
  - Pick assignees from the roster by matching the task to the profile's
    DESCRIPTION (not just the name). When nothing matches well, use null
    and the system will route to the default_assignee.
  - Each child task body is what a fresh worker will read with no other
    context — be specific about goal, approach, and acceptance criteria.

When the task is genuinely a single unit of work (no useful decomposition),
return:

  {
    "fanout": false,
    "rationale": "<one sentence>",
    "title": "<tightened title>",
    "body":  "<concrete spec for a single worker>",
    "assignee": "<profile name from the roster, or null for default>"
  }

In that case the task stays as one work item, just with a tightened spec and
a concrete assignee. If no profile fits, use null and the system will route to
the default_assignee.

No preamble, no closing remarks, no code fences. Output only the JSON object.
"""


_USER_TEMPLATE = """Task id: {task_id}
Title: {title}
Body:
{body}

Available profiles (assignees you may pick from):
{roster}

Default assignee (used when no profile fits a task): {default_assignee}
"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class DecomposeOutcome:
    """Result of decomposing a single triage task."""

    task_id: str
    ok: bool
    reason: str = ""
    fanout: bool = False
    child_ids: list[str] | None = None
    new_title: Optional[str] = None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_json_blob(raw: str) -> Optional[dict]:
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(val, dict):
        return None
    return val


def _profile_author() -> str:
    """Mirror of ``hermes_cli.kanban._profile_author``."""
    return (
        os.environ.get("HERMES_PROFILE")
        or os.environ.get("USER")
        or "decomposer"
    )


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception:
        return {}


def _resolve_orchestrator_profile(cfg: dict) -> str:
    """Resolve which profile owns the root/orchestration task after fan-out.

    NEVER returns a human-driven profile (default/work): the root task
    reactivates to 'ready' once its children finish, and the dispatcher refuses
    to auto-spawn a human-driven assignee — an HD orchestrator would strand the
    whole decomposition with no completion. Order (each non-hd): explicit
    orchestrator_profile → active profile → first non-hd installed profile;
    the degenerate all-hd case keeps the active/"default" (dispatcher HD-guard
    keeps it visible) so a task is never stranded for lack of an orchestrator.
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        hd = kb.human_driven_profiles()
    except Exception:
        hd = frozenset()
    explicit = (kanban_cfg.get("orchestrator_profile") or "").strip()
    if explicit:
        try:
            explicit = profiles_mod.normalize_profile_name(explicit)
        except Exception:
            explicit = explicit.lower()
    if explicit and explicit in hd:
        logger.warning(
            "decompose: orchestrator_profile=%r is human-driven; ignoring "
            "(root task would never reactivate after fan-out)", explicit,
        )
    elif explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass
    try:
        active = profiles_mod.get_active_profile_name() or "default"
    except Exception:
        active = "default"
    if active and active not in hd:
        return active
    try:
        for p in profiles_mod.list_profiles():
            if p.name and p.name not in hd:
                return p.name
    except Exception:
        pass
    logger.warning(
        "decompose: no non-human-driven orchestrator available; falling back "
        "to %r (dispatcher HD-guard keeps the root task visible)", active,
    )
    return active


def _resolve_default_assignee(cfg: dict) -> str:
    """Resolve which profile catches child tasks the orchestrator can't route.

    NEVER returns a human-driven profile: it is the rewrite target for every
    LLM miss (empty/invalid/excluded choice), so a human-driven value would
    funnel unrouted children to an operator-driven profile the dispatcher then
    refuses to spawn — stranding them. Preference order, each candidate must be
    non-human-driven (explicit must also exist): explicit ``default_assignee``
    → ``orchestrator_profile`` → active profile → first non-human-driven
    installed profile. Children never end up unassigned, so in the degenerate
    case where EVERY candidate is human-driven the active/"default" name is
    kept with a warning (the dispatcher's HD-guard keeps such a child visible
    rather than silently spawning it).
    """
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        hd = kb.human_driven_profiles()
    except Exception:
        hd = frozenset()

    def _non_hd(name: str) -> bool:
        return bool(name) and name not in hd

    explicit = (kanban_cfg.get("default_assignee") or "").strip()
    if explicit:
        try:
            explicit = profiles_mod.normalize_profile_name(explicit)
        except Exception:
            explicit = explicit.lower()
    if explicit and explicit in hd:
        logger.warning(
            "decompose: default_assignee=%r is human-driven; ignoring "
            "(unrouted children will not fall back to a human-driven profile)",
            explicit,
        )
    elif explicit:
        try:
            if profiles_mod.profile_exists(explicit):
                return explicit
        except Exception:
            pass

    # Prefer the configured orchestrator (a real worker like 'pm') over the
    # active profile, which in the gateway/dispatcher runs under the 'default'
    # (human-driven) home and would otherwise leak here. Canonicalize before the
    # HD-check (mirroring `explicit` above): a raw casing like 'Work' would
    # otherwise slip past `_non_hd` and be returned, violating the NEVER-returns-
    # human-driven guarantee.
    orchestrator = (kanban_cfg.get("orchestrator_profile") or "").strip()
    if orchestrator:
        try:
            orchestrator = profiles_mod.normalize_profile_name(orchestrator)
        except Exception:
            orchestrator = orchestrator.lower()
    if _non_hd(orchestrator):
        try:
            if profiles_mod.profile_exists(orchestrator):
                return orchestrator
        except Exception:
            pass

    try:
        active = profiles_mod.get_active_profile_name() or "default"
    except Exception:
        active = "default"
    if _non_hd(active):
        return active

    # active is human-driven (e.g. dispatcher under the 'default' home): take
    # the first non-human-driven installed profile as a safe catch-all.
    try:
        for p in profiles_mod.list_profiles():
            if _non_hd(p.name):
                return p.name
    except Exception:
        pass

    logger.warning(
        "decompose: no non-human-driven default_assignee available; falling "
        "back to %r (dispatcher HD-guard keeps such children visible)", active,
    )
    return active


def _build_roster() -> tuple[list[dict], set[str]]:
    """Return (roster_for_prompt, valid_assignee_names).

    Each roster entry is ``{name, description, has_description}``. The
    valid-set is used after the LLM responds to rewrite invalid
    assignees to the default fallback.
    """
    roster: list[dict] = []
    valid: set[str] = set()
    try:
        all_profiles = profiles_mod.list_profiles()
    except Exception as exc:
        logger.warning("decompose: failed to list profiles: %s", exc)
        return roster, valid
    # Human-driven profiles (default/work class) are operator-driven and must
    # never receive autonomously-routed work. Exclude them from BOTH the
    # roster the LLM sees AND the valid set, so if the model names one anyway
    # `_normalize_assignee_choice` rewrites it to the default fallback.
    try:
        hd = kb.human_driven_profiles()
    except Exception:
        hd = frozenset()
    for p in all_profiles:
        if p.name in hd:
            continue
        desc = (p.description or "").strip()
        roster.append({
            "name": p.name,
            "description": desc or f"(no description; profile named {p.name!r})",
            "has_description": bool(desc),
        })
        valid.add(p.name)
    return roster, valid


def _format_roster(roster: list[dict]) -> str:
    if not roster:
        return "  (no profiles installed — decomposer cannot route work)"
    lines = []
    for entry in roster:
        tag = "" if entry["has_description"] else " ⚠ undescribed"
        lines.append(f"  - {entry['name']}{tag}: {entry['description']}")
    return "\n".join(lines)


def _normalize_assignee_choice(
    assignee: object,
    *,
    default_assignee: str,
    valid_names: set[str],
) -> str:
    """Return a valid assignee, falling back to ``default_assignee``.

    Fan-out children and the single-task fallback should share the same
    routing guarantee: promoted work must not be left unassigned.
    """
    if not isinstance(assignee, str) or not assignee.strip():
        return default_assignee
    chosen = assignee.strip()
    if chosen not in valid_names:
        return default_assignee
    return chosen


def decompose_task(
    task_id: str,
    *,
    author: Optional[str] = None,
    timeout: Optional[int] = None,
) -> DecomposeOutcome:
    """Decompose a triage task into a graph of child tasks.

    Returns an outcome describing what happened. Never raises for
    expected failure modes (task not in triage, no aux client
    configured, API error, malformed response, decomposer returned
    fanout=true with empty task list) — those surface via ``ok=False``.
    """
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        pending_action = kb.get_pending_action(conn, task_id)
    if task is None:
        return DecomposeOutcome(task_id, False, "unknown task id")
    if task.status != "triage":
        return DecomposeOutcome(
            task_id, False, f"task is not in triage (status={task.status!r})"
        )
    if pending_action is not None:
        return DecomposeOutcome(
            task_id, False, "task has unresolved exact terminal action approval"
        )

    cfg = _load_config()
    orchestrator = _resolve_orchestrator_profile(cfg)
    default_assignee = _resolve_default_assignee(cfg)
    kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    auto_promote = bool(kanban_cfg.get("auto_promote_children", True))

    # Decompose guardrails (2026-07-02, landscape-research #3). Generous safety
    # nets that only fire on genuine runaway; a value <= 0 (or non-int) disables
    # that cap. See decompose_triage_task for enforcement semantics.
    def _cap(key: str, default: int) -> Optional[int]:
        try:
            val = int(kanban_cfg.get(key, default))
        except (TypeError, ValueError):
            return default
        return val if val >= 1 else None

    max_fanout = _cap("max_fanout", 12)
    max_tasks_per_root = _cap("max_tasks_per_root", 60)
    roster, valid_names = _build_roster()

    try:
        from agent.auxiliary_client import call_llm  # type: ignore
    except Exception as exc:
        logger.debug("decompose: auxiliary client import failed: %s", exc)
        return DecomposeOutcome(task_id, False, "auxiliary client unavailable")

    # local(tars) Class-specific planner routing (Quality-Class-Routing,
    # 2026-07-09), re-expressed over upstream's call_llm refactor (#35566):
    # a task tagged ``task_class=X`` uses the auxiliary role
    # ``kanban_decomposer_<x>`` IFF that role is configured; otherwise the
    # default ``kanban_decomposer``. The existence check is essential —
    # passing an unconfigured role name to call_llm would route to "auto"
    # (main model), not to the default decomposer.
    decomposer_role = "kanban_decomposer"
    if task.task_class:
        candidate = "kanban_decomposer_" + task.task_class.strip().lower()
        aux_cfg = cfg.get("auxiliary", {}) if isinstance(cfg, dict) else {}
        if isinstance(aux_cfg, dict) and isinstance(aux_cfg.get(candidate), dict):
            decomposer_role = candidate
            logger.debug(
                "decompose: task %s class=%r -> planner role %r",
                task.id, task.task_class, decomposer_role,
            )

    user_msg = _USER_TEMPLATE.format(
        task_id=task.id,
        title=_truncate(task.title or "", 400),
        body=_truncate(task.body or "(no body)", 4000),
        roster=_format_roster(roster),
        default_assignee=default_assignee,
    )

    def _call_decomposer(role: str):
        # Route through call_llm so auxiliary.<role>.* config
        # (provider/model/base_url, extra_body, reasoning_effort, retries)
        # all apply — the previous direct client.chat.completions.create()
        # path dropped auxiliary.<task>.extra_body entirely (#35566).
        return call_llm(
            task=role,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4000,
            timeout=timeout or 180,
        )

    try:
        resp = _call_decomposer(decomposer_role)
    except Exception as exc:
        # local(tars) runtime fallback: a failing class-specific planner
        # (quota gone, endpoint down) must not make ``class=hard`` tasks MORE
        # fragile than unclassified ones — retry once on the default role.
        if decomposer_role != "kanban_decomposer":
            logger.warning(
                "decompose: class planner role %r failed for %s (%s); "
                "falling back to default decomposer",
                decomposer_role, task_id, exc,
            )
            try:
                resp = _call_decomposer("kanban_decomposer")
            except Exception as exc2:
                logger.info(
                    "decompose: API call failed for %s (%s)", task_id, exc2,
                )
                return DecomposeOutcome(
                    task_id, False, f"LLM error: {type(exc2).__name__}"
                )
        else:
            logger.info(
                "decompose: API call failed for %s (%s)", task_id, exc,
            )
            return DecomposeOutcome(task_id, False, f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = _extract_json_blob(raw)
    if parsed is None:
        return DecomposeOutcome(task_id, False, "LLM returned malformed JSON")

    fanout = bool(parsed.get("fanout"))
    audit_author = author or _profile_author()

    if not fanout:
        # Fall back to single-task spec promotion (same effect as specify).
        new_title = parsed.get("title")
        new_body = parsed.get("body")
        title_val = new_title.strip() if isinstance(new_title, str) and new_title.strip() else None
        body_val = new_body if isinstance(new_body, str) and new_body.strip() else None
        assignee_val = None
        if not task.assignee:
            assignee_val = _normalize_assignee_choice(
                parsed.get("assignee"),
                default_assignee=default_assignee,
                valid_names=valid_names,
            )
        if title_val is None and body_val is None:
            return DecomposeOutcome(
                task_id, False, "decomposer returned fanout=false with no title/body",
            )
        with kb.connect_closing() as conn:
            ok = kb.specify_triage_task(
                conn,
                task_id,
                title=title_val,
                body=body_val,
                assignee=assignee_val,
                author=audit_author,
            )
        if not ok:
            return DecomposeOutcome(
                task_id, False, "task moved out of triage before promotion",
            )
        return DecomposeOutcome(
            task_id, True, "single task (no fanout)",
            fanout=False, new_title=title_val,
        )

    raw_tasks = parsed.get("tasks") or []
    if not isinstance(raw_tasks, list) or not raw_tasks:
        return DecomposeOutcome(
            task_id, False, "decomposer returned fanout=true with empty tasks list",
        )

    # Rewrite invalid assignees to the default fallback. Never leave a
    # task with assignee=None — the user explicitly does not want that.
    children: list[dict] = []
    for idx, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}] is not an object",
            )
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            return DecomposeOutcome(
                task_id, False, f"tasks[{idx}].title is missing or empty",
            )
        body = entry.get("body")
        if not isinstance(body, str):
            body = ""
        assignee = entry.get("assignee")
        chosen = _normalize_assignee_choice(
            assignee,
            default_assignee=default_assignee,
            valid_names=valid_names,
        )
        if (
            isinstance(assignee, str)
            and assignee.strip()
            and assignee.strip() not in valid_names
        ):
            logger.info(
                "decompose: task %s child %d picked unknown assignee %r — "
                "routing to default_assignee %r",
                task_id, idx, assignee, default_assignee,
            )
        parents = entry.get("parents") or []
        if not isinstance(parents, list):
            parents = []
        # Clean parent indices: drop non-int and out-of-range.
        clean_parents = [p for p in parents if isinstance(p, int) and 0 <= p < len(raw_tasks) and p != idx]
        children.append({
            "title": title.strip()[:200],
            "body": body.strip(),
            "assignee": chosen,
            "parents": clean_parents,
        })

    try:
        with kb.connect_closing() as conn:
            child_ids = kb.decompose_triage_task(
                conn,
                task_id,
                root_assignee=orchestrator,
                children=children,
                author=audit_author,
                auto_promote=auto_promote,
                max_fanout=max_fanout,
                max_tasks_per_root=max_tasks_per_root,
            )
    except ValueError as exc:
        return DecomposeOutcome(task_id, False, f"DB rejected graph: {exc}")
    except Exception as exc:
        logger.exception("decompose: DB error on task %s", task_id)
        return DecomposeOutcome(task_id, False, f"DB error: {type(exc).__name__}")

    if child_ids is None:
        return DecomposeOutcome(
            task_id, False, "task moved out of triage before decomposition",
        )

    return DecomposeOutcome(
        task_id, True, f"decomposed into {len(child_ids)} children",
        fanout=True, child_ids=child_ids,
    )


def list_triage_ids(*, tenant: Optional[str] = None) -> list[str]:
    """Return task ids currently in the triage column."""
    with kb.connect_closing() as conn:
        rows = kb.list_tasks(
            conn,
            status="triage",
            tenant=tenant,
            limit=1000,
        )
        return [
            row.id for row in rows
            if kb.get_pending_action(conn, row.id) is None
            and kb.triage_auto_eligible(row)
        ]

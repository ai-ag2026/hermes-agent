"""Lightweight, dependency-free shape validation for orchestration payloads.

Used by ``kanban_decompose`` (aux-LLM task graph) and ``kanban_swarm`` (worker
blackboard updates) to enforce a minimal structural contract and drive bounded
retries — mirroring the schema-forced structured returns of the Claude Code
Workflow harness (audit finding CC-PARITY-A2).

Deliberately NOT full JSON-Schema: the orchestration hot path must never grow a
hard third-party dependency, and all we actually need is presence + type +
non-empty checks. Every function returns a list of human-readable error strings
(empty list == valid), so callers can feed the errors straight back to an LLM as
corrective feedback.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(getattr(t, "__name__", str(t)) for t in expected)
    return getattr(expected, "__name__", str(expected))


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def validate_fields(payload: Any, spec: Dict[str, Dict[str, Any]]) -> List[str]:
    """Validate ``payload`` against a simple field spec.

    ``spec`` maps a field name to a rule dict with optional keys:
      * ``type``      — a type or tuple of types the value must be an instance of.
      * ``required``  — bool (default True); if False the field may be absent.
      * ``non_empty`` — bool (default False); reject empty str/list/dict/None.

    Note ``bool`` is a subclass of ``int`` in Python; specs that need a real int
    should not accept bools implicitly — pass ``type=int`` knowing bools slip
    through, or check separately. Returns a list of error strings.
    """
    if not isinstance(payload, dict):
        return [f"expected a JSON object, got {type(payload).__name__}"]

    errors: List[str] = []
    for field, rule in spec.items():
        required = rule.get("required", True)
        if field not in payload:
            if required:
                errors.append(f"missing required field '{field}'")
            continue
        value = payload[field]
        expected = rule.get("type")
        if expected is not None and not isinstance(value, expected):
            errors.append(
                f"field '{field}' must be {_type_name(expected)}, "
                f"got {type(value).__name__}"
            )
            continue
        if rule.get("non_empty") and _is_empty(value):
            errors.append(f"field '{field}' must not be empty")
    return errors


def validate_decompose_graph(payload: Any) -> List[str]:
    """Validate a Kanban decomposer task-graph payload.

    Mirrors the structural conditions ``decompose_task`` already enforces inline,
    so enabling retry never changes which payloads are accepted — it only gives
    the aux LLM a chance to fix an invalid one before we give up.
    """
    if not isinstance(payload, dict):
        return [f"expected a JSON object, got {type(payload).__name__}"]

    if bool(payload.get("fanout")):
        errors = validate_fields(payload, {"tasks": {"type": list, "non_empty": True}})
        if errors:
            return errors
        for idx, entry in enumerate(payload["tasks"]):
            for err in validate_fields(entry, {"title": {"type": str, "non_empty": True}}):
                errors.append(f"tasks[{idx}]: {err}")
        return errors

    # fanout=false → single-task promotion needs at least a title or a body.
    title = payload.get("title")
    body = payload.get("body")
    has_title = isinstance(title, str) and title.strip()
    has_body = isinstance(body, str) and body.strip()
    if not has_title and not has_body:
        return ["fanout=false requires a non-empty 'title' or 'body'"]
    return []

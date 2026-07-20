"""Idempotent promotion of producer events into work items (D1).

The problem
-----------
Four producers can create work: conversations (``delegate_task``), cron,
background processes and ACP. Each of them retries. Today a retry that lands
after a partial failure creates a *second* card, because the dedup check is a
read followed by an insert -- and between those two statements another caller
can do the same thing.

D0 replaced that with a unique index on ``(origin_kind, origin_key)``. This
module is what actually uses it: promotion happens as a single INSERT whose
conflict is resolved by SQLite, not by the caller's earlier read.

The distinction that matters
----------------------------
Two different things look like a duplicate:

* **The same request, retried.** Same key, same payload digest. The right
  answer is to return the existing work item and report that nothing new was
  created. Silently succeeding is correct here.
* **A different request reusing a key.** Same key, *different* payload
  digest. This is a bug or a collision at the producer, and the right answer
  is to fail loudly. Treating it as a retry would silently discard the second
  request -- the work would be accepted and never done.

Conflating these two is the failure mode this module exists to prevent, which
is why ``payload_digest`` is compared rather than merely stored.

Scope
-----
D1 ends at durable producer completion, idempotent promotion and review
state. There is no core inbox -- that arrives in F3/F4. The existing
compatibility wake stays best-effort, and this module does not pretend
otherwise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VALID_ORIGIN_KINDS = frozenset({"conversation", "cron", "process", "webhook", "acp"})


class OriginKeyConflict(Exception):
    """Same origin key, different payload.

    Loud on purpose. The alternative -- treating it as a retry -- accepts a
    request and then never performs it.
    """

    def __init__(self, origin_kind: str, origin_key: str,
                 existing_task_id: str, existing_digest: Optional[str],
                 incoming_digest: str):
        super().__init__(
            f"origin_key {origin_kind}:{origin_key} already belongs to "
            f"{existing_task_id} with a different payload "
            f"(existing digest {existing_digest}, incoming {incoming_digest}). "
            "Refusing to silently discard the incoming request."
        )
        self.origin_kind = origin_kind
        self.origin_key = origin_key
        self.existing_task_id = existing_task_id
        self.existing_digest = existing_digest
        self.incoming_digest = incoming_digest


@dataclass(frozen=True)
class PromotionResult:
    task_id: str
    created: bool           # False => an existing item was returned
    origin_kind: str
    origin_key: str
    payload_digest: str

    @property
    def deduplicated(self) -> bool:
        return not self.created


def payload_digest(payload: Any) -> str:
    """Stable digest of a producer payload.

    ``sort_keys`` matters: two dicts that differ only in insertion order are
    the same request, and a digest that disagreed would turn every retry into
    a spurious conflict.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _existing_by_origin(conn: sqlite3.Connection, origin_kind: str,
                        origin_key: str) -> Optional[sqlite3.Row]:
    row = conn.execute(
        "SELECT id, payload_digest FROM tasks "
        "WHERE origin_kind = ? AND origin_key = ?",
        (origin_kind, origin_key),
    ).fetchone()
    return row


def promote(
    conn: sqlite3.Connection,
    *,
    origin_kind: str,
    origin_key: str,
    title: str,
    payload: Any = None,
    body: Optional[str] = None,
    work_kind: str = "one_shot",
    owner_core_id: Optional[str] = None,
    origin_conversation_ref: Optional[str] = None,
    acceptance_required: bool = False,
    definition_of_done: Optional[str] = None,
    commitment: Optional[str] = None,
    commitment_due_at: Optional[int] = None,
    tenant: Optional[str] = None,
    **create_kwargs: Any,
) -> PromotionResult:
    """Promote a producer event into a work item, exactly once.

    Returns the existing item when the same request is retried; raises
    ``OriginKeyConflict`` when the key is reused for a different payload.
    """
    if origin_kind not in VALID_ORIGIN_KINDS:
        raise ValueError(
            f"unknown origin_kind {origin_kind!r}; "
            f"expected one of {sorted(VALID_ORIGIN_KINDS)}"
        )
    if not origin_key:
        # A promotion without a key cannot be deduplicated, so accepting one
        # would quietly reintroduce the duplicate-card problem for that
        # producer. Callers that genuinely have no stable key must say so by
        # using create_task directly.
        raise ValueError("origin_key is required for idempotent promotion")

    from hermes_cli.kanban_db import create_task

    digest = payload_digest(payload)

    existing = _existing_by_origin(conn, origin_kind, origin_key)
    if existing is not None:
        return _resolve_existing(existing, origin_kind, origin_key, digest)

    # The contract fields travel INSIDE ``create_task``'s own INSERT rather
    # than in a follow-up UPDATE. An UPDATE would leave a window in which the
    # card exists without its origin key -- and therefore without the
    # uniqueness that makes promotion idempotent at all. Wrapping the call in
    # an outer transaction is not an option either: ``write_txn`` is not
    # reentrant and ``create_task`` opens its own. (This is the contradiction
    # ADR-9 flagged: "same transaction" and "no core change needed" could not
    # both hold. Extending the INSERT is the resolution.)
    work_contract = {
        "origin_kind": origin_kind,
        "origin_key": origin_key,
        "payload_digest": digest,
        "work_kind": work_kind,
        "owner_core_id": owner_core_id,
        "origin_conversation_ref": origin_conversation_ref,
        "acceptance_required": bool(acceptance_required),
        "definition_of_done": definition_of_done,
        "commitment": commitment,
        "commitment_due_at": commitment_due_at,
        "route_tenant_snapshot": tenant,
        "authority_epoch": 0,
    }
    try:
        task_id = create_task(
            conn, title=title, body=body, tenant=tenant,
            work_contract=work_contract, **create_kwargs
        )
    except sqlite3.IntegrityError:
        # Another writer won the race between our SELECT and this INSERT.
        # That is the expected outcome, not an error: the unique index did its
        # job. Re-read and treat it exactly like any other retry.
        existing = _existing_by_origin(conn, origin_kind, origin_key)
        if existing is None:
            raise
        logger.debug(
            "promotion race on %s:%s resolved by the unique index",
            origin_kind, origin_key,
        )
        return _resolve_existing(existing, origin_kind, origin_key, digest)

    logger.info("promoted %s:%s -> %s", origin_kind, origin_key, task_id)
    return PromotionResult(task_id=task_id, created=True, origin_kind=origin_kind,
                           origin_key=origin_key, payload_digest=digest)


def _resolve_existing(row: sqlite3.Row, origin_kind: str, origin_key: str,
                      digest: str) -> PromotionResult:
    existing_id = row[0] if not isinstance(row, sqlite3.Row) else row["id"]
    existing_digest = row[1] if not isinstance(row, sqlite3.Row) else row["payload_digest"]

    if existing_digest is not None and existing_digest != digest:
        raise OriginKeyConflict(origin_kind, origin_key, existing_id,
                                existing_digest, digest)

    return PromotionResult(task_id=existing_id, created=False,
                           origin_kind=origin_kind, origin_key=origin_key,
                           payload_digest=digest)

"""Kanban invariant reconciler — read-only audit + strictly allowlisted repair.

Card t_0cb8668d ("S5 Invariant-Reconciler"). Task, run, event, claim and
durable completion-artifact state form one logical projection that can
drift out of sync with itself: a crash between two statements outside a
transaction, a manual DB edit, a table-rebuild bug, or a stale worker
still holding a superseded ``expected_run_id``. This module answers two
separate questions:

* :func:`run_audit` — **read-only**. Walks every non-archived-forever
  invariant and returns a typed :class:`Finding` for each violation. Never
  writes to the database.
* :func:`run_repair` — dry-run by default. Re-derives the same findings
  and, for the strict allowlist of *mechanically unambiguous* cases,
  applies an idempotent, CAS-guarded fix inside its own transaction and
  records a ``kanban_repair_applied`` audit event. Everything else is
  left exactly as found — a typed finding routed to ``bucket="triage"``
  for a human, never an optimistic promotion or a fabricated completion.

Design constraints (from the card):

* No heuristic completion from prose. A "done" transition is only ever
  reconstructed from **durable, already-validated** evidence that was
  provably produced atomically by an earlier ``complete_task``/
  ``decide_task_review`` call (the closed run's ``outcome`` + the
  matching ``completed`` event + intact artifact bytes) — never from
  a judge verdict or free-text summary.
* Every repair is idempotent: running it twice produces zero additional
  changes the second time, because each repair re-checks its own
  precondition via a CAS ``UPDATE ... WHERE`` (or an ``INSERT`` guarded
  by the same uniqueness the schema already enforces) inside a single
  ``kanban_db.write_txn``. Two concurrent repair passes (or a repair
  racing the live dispatcher) can therefore have at most one winner per
  row; the loser observes rowcount 0 and reports "precondition changed"
  rather than erroring or double-applying.
* This module never touches more than one board's connection at a time
  and never runs a naive fleet-wide bulk write — callers loop over
  boards themselves (see ``hermes kanban audit --all-boards`` /
  ``hermes kanban repair --all-boards``), each iteration opening and
  closing its own connection, so there is exactly one writer per board
  per call, matching the rest of this codebase's board-scoping model.
* Where a fix already exists elsewhere in the kernel (``recompute_ready``
  for stale ``todo``/``blocked`` promotion, ``_has_sticky_block`` for the
  sticky-block exclusion, ``_append_event``/``write_txn`` for the audit
  trail), this module calls that code directly instead of re-implementing
  it, so the reconciler can never disagree with the dispatcher about when
  a promotion is safe.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb

# ---------------------------------------------------------------------------
# Typed findings / reports
# ---------------------------------------------------------------------------

# A finding either has a mechanically-unambiguous, allowlisted repair
# (``safe_repair`` names it and ``bucket == "safe_repair"``), or it doesn't
# and is routed to a human (``bucket == "triage"``, ``safe_repair is None``).
# There is no third bucket: the reconciler never auto-applies anything that
# isn't on the allowlist below.
FindingBucket = str  # "safe_repair" | "triage"


@dataclass
class Finding:
    """One classified invariant violation."""

    kind: str
    task_id: str
    bucket: FindingBucket
    detail: str
    safe_repair: Optional[str] = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "bucket": self.bucket,
            "detail": self.detail,
            "safe_repair": self.safe_repair,
            "data": self.data,
        }


@dataclass
class AuditReport:
    findings: list[Finding]
    scanned_tasks: int
    board: Optional[str] = None
    generated_at: int = 0

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "board": self.board,
            "generated_at": self.generated_at,
            "scanned_tasks": self.scanned_tasks,
            "findings": [f.to_dict() for f in self.findings],
            "counts_by_kind": self.counts_by_kind(),
        }


@dataclass
class RepairResult:
    finding: Finding
    applied: bool
    reason: Optional[str] = None  # why not applied, when applied=False

    def to_dict(self) -> dict:
        return {
            "finding": self.finding.to_dict(),
            "applied": self.applied,
            "reason": self.reason,
        }


@dataclass
class RepairReport:
    dry_run: bool
    results: list[RepairResult]
    board: Optional[str] = None
    generated_at: int = 0

    def applied_count(self) -> int:
        return sum(1 for r in self.results if r.applied)

    def to_dict(self) -> dict:
        return {
            "board": self.board,
            "dry_run": self.dry_run,
            "generated_at": self.generated_at,
            "applied_count": self.applied_count(),
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Shared evidence helpers
# ---------------------------------------------------------------------------

def _verify_manifest_entry_on_disk(entry: dict) -> tuple[bool, Optional[str]]:
    """Re-validate one completion-artifact manifest entry against disk.

    Never trusts a cached hash: re-reads the file and recomputes sha256 +
    size, exactly like ``_copy_artifact_atomically`` did the moment the
    artifact was promoted. Returns ``(ok, reason_if_not)``.
    """
    durable_path = entry.get("durable_path")
    if not durable_path:
        return False, "manifest entry has no durable_path"
    p = Path(durable_path)
    try:
        if not p.is_file():
            return False, "durable artifact file is missing"
        digest = hashlib.sha256()
        size = 0
        with open(p, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        return False, f"durable artifact unreadable: {exc}"
    expected_size = entry.get("size")
    if expected_size is not None and size != int(expected_size):
        return False, "durable artifact size mismatch"
    expected_sha = entry.get("sha256")
    if expected_sha and digest.hexdigest() != expected_sha:
        return False, "durable artifact sha256 mismatch"
    return True, None


def _completed_event_manifest(conn, task_id: str) -> list[dict]:
    """Return the artifact_manifest recorded on the task's 'completed' event.

    The event log is append-only and immutable once written, so this is a
    second, independent durable source of truth for what was promoted at
    completion time -- distinct from (and useful for reconstructing) the
    mutable ``task_artifacts`` table.
    """
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'completed' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return []
    import json

    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return []
    manifest = payload.get("artifact_manifest") if isinstance(payload, dict) else None
    return manifest if isinstance(manifest, list) else []


def _has_completion_evidence(conn, task_id: str) -> bool:
    """A 'done' task has completion evidence iff either durable source of
    truth for a successful terminal transition exists:

    * a closed run whose ``outcome == 'completed'`` (the ``complete_task``
      path), or
    * a ``completed`` task_event (also covers ``decide_task_review``'s
      ACCEPT path, which ends the run with ``outcome='accept'`` but always
      appends a ``completed`` event).
    """
    run_row = conn.execute(
        "SELECT 1 FROM task_runs WHERE task_id = ? AND outcome = 'completed' LIMIT 1",
        (task_id,),
    ).fetchone()
    if run_row is not None:
        return True
    ev_row = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'completed' LIMIT 1",
        (task_id,),
    ).fetchone()
    return ev_row is not None


def _latest_goal_progress_shows_closeout_evidence(conn, task_id: str) -> Optional[dict]:
    """Return the latest goal_progress payload if it shows the goal-loop's
    judge believed the work was finished (closeout/finalize was attempted),
    else None.

    This is *not* heuristic completion: ``nudged_to_finalize`` /
    ``closeout_turns_used`` are structured counters the goal loop itself
    persists via :func:`kanban_db.record_goal_progress_event` -- durable
    telemetry, not free text -- so surfacing them is classification, not
    fabrication. The reconciler never uses this to flip status; it only
    ever produces a triage-only finding.
    """
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'goal_progress' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    import json

    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("nudged_to_finalize") or (payload.get("closeout_turns_used") or 0) > 0:
        return payload
    return None


# ---------------------------------------------------------------------------
# Audit: one function per invariant class
# ---------------------------------------------------------------------------

def _audit_run_pointer_invariants(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute(
        "SELECT id, status, current_run_id FROM tasks "
        "WHERE current_run_id IS NOT NULL"
    ).fetchall()
    for row in rows:
        task_id = row["id"]
        status = row["status"]
        run_id = int(row["current_run_id"])
        run_row = conn.execute(
            "SELECT ended_at FROM task_runs WHERE id = ?", (run_id,),
        ).fetchone()
        run_is_live = run_row is not None and run_row["ended_at"] is None
        if status == "running":
            if run_row is None or not run_is_live:
                findings.append(
                    Finding(
                        kind="running_task_missing_live_run",
                        task_id=task_id,
                        bucket="safe_repair",
                        safe_repair="reclaim_orphaned_running_task",
                        detail=(
                            "task is 'running' but its current_run_id does not "
                            "point to a live (unended) run row"
                        ),
                        data={"current_run_id": run_id},
                    )
                )
        else:
            # Every non-running transition clears current_run_id in the
            # same transaction that flips status; a non-NULL pointer here
            # is always drift, whether or not the pointed-to run is live.
            findings.append(
                Finding(
                    kind="nonrunning_task_dangling_run_pointer",
                    task_id=task_id,
                    bucket="safe_repair",
                    safe_repair="clear_dangling_run_pointer",
                    detail=(
                        f"task status is {status!r} but current_run_id={run_id} "
                        "is still set"
                    ),
                    data={"current_run_id": run_id, "run_is_live": run_is_live},
                )
            )
    return findings


def _audit_multiple_open_runs(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute(
        "SELECT task_id, COUNT(*) AS n FROM task_runs "
        "WHERE ended_at IS NULL GROUP BY task_id HAVING COUNT(*) > 1"
    ).fetchall()
    for row in rows:
        task_id = row["task_id"]
        open_runs = conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL "
            "ORDER BY id",
            (task_id,),
        ).fetchall()
        open_ids = [int(r["id"]) for r in open_runs]
        trow = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        current = int(trow["current_run_id"]) if trow and trow["current_run_id"] else None
        if current is not None and current in open_ids:
            findings.append(
                Finding(
                    kind="multiple_open_runs",
                    task_id=task_id,
                    bucket="safe_repair",
                    safe_repair="close_duplicate_open_runs",
                    detail=(
                        f"{len(open_ids)} open run rows for one task; "
                        f"current_run_id={current} unambiguously identifies the survivor"
                    ),
                    data={"open_run_ids": open_ids, "current_run_id": current},
                )
            )
        else:
            findings.append(
                Finding(
                    kind="multiple_open_runs",
                    task_id=task_id,
                    bucket="triage",
                    detail=(
                        f"{len(open_ids)} open run rows for one task and "
                        "current_run_id does not unambiguously identify a survivor"
                    ),
                    data={"open_run_ids": open_ids, "current_run_id": current},
                )
            )
    return findings


def _audit_stale_promotable(conn) -> list[Finding]:
    """Mirror ``recompute_ready``'s own promotion decision, read-only.

    Reuses ``_has_sticky_block`` and the same failure-limit resolution
    order so this can never disagree with the dispatcher about whether a
    promotion is safe.
    """
    findings: list[Finding] = []
    rows = conn.execute(
        "SELECT id, status, consecutive_failures, max_retries "
        "FROM tasks WHERE status IN ('todo', 'blocked')"
    ).fetchall()
    for row in rows:
        task_id = row["id"]
        status = row["status"]
        if status == "blocked" and kb._has_sticky_block(conn, task_id):
            continue
        parents = conn.execute(
            "SELECT t.status FROM tasks t "
            "JOIN task_links l ON l.parent_id = t.id "
            "WHERE l.child_id = ?",
            (task_id,),
        ).fetchall()
        if not all(p["status"] in ("done", "archived") for p in parents):
            continue
        if status == "blocked":
            failures = int(row["consecutive_failures"] or 0)
            task_limit = row["max_retries"]
            effective_limit = (
                int(task_limit) if task_limit is not None else kb.DEFAULT_FAILURE_LIMIT
            )
            if failures >= effective_limit:
                continue
        findings.append(
            Finding(
                kind="stale_promotable_task",
                task_id=task_id,
                bucket="safe_repair",
                safe_repair="promote_stale_task",
                detail=(
                    f"task is {status!r} with all parents terminal but was "
                    "never promoted to 'ready'"
                ),
                data={"from_status": status},
            )
        )
    return findings


def _audit_done_completion_evidence(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute("SELECT id FROM tasks WHERE status = 'done'").fetchall()
    for row in rows:
        task_id = row["id"]
        if not _has_completion_evidence(conn, task_id):
            findings.append(
                Finding(
                    kind="done_missing_completion_evidence",
                    task_id=task_id,
                    bucket="triage",
                    detail=(
                        "task is 'done' but neither a completed run outcome "
                        "nor a 'completed' event exists"
                    ),
                )
            )
    return findings


def _audit_done_artifact_manifests(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute("SELECT id FROM tasks WHERE status = 'done'").fetchall()
    for row in rows:
        task_id = row["id"]
        db_rows = conn.execute(
            "SELECT * FROM task_artifacts WHERE task_id = ?", (task_id,),
        ).fetchall()
        db_keys = {
            (r["producer_run_id"], r["original_path"], r["sha256"]) for r in db_rows
        }
        invalid: list[dict] = []
        for r in db_rows:
            ok, why = _verify_manifest_entry_on_disk(dict(r))
            if not ok:
                invalid.append({"durable_path": r["durable_path"], "reason": why})

        event_manifest = _completed_event_manifest(conn, task_id)
        lost_but_valid: list[dict] = []
        lost_and_invalid: list[dict] = []
        for entry in event_manifest:
            key = (entry.get("producer_run_id"), entry.get("original_path"), entry.get("sha256"))
            if key in db_keys:
                continue  # already reconciled / present
            ok, why = _verify_manifest_entry_on_disk(entry)
            if ok:
                lost_but_valid.append(entry)
            else:
                lost_and_invalid.append({"durable_path": entry.get("durable_path"), "reason": why})

        if invalid or lost_and_invalid:
            findings.append(
                Finding(
                    kind="done_artifact_manifest_invalid",
                    task_id=task_id,
                    bucket="triage",
                    detail=(
                        "one or more completion artifacts no longer match "
                        "their recorded manifest (missing file, size or "
                        "sha256 mismatch) — cannot fabricate lost bytes"
                    ),
                    data={"invalid": invalid + lost_and_invalid},
                )
            )
        if lost_but_valid:
            findings.append(
                Finding(
                    kind="done_artifact_manifest_lost",
                    task_id=task_id,
                    bucket="safe_repair",
                    safe_repair="reattach_artifact_manifest",
                    detail=(
                        "the 'completed' event's artifact_manifest references "
                        "entries with no task_artifacts row, but the durable "
                        "file still exists and matches the recorded hash — "
                        "safe to reattach"
                    ),
                    data={"entries": lost_but_valid},
                )
            )
    return findings


def _audit_orphaned_completed_runs(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute(
        "SELECT id, status, current_run_id FROM tasks "
        "WHERE status NOT IN ('done', 'archived')"
    ).fetchall()
    for row in rows:
        task_id = row["id"]
        latest = conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if latest is None or latest["ended_at"] is None or latest["outcome"] != "completed":
            continue
        run_id = int(latest["id"])
        ev = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'completed' "
            "AND run_id = ? LIMIT 1",
            (task_id, run_id),
        ).fetchone()
        if ev is None:
            continue
        current_run_id = row["current_run_id"]
        if current_run_id is not None and int(current_run_id) != run_id:
            # A newer run pointer exists — not orphaned, a real retry is
            # (or was) in flight. Leave it alone entirely.
            continue

        manifest = _completed_event_manifest(conn, task_id)
        invalid_entries = []
        for entry in manifest:
            ok, why = _verify_manifest_entry_on_disk(entry)
            if not ok:
                invalid_entries.append({"durable_path": entry.get("durable_path"), "reason": why})

        if invalid_entries:
            findings.append(
                Finding(
                    kind="nonterminal_task_completed_run_orphaned_evidence_invalid",
                    task_id=task_id,
                    bucket="triage",
                    detail=(
                        "the latest run completed and its 'completed' event "
                        "is intact, but the referenced completion artifacts "
                        "no longer validate — refusing to reconcile to done"
                    ),
                    data={"run_id": run_id, "invalid": invalid_entries},
                )
            )
        else:
            findings.append(
                Finding(
                    kind="nonterminal_task_completed_run_orphaned",
                    task_id=task_id,
                    bucket="safe_repair",
                    safe_repair="reconcile_task_done_from_completed_run",
                    detail=(
                        "the latest (and only current) run for this task "
                        "ended with outcome='completed' and a matching "
                        "'completed' event exists, but the task itself "
                        f"is still {row['status']!r} — restoring the done "
                        "status from durable, already-validated evidence"
                    ),
                    data={"run_id": run_id},
                )
            )
    return findings


def _audit_closeout_evidence_on_blocked(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute(
        "SELECT id, status FROM tasks WHERE status = 'blocked'"
    ).fetchall()
    for row in rows:
        task_id = row["id"]
        latest_run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        latest_outcome = latest_run["outcome"] if latest_run else None
        evidence = _latest_goal_progress_shows_closeout_evidence(conn, task_id)
        if evidence is None:
            continue
        findings.append(
            Finding(
                kind="blocked_with_verified_closeout_evidence",
                task_id=task_id,
                bucket="triage",
                detail=(
                    "task is blocked (gave_up/needs_input) but the goal-loop "
                    "telemetry shows a closeout/finalize attempt was made — "
                    "needs a human to check whether the contract was actually "
                    "satisfied; never auto-completed"
                ),
                data={"latest_run_outcome": latest_outcome, "goal_progress": evidence},
            )
        )
    # timed_out tasks that were released back to 'ready' without tripping
    # the breaker are the other half of the "gave_up/timed_out" bullet.
    ready_rows = conn.execute(
        "SELECT id FROM tasks WHERE status = 'ready'"
    ).fetchall()
    for row in ready_rows:
        task_id = row["id"]
        latest_run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if latest_run is None or latest_run["outcome"] != "timed_out":
            continue
        evidence = _latest_goal_progress_shows_closeout_evidence(conn, task_id)
        if evidence is None:
            continue
        findings.append(
            Finding(
                kind="blocked_with_verified_closeout_evidence",
                task_id=task_id,
                bucket="triage",
                detail=(
                    "task's latest run timed out but the goal-loop telemetry "
                    "shows a closeout/finalize attempt was made — needs a "
                    "human to check whether the contract was actually "
                    "satisfied; never auto-completed"
                ),
                data={"latest_run_outcome": "timed_out", "goal_progress": evidence},
            )
        )
    return findings


def _audit_archived_with_pending_review(conn) -> list[Finding]:
    findings: list[Finding] = []
    rows = conn.execute("SELECT id FROM tasks WHERE status = 'archived'").fetchall()
    for row in rows:
        task_id = row["id"]
        pending = kb._pending_review_request(conn, task_id)
        if pending is not None:
            findings.append(
                Finding(
                    kind="archived_with_pending_review",
                    task_id=task_id,
                    bucket="triage",
                    detail=(
                        "task was archived while a review handoff was still "
                        "pending — the reviewer's decision can never land"
                    ),
                    data={"pending_review": pending},
                )
            )
    return findings


_AUDIT_FUNCS: tuple[Callable[[Any], list[Finding]], ...] = (
    _audit_run_pointer_invariants,
    _audit_multiple_open_runs,
    _audit_stale_promotable,
    _audit_done_completion_evidence,
    _audit_done_artifact_manifests,
    _audit_orphaned_completed_runs,
    _audit_closeout_evidence_on_blocked,
    _audit_archived_with_pending_review,
)


def run_audit(conn, *, board: Optional[str] = None) -> AuditReport:
    """Read-only invariant audit. Never writes to the database."""
    findings: list[Finding] = []
    for fn in _AUDIT_FUNCS:
        findings.extend(fn(conn))
    scanned = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    return AuditReport(
        findings=findings,
        scanned_tasks=int(scanned),
        board=board,
        generated_at=int(time.time()),
    )


# ---------------------------------------------------------------------------
# Repair: strict allowlist, one function per ``safe_repair`` name
# ---------------------------------------------------------------------------

def _append_repair_event(conn, task_id: str, finding: Finding, *, actor: str, reason: str) -> None:
    kb._append_event(
        conn, task_id, "kanban_repair_applied",
        {
            "finding_kind": finding.kind,
            "safe_repair": finding.safe_repair,
            "actor": actor,
            "reason": reason,
            "data": finding.data,
        },
    )


def _repair_reclaim_orphaned_running_task(conn, finding: Finding, *, actor: str, reason: str) -> tuple[bool, Optional[str]]:
    task_id = finding.task_id
    observed_run_id = finding.data.get("current_run_id")
    with kb.write_txn(conn):
        if observed_run_id is None:
            cur = conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL, current_run_id=NULL "
                "WHERE id=? AND status='running' AND current_run_id IS NULL",
                (task_id,),
            )
        else:
            cur = conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL, current_run_id=NULL "
                "WHERE id=? AND status='running' AND current_run_id=?",
                (task_id, int(observed_run_id)),
            )
        if cur.rowcount != 1:
            return False, "precondition changed (task no longer running with that run pointer)"
        if observed_run_id is not None:
            conn.execute(
                "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
                "ended_at=COALESCE(ended_at, ?), claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL "
                "WHERE id=? AND ended_at IS NULL",
                (int(time.time()), int(observed_run_id)),
            )
        _append_repair_event(conn, task_id, finding, actor=actor, reason=reason)
    return True, None


def _repair_clear_dangling_run_pointer(conn, finding: Finding, *, actor: str, reason: str) -> tuple[bool, Optional[str]]:
    task_id = finding.task_id
    observed_run_id = int(finding.data["current_run_id"])
    with kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET current_run_id=NULL "
            "WHERE id=? AND status != 'running' AND current_run_id=?",
            (task_id, observed_run_id),
        )
        if cur.rowcount != 1:
            return False, "precondition changed"
        if finding.data.get("run_is_live"):
            conn.execute(
                "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
                "ended_at=COALESCE(ended_at, ?), claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL "
                "WHERE id=? AND ended_at IS NULL",
                (int(time.time()), observed_run_id),
            )
        _append_repair_event(conn, task_id, finding, actor=actor, reason=reason)
    return True, None


def _repair_close_duplicate_open_runs(conn, finding: Finding, *, actor: str, reason: str) -> tuple[bool, Optional[str]]:
    task_id = finding.task_id
    survivor = finding.data["current_run_id"]
    others = [rid for rid in finding.data["open_run_ids"] if rid != survivor]
    if not others:
        return False, "no duplicate open runs remain"
    closed_any = False
    with kb.write_txn(conn):
        for rid in others:
            cur = conn.execute(
                "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
                "ended_at=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL "
                "WHERE id=? AND task_id=? AND ended_at IS NULL AND id != ?",
                (int(time.time()), rid, task_id, survivor),
            )
            closed_any = closed_any or cur.rowcount == 1
        if closed_any:
            _append_repair_event(conn, task_id, finding, actor=actor, reason=reason)
    if not closed_any:
        return False, "precondition changed (duplicates already closed)"
    return True, None


def _repair_promote_stale_task(conn, finding: Finding, *, actor: str, reason: str) -> tuple[bool, Optional[str]]:
    task_id = finding.task_id
    before = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    if before is None:
        return False, "task no longer exists"
    if before["status"] not in ("todo", "blocked"):
        return False, "precondition changed (task already left todo/blocked)"
    # Reuse the dispatcher's own promotion function rather than
    # re-implementing the CAS/eligibility logic here. It re-derives
    # eligibility from scratch (including the sticky-block and
    # failure-limit exclusions) and is itself idempotent.
    kb.recompute_ready(conn)
    after = conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
    if after is None or after["status"] == before["status"]:
        return False, "recompute_ready did not promote this task (precondition changed)"
    with kb.write_txn(conn):
        _append_repair_event(conn, task_id, finding, actor=actor, reason=reason)
    return True, None


def _repair_reattach_artifact_manifest(conn, finding: Finding, *, actor: str, reason: str) -> tuple[bool, Optional[str]]:
    task_id = finding.task_id
    reattached = 0
    with kb.write_txn(conn):
        for entry in finding.data.get("entries", []):
            # Never trust the audit-time snapshot blindly — re-verify right
            # before writing, and skip (not fail) an entry that no longer
            # validates or was already reattached by a concurrent repair.
            ok, _why = _verify_manifest_entry_on_disk(entry)
            if not ok:
                continue
            exists = conn.execute(
                "SELECT 1 FROM task_artifacts WHERE task_id=? AND producer_run_id=? "
                "AND original_path=? AND sha256=?",
                (task_id, entry.get("producer_run_id"), entry.get("original_path"), entry.get("sha256")),
            ).fetchone()
            if exists is not None:
                continue
            try:
                conn.execute(
                    """
                    INSERT INTO task_artifacts (
                        task_id, producer_run_id, original_path, durable_path,
                        sha256, size, content_type, validated_at, retention_class
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        entry.get("producer_run_id"),
                        entry.get("original_path"),
                        entry.get("durable_path"),
                        entry.get("sha256"),
                        entry.get("size"),
                        entry.get("content_type"),
                        int(time.time()),
                        entry.get("retention_class") or "task_completion",
                    ),
                )
                reattached += 1
            except Exception:
                # UNIQUE collision: another repair pass (or the row itself)
                # already restored this exact entry. Idempotent no-op.
                continue
        if reattached:
            _append_repair_event(conn, task_id, finding, actor=actor, reason=reason)
    if not reattached:
        return False, "nothing left to reattach (already restored or no longer valid)"
    return True, None


def _repair_reconcile_task_done_from_completed_run(conn, finding: Finding, *, actor: str, reason: str) -> tuple[bool, Optional[str]]:
    task_id = finding.task_id
    run_id = int(finding.data["run_id"])
    now = int(time.time())
    with kb.write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = 'done',
                   completed_at = ?,
                   current_run_id = NULL,
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL,
                   block_kind = NULL,
                   block_recurrences = 0
             WHERE id = ?
               AND status NOT IN ('done', 'archived')
               AND (current_run_id IS NULL OR current_run_id = ?)
            """,
            (now, task_id, run_id),
        )
        if cur.rowcount != 1:
            return False, "precondition changed"
        _append_repair_event(conn, task_id, finding, actor=actor, reason=reason)
    kb._clear_failure_counter(conn, task_id)
    kb.recompute_ready(conn)
    return True, None


_REPAIR_FUNCS: dict[str, Callable[..., tuple[bool, Optional[str]]]] = {
    "reclaim_orphaned_running_task": _repair_reclaim_orphaned_running_task,
    "clear_dangling_run_pointer": _repair_clear_dangling_run_pointer,
    "close_duplicate_open_runs": _repair_close_duplicate_open_runs,
    "promote_stale_task": _repair_promote_stale_task,
    "reattach_artifact_manifest": _repair_reattach_artifact_manifest,
    "reconcile_task_done_from_completed_run": _repair_reconcile_task_done_from_completed_run,
}


def run_repair(
    conn,
    *,
    dry_run: bool = True,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
    board: Optional[str] = None,
    only_kinds: Optional[set[str]] = None,
) -> RepairReport:
    """Dry-run by default. Applying requires an explicit actor + reason.

    Re-derives findings via :func:`run_audit` at call time (never accepts
    a pre-computed, possibly-stale report) so every repair attempt checks
    the live database immediately before mutating it.
    """
    if not dry_run:
        if not actor or not str(actor).strip():
            raise ValueError("apply mode requires a non-empty actor")
        if not reason or not str(reason).strip():
            raise ValueError("apply mode requires a non-empty reason")

    report = run_audit(conn, board=board)
    results: list[RepairResult] = []
    for finding in report.findings:
        if only_kinds is not None and finding.kind not in only_kinds:
            continue
        if finding.safe_repair is None:
            results.append(
                RepairResult(finding=finding, applied=False, reason="no allowlisted repair (triage)")
            )
            continue
        repair_fn = _REPAIR_FUNCS.get(finding.safe_repair)
        if repair_fn is None:
            # Defensive: a finding claims a repair name this module doesn't
            # (yet) implement. Never silently no-op as "applied".
            results.append(
                RepairResult(
                    finding=finding, applied=False,
                    reason=f"unknown repair action {finding.safe_repair!r}",
                )
            )
            continue
        if dry_run:
            results.append(
                RepairResult(finding=finding, applied=False, reason="dry_run")
            )
            continue
        applied, why = repair_fn(conn, finding, actor=actor, reason=reason)
        results.append(RepairResult(finding=finding, applied=applied, reason=why))

    return RepairReport(
        dry_run=dry_run, results=results, board=board, generated_at=int(time.time()),
    )

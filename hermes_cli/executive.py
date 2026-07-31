"""Executive control loop: notice, classify, repair, and only then escalate.

Why this exists
---------------
The complaint that started this programme was not "the agent is bad at
tasks". It was: *"Ständig gebe ich dir ok damit du schritte ausführst oder
anweisungen damit du aufgaben beginnst und weiterführst."* Every stall became
a question. The system noticed problems and handed them straight back.

So the loop is deliberately ordered **recovery before escalation** (W5). A
work item that has gone quiet is first classified, then repaired on the *same*
``work_uid``, and only escalated when repair is not available or has been
exhausted. A new card is created only for a genuinely separate sub-goal --
opening one for every retry is how a board turns into a list of the same
problem written down repeatedly.

The escalation itself is shaped by acceptance test 8: **exactly one bundled
decision package** with situation, attempts, options, recommendation and
impact -- never a bare "how do you want to proceed?". A question without a
recommendation moves the work back to the person; that is the behaviour this
module exists to remove.

Deduplication is not a nicety here. Without a coalesce key, a watcher that
runs every few minutes re-reports the same stall until someone answers, which
is precisely the "zugespamt" experience that made the existing channels
useless.

Scope
-----
Phase E: commitment watcher, staleness classification, repair on the same
work_uid, retry budgets with strategy change, decision packages, WIP control.
No delivery guarantee -- the E form of acceptance test 3 asserts that the
commitment and its next step *survive*, not that a message arrived. That needs
F4/F6.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# How long a claimed-but-silent item may stay quiet before it counts as stale.
DEFAULT_STALE_AFTER_SECONDS = 30 * 60

# Retry budget per work item before escalation. Small on purpose: a fourth
# identical attempt is not persistence, it is a loop.
DEFAULT_RETRY_BUDGET = 3

# A suppressed decision stays suppressed this long. Long enough that a busy
# watcher cannot re-ask within one working session.
SUPPRESSION_TTL_SECONDS = 12 * 3600

QUIET_HOURS = (23, 7)  # inclusive start hour, exclusive end hour


class Condition(str, Enum):
    """What is wrong with a work item, in the vocabulary W5 uses."""

    HEALTHY = "healthy"
    STALE = "stale"                  # claimed, running, but silent too long
    LOST = "lost"                    # owner process is provably gone
    FAILED = "failed"                # ran and failed
    REVIEW_PENDING = "review_pending"
    COMMITMENT_DUE = "commitment_due"
    BLOCKED = "blocked"


class Action(str, Enum):
    NONE = "none"
    REPAIR = "repair"                # retry on the SAME work_uid
    STRATEGY_CHANGE = "strategy_change"
    ESCALATE = "escalate"            # one decision package to the person
    WAIT = "wait"


@dataclass(frozen=True)
class Finding:
    work_uid: str
    task_id: str
    condition: Condition
    detail: str
    attempts: int = 0

    @property
    def coalesce_key(self) -> str:
        """Stable identity of *this problem on this item*.

        Deliberately excludes the attempt count and any timestamp: the second
        occurrence of the same stall is the same problem, and a key that
        changed on every tick would defeat suppression entirely -- which is
        how a watcher turns into a spam source.
        """
        return f"{self.work_uid}:{self.condition.value}"


@dataclass
class DecisionPackage:
    """One bundled escalation. The shape is acceptance test 8's PASS criterion."""

    coalesce_key: str
    situation: str
    attempts: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None
    impact: Optional[str] = None
    work_uids: List[str] = field(default_factory=list)

    def is_well_formed(self) -> bool:
        """A package without options and a recommendation is a bare question.

        Enforced rather than documented, because "how do you want to proceed?"
        is exactly the output this module was built to stop producing.
        """
        return bool(
            self.situation
            and self.attempts
            and len(self.options) >= 2
            and self.recommendation
            and self.impact
        )

    def render(self) -> str:
        lines = [f"Lage: {self.situation}", "", "Bisherige Versuche:"]
        lines += [f"  - {a}" for a in self.attempts]
        lines += ["", "Optionen:"]
        lines += [f"  {i}. {o}" for i, o in enumerate(self.options, 1)]
        lines += ["", f"Empfehlung: {self.recommendation}",
                  f"Auswirkung: {self.impact}"]
        return "\n".join(lines)


# --------------------------------------------------------------- suppression


class SuppressionLedger:
    """Remembers what has already been asked, so it is not asked again.

    Backed by ``task_events`` rather than a new table: the suppression state
    of a work item belongs to that item's history, and a separate store would
    be one more thing that can drift out of sync with the board.
    """

    def __init__(self, conn: sqlite3.Connection, ttl: int = SUPPRESSION_TTL_SECONDS):
        self.conn = conn
        self.ttl = ttl

    def is_suppressed(self, coalesce_key: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        # Compare the LATEST event of either kind: a resolve newer than the
        # escalation clears suppression. Querying only 'executive_escalated'
        # made resolve() a no-op (it stayed suppressed for the full TTL even
        # after recovery).
        row = self.conn.execute(
            """SELECT kind, created_at FROM task_events
                WHERE kind IN ('executive_escalated', 'executive_resolved')
                  AND payload LIKE ?
                ORDER BY id DESC LIMIT 1""",
            (f'%"coalesce_key": "{coalesce_key}"%',),
        ).fetchone()
        if row is None:
            return False
        kind, created_at = row[0], row[1]
        if kind != "executive_escalated":
            return False
        return (now - float(created_at)) < self.ttl

    def open_escalations(self, now: Optional[float] = None) -> List[tuple]:
        """``(coalesce_key, task_id)`` for each escalation still within TTL
        whose latest event is the escalation itself (not a later resolve).

        Used to drive resolve() when a previously-escalated problem no longer
        appears in the scan.
        """
        now = time.time() if now is None else now
        rows = self.conn.execute(
            """SELECT task_id, payload, created_at, kind FROM task_events
                WHERE kind IN ('executive_escalated', 'executive_resolved')
                ORDER BY id DESC"""
        ).fetchall()
        seen: set = set()
        out: List[tuple] = []
        for task_id, payload, created_at, kind in rows:
            try:
                key = json.loads(payload).get("coalesce_key")
            except Exception:
                continue
            if not key or key in seen:
                continue
            seen.add(key)  # first (newest) event per key decides
            if kind == "executive_escalated" and (now - float(created_at)) < self.ttl:
                out.append((key, task_id))
        return out

    def record(self, task_id: str, package: DecisionPackage) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?,?,?,?)",
            (task_id, "executive_escalated",
             json.dumps({"coalesce_key": package.coalesce_key,
                         "situation": package.situation,
                         "recommendation": package.recommendation},
                        sort_keys=True),
             int(time.time())),
        )

    def resolve(self, task_id: str, coalesce_key: str) -> None:
        """Clear suppression once the underlying problem is gone.

        Without this the item stays silent even after it recovers and breaks
        again for the same reason -- suppression would turn into blindness.
        """
        self.conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?,?,?,?)",
            (task_id, "executive_resolved",
             json.dumps({"coalesce_key": coalesce_key}, sort_keys=True),
             int(time.time())),
        )


# ------------------------------------------------------------- classification


def classify(row: Any, *, now: Optional[float] = None,
             stale_after: int = DEFAULT_STALE_AFTER_SECONDS,
             last_signal: Optional[float] = None) -> Condition:
    """Decide what, if anything, is wrong with one work item.

    ``last_signal`` is the most recent heartbeat or activity timestamp from
    the item's runs. It is passed in rather than read off ``row`` because
    those columns live on ``task_runs``, not on ``tasks`` -- an earlier
    version of this function looked for them on the task row, found nothing,
    and silently fell back to the creation time. That "worked" in the sense
    that the tests passed, while measuring the wrong thing.

    Order matters. ``blocked`` is checked before staleness because a blocked
    item is *supposed* to be quiet -- reporting it as stale would generate a
    stream of findings for items that are behaving correctly.
    """
    now = time.time() if now is None else now
    status = row["status"]

    if status in ("done", "archived"):
        return Condition.HEALTHY
    if status == "blocked":
        return Condition.BLOCKED
    if status == "review":
        return Condition.REVIEW_PENDING

    keys = row.keys() if hasattr(row, "keys") else ()
    due = row["commitment_due_at"] if "commitment_due_at" in keys else None
    if due is not None and float(due) <= now:
        return Condition.COMMITMENT_DUE

    if status == "running":
        if last_signal:
            if (now - float(last_signal)) > stale_after:
                return Condition.STALE
        else:
            # A running item that has never reported anything is not "fine
            # because we have no evidence against it". Absence of a heartbeat
            # is the signal, not the lack of one.
            created = float(row["created_at"] or 0) if "created_at" in keys else 0
            if created and (now - created) > stale_after:
                return Condition.STALE
    return Condition.HEALTHY


def scan(conn: sqlite3.Connection, *, board_uuid: Optional[str] = None,
         now: Optional[float] = None,
         stale_after: int = DEFAULT_STALE_AFTER_SECONDS) -> List[Finding]:
    """Find everything that needs attention, across all work items.

    Liveness and attempt count come from ``task_runs``: a task row records
    what the work *is*, a run row records what happened when it was tried.
    Counting runs is also the honest attempt count -- there is no ``retries``
    column on ``tasks``, and inventing one would duplicate a truth the runs
    table already holds.
    """
    from hermes_cli.kanban_db import build_work_uid, get_board_uuid

    board_uuid = board_uuid or get_board_uuid(conn) or "unknown"
    findings: List[Finding] = []
    for row in conn.execute(
        "SELECT * FROM tasks WHERE status NOT IN ('done', 'archived')"
    ):
        stats = conn.execute(
            """SELECT COUNT(*) AS attempts,
                      MAX(COALESCE(last_heartbeat_at, 0),
                          COALESCE(last_activity_at, 0),
                          COALESCE(started_at, 0)) AS last_signal
                 FROM task_runs WHERE task_id = ?""",
            (row["id"],),
        ).fetchone()
        attempts = int(stats["attempts"] or 0)
        last_signal = float(stats["last_signal"] or 0) or None

        condition = classify(row, now=now, stale_after=stale_after,
                             last_signal=last_signal)
        if condition is Condition.HEALTHY:
            continue
        findings.append(Finding(
            work_uid=build_work_uid(board_uuid, row["id"]),
            task_id=row["id"],
            condition=condition,
            detail=f"status={row['status']}, attempts={attempts}",
            attempts=attempts,
        ))
    return findings


# ------------------------------------------------------------------ decide


def decide(finding: Finding, *, retry_budget: int = DEFAULT_RETRY_BUDGET) -> Action:
    """Recovery before escalation (W5).

    The budget exists so that "keep trying" cannot become the permanent
    answer. Below it, repair on the same work_uid. At the halfway mark the
    strategy has to change -- a third identical attempt is not persistence,
    it is a loop, and repeating it wastes the budget that would otherwise buy
    a different approach. Only an exhausted budget escalates.
    """
    if finding.condition is Condition.REVIEW_PENDING:
        # Waiting for a reviewer is the system working, not a problem.
        return Action.WAIT
    if finding.condition is Condition.BLOCKED:
        return Action.ESCALATE
    if finding.condition is Condition.COMMITMENT_DUE:
        return Action.REPAIR
    if finding.condition in (Condition.STALE, Condition.LOST, Condition.FAILED):
        if finding.attempts >= retry_budget:
            return Action.ESCALATE
        if finding.attempts >= max(1, retry_budget // 2):
            return Action.STRATEGY_CHANGE
        return Action.REPAIR
    return Action.NONE


def in_quiet_hours(now: Optional[float] = None,
                   quiet: Sequence[int] = QUIET_HOURS) -> bool:
    now = time.time() if now is None else now
    hour = time.localtime(now).tm_hour
    start, end = quiet
    return hour >= start or hour < end


def build_decision_package(findings: Sequence[Finding]) -> DecisionPackage:
    """Bundle related findings into ONE package.

    Acceptance test 8 asks for exactly one package, not one per finding. Three
    items stalled by the same cause are one decision; sending three messages
    makes the person do the grouping the system should have done.
    """
    if not findings:
        raise ValueError("cannot build a decision package from nothing")
    primary = findings[0]
    conditions = {f.condition for f in findings}
    situation = (
        f"{len(findings)} Work Item(s) in Zustand "
        f"{', '.join(sorted(c.value for c in conditions))}"
    )
    # Fill the fields is_well_formed() requires, from the findings themselves.
    # An empty package renders as the bare "wie soll ich weitermachen?" this
    # module exists to prevent, and run_once escalated it anyway.
    attempts = [
        f"{f.work_uid}: {f.detail} (bisher {f.attempts} Versuch(e))"
        for f in findings
    ]
    uids = ", ".join(f.work_uid for f in findings)
    if primary.condition is Condition.BLOCKED:
        options = [
            "Blocker auflösen und Item entsperren (kanban_unblock)",
            "Aufgabe abbrechen bzw. archivieren, falls obsolet",
        ]
        recommendation = (
            "Blocker prüfen und entsperren; nur archivieren, wenn die Aufgabe "
            "nicht mehr gebraucht wird."
        )
        impact = f"Ohne Entscheidung bleibt {uids} blockiert und hält abhängige Arbeit auf."
    else:  # STALE / LOST / FAILED with an exhausted retry budget
        options = [
            "Neu zuweisen mit geänderter Strategie (frischer Versuch)",
            "Aufgabe abbrechen bzw. archivieren",
            "Manuell übernehmen und selbst abschließen",
        ]
        recommendation = (
            "Das Retry-Budget ist erschöpft — nur mit geänderter Strategie neu "
            "zuweisen, sonst abbrechen; ein identischer Neuversuch wäre eine Schleife."
        )
        impact = (
            f"Ohne Entscheidung hängt {uids} weiter und bindet Ressourcen ohne "
            "Fortschritt."
        )
    return DecisionPackage(
        coalesce_key=primary.coalesce_key,
        situation=situation,
        attempts=attempts,
        options=options,
        recommendation=recommendation,
        impact=impact,
        work_uids=[f.work_uid for f in findings],
    )


def coalesce(findings: Sequence[Finding]) -> Dict[str, List[Finding]]:
    """Group findings by what would be one decision."""
    grouped: Dict[str, List[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.coalesce_key, []).append(finding)
    return grouped


@dataclass
class LoopResult:
    repaired: List[Finding] = field(default_factory=list)
    strategy_changed: List[Finding] = field(default_factory=list)
    escalated: List[DecisionPackage] = field(default_factory=list)
    suppressed: List[str] = field(default_factory=list)
    waiting: List[Finding] = field(default_factory=list)
    deferred_quiet_hours: List[str] = field(default_factory=list)


def run_once(conn: sqlite3.Connection, *, now: Optional[float] = None,
             retry_budget: int = DEFAULT_RETRY_BUDGET,
             stale_after: int = DEFAULT_STALE_AFTER_SECONDS,
             respect_quiet_hours: bool = True) -> LoopResult:
    """One pass of the loop. Pure classification and decision; no side effects
    beyond recording suppression, so it is safe to call from a watcher.

    The actual repair execution is the caller's job -- this returns *what*
    should happen. Keeping the decision separable from the act is what makes
    the policy testable without spawning work.
    """
    now = time.time() if now is None else now
    ledger = SuppressionLedger(conn)
    result = LoopResult()

    for key, group in coalesce(scan(conn, now=now, stale_after=stale_after)).items():
        action = decide(group[0], retry_budget=retry_budget)

        if action is Action.WAIT:
            result.waiting.extend(group)
            continue
        if action is Action.REPAIR:
            result.repaired.extend(group)
            continue
        if action is Action.STRATEGY_CHANGE:
            result.strategy_changed.extend(group)
            continue
        if action is not Action.ESCALATE:
            continue

        if ledger.is_suppressed(key, now=now):
            # Already asked. Asking again is the spam behaviour, not diligence.
            result.suppressed.append(key)
            continue
        if respect_quiet_hours and in_quiet_hours(now) and \
                group[0].condition is not Condition.LOST:
            result.deferred_quiet_hours.append(key)
            continue

        package = build_decision_package(group)
        if not package.is_well_formed():
            # Fail closed: never escalate a bare "how should I proceed?" — the
            # exact output this module exists to stop. Treat it as waiting
            # rather than sending the person an empty question.
            result.waiting.extend(group)
            continue
        result.escalated.append(package)
        ledger.record(group[0].task_id, package)

    # Clear suppression for problems that have since recovered: a coalesce_key
    # escalated within the TTL but absent from this scan is gone. Without this,
    # suppression outlives the problem and the item stays silent if it breaks
    # the same way again (resolve() otherwise had no caller).
    active_keys = {f.coalesce_key for f in scan(conn, now=now, stale_after=stale_after)}
    for key, task_id in ledger.open_escalations(now=now):
        if key not in active_keys:
            ledger.resolve(task_id, key)

    return result

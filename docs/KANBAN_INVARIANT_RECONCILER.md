# Kanban Invariant Reconciler (S5)

Stand: 2026-07-10. Karte `t_0cb8668d`.

## Zweck

Task, Run, Event, Claim und durable Completion-Artifact-Manifeste bilden
zusammen eine logische Projektion, die über viele einzeln korrekte
Code-Pfade hinweg konsistent bleiben muss. Ein Prozessabbruch zwischen zwei
Statements außerhalb einer Transaktion, eine manuelle DB-Änderung, ein
Tabellen-Rebuild-Bug oder ein Worker, der noch eine überholte
`expected_run_id` hält, kann diese Projektion auseinanderdriften lassen.

`hermes_cli/kanban_repair.py` beantwortet zwei getrennte Fragen:

* `run_audit(conn)` — **rein lesend**. Läuft über jede Invariante und gibt
  eine Liste typisierter `Finding`-Objekte zurück. Schreibt nie in die DB.
* `run_repair(conn, dry_run=..., actor=..., reason=...)` — Dry-Run per
  Default. Leitet dieselben Findings neu her und wendet für die strikte
  Allowlist *mechanisch eindeutiger* Fälle eine idempotente,
  CAS-gebundene Reparatur in einer eigenen Transaktion an, mit
  `kanban_repair_applied`-Audit-Event. Alles andere bleibt unangetastet —
  ein typisiertes Finding im Bucket `"triage"`, niemals eine optimistische
  Promotion oder eine fabrizierte Fertigstellung.

## Finding-Kataloge (Invariante → Kind → Bucket)

| Invariante (Karte)                                              | Finding-`kind`                                              | Bucket        | Repair |
|---|---|---|---|
| Task-Status/current_run_id/Claim/PID                             | `running_task_missing_live_run`                              | safe_repair   | `reclaim_orphaned_running_task` — CAS-Demote auf `ready`, schließt den toten Run als `reclaimed` |
| dito, für Nicht-`running`-Status                                 | `nonrunning_task_dangling_run_pointer`                       | safe_repair   | `clear_dangling_run_pointer` |
| max. ein aktiver Run pro Task (eindeutiger Fall)                 | `multiple_open_runs` (current_run_id ∈ offenen Runs)         | safe_repair   | `close_duplicate_open_runs` — schließt alle außer dem durch `current_run_id` referenzierten |
| max. ein aktiver Run pro Task (mehrdeutiger Fall)                | `multiple_open_runs`                                         | triage        | keine — kein Gewinner mechanisch bestimmbar |
| terminale Eltern vs. stale `todo`/`blocked`                      | `stale_promotable_task`                                      | safe_repair   | `promote_stale_task` — ruft `kanban_db.recompute_ready()` (Wiederverwendung, keine Duplizierung der Sticky-/Failure-Limit-Logik) |
| Sticky-/Review-Block (negative Invariante)                       | — (bewusst NICHT geflaggt, per `_has_sticky_block`)           | —             | — |
| Task-/Runstatus-Divergenz bei `dependency`-Wait                  | — (bewusst NICHT geflaggt, dokumentiertes By-Design-Verhalten)| —             | — |
| `done` ohne completed Run/Event                                  | `done_missing_completion_evidence`                            | triage        | keine (fail-closed) |
| `done` mit fehlendem/invalidem Manifest                          | `done_artifact_manifest_invalid`                              | triage        | keine (Bytes nicht rekonstruierbar) |
| `done`-Manifest-Zeile verloren, Datei+Event-Beleg intakt          | `done_artifact_manifest_lost`                                 | safe_repair   | `reattach_artifact_manifest` — restauriert die `task_artifacts`-Zeile aus dem unveränderlichen `completed`-Event + erneut verifizierter Datei |
| eindeutige Run-/Taskabschluss-Reconciliation                     | `nonterminal_task_completed_run_orphaned`                     | safe_repair   | `reconcile_task_done_from_completed_run` |
| dito, aber Artefakt-Beleg ungültig                               | `nonterminal_task_completed_run_orphaned_evidence_invalid`    | triage        | keine (fail-closed) |
| `gave_up`/`timed_out` mit verifizierter Closeout-Evidenz          | `blocked_with_verified_closeout_evidence`                     | triage        | keine — niemals automatische Fertigstellung aus Judge-Verdikt |
| stale Worker-Entscheidung via `expected_run_id`                  | abgedeckt durch `running_task_missing_live_run` / `nonrunning_task_dangling_run_pointer` | safe_repair | s.o. |
| archive-vs-complete Terminal-Konflikt                             | `archived_with_pending_review`                                | triage        | keine — Reviewer-Handshake ist verwaist, braucht Mensch |

## Warum genau diese Repairs sicher sind

Jede Repair-Funktion prüft ihre Vorbedingung unmittelbar vor dem Schreiben
per `UPDATE ... WHERE` (CAS) innerhalb eines eigenen
`kanban_db.write_txn`. Läuft ein zweiter Reparatur-Durchlauf (oder der
Dispatcher) gleichzeitig, sieht der Verlierer `rowcount == 0` und meldet
`applied=False, reason="precondition changed"` — kein Fehler, keine
Doppelanwendung. Das ist mit einem gezielten Zwei-Threads-Test abgesichert
(`test_concurrent_repair_and_claim_yield_single_winner_no_violation`).

`reattach_artifact_manifest` verifiziert die Datei auf der Platte *erneut*
(Größe + SHA-256) unmittelbar vor dem `INSERT`, verlässt sich also nie auf
den zum Audit-Zeitpunkt gecachten Zustand. Ein `UNIQUE`-Constraint-Konflikt
beim `INSERT` (weil ein Parallel-Lauf die Zeile bereits restauriert hat)
wird als idempotenter No-Op behandelt, nicht als Fehler.

`reconcile_task_done_from_completed_run` fabriziert nie eine Fertigstellung
aus Prosa: die einzige Quelle ist ein bereits **atomar** durch
`complete_task`/`decide_task_review` erzeugtes Paar aus geschlossenem Run
(`outcome='completed'`) und `completed`-Event mit passender `run_id` — beide
konnten historisch nur gemeinsam entstehen. Referenzierte
Completion-Artefakte werden vor der Reconciliation erneut verifiziert;
sind sie ungültig, gibt es nur ein Triage-Finding, keine Reparatur.

## Multi-Board

`run_audit`/`run_repair` operieren auf einer einzelnen bereits verbundenen
Board-Connection — genau wie `recompute_ready`/`gc_events`/etc. Für mehrere
Boards iteriert der Aufrufer (CLI: `--board <slug>` pro Aufruf) und öffnet
je Board eine eigene Connection nacheinander. Es gibt keinen globalen,
unkoordinierten Fleet-Write — ein Writer pro Board pro Aufruf, konsistent
mit dem Rest der Codebase.

## CLI

```
hermes kanban audit [--board <slug>] [--json]
hermes kanban repair [--board <slug>] [--apply --actor <wer> --reason <warum>] [--json]
```

`repair` ist ohne `--apply` ein reiner Dry-Run (keine Schreibzugriffe).
`--apply` ohne `--actor`/`--reason` schlägt mit `ValueError` fehl (CLI:
Exit-Code 2, Fehlermeldung auf stderr).

## Bewusst ausgelassen / offene Anschlüsse

* **Historische Sticky-Block-Verletzung** (ein `promoted`-Event direkt nach
  einem unresolved `blocked`-Event in der Vergangenheit) wurde NICHT als
  eigenes Finding gebaut. Die Karte verlangt, dass sticky Blocks nicht
  (weiterhin) automatisch promoted werden — das ist über die negative
  Fixture (`test_sticky_blocked_task_is_never_flagged_as_stale_promotable`)
  abgedeckt. Ein forensischer Scan vergangener Verletzungen wäre ungetestet
  Mehraufwand ohne zusätzlichen operativen Nutzen (die Vergangenheit ist
  ohnehin nicht reparierbar) und wurde bewusst nicht gebaut statt halbgar
  ergänzt.
* **`workflow_metrics`-Anschluss**: Die Karte nennt neue Outcome-/Finding-
  Klassen (`no-progress`, `terminal-step-not-taken`, `blind retry`,
  `evidence-invalid`, `artifact loss`, `pre-closeout budget exhaustion`).
  Diese sind in diesem Repo als typisierte `Finding.kind`-Werte vorhanden
  (`blocked_with_verified_closeout_evidence`,
  `done_artifact_manifest_invalid`,
  `nonterminal_task_completed_run_orphaned_evidence_invalid`, …) und über
  `AuditReport.counts_by_kind()` strukturiert/begrenzt abrufbar. Der in der
  Karte erwähnte `workflow_metrics`-Code liegt in
  `~/.hermes/plugins/tars-workflow/__init__.py` — einem separaten Plugin,
  nicht in diesem Repo (`hermes-agent`). Dieser Slice erweitert bewusst nur
  die Repo-seitige Audit-Fläche (`kanban_repair.py` + `hermes kanban audit
  --json`); der Plugin-seitige Anschluss (das Plugin gegen
  `AuditReport.to_dict()`/`counts_by_kind()` verdrahten) ist NICHT
  Bestandteil dieses Commits und bleibt offen.
* Keine Anwendung auf Live-Boards, keine Restarts, kein Push/PR in diesem
  Implementierungs-Slice (Karten-Vorgabe).

## Verifikation

```bash
PYTHONPATH=$PWD hermes-agent/venv/bin/python -m pytest \
  tests/hermes_cli/test_kanban_repair.py --collect-only -q

PYTHONPATH=$PWD hermes-agent/venv/bin/python -m pytest \
  tests/hermes_cli/test_kanban_db.py \
  tests/hermes_cli/test_kanban_completion_evidence.py \
  tests/cli/test_kanban_worker_quiet_exit.py \
  tests/tools/test_kanban_tools.py \
  tests/plugins/test_kanban_dashboard_plugin.py \
  tests/hermes_cli/test_kanban_notify.py \
  tests/hermes_cli/test_kanban_goal_mode.py \
  tests/hermes_cli/test_goals.py \
  tests/hermes_cli/test_kanban_repair.py -q

ruff check hermes_cli/kanban_repair.py hermes_cli/kanban.py tests/hermes_cli/test_kanban_repair.py
python -m compileall -q hermes_cli/kanban_repair.py hermes_cli/kanban.py tests/hermes_cli/test_kanban_repair.py
git diff --check
```

### Ausgeführte Evidenz am 2026-07-10

* Baseline vor dieser Karte: 664 passed (8 Kern-Testdateien).
* Kombinierter Lauf inkl. 29 neuer Tests
  (`tests/hermes_cli/test_kanban_repair.py`): **693 passed** (collect-only
  vorher grün, keine übersprungenen Collection-Errors).
* `ruff check`: bestanden.
* `compileall`: bestanden.
* `git diff --check`: bestanden.

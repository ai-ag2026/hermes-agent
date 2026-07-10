# Kanban Completion Evidence: Sicherheits- und Betriebsmodell

Stand: 2026-07-10

## Zweck

Eine Kanban-Aufgabe darf nur dann auf `done` wechseln, wenn ihre deklarierte Completion-Evidence nachweisbar, dauerhaft und dem richtigen Board/Run zugeordnet ist. Diese Härtung wurde nach einer forensischen Übernahme des S3-Diffs und einem unabhängigen Review von Commit `5407f3f73` umgesetzt.

Der Review meldete sieben Klassen von Problemen:

1. TOCTOU-/Symlink-Races in Quell- und Zielpfaden.
2. Zu schmale Erkennung typischer Secret-Pfade.
3. Verwaiste Dateien bei Crash oder Rollback.
4. Verlust der Named-Board-Isolation in Dashboard-Endpunkten.
5. Nicht durchgängig behandelter `CompletionEvidenceError`.
6. Stiller Manifestverlust durch `INSERT OR IGNORE` und kollidierende Durable-Pfade.
7. Auslieferungsfilter, die gehärtete Completion-Artefakte verwerfen konnten.

Ein zweiter unabhängiger Review des ersten Reparatur-Commits `ef125f5a7`
lieferte erneut `NEEDS_REPAIR`: fehlende Parent-Directory-`fsync`s,
pfadbasiertes Cleanup/Scavenging, ein `init_db()`-Bypass des Scavengers,
maskierbare Domain-Fehler bei kaputtem Audit und falsche Board-Attribution im
Lifecycle-Hook. Diese fünf Punkte wurden im Folge-Diff behoben und jeweils
regressionsgetestet.

Der Review des Folge-Commits `aa4a998e3` fand eine verbleibende Race Condition:
Ein Scavenger konnte einen alten wiederverwendeten Artifact-Pfad oder ein noch
leeres, bereits per FD geöffnetes Promotion-Verzeichnis entfernen. Der finale
Folge-Diff serialisiert deshalb Scavenging und den gesamten
Promotion-bis-Manifest-Commit-Pfad über einen board-/DB-spezifischen,
prozessübergreifenden Lock. Leere Artifact-Verzeichnisse werden vom Scavenger
nicht mehr entfernt.

## Sicherheitsinvarianten

- Die Task-/Run-Transition bleibt CAS-geschützt. Ein konkurrierender oder veralteter Run darf keinen Task abschließen.
- Ein Artifact wird aus einem erlaubten Root gelesen: Board-Workspace, Attachments, Completion-Root oder expliziter Task-Workspace.
- Auf POSIX/Linux wird jede Pfadkomponente ab dem Dateisystem-Anchor per Directory-FD geöffnet. `O_DIRECTORY` und `O_NOFOLLOW` verhindern, dass ein zwischen Prüfung und Öffnen ausgetauschter Symlink verfolgt wird.
- Der Zielbaum wird ebenfalls komponentenweise per `mkdir(..., dir_fd=...)` und `open(..., dir_fd=...)` erstellt/geöffnet. Jede neue Verzeichnisebene wird durch `fsync` ihres Parent-Directory-FDs persistiert. Temporärdatei, Kollisionstest, `replace`, `unlink` und Directory-`fsync` laufen relativ zum bereits geöffneten Ziel-Directory-FD.
- Nur reguläre Dateien werden akzeptiert.
- Der Durable-Pfad enthält ein Unterverzeichnis aus dem Hash des kanonischen Quellpfads sowie einen Dateinamen aus Content-SHA-256 und bereinigtem Basename. Zwei verschiedene Quellen mit gleichem Namen und Inhalt behalten dadurch zwei Manifeste, während das etablierte Dateinamenformat kompatibel bleibt.
- Manifestpersistierung verwendet kein `INSERT OR IGNORE`. Ein Konflikt ist ein Fehler, kein stiller Erfolg.
- Wenn Promotion, Manifestpersistierung oder CAS scheitern, bleibt der Task nicht `done`. Neu erzeugte, unreferenzierte Dateien werden entfernt und das Verzeichnis wird synchronisiert.

## Ablauf einer Completion

1. Completion-Contract gegen `summary`, `metadata`, `artifacts` und `tests_or_smokes` prüfen.
2. Artifact-Pfade kanonisieren, auf erlaubte Roots begrenzen und gegen die Secret-Denylist prüfen.
3. Quellen race-sicher öffnen und in eine exklusive `.promoting-*`-Datei kopieren.
4. Dateiinhalt beim Kopieren hashen, Datei-`fsync`, atomarer `replace`, danach Directory-`fsync`.
5. Manifestzeilen in `task_artifacts` persistieren.
6. Task und aktiven Run in derselben SQLite-Transaktion per CAS abschließen; Event- und Run-Metadaten enthalten die manifestierten Durable-Pfade.
7. Bei einem Verlierer-CAS oder einer Exception ausschließlich Dateien entfernen, die in keinem committed Manifest referenziert sind.

## Crash-Durability und Orphan-Scavenger

Dateisystem und SQLite bilden keine gemeinsame atomare Transaktion. Ein Prozessabbruch zwischen erfolgreichem Dateisystem-`fsync` und SQLite-Commit kann daher eine nicht referenzierte Datei hinterlassen. Das ist kein akzeptierter Dauerzustand.

Beim ersten DB-Connect eines Prozesses – einschließlich des kanonischen `init_db()`-Pfads von CLI und Dashboard – läuft deshalb ein best-effort Scavenger:

- Scavenger und `complete_task()` halten denselben DB-spezifischen Cross-Process-Lock. Damit kann kein Scan zwischen Promotion, CAS und Manifest-Commit eingreifen. Der Lock ist auf 30 Sekunden begrenzt; Completion schlägt bei Nichtverfügbarkeit typisiert fehl, der Startpfad protokolliert den best-effort Scavenger-Fehler.
- Er liest alle committed `durable_path`-Werte aus `task_artifacts`.
- Er traversiert per `os.fwalk` und Directory-FDs, folgt keinen Directory-Symlinks und weist einen symlinkenden Top-Level-Root fail-closed ab.
- Er entfernt nur reguläre Dateien oder Symlinks, die älter als eine Stunde und nicht manifestiert sind.
- Er entfernt keine regulären Verzeichnisse; ein offenes, noch leeres Promotion-Verzeichnis darf nicht aus dem Namespace gelöst werden.
- `stat`, `unlink` und `fsync` bleiben FD-relativ.
- Die Stunde Grace schützt Crash-Orphans und der Cross-Process-Lock schützt den vollständigen parallelen Promotion-bis-Commit-Zeitraum.
- Ein Scavenger-Fehler verhindert nicht den DB-Start, wird aber als Warnung geloggt.

## Secret-Pfade

Die Filterung ist bewusst fail-closed und pfadbasiert. Unter anderem werden abgewiesen:

- `.env*`, `.npmrc`, `.pypirc`, `.netrc`
- `.ssh`, `.gnupg`, `.aws`, `.kube`, `.docker`, `.azure`, `.gcloud`, `.config`
- Pairing-Verzeichnisse
- bekannte Credential-/Auth-/Token-Dateinamen
- private Schlüssel und Containerformate (`.pem`, `.key`, `.p12`, `.pfx`)
- Basenames mit `credential`, `private-key`, `secret` oder `token`

Das ist keine Inhaltsklassifizierung. Ein harmlos benannter Text mit eingebettetem Secret kann damit weiterhin nicht erkannt werden; Producer dürfen deshalb keine sensitiven Inhalte als Completion-Artefakt deklarieren.

## Named Boards und Delivery

Dashboard-Single-Update und Bulk-Update reichen `board` bis `complete_task()` durch. Damit werden Quelle, Durable-Root, Manifest, Cleanup und der Completion-Lifecycle-Hook gegen dasselbe Board aufgelöst.

Der Gateway-Medienfilter nimmt zusätzlich auf:

- den Default-Board-Root `kanban/artifacts`,
- alle vorhandenen Named-Board-Roots `kanban/boards/<board>/artifacts`,
- einen expliziten `HERMES_KANBAN_ARTIFACTS_ROOT`.

Dadurch bleiben gehärtete Completion-Artefakte auch bei Strict-Delivery und deaktiviertem Recency-Trust zustellbar.

## Fehlervertrag

`CompletionEvidenceError` trägt einen stabilen `kind`, eine lesbare Nachricht und strukturierte Details.

Behandelte Entry-Points:

- Model-Tool: strukturierte Tool-Antwort.
- CLI `kanban complete`: knappe Fehlermeldung ohne Traceback und Nonzero-Resultat.
- Dashboard Single-Update: HTTP 409 mit `detail.kind`, `detail.message` und Details.
- Dashboard Bulk-Update: per Task `ok=false`, `error_kind` und `error_details`; andere Tasks laufen unabhängig weiter.

Das Rejection-Audit ist best-effort. Schlägt nur das Schreiben des Audit-Events
fehl, wird dies geloggt, aber der ursprüngliche `CompletionEvidenceError` bleibt
für Model-Tool, CLI und Dashboard erhalten.

Relevante Fehlerarten sind unter anderem `evidence_missing`, `artifact_not_durable` und `artifact_promotion_failed`.

## Verifikation

Fokussierte Befehle:

```bash
scripts/run_tests.sh \
  tests/hermes_cli/test_kanban_completion_evidence.py \
  tests/plugins/test_kanban_dashboard_plugin.py \
  -q --tb=short

ruff check \
  hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py \
  plugins/kanban/dashboard/plugin_api.py gateway/platforms/base.py \
  gateway/kanban_watchers.py \
  tests/hermes_cli/test_kanban_completion_evidence.py \
  tests/hermes_cli/test_kanban_notify.py tests/tools/test_kanban_tools.py \
  tests/plugins/test_kanban_dashboard_plugin.py

python -m compileall -q \
  hermes_cli/kanban_db.py hermes_cli/kanban.py tools/kanban_tools.py \
  plugins/kanban/dashboard/plugin_api.py gateway/platforms/base.py \
  gateway/kanban_watchers.py

git diff --check
```

Die Regressionen decken ab:

- finalen Symlink-Swap und Ancestor-Symlink-Swap der Quelle,
- Ancestor-Symlink-Swap des Zielbaums,
- Secret-Pfadklassen,
- identischen Inhalt/Basename aus verschiedenen Quellen,
- stale Orphan-Cleanup bei Erhalt manifestierter Dateien,
- Cross-Process-Serialisierung von Scavenger und Completion sowie Erhalt leerer Promotion-Verzeichnisse,
- Scavenger-Aufruf über `init_db()` und fail-closed Verhalten bei symlinkendem Top-Level-Root,
- FD-relatives Rollback-Cleanup bei ausgetauschtem Ancestor,
- Parent-Directory-`fsync` für Task-, Run- und Source-Verzeichnisse,
- Named-Board-Durable-Root und Lifecycle-Attribution im Dashboard,
- Erhalt des typisierten Fehlers bei Audit-Schreibfehlern,
- strukturierte Single-/Bulk-/CLI-Fehler,
- Strict-Notifier-Zustellung aus dem Durable-Root.

### Ausgeführte Evidenz am 2026-07-10

- Completion-Evidence-Suite: **37/37 bestanden**.
- Completion-Evidence plus Kanban-Tool- und Dashboard-Suite: **235/235 bestanden**.
- Breite Kanban-/Tool-/Dashboard-/Gateway-Gruppe: **1048 bestanden, 2 fehlgeschlagen**.
- Die beiden verbleibenden Fehler wurden in einem frischen Detached-Worktree auf dem Parent `0af4ec0de` mit denselben Tests und denselben Failure-Signaturen reproduziert: **5 bestanden, 2 fehlgeschlagen**. Betroffen sind `test_rebuilt_schema_matches_fresh_db` und `test_first_init_connect_is_bounded_when_lock_held`; beide sind damit vorbestehend und nicht durch S3R verursacht.
- Ruff für alle geänderten Python-Dateien: bestanden.
- `compileall` für alle geänderten Python-Dateien: bestanden.
- `git diff --check`: bestanden.

Diese Evidence ist eine breite Kanban-bezogene Verifikation, aber ausdrücklich **keine vollständige Gesamtrepo-Suite**. Ein unabhängiger Review des finalen Commit-Hashes bleibt vor Akzeptanz erforderlich.

## Verbleibende Grenzen

- Auf Plattformen ohne `dir_fd`, `O_NOFOLLOW` und `O_DIRECTORY` fällt die Quellöffnung auf `lstat`/`fstat`-Identitätsprüfung zurück. Das ist kompatibel, erreicht aber nicht dieselbe Ancestor-Race-Garantie wie Linux.
- Der Scavenger ist absichtlich zeitverzögert; ein Crash-Orphan kann bis zum nächsten Prozessstart plus Grace-Intervall bestehen bleiben.
- Ein privilegierter lokaler Angreifer, der bereits geöffnete Verzeichnisstrukturen nach Abschluss austauschen oder die DB manipulieren kann, liegt außerhalb dieses Schutzmodells.
- Der Secret-Schutz ist pfadheuristisch und ersetzt keine Inhaltsprüfung oder vorgelagerte Datenklassifizierung.

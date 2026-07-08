# Quality-Class Routing (Fable-Planer + Opus-Design)

**Branch:** `tars/quality-class-routing` · **Erstellt:** 2026-07-09 · **Autor:** Claude Code
**Plan:** `~/.claude/plans/swift-chasing-tide.md` · **Recherche:** `~/.claude/plans/fable-planner-hermes-kanban.md`

## Ziel
Aufgaben im Hermes-Kanban **selektiv** an bestimmte Modelle routen (nicht pauschal), damit teure
Modelle nur dort laufen, wo sie sich lohnen:

- **Achse A — härteste/wichtigste Aufgaben → Fable als Planer.** Eine Triage-Card wird mit
  `--class hard` getaggt; der Decomposer (Card→Sub-Cards) läuft dann auf `anthropic/claude-fable-5`.
  Der Decompose-Call ist **toollos** → über die auto-erkannten Claude-Code-Credentials (Max/OAuth)
  liefert Anthropic `200` (die HTTP-400-Tool-Hürde trifft nur tool-tragende Requests).
- **Achse B — Design-Aufgaben → Opus 4.8 als Executor.** Abgeleitet aus `assignee=designer`;
  reine Profil-Config-Änderung. Läuft über Max/OAuth + `mcp__`-Rewrite; Designer-Profil ist
  tool-arm (terminal/code_execution deaktiviert). ⚠️ native Tools über OAuth können 400 werfen → Test-Gate.

## Design in einem Satz
`task_class` (freies Label auf der Card) → der Decomposer wählt per **Konvention** die Aux-Rolle
`kanban_decomposer_<class>`, **falls sie in der Config existiert**, sonst den Default `kanban_decomposer`.
Kein neues Config-Schema, kein Spawn-Chokepoint-Umbau, keine LLM-Klassifikation.

## Was geändert wurde

### Code (Fork-Repo, git-getrackt)
| Datei | Änderung |
|---|---|
| `hermes_cli/kanban_db.py` | `tasks.task_class TEXT` (CREATE + idempotente Migration); `Task.task_class`-Feld **und `from_row`-Read** (anders als die tote `effort`-Spalte); `create_task(task_class=...)`; neuer Setter `set_task_class()` |
| `hermes_cli/kanban_decompose.py` | Rollen-Switch: `kanban_decomposer_<class>` wenn in `auxiliary`-Config vorhanden, sonst Default. **Existenz-Check ist kritisch** — eine unkonfigurierte Rolle würde sonst auf „auto" (Hauptmodell) statt auf den Default-Decomposer routen. |
| `hermes_cli/kanban.py` | `hermes kanban create --class <name>`; neues Subkommando `hermes kanban reclassify <id> <class|none>`; Anzeige in `show`; `task_class` in JSON-Ausgabe |

### Config (unter `~/.hermes`, **NICHT** git-getrackt → nur per Backup gesichert)
| Datei | Änderung |
|---|---|
| `~/.hermes/config.yaml` | Neue Aux-Rolle `auxiliary.kanban_decomposer_hard` → `provider: anthropic`, `model: claude-fable-5`, Fallback `gpt-5.5` → `deepseek-v4-pro` |
| `~/.hermes/profiles/designer/config.yaml` | `model` → `provider: anthropic`, `default: claude-opus-4-8` (vorher openrouter/deepseek-v4-pro) |

**Backup der Config vor Änderung:** siehe `ROLLBACK.md` (Verzeichnis
`~/.hermes/backups/quality-class-routing-<timestamp>/`).

## Verhaltens-Neutralität (Beleg)
Ohne Klassen-Tag **und** ohne konfigurierte Klassen-Rolle ist das Verhalten identisch zu vorher:
- `task_class` ist NULL für alle bestehenden Cards (nullable Migration).
- Der Decomposer wählt nur dann eine andere Rolle, wenn `task.task_class` gesetzt **und**
  `auxiliary.kanban_decomposer_<class>` konfiguriert ist — sonst exakt der alte `kanban_decomposer`.
- Achse B betrifft nur das `designer`-Profil.

## Verifikation (Stand 2026-07-09, lokal, ohne Live-Account)
- ✅ `py_compile` aller 3 geänderten Dateien
- ✅ DB-Roundtrip: `create_task(task_class="hard")` → `from_row` liest `"hard"`; no-class → NULL; `set_task_class` set/clear; Whitespace-Normalisierung (**genau der Test, der den `effort`-Bug gefangen hätte**)
- ✅ Rollen-Switch (4 Fälle): `hard`+konfiguriert → `kanban_decomposer_hard`; `hard`+unkonfiguriert → `kanban_decomposer` (**nicht** auto); ohne Klasse → Default; `design` ohne `_design`-Rolle → Default
- ✅ CLI E2E: `create --class`, `show` (class-Zeile), `reclassify` set/clear, Events
- ✅ Bestehende Suite: 474 passed über die Kern-Kanban-Testdateien; 4 Failures allesamt **vorbestehend/umgebungsbedingt** (fehlendes `fastapi`/`prompt_toolkit`, vorbestehender `task_runs`-Schema-Rebuild-Drift) — gegen Basis-Stand verifiziert
- ✅ Beide Config-YAMLs parsen sauber

### Noch offen (brauchen Live-Account/Quota — separat, mit Go)
- ⏳ **P1:** Fable unter Max-OAuth erreichbar? (toolloser Aux-Call → 200?) — Fable-Limit kann gerade rate-limited sein; Fallback fängt das ab.
- ⏳ **A2 live:** Triage-Card `--class hard` → Decompose läuft real auf Fable, Sub-Cards entstehen.
- ⏳ **B4 live (Gate):** Design-Card → Opus-Executor mit Tools → 200 vs. 400. Bei 400: Rollback/API-Key (siehe ROLLBACK.md).

## Deploy
Code-Änderungen brauchen einen **Gateway-Restart**, damit Dispatcher/Worker den neuen Code + die
DB-Migration laden. Config-Tuning danach ist live (mtime-Cache pro Turn). Details im `RUNBOOK.md`.

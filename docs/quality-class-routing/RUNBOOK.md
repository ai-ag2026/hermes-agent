# RUNBOOK — Quality-Class Routing

## Benutzung

### Härteste Aufgaben von Fable planen lassen (Achse A)
Eine Triage-Card mit `--class hard` erstellen — der Decomposer plant sie dann auf Fable:
```bash
hermes kanban create "Großes Epic X" --triage --class hard
# oder bestehende Card umklassifizieren:
hermes kanban reclassify t_abc123 hard
# Klasse wieder entfernen:
hermes kanban reclassify t_abc123 none
```
Nur `--class`-Werte, für die eine Rolle `auxiliary.kanban_decomposer_<class>` existiert, ändern das
Planer-Modell. Aktuell konfiguriert: **`hard` → `kanban_decomposer_hard` → Fable**. Andere Werte
(z.B. `--class foo`) sind erlaubt, wirken aber erst, wenn man die passende Aux-Rolle anlegt.

### Weitere Klassen hinzufügen (rein per Config, live)
Neue Aux-Rolle nach dem Muster `kanban_decomposer_<klasse>` in `~/.hermes/config.yaml` unter `auxiliary:`
anlegen (z.B. `kanban_decomposer_research` → ein Recherche-Planer-Modell). Kein Code, kein Restart nötig
(mtime-Cache pro Turn). Dann Cards mit `--class research` taggen.

### Design → Opus (Achse B)
Automatisch: Cards mit `assignee=designer` laufen auf Opus 4.8 (Designer-Profil-Modell). Nichts zu taggen.

## Deploy / Restart
Die **Code**-Änderungen (Task/from_row/create_task/decompose/CLI) brauchen einen Gateway-Restart, damit
Dispatcher + Worker den neuen Code laden und die DB-Migration (`tasks.task_class`) auf dem Live-Board läuft.

```bash
# Gateway-Status / Restart — bevorzugt über die vorhandene systemd-Unit bzw. das Projekt-Skript.
# (Restart-Autorität: Gateway + WebUI dürfen evidenzbasiert neu gestartet werden.)
systemctl --user restart hermes-gateway    # falls so benannt; sonst die tatsächliche Unit/Skript
# danach: Logs auf saubere Migration prüfen
journalctl --user -u hermes-gateway -n 50 --no-pager | grep -i "kanban migration\|task_class\|error"
```
Config-Tuning (Modellwahl je Rolle/Profil) danach ist **live** ohne Restart.

## Verifikation nach Deploy (Live, verbraucht Account/Quota)
```bash
# A) Fable-Planer: harte Triage-Card decomposen und Provider/Modell im Log prüfen
hermes kanban create "Testepic hart" --triage --class hard --body "…genug Kontext zum Zerlegen…"
#   → Auto-Decompose (kanban.auto_decompose) ODER manuell anstoßen; erwartet: provider=anthropic,
#     model=claude-fable-5, HTTP 200, Sub-Cards entstehen. Bei Fable-Rate-Limit: Fallback gpt-5.5.
# B) Design-Executor-Gate: Design-Card an assignee=designer, Tool-Nutzung provozieren → 200 vs 400.
#   Bei 400 "extra usage": ROLLBACK.md → Teil-Rollback Achse B.
```

## Auth-Hinweise / Fallstricke
- Fable-Planer nutzt auto-erkannte Claude-Code-Credentials (Max/OAuth); **toollos → kein 400**.
  Fable-Limit setzt nur alle paar Stunden zurück → Fallback `gpt-5.5`/`deepseek` fängt Rate-Limits ab.
- Opus-Design läuft über OAuth + `mcp__`-Rewrite; **native Tools (edit/terminal) können 400 werfen**.
  Designer-Profil ist tool-arm gehalten; falls doch 400 → Rollback oder `ANTHROPIC_API_KEY`.
- Der Konventions-Router schaltet nur auf eine Klassen-Rolle um, wenn sie **konfiguriert** ist —
  sonst Default-Decomposer (bewusst, sonst würde eine unbekannte Rolle auf „auto"/Hauptmodell routen).

## Referenzen
- Design/Audit: `README.md` (dieses Verzeichnis)
- Rollback: `ROLLBACK.md`
- Plan: `~/.claude/plans/swift-chasing-tide.md`
- Recherche/Hintergrund: `~/.claude/plans/fable-planner-hermes-kanban.md`

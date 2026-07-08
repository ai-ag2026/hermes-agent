# ROLLBACK — Quality-Class Routing

Zwei getrennte Rückbau-Achsen: **Code** (Fork-Git) und **Config** (`~/.hermes`, nur per Backup).

Config-Backup dieser Änderung:
`~/.hermes/backups/quality-class-routing-20260709-003955/`
- `root-config.yaml.bak`      → `~/.hermes/config.yaml`
- `designer-config.yaml.bak`  → `~/.hermes/profiles/designer/config.yaml`

---

## Schnell-Rollback (alles zurück)

```bash
# 1) Config zurückspielen (Live, unter ~/.hermes)
BK=~/.hermes/backups/quality-class-routing-20260709-003955
cp "$BK/root-config.yaml.bak"     ~/.hermes/config.yaml
cp "$BK/designer-config.yaml.bak" ~/.hermes/profiles/designer/config.yaml

# 2) Code zurück (Fork-Repo)
cd ~/.hermes/hermes-agent
git checkout tars/completion-drive-20260707     # Basis-Branch vor dieser Arbeit
# (oder auf der Feature-Branch bleiben und die Datei-Änderungen verwerfen:
#  git checkout -- hermes_cli/kanban_db.py hermes_cli/kanban.py hermes_cli/kanban_decompose.py)

# 3) Gateway neu starten, damit alter Code + Config wieder greifen
#    (siehe RUNBOOK.md → Deploy/Restart)
```

Die **DB-Migration** (`tasks.task_class`) muss NICHT zurückgebaut werden: die Spalte ist nullable und
wird von altem Code schlicht ignoriert. SQLite-Spalten lassen sich ohnehin nicht trivial droppen; die
Spalte ist harmlos. (Falls dennoch gewünscht: Tabelle rebuilden — nicht empfohlen.)

---

## Teil-Rollback

### Nur Achse B (Opus-Design) zurück — z.B. wenn Opus+Tools 400 wirft
Nur das Designer-Profil-Modell zurück auf den vorherigen Default:
```bash
cp ~/.hermes/backups/quality-class-routing-20260709-003955/designer-config.yaml.bak \
   ~/.hermes/profiles/designer/config.yaml
```
Alternativ manuell im `model:`-Block von `~/.hermes/profiles/designer/config.yaml` den auskommentierten
Rollback-Block reaktivieren (`provider: openrouter`, `default: deepseek/deepseek-v4-pro`).
Greift beim nächsten Designer-Worker-Spawn (kein Gateway-Restart nötig).

### Nur Achse A (Fable-Planer) deaktivieren — ohne Code-Rückbau
Die Aux-Rolle aus `~/.hermes/config.yaml` entfernen (oder umbenennen), dann tut der Decomposer für
`--class hard`-Cards wieder das Default-Verhalten (Konventions-Router findet keine Rolle → Fallback):
```bash
cp ~/.hermes/backups/quality-class-routing-20260709-003955/root-config.yaml.bak \
   ~/.hermes/config.yaml
```
Greift pro Turn live (mtime-Cache). Der Code bleibt drin, ist aber ohne konfigurierte Rolle inert.

### Nur den Code zurück, Config behalten
```bash
cd ~/.hermes/hermes-agent
git checkout -- hermes_cli/kanban_db.py hermes_cli/kanban.py hermes_cli/kanban_decompose.py
```
Achtung: dann ist `kanban_decomposer_hard` in der Config zwar vorhanden, wird aber nicht mehr per
`task_class` angesteuert (der Rollen-Switch ist weg) → effektiv inert. Sauberer ist der Config-Rollback oben.

---

## Verhaltens-Neutralität als „weicher" Rollback
Ohne `--class`-Tags **und** ohne `auxiliary.kanban_decomposer_*`-Rollen verhält sich alles wie vorher.
Man kann die Funktion also auch einfach „nicht benutzen", ohne irgendetwas zurückzubauen — außer der
Achse-B-Profiländerung, die sofort wirkt und daher aktiv zurückgesetzt werden muss, wenn nicht gewünscht.

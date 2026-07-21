#!/usr/bin/env python3
"""Hermetic test runner: the process-level isolation the 2026-07-20 incident
demanded.

Why this exists
---------------
On 2026-07-20 23:55 a review subagent ran ``python -m pytest -q tests/`` from
a branch archive without process-wide isolation. The per-test fixture in
``tests/conftest.py`` redirects ``HERMES_HOME`` but deliberately not ``HOME``,
does not cover import-time path resolution, and cannot cover subprocesses or
surviving threads. One of those gaps deleted most of the real ``~/.hermes``
(state.db 7.7 GB, kanban.db, profiles, the repo itself). The fixture's own
comment had announced the gap; the store guard works at the SQLite layer and
could not have prevented file-level deletion.

Consequence (WIEDERHERSTELLUNG.md §1): never again a suite run without
process-wide ``HOME``/``HERMES_HOME``/``XDG_*``/``TMPDIR`` redirection and a
read-only real ``~/.hermes``. This runner is that rule, made executable.

What it does
------------
1. Builds a throwaway sandbox root and redirects, **process-wide for the
   whole test process tree**: ``HOME``, ``HERMES_HOME``, ``XDG_CONFIG_HOME``,
   ``XDG_DATA_HOME``, ``XDG_STATE_HOME``, ``XDG_CACHE_HOME``, ``TMPDIR`` and
   pytest's ``--basetemp``. Subprocesses inherit all of it — the exact hole
   the per-test fixture cannot close.
2. Remounts the real ``~/.hermes`` **read-only** for the test process via
   bubblewrap (kernel-enforced; a repeat of the incident becomes ``EROFS``
   instead of data loss). ``--no-sandbox`` skips this only if bwrap is
   missing; the env redirection always applies.
3. **Brackets** the run: records an inventory (path, size, mtime) of the real
   ``~/.hermes`` before and after, and reports every drifted path. With
   ``--strict`` any drift fails the run; without it, drift in known
   live-noise paths (running gateway) is listed but tolerated. A green suite
   alone proves nothing about isolation — only the bracket does.
4. Writes a durable log with the header the R7 review required: command,
   revision, start/end, exit code, environment overrides, and the semantic
   cron manifest (sha256 over the sorted job ids — the gate's formula) before
   and after.

Usage
-----
    scripts/run_tests_hermetic.py [--strict] [--log-dir DIR] [--keep]
                                  [pytest args ...]

Defaults: ``-q -p no:cacheprovider --timeout=60 tests/`` from the repo root
the script lives in. Bytecode is suppressed (``PYTHONDONTWRITEBYTECODE``) so
a read-only repo checkout works too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL_HOME = Path.home()
REAL_HERMES = REAL_HOME / ".hermes"

# Paths inside ~/.hermes a RUNNING gateway legitimately touches. Drift here is
# reported but does not fail a non---strict run. Everything else failing the
# bracket means the isolation leaked and the run is invalid.
LIVE_NOISE_PREFIXES = (
    "state.db", "kanban.db", "cron.db", "lcm.db", "projects.db",
    "response_store.db", "verification_evidence.db", "processes.json",
    "gateway_state.json", "gateway.lock", "auth.lock", "logs/", "telemetry/",
    "cache/", "audio_cache/", "image_cache/", "sessions/", "state/",
    "tool-state/", "temporal_context/", "hindsight/", "cron/jobs.json",
    "cron/last_tick", "cron/scheduler", "cron/output/", "cron/ticker_",
    "cron/.tick.lock", "kanban/logs/", "kanban/current/",
    "ops/detached-work-watchdog", "workspace/reports/", "workspace/.state/",
)


def cron_manifest() -> str:
    """The gate's semantic manifest: sha256 over the sorted job ids.

    Reproduces the G0 anchor (``49106a69…`` for the 57-job baseline);
    verified against the anchor on 2026-07-21. Raw file bytes are useless
    here — the scheduler rewrites ``last_run`` timestamps on every tick.
    """
    jobs_file = REAL_HERMES / "cron" / "jobs.json"
    if not jobs_file.exists():
        return "absent"
    data = json.loads(jobs_file.read_text())
    jobs = data.get("jobs", data) if isinstance(data, dict) else data
    ids = sorted(jobs.keys()) if isinstance(jobs, dict) else sorted(
        j["id"] for j in jobs)
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    return f"{len(ids)} Jobs | {digest}"


def inventory(root: Path) -> dict[str, tuple]:
    """Path → (size, mtime) for every file under root. The bracket."""
    inv: dict[str, tuple] = {}
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        # The sandbox must never recurse into itself via symlinks.
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            inv[str(p.relative_to(root))] = (st.st_size, int(st.st_mtime))
    return inv


def classify_drift(before: dict, after: dict) -> tuple[list, list]:
    """Changed/added/removed paths, split into (noise, violations)."""
    drifted = []
    for path in set(before) | set(after):
        if before.get(path) != after.get(path):
            kind = ("geändert" if path in before and path in after
                    else "NEU" if path in after else "GELÖSCHT")
            drifted.append((path, kind))
    noise = [d for d in drifted if any(d[0].startswith(p) for p in LIVE_NOISE_PREFIXES)]
    violations = [d for d in drifted if d not in noise]
    return noise, violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--strict", action="store_true",
                        help="jede Drift im echten ~/.hermes ist ein Fehler")
    parser.add_argument("--log-dir", type=Path,
                        default=REPO / ".hermetic-logs",
                        help="Ablage für Log + Header (Default: repo/.hermetic-logs)")
    parser.add_argument("--keep", action="store_true",
                        help="Sandbox-Root nach dem Lauf behalten")
    parser.add_argument("--no-sandbox", action="store_true",
                        help="bwrap weglassen (nur Env-Umleitung; NICHT für Vollscope)")
    parser.add_argument("pytest_args", nargs="*",
                        help="Argumente für pytest (Default: -q tests/)")
    args, passthrough = parser.parse_known_args()

    pytest_args = [*args.pytest_args, *passthrough] or ["-q", "tests/"]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"hermetic-{stamp}.log"

    sandbox = Path(tempfile.mkdtemp(prefix="hermetic-", dir="/tmp"))
    sub = {name: sandbox / name for name in
           ("home", "hermes", "xdg-config", "xdg-data", "xdg-state",
            "xdg-cache", "tmp", "basetemp")}
    for p in sub.values():
        p.mkdir()

    env = os.environ.copy()
    overrides = {
        "HOME": str(sub["home"]),
        "HERMES_HOME": str(sub["hermes"]),
        "XDG_CONFIG_HOME": str(sub["xdg-config"]),
        "XDG_DATA_HOME": str(sub["xdg-data"]),
        "XDG_STATE_HOME": str(sub["xdg-state"]),
        "XDG_CACHE_HOME": str(sub["xdg-cache"]),
        "TMPDIR": str(sub["tmp"]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(overrides)

    cmd = [sys.executable, "-m", "pytest",
           "-p", "no:cacheprovider", f"--basetemp={sub['basetemp']}",
           *pytest_args]

    bwrap = shutil.which("bwrap")
    if not args.no_sandbox:
        if not bwrap:
            print("FEHLER: bwrap fehlt. Entweder installieren oder bewusst "
                  "--no-sandbox setzen (dann NUR Env-Umleitung, kein "
                  "Schreibschutz).", file=sys.stderr)
            return 2
        cmd = [bwrap, "--dev-bind", "/", "/",
               "--ro-bind", str(REAL_HERMES), str(REAL_HERMES),
               "--die-with-parent", "--", *cmd]

    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True).stdout.strip() or "unbekannt"

    manifest_before = cron_manifest()
    print("Bracket: Inventar des echten ~/.hermes …", flush=True)
    inv_before = inventory(REAL_HERMES)

    header = [
        f"Start:      {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Repo:       {REPO}",
        f"Revision:   {git_rev}",
        f"Befehl:     {' '.join(cmd)}",
        f"Sandbox:    {sandbox}  (bwrap: {'ja' if not args.no_sandbox else 'NEIN'})",
        f"Overrides:  {json.dumps(overrides)}",
        f"Inventar:   {len(inv_before)} Dateien unter {REAL_HERMES}",
        f"Cron-Manifest vor Lauf: {manifest_before}",
    ]
    print("\n".join(header), flush=True)

    t0 = time.time()
    with log_path.open("w") as log:
        log.write("\n".join(header) + "\n" + "=" * 70 + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            sys.stdout.write(line)
        rc = proc.wait()
    duration = time.time() - t0

    manifest_after = cron_manifest()
    inv_after = inventory(REAL_HERMES)
    noise, violations = classify_drift(inv_before, inv_after)

    footer = [
        "=" * 70,
        f"Ende:       {time.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"({duration:.0f}s)",
        f"Exitcode:   {rc}",
        f"Cron-Manifest nach Lauf: {manifest_after}",
        f"Manifest unverändert:    "
        f"{'JA' if manifest_after == manifest_before else 'NEIN — PRÜFEN'}",
        f"Bracket-Drift: {len(noise)} Live-Rauschen, "
        f"{len(violations)} VERLETZUNGEN",
    ]
    for path, kind in violations:
        footer.append(f"  VERLETZUNG: {kind} {path}")
    for path, kind in noise[:20]:
        footer.append(f"  rauschen:   {kind} {path}")
    if len(noise) > 20:
        footer.append(f"  … {len(noise) - 20} weitere Rausch-Pfade")

    isolation_ok = not violations and (not args.strict or not noise)
    footer.append(f"ISOLATION:  {'OK' if isolation_ok else 'VERLETZT'}"
                  + ("" if isolation_ok else " — Lauf ungültig, Befund sichern"))
    footer.append(f"Log:        {log_path}")
    print("\n".join(footer), flush=True)
    with log_path.open("a") as log:
        log.write("\n".join(footer) + "\n")

    if args.keep:
        print(f"Sandbox behalten: {sandbox}")
    else:
        shutil.rmtree(sandbox, ignore_errors=True)

    if not isolation_ok:
        return 3
    return rc


if __name__ == "__main__":
    sys.exit(main())

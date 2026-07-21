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

Trust model (hardened after the 2026-07-21 TARS review)
-------------------------------------------------------
There is **no attestation flag**. The conftest tripwire verifies isolation
mechanically: it resolves the real home from the passwd database (immune to a
redirected ``$HOME``) and requires every live store to be kernel-read-only
for the test process. This runner *produces* that state; it cannot fake it,
and neither can anyone else — ``HERMES_HERMETIC=1`` in the environment is
dead and changes nothing.

What it does
------------
1. Builds a throwaway sandbox root and constructs the child environment from
   an **allowlist** — inherited Hermes store/gateway/workspace pins
   (``HERMES_*``), credentials and ``XDG_*``/``TMPDIR`` never reach the test
   process. ``HOME``, ``HERMES_HOME``, all ``XDG_*`` dirs (including
   ``XDG_RUNTIME_DIR``), ``TMPDIR`` and pytest's ``--basetemp`` point into
   the sandbox, process-wide for the whole test process tree.
2. Runs the tests under bubblewrap with the **host root read-only**
   (``--ro-bind / /``). Writable are only: a fresh tmpfs ``/tmp``, the
   sandbox, and the repo checkout. Every known state-bearing root (real
   ``~/.hermes``, an original custom ``HERMES_HOME``, the datastore mount)
   is additionally re-bound read-only *after* the writable binds, so it
   stays read-only even if it overlaps one of them. A repeat of the
   incident becomes ``EROFS`` against any of those targets, not just the
   one path the incident happened to hit.
3. **Refuses** ``--no-sandbox`` when any live store is writable. The only
   unhermetic path is the human-set, spelled-out
   ``HERMES_ALLOW_UNHERMETIC=yes-i-accept-the-risk`` directly against
   pytest — a deliberate forensic escape hatch that this runner never
   attests and never takes itself.
4. **Brackets** the run: records an inventory (size, ``st_mtime_ns``, and a
   content sha256 for critical manifests) of every protected root before and
   after, and reports every drifted path. With ``--strict`` any drift fails
   the run; without it, drift in known live-noise paths (running gateway) is
   listed but tolerated. A green suite alone proves nothing about isolation
   — only the bracket does.
5. Writes a durable log **outside the observed roots** (default:
   ``<real home>/.hermetic-logs``; a log dir under a protected root is
   refused, so a ``--strict`` bracket can never be polluted by its own
   evidence). The header records target and runner identity **separately**:
   target SHA, runner/conftest/store-guard content hashes, command,
   environment overrides, and the semantic cron manifest (sha256 over the
   sorted job ids — the gate's formula) before and after.

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


def real_home() -> Path:
    """The invoking user's home from the passwd database, NOT ``$HOME``.

    ``$HOME`` is exactly the variable this runner redirects — and the one a
    spoofing caller would redirect too. The passwd entry is the mechanical
    ground truth the tripwire and the runner must agree on.
    """
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):  # non-POSIX fallback
        return Path.home()


REAL_HOME = real_home()
REAL_HERMES = REAL_HOME / ".hermes"

# Known state-bearing roots beyond the store itself. Existing entries are
# re-bound read-only inside the sandbox even though the host root already is:
# an explicit late bind wins over any earlier writable bind they might
# overlap, today or after a future edit.
STATE_BEARING_MOUNTS = (Path("/mnt/hermes-datastore"),)

# Paths inside a protected root a RUNNING gateway legitimately touches. Drift
# here is reported but does not fail a non---strict run. Everything else
# failing the bracket means the isolation leaked and the run is invalid.
LIVE_NOISE_PREFIXES = (
    "state.db", "kanban.db", "cron.db", "lcm.db", "projects.db",
    "response_store.db", "verification_evidence.db", "processes.json",
    "gateway_state.json", "gateway.lock", "auth.lock", "logs/", "telemetry/",
    "cache/", "audio_cache/", "image_cache/", "sessions/", "state/",
    "tool-state/", "temporal_context/", "hindsight/", "channel_directory.json",
    "cron/jobs.json",
    "cron/last_tick", "cron/scheduler", "cron/output/", "cron/ticker_",
    "cron/.tick.lock", "kanban/logs/", "kanban/current/",
    "ops/detached-work-watchdog", "workspace/reports/", "workspace/.state/",
)

# Files whose semantic content matters more than size/mtime: the bracket
# records a content hash so a same-size same-second rewrite cannot hide.
CRITICAL_CONTENT = ("config.yaml", ".env", "cron/jobs.json")

# Child environment allowlist. Everything not named here (or overridden by
# the runner) is dropped — including every inherited ``HERMES_*`` pin
# (HERMES_KANBAN_DB, HERMES_MANAGED_DIR, HERMES_REAL_HOME, …), credentials,
# and ``XDG_RUNTIME_DIR``. Negative tests pin this behaviour.
ENV_ALLOWLIST = {
    "PATH", "TERM", "USER", "LOGNAME", "SHELL", "TZ", "HOSTNAME",
    "COLUMNS", "LINES", "LANG", "LANGUAGE", "VIRTUAL_ENV",
}
ENV_ALLOWLIST_PREFIXES = ("LC_",)


def protected_roots(environ: dict | None = None) -> list[Path]:
    """Every existing root the test process must never be able to write.

    Order matters downstream: these become the *last* (winning) bwrap binds.
    Covers the real ``~/.hermes``, an original custom ``HERMES_HOME`` (which
    ``Path.home()``-based code would miss), and known state-bearing mounts.
    """
    environ = os.environ if environ is None else environ
    roots: list[Path] = []
    candidates = [REAL_HERMES]
    custom = environ.get("HERMES_HOME")
    if custom:
        candidates.append(Path(custom))
    candidates.extend(STATE_BEARING_MOUNTS)
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def live_writable_roots(roots: list[Path]) -> list[Path]:
    """Protected roots that hold a live store AND are writable right now."""
    return [r for r in roots
            if (r / "state.db").exists() and os.access(r, os.W_OK)]


def build_child_env(sub: dict[str, Path], base_env: dict | None = None,
                    protected: list[Path] | None = None) -> tuple[dict, dict]:
    """(child_env, overrides): allowlisted base + sandbox redirections.

    No attestation flag is set — hermeticity is proven to the tripwire by the
    kernel-read-only state of the protected roots, never declared.
    ``HERMES_HERMETIC_PROTECT`` only *adds* roots for the tripwire to check;
    forging it can tighten the check, not loosen it.
    """
    base_env = os.environ.copy() if base_env is None else dict(base_env)
    env = {k: v for k, v in base_env.items()
           if k in ENV_ALLOWLIST or k.startswith(ENV_ALLOWLIST_PREFIXES)}
    overrides = {
        "HOME": str(sub["home"]),
        "HERMES_HOME": str(sub["hermes"]),
        "XDG_CONFIG_HOME": str(sub["xdg-config"]),
        "XDG_DATA_HOME": str(sub["xdg-data"]),
        "XDG_STATE_HOME": str(sub["xdg-state"]),
        "XDG_CACHE_HOME": str(sub["xdg-cache"]),
        "XDG_RUNTIME_DIR": str(sub["xdg-runtime"]),
        "TMPDIR": str(sub["tmp"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HERMES_HERMETIC_PROTECT": os.pathsep.join(
            str(p) for p in (protected or [])),
    }
    env.update(overrides)
    return env, overrides


def build_bwrap_command(bwrap: str, sandbox: Path, protected: list[Path],
                        inner_cmd: list[str]) -> list[str]:
    """Default-deny mount plan: host read-only, explicit writes only.

    Bind order is the contract: later binds win, so the protected roots come
    last and stay read-only even where they overlap the writable repo or
    sandbox (e.g. a live-checkout repo inside ``~/.hermes``).
    """
    cmd = [bwrap,
           "--ro-bind", "/", "/",
           "--dev", "/dev",
           "--proc", "/proc",
           "--tmpfs", "/tmp",
           "--bind", str(sandbox), str(sandbox),
           "--bind", str(REPO), str(REPO)]
    for root in protected:
        cmd += ["--ro-bind", str(root), str(root)]
    cmd += ["--die-with-parent", "--", *inner_cmd]
    return cmd


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "unlesbar"
    return h.hexdigest()


def inventory(root: Path) -> dict[str, tuple]:
    """Path → (size, mtime_ns[, content-sha256]) for every file under root.

    ``st_mtime_ns`` because second-resolution lets a same-size rewrite within
    one second pass unseen; content hashes for the critical manifests because
    even nanoseconds can in principle be restored by an attacker-shaped bug.
    """
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
            rel = str(p.relative_to(root))
            entry: tuple = (st.st_size, st.st_mtime_ns)
            if rel in CRITICAL_CONTENT:
                entry += (_sha256_file(p),)
            inv[rel] = entry
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
                        help="jede Drift in den geschützten Roots ist ein Fehler")
    parser.add_argument("--log-dir", type=Path,
                        default=REAL_HOME / ".hermetic-logs",
                        help="Ablage für Log + Header (Default: ~/.hermetic-logs; "
                             "muss außerhalb der geschützten Roots liegen)")
    parser.add_argument("--keep", action="store_true",
                        help="Sandbox-Root nach dem Lauf behalten")
    parser.add_argument("--no-sandbox", action="store_true",
                        help="bwrap weglassen — wird bei vorhandenem "
                             "beschreibbarem Live-Store HART verweigert")
    parser.add_argument("pytest_args", nargs="*",
                        help="Argumente für pytest (Default: -q tests/)")
    args, passthrough = parser.parse_known_args()

    protected = protected_roots()
    live = live_writable_roots(protected)

    if args.no_sandbox and live:
        print("VERWEIGERT: --no-sandbox bei beschreibbarem Live-Store "
              f"({', '.join(str(p) for p in live)}). Dieser Runner attestiert "
              "keinen unhermetischen Lauf. Forensik läuft — wenn überhaupt — "
              "nur bewusst und menschlich freigegeben direkt gegen pytest mit "
              "HERMES_ALLOW_UNHERMETIC=yes-i-accept-the-risk, besser auf "
              "einer Wegwerf-VM.", file=sys.stderr)
        return 2

    log_dir = args.log_dir.resolve()
    for root in protected:
        if log_dir == root or root in log_dir.parents:
            print(f"VERWEIGERT: --log-dir {log_dir} liegt im geschützten "
                  f"Root {root} — das Bracket würde seinen eigenen Beweis "
                  "verschmutzen. Logziel außerhalb wählen.", file=sys.stderr)
            return 2

    pytest_args = [*args.pytest_args, *passthrough] or ["-q", "tests/"]
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hermetic-{stamp}.log"

    sandbox = Path(tempfile.mkdtemp(prefix="hermetic-", dir="/tmp"))
    sub = {name: sandbox / name for name in
           ("home", "hermes", "xdg-config", "xdg-data", "xdg-state",
            "xdg-cache", "xdg-runtime", "tmp", "basetemp")}
    for p in sub.values():
        p.mkdir()
    sub["xdg-runtime"].chmod(0o700)  # XDG spec: runtime dir must be 0700

    env, overrides = build_child_env(sub, protected=protected)

    cmd = [sys.executable, "-m", "pytest",
           "-p", "no:cacheprovider", f"--basetemp={sub['basetemp']}",
           *pytest_args]

    bwrap = shutil.which("bwrap")
    if not args.no_sandbox:
        if not bwrap:
            print("FEHLER: bwrap fehlt. Entweder installieren oder — NUR "
                  "ohne Live-Store — bewusst --no-sandbox setzen.",
                  file=sys.stderr)
            return 2
        cmd = build_bwrap_command(bwrap, sandbox, protected, cmd)

    git_rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
        text=True).stdout.strip() or "unbekannt"
    # Runner identity, SEPARATE from the target SHA (TARS review §4): the
    # content hashes of the trust set actually loaded for this run.
    runner_ident = {
        rel: _sha256_file(REPO / rel)
        for rel in ("scripts/run_tests_hermetic.py", "tests/conftest.py",
                    "tests/store_guard.py")
    }

    manifest_before = cron_manifest()
    # Bracketed are the STORE roots only. The state-bearing mounts are
    # kernel-protected via their read-only binds, but walking a
    # multi-terabyte datastore for an inventory is neither feasible nor
    # meaningful evidence.
    bracket_roots = [r for r in protected
                     if not any(r == m for m in STATE_BEARING_MOUNTS)]
    print("Bracket: Inventar der Store-Roots …", flush=True)
    inv_before = {root: inventory(root) for root in bracket_roots}

    header = [
        f"Start:      {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Repo:       {REPO}",
        f"Target-SHA: {git_rev}",
        f"Python:     {sys.version.split()[0]} ({sys.executable})",
        *[f"Runner-Blob: {rel} sha256={digest}"
          for rel, digest in runner_ident.items()],
        f"Befehl:     {' '.join(cmd)}",
        f"Sandbox:    {sandbox}  (bwrap: {'ja' if not args.no_sandbox else 'NEIN'})",
        f"Geschützt:  {', '.join(str(p) for p in protected)}",
        f"Overrides:  {json.dumps(overrides)}",
        *[f"Inventar:   {len(inv)} Dateien unter {root}"
          for root, inv in inv_before.items()],
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
    noise, violations = [], []
    for root in bracket_roots:
        n, v = classify_drift(inv_before[root], inventory(root))
        noise += [(f"{root}/{p}", k) for p, k in n]
        violations += [(f"{root}/{p}", k) for p, k in v]

    sandboxed = not args.no_sandbox
    # Verdict logic: under bwrap the test process holds a KERNEL-enforced
    # read-only view of every protected root — drift there cannot come from
    # the tests, it is live operation (gateway, cron jobs) by definition. The
    # violation category is therefore only a verdict-carrier without the
    # sandbox, or in --strict mode (maintenance window, services stopped),
    # where any drift at all invalidates the proof.
    if args.strict:
        isolation_ok = not violations and not noise
    elif sandboxed:
        isolation_ok = True
    else:
        isolation_ok = not violations

    drift_label = ("extern:    " if sandboxed and not args.strict
                   else "VERLETZUNG:")
    footer = [
        "=" * 70,
        f"Ende:       {time.strftime('%Y-%m-%d %H:%M:%S %z')}  "
        f"({duration:.0f}s)",
        f"Exitcode:   {rc}",
        f"Cron-Manifest nach Lauf: {manifest_after}",
        f"Manifest unverändert:    "
        f"{'JA' if manifest_after == manifest_before else 'NEIN — PRÜFEN'}",
        f"Bracket-Drift: {len(noise)} Live-Rauschen, {len(violations)} "
        f"{'extern beobachtete Pfade' if sandboxed and not args.strict else 'VERLETZUNGEN'}",
    ]
    for path, kind in violations:
        footer.append(f"  {drift_label} {kind} {path}")
    for path, kind in noise[:20]:
        footer.append(f"  rauschen:   {kind} {path}")
    if len(noise) > 20:
        footer.append(f"  … {len(noise) - 20} weitere Rausch-Pfade")

    footer.append(f"ISOLATION:  {'OK' if isolation_ok else 'VERLETZT'}"
                  + (" (kernel-RO erzwungen; Drift = Live-Betrieb)"
                     if isolation_ok and sandboxed and (violations or noise)
                     else "" if isolation_ok
                     else " — Lauf ungültig, Befund sichern"))
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

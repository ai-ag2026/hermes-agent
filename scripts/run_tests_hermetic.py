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
3. **Refuses** ``--no-sandbox`` when any live store is writable, with no
   escape hatch: there is no env override anywhere in this path. Forensics
   that must run unhermetically belong on a disposable VM with no live
   store, where the tripwire does not fire in the first place.
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
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The runner and the conftest tripwire MUST classify roots identically —
# two copies of that rule would drift silently. Single source of truth:
from tests.hermetic_policy import (  # noqa: E402
    candidate_roots, is_live, is_readonly_mount, passwd_home,
)

REAL_HOME = passwd_home()
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
    Covers the canonical ``~/.hermes``, an ambient custom ``HERMES_HOME``
    (which ``Path.home()``-based code would miss) and known state-bearing
    mounts. Root discovery is shared with the tripwire.
    """
    environ = os.environ if environ is None else environ
    roots = candidate_roots(environ, REAL_HOME)
    for mount in STATE_BEARING_MOUNTS:
        try:
            resolved = mount.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def live_writable_roots(roots: list[Path]) -> list[Path]:
    """Roots holding live state that this process could still damage.

    Same classification the tripwire applies: liveness by canonical markers
    (not just ``state.db``), protection only by a read-only mount — never by
    directory permissions.
    """
    return [r for r in roots if is_live(r) and not is_readonly_mount(r)]


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
    # F-3 (2026-08-02): eigener PID-Namespace. Tests feuern echte
    # Kill-Primitive (test_live_system_guard_self_test.py: os.kill(-1),
    # ``pkill -f python``, killall) und verlassen sich auf den conftest-Guard —
    # ohne --unshare-pid schlug jeder Durchrutscher auf HOST-Prozesse durch
    # (beobachtet: parallele Suiten-Läufe per SIGTERM abgeräumt). Mit dem
    # Namespace sieht /proc nur noch Sandbox-Prozesse und kill(-1) endet an
    # der Namespace-Grenze; bwrap reapt als PID-1-Init die Zombies.
    cmd += ["--unshare-pid", "--die-with-parent", "--", *inner_cmd]
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


#: The trust set whose content decides what a run actually enforced. Hashed
#: into every log header SEPARATELY from the target SHA (TARS review §4): a
#: run pins which code was under test *and* which runner enforced it — those
#: are different questions, and a log that conflates them proves neither.
RUNNER_TRUST_SET = (
    "scripts/run_tests_hermetic.py",
    "tests/hermetic_policy.py",
    "tests/hermetic_guard.py",  # P0-RUN-5: the guard that --noconftest can't skip
    "tests/conftest.py",
    "tests/store_guard.py",
    "pyproject.toml",  # carries the addopts that load the guard at all
)


def runner_identity(repo: Path | None = None) -> dict[str, str]:
    """Content hashes of the runner's trust set as loaded for this run."""
    repo = REPO if repo is None else repo
    return {rel: _sha256_file(repo / rel) for rel in RUNNER_TRUST_SET}


#: Ignored paths that can hold *executable* code but are never the code under
#: test — hashing them would mean walking a venv or node_modules on every run.
_UNHASHED_IGNORED_PREFIXES = (
    ".venv/", "venv/", "node_modules/", ".git/", ".worktrees/",
    "__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
)

#: Ignored files that ARE hashed: anything Python can import or execute.
_HASHED_IGNORED_SUFFIXES = (".py", ".pth", ".so", ".sh")

#: …and anything that *configures* the run. TARS review R10, P1-RUN-6: an ignored
#: ``pytest.ini`` containing ``addopts=-p no:tests.hermetic_guard`` changes which plugins
#: load — it decides whether the tripwire runs at all — yet it matched none of the
#: suffixes above, so the attestation did not move. Effective configuration is executable
#: in every sense that matters here.
_HASHED_IGNORED_NAMES = (
    "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
    "conftest.py", "sitecustomize.py", "usercustomize.py",
)


class SpecialFileInAttestation(RuntimeError):
    """A FIFO/socket/device sits in the attested set — refuse, never guess."""


def _attest_records(repo: Path, paths: list[str]) -> list[tuple[str, str, str]]:
    """``(path, kind, digest)`` per path, typed via ``lstat`` — never followed.

    P1-RUN-6 (R10): hashing through ``open()`` bound neither the file *type* nor a
    symlink's *target*, and a FIFO could block the runner forever. Symlinks are attested
    by their target string, regular files by content, and anything else is fail-closed.
    """
    records: list[tuple[str, str, str]] = []
    for rel in sorted(paths):
        p = repo / rel
        try:
            st = p.lstat()
        except OSError:
            records.append((rel, "missing", ""))
            continue
        mode = st.st_mode
        if stat.S_ISLNK(mode):
            # P1-R11-2B: der ZielsTRING allein genügt nicht — zeigt der Link aus
            # dem Repo hinaus, importiert Python die dortigen BYTES, und die
            # ändern sich, ohne dass der Pfad sich ändert (TARS' Gegenbeleg:
            # VALUE=1 → VALUE=2, Attest unverändert). Repo-interne Ziele deckt
            # der Tree-Hash bereits ab; externe werden hier mitgehasht.
            target = os.readlink(p)
            h = hashlib.sha256(target.encode("utf-8", "surrogateescape"))
            resolved = os.path.realpath(p)
            kind = "symlink"
            if not resolved.startswith(str(repo) + os.sep):
                kind = "symlink-extern"
                try:
                    if stat.S_ISREG(os.stat(resolved).st_mode):
                        h.update(b"\0" + _sha256_file(Path(resolved)).encode())
                    else:
                        h.update(b"\0unlesbar")
                except OSError:
                    h.update(b"\0unlesbar")
            records.append((rel, kind, h.hexdigest()))
        elif stat.S_ISREG(mode):
            records.append((rel, "file", _sha256_file(p)))
        elif stat.S_ISDIR(mode):
            records.append((rel, "dir", ""))
        else:
            raise SpecialFileInAttestation(
                f"{rel} ist eine Sonderdatei (FIFO/Socket/Device) im attestierten Satz — "
                "ihr Inhalt ist nicht reproduzierbar messbar; Lauf abgebrochen"
            )
    return records


def _hash_records(records: list[tuple[str, str, str]]) -> str:
    """Stable digest over ``path\\0kind\\0digest`` for typed records."""
    h = hashlib.sha256()
    for rel, kind, digest in records:
        h.update(rel.encode("utf-8", "surrogateescape"))
        h.update(b"\0" + kind.encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def _special_files(repo: Path) -> list[str]:
    """Non-regular, non-symlink entries in the repo — outside git's view.

    ``git ls-files --others`` does not list FIFOs, sockets or devices at all, so a
    ``pipe.py`` sitting in the tree was neither hashed nor refused: it was simply
    invisible to the attestation. Since importing one would block the interpreter
    rather than fail, the walk finds them directly and the run is refused.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo, onerror=lambda e: None):
        rel_dir = os.path.relpath(dirpath, repo)
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        if rel_dir.startswith(_UNHASHED_IGNORED_PREFIXES):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if not (rel_dir + d + "/").startswith(_UNHASHED_IGNORED_PREFIXES)]
        for name in filenames:
            try:
                mode = os.lstat(os.path.join(dirpath, name)).st_mode
            except OSError:
                continue
            if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                found.append(rel_dir + name)
    return found


def _untracked(repo: Path, ignored: bool) -> list[str]:
    """Untracked paths, either the normal ones or the *relevant* ignored ones."""
    cmd = ["git", "ls-files", "--others", "-z"]
    cmd += ["--ignored", "--exclude-standard"] if ignored else ["--exclude-standard"]
    out = subprocess.run(cmd, cwd=repo, capture_output=True, text=True).stdout
    paths = [p for p in out.split("\0") if p]
    if not ignored:
        return paths
    return [p for p in paths
            if (p.endswith(_HASHED_IGNORED_SUFFIXES)
                or os.path.basename(p) in _HASHED_IGNORED_NAMES)
            and not p.startswith(_UNHASHED_IGNORED_PREFIXES)]


def worktree_state(repo: Path | None = None) -> dict:
    """What the repo tree actually looked like at run time, not just its SHA.

    ``git rev-parse HEAD`` pins the commit; a dirty worktree runs *other*
    code under that same SHA. This records ``git status --porcelain`` (the
    dirty file list) and a content hash of the tracked tree WITH those
    modifications applied (``git stash create`` when dirty, else the HEAD
    tree), so the log attests the bytes that ran — clean or not.

    TARS review R9, **P1-RUN-6**: ``git stash create`` builds its tree from
    the *index* — untracked and ignored files are not in it. A run could
    therefore execute an untracked ``tests/test_evil.py`` or an ignored
    ``sitecustomize.py`` while ``Tree-Hash`` looked exactly like a clean
    checkout. So the attestation now has three parts, combined into one
    ``attest_hash``:

    * the tracked tree (stash tree when dirty, HEAD tree when clean);
    * every untracked non-ignored file, hashed individually;
    * every ignored file that Python can import or execute
      (``.py/.pth/.so/.sh``), except the ones under a venv, ``node_modules``
      or a cache — those are hashed by *name set* would be meaningless and
      walking them costs minutes. That exclusion is listed in the header, so
      what is NOT covered stays visible instead of being implied clean.
    """
    repo = REPO if repo is None else repo

    def _git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True).stdout.strip()

    # NB: do NOT strip() porcelain — its leading column is significant
    # (" M path" for a worktree modification); stripping it shifts every
    # path left by one and corrupts the dirty-file list.
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo,
        capture_output=True, text=True).stdout.rstrip("\n")

    special = _special_files(repo)
    if special:
        raise SpecialFileInAttestation(
            "Sonderdateien (FIFO/Socket/Device) im Repo-Baum: "
            f"{', '.join(special[:10])} — nicht reproduzierbar messbar, Lauf abgebrochen"
        )

    untracked = _untracked(repo, ignored=False)
    ignored_code = _untracked(repo, ignored=True)
    untracked_records = _attest_records(repo, untracked)
    ignored_records = _attest_records(repo, ignored_code)
    untracked_hash = _hash_records(untracked_records)
    ignored_hash = _hash_records(ignored_records)

    if not porcelain:
        tree = _git("rev-parse", "HEAD^{tree}") or "unbekannt"
        state, dirty = "clean", ""
    else:
        # Porcelain v1: two status columns, a space, then the path (index 3);
        # renames appear as "orig -> new" — keep the whole thing.
        dirty = " ".join(line[3:] for line in porcelain.splitlines())
        # ``git stash create`` builds a commit object capturing the dirty
        # state without touching the worktree.
        stash = _git("stash", "create")
        tree = (_git("rev-parse", f"{stash}^{{tree}}") if stash else "") \
            or "unbekannt (stash create fehlgeschlagen)"
        state = "DIRTY"

    attest = hashlib.sha256(
        f"{tree}\n{untracked_hash}\n{ignored_hash}\n".encode()).hexdigest()
    return {
        "state": state, "porcelain": porcelain, "dirty": dirty,
        "tree_hash": tree,
        "untracked": untracked, "untracked_hash": untracked_hash,
        "ignored_code": ignored_code, "ignored_hash": ignored_hash,
        "attest_hash": attest,
        # Full typed inventory — the header prints a summary, the log carries every
        # single record (R10: "höchstens acht Pfade" was not an inventory).
        "records": untracked_records + ignored_records,
    }


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

    # P1-R11-2A: eine externe Ini (``-c /tmp/x.ini``) entscheidet, welche Plugins
    # laden — sie kann den Guard abwählen — liegt aber außerhalb des Repos und
    # damit außerhalb der Attestierung. TARS' Gegenbeleg: Ini geändert,
    # Attest-Hash identisch. Es gibt keinen Grund, eine repo-fremde Testkonfig
    # durch den offiziellen Runner zu erlauben, also: fail-closed abweisen.
    # NB: gegen das ROHE argv prüfen, nicht gegen ``args.pytest_args + passthrough``.
    # argparse reißt ``-c pfad`` auseinander: ``-c`` landet als unbekannte Option im
    # passthrough, der Pfad wird als Positional geschluckt — in der rekonstruierten
    # Liste stehen sie nicht mehr nebeneinander, und die Prüfung liefe ins Leere.
    # TARS R12: die erste Fassung deckte ``-c pfad`` und ``-cpfad`` ab, aber
    # ``-c=/tmp/x.ini`` löste zu ``=/tmp/x.ini`` auf — ein relativer Pfad, der
    # unter dem Repo landete und damit als "innerhalb" durchging, während pytest
    # ihn als externe ``/tmp/x.ini`` liest. Alle Schreibweisen, die pytest
    # akzeptiert, müssen durch denselben Grenzcheck.
    requested_args = sys.argv[1:]
    for i, arg in enumerate(requested_args):
        cfg = None
        if arg in ("-c", "--config-file") and i + 1 < len(requested_args):
            cfg = requested_args[i + 1]
        elif arg.startswith("--config-file="):
            cfg = arg[len("--config-file="):]
        elif arg.startswith("-c") and len(arg) > 2:
            cfg = arg[2:]
            if cfg.startswith("="):      # ``-c=pfad``
                cfg = cfg[1:]
        if not cfg:
            continue
        resolved = Path(cfg).expanduser().resolve()
        if not str(resolved).startswith(str(REPO) + os.sep):
            print(f"VERWEIGERT: externe pytest-Konfiguration {resolved} liegt außerhalb "
                  f"des Repos ({REPO}) und wäre nicht attestiert. Eine Ini entscheidet, "
                  "welche Plugins laden — sie muss versioniert im Repo liegen.",
                  file=sys.stderr)
            return 2


    protected = protected_roots()
    live = live_writable_roots(protected)

    if args.no_sandbox and live:
        print("VERWEIGERT: --no-sandbox bei beschreibbarem Live-Store "
              f"({', '.join(str(p) for p in live)}). Dieser Runner attestiert "
              "keinen unhermetischen Lauf und kennt keinen Env-Override. "
              "Forensik läuft auf einer Wegwerf-VM ohne Live-Store.",
              file=sys.stderr)
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

    # Ablageort des Sandkastens überschreibbar. Vorgabe bleibt /tmp, damit sich
    # für niemanden etwas ändert.
    #
    # Warum es den Schalter braucht (30.07.2026): /tmp ist hier ein tmpfs mit
    # 16 GB, also RAM-gedeckt. Ein Volllauf der Agent-Suite füllt das — und ein
    # per SIGTERM abgebrochener Lauf lässt seinen basetemp liegen, weil pytest
    # das Aufräumen dann überspringt. Zwei solcher Leichen plus ein laufender
    # Lauf haben /tmp zum Überlaufen gebracht; die Suite kippte ab 71 % in eine
    # reine ENOSPC-Fehlerkaskade, das Ergebnis war wertlos. Mit
    # HERMETIC_SANDBOX_ROOT=/var/tmp landet der Sandkasten auf echter Platte.
    _sandbox_root = os.environ.get("HERMETIC_SANDBOX_ROOT") or "/tmp"
    os.makedirs(_sandbox_root, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="hermetic-", dir=_sandbox_root))
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
    worktree = worktree_state(REPO)
    runner_ident = runner_identity()

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
        # The SHA alone attests the commit, not the tree that actually ran
        # (TARS review P1-RUN-3): a dirty worktree tests uncommitted code
        # under an unchanged SHA. Record the exact working-tree state.
        f"Worktree:   {worktree['state']}"
        + (f" — dirty: {worktree['dirty']}" if worktree['porcelain'] else ""),
        f"Tree-Hash:  {worktree['tree_hash']}  (tracked)",
        # P1-RUN-6: the tracked tree says nothing about untracked/ignored code
        # that the run can still import. Inventory both, and name the
        # deliberate blind spot instead of implying full coverage.
        f"Untracked:  {len(worktree['untracked'])} Datei(en) "
        f"sha256={worktree['untracked_hash'][:16]}…"
        + (f" — {' '.join(worktree['untracked'][:8])}"
           + (" …" if len(worktree['untracked']) > 8 else "")
           if worktree['untracked'] else ""),
        f"Ignored-Code: {len(worktree['ignored_code'])} ausführbare Datei(en) "
        f"sha256={worktree['ignored_hash'][:16]}…  "
        f"(nicht gehasht: {', '.join(_UNHASHED_IGNORED_PREFIXES)})",
        f"Attest-Hash: {worktree['attest_hash']}  "
        "(tracked+untracked+ignored-code)",
        # Full typed inventory: every attested path with its kind and digest. The
        # summary lines above are for reading; this is the evidence (R10 P1-RUN-6).
        # P2-R11-2C: volle 64-Hex-Digests — mit 16 Hex war die "vollständige
        # Inventaranlage" aus dem Log heraus nicht nachrechenbar.
        *[f"Attest-Record: {kind:<15} {digest or '-':<64} {rel}"
          for rel, kind, digest in worktree["records"]],
        "Attest-Ausschluss: " + ", ".join(_UNHASHED_IGNORED_PREFIXES)
        + "  (NICHT abgedeckt — weder gehasht noch auf Sonderdateien geprüft)",
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

    # P1-RUN-6 (R10), TOCTOU: the attestation above was taken BEFORE the suite ran.
    # Anything that changed the attested set during the run — a test dropping a
    # conftest, an ignored pytest.ini appearing mid-run — would have gone unrecorded.
    # Re-attest and treat any drift as fail-closed: the header's Attest-Hash must
    # describe the bytes that actually ran, start to finish.
    worktree_after = worktree_state()
    attest_drift = worktree_after["attest_hash"] != worktree["attest_hash"]

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

    footer.append(
        f"Attest-Hash nach Lauf: {worktree_after['attest_hash']}"
        + ("  — UNVERÄNDERT" if not attest_drift else
           "  — ABWEICHUNG! Der attestierte Satz hat sich WÄHREND des Laufs geändert"))
    if attest_drift:
        before = {(r, k): d for r, k, d in worktree["records"]}
        after = {(r, k): d for r, k, d in worktree_after["records"]}
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                footer.append(f"  ATTEST-DRIFT: {key[1]} {key[0]}")
        isolation_ok = False

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

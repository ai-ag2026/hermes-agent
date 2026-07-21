"""Single source of truth for "may this process run tests here?".

Both the conftest tripwire and ``scripts/run_tests_hermetic.py`` decide the
same question — which roots hold live Hermes state, and whether this process
can damage them. Two copies of that rule would drift, and the drift would be
invisible until it mattered. So the rule lives here, once.

Contract (hardened 2026-07-21 after two TARS reviews):

* **Nothing is trusted from the environment.** There is no attestation flag.
  Protection is *observed*: the filesystem carrying a live root must be
  mounted read-only (``statvfs().f_flag & ST_RDONLY``) — what the runner's
  bubblewrap ``--ro-bind`` produces and what makes a repeat of the
  2026-07-20 deletion an ``EROFS`` instead of data loss.
* **Directory permissions are NOT protection.** A ``0555`` directory still
  allows an existing ``state.db`` inside it to be opened, written and
  truncated; only new entries and unlinks are blocked. The earlier check
  (``os.access(root, W_OK)``) accepted exactly that state as safe.
* **Liveness is not "state.db exists".** After a partial loss — precisely
  when the tripwire is needed most — the store can be missing ``state.db``
  and still hold ``profiles/``, ``config.yaml``, ``.env`` or the repo. Any
  canonical marker makes a root live.
* **Every candidate root is found without cooperation.** The real home comes
  from the passwd database (not ``$HOME``, which a hermetic runner redirects
  and a spoofing caller would too), plus the *ambient* ``HERMES_HOME`` — a
  custom/profile store that passwd-based lookup cannot see.
  ``HERMES_HERMETIC_PROTECT`` may name additional roots; it is strictly
  additive, so forging it can only tighten the check, never loosen it.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Files/dirs whose presence makes a root "live Hermes state worth losing".
#: Deliberately broader than ``state.db`` — see the module docstring.
LIVE_MARKERS = (
    "state.db",
    "kanban.db",
    "config.yaml",
    ".env",
    "cron/jobs.json",
    "profiles",
    "memories",
    "hermes-agent",
)

#: Spelled-out human override for forensic setups on a disposable machine.
OVERRIDE_ENV = "HERMES_ALLOW_UNHERMETIC"
OVERRIDE_VALUE = "yes-i-accept-the-risk"

#: Additive-only list of extra roots to protect (the runner passes what it
#: found so a redirected child still guards the original store).
PROTECT_ENV = "HERMES_HERMETIC_PROTECT"


def passwd_home() -> Path:
    """The invoking user's home from the passwd database, NOT ``$HOME``."""
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):  # non-POSIX fallback
        return Path.home()


def candidate_roots(environ=None, home: Path | None = None) -> list[Path]:
    """Every root that could hold Hermes state, deduplicated, existing only.

    Order: canonical ``~/.hermes``, ambient ``HERMES_HOME``, then the
    additive ``HERMES_HERMETIC_PROTECT`` entries.
    """
    environ = os.environ if environ is None else environ
    home = passwd_home() if home is None else home

    raw = [home / ".hermes"]
    ambient = environ.get("HERMES_HOME")
    if ambient:
        raw.append(Path(ambient))
    for extra in environ.get(PROTECT_ENV, "").split(os.pathsep):
        if extra:
            raw.append(Path(extra))

    roots: list[Path] = []
    for cand in raw:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def is_live(root: Path) -> bool:
    """True if *root* holds state a test run must never be able to destroy."""
    return any((root / marker).exists() for marker in LIVE_MARKERS)


def is_readonly_mount(root: Path) -> bool:
    """True if *root* sits on a read-only mount — the only real protection.

    This is what bubblewrap's ``--ro-bind`` produces; a write attempt then
    fails with ``EROFS`` in the kernel rather than relying on any userspace
    check. Permissions are deliberately NOT consulted here.
    """
    statvfs = getattr(os, "statvfs", None)
    if statvfs is None:  # non-POSIX: no such guarantee available
        return False
    try:
        return bool(statvfs(root).f_flag & os.ST_RDONLY)
    except OSError:
        return False


def unprotected_live_roots(environ=None, home: Path | None = None) -> list[Path]:
    """Live roots this process could still damage. Empty means safe to run."""
    return [r for r in candidate_roots(environ, home)
            if is_live(r) and not is_readonly_mount(r)]


def refusal_reason(environ=None, home: Path | None = None) -> str | None:
    """Why this process must not run tests here, or None if it may.

    The human override is honoured last: it can only excuse roots that were
    already identified, never hide them.
    """
    environ = os.environ if environ is None else environ
    exposed = unprotected_live_roots(environ, home)
    if not exposed:
        return None
    if environ.get(OVERRIDE_ENV) == OVERRIDE_VALUE:
        return None
    return (
        "ABGEBROCHEN: Testlauf mit beschreibbarem echten Live-Store "
        f"({', '.join(str(s) for s in exposed)}). Genau so wurde am "
        "20.07.2026 die Installation gelöscht. Diese Roots liegen NICHT auf "
        "einem read-only gemounteten Dateisystem — Verzeichnisrechte zählen "
        "nicht als Schutz. Nutze scripts/run_tests_hermetic.py (bwrap "
        "--ro-bind, kernel-erzwungen; ein Attestierungs-Flag gibt es nicht) "
        f"— oder setze {OVERRIDE_ENV}={OVERRIDE_VALUE}, wenn du wirklich "
        "weißt, was du tust."
    )

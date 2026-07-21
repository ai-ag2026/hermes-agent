"""Mechanical guard: no test may create a durable store in the real home.

Why a guard and not more discipline
-----------------------------------
Between 2026-07-19 and 2026-07-20 the production home was written to by tests
**four times**, each with a different immediate cause:

1. ``cron/jobs.py`` froze its paths at import, so a fixture redirect never
   reached them (seven dummy jobs, one of which the live scheduler executed);
2. the process ledger resolved its path at write time, so a reader thread
   outliving its fixture wrote to whatever ``HERMES_HOME`` had become;
3. a test called ``monkeypatch.undo()``, which reverts the ``HERMES_HOME``
   redirect too, and a later read created an empty database in the real home;
4. before that, a baseline comparison run deliberately removed the isolation
   fixture and polluted the store again.

Each was fixed individually. The pattern did not stop, and after-the-fact hash
bracketing only ever *detected* the damage -- and only when the bracketed set
happened to include the right file and the right test scope. It missed case 2
entirely because the triggering test lived in a directory the bracketed runs
did not cover.

So this is not another hash comparison. It makes the mistake **fail at the
moment it is made**, with a traceback pointing at the test that did it.

How it works
------------
``sqlite3.connect`` is wrapped for the duration of the test session. Any
write-capable open of a path under the real ``~/.hermes`` raises. Read-only
URIs (``mode=ro``) are allowed, because tests legitimately read production
data for probes.

The real home is resolved ONCE, at import, from the ambient environment --
before any fixture has had a chance to redirect it. That is the one place
where import-time freezing is correct rather than a bug: we want the *actual*
home, not whatever the current test says it is.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _passwd_home() -> Path:
    """The real home from the passwd database, NOT ``$HOME``.

    ``$HOME`` is exactly what a hermetic runner (or a spoofing caller)
    redirects; the passwd entry is ground truth no environment can move.
    Hardened after the 2026-07-21 TARS review alongside the conftest
    tripwire — the two must agree on what "real" means.
    """
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):  # non-POSIX fallback
        return Path.home()


# Resolved once, deliberately: these must be the real stores, not redirected
# ones. Everything else in this codebase resolves paths at call time; this is
# the exception that proves why the rule exists. Protected are BOTH the
# passwd-home store (immune to a redirected $HOME) and whatever HERMES_HOME
# pointed at when this module loaded — a custom/profile store the passwd
# path would miss.
_REAL_HOME = (_passwd_home() / ".hermes").resolve()
_PROTECTED = tuple(dict.fromkeys(
    p for p in (
        _REAL_HOME,
        Path(os.environ["HERMES_HOME"]).resolve()
        if os.environ.get("HERMES_HOME") else None,
    ) if p is not None
))

_original_connect = sqlite3.connect
_enabled = False


class ProductionStoreWriteAttempt(AssertionError):
    """A test tried to open a production store for writing."""


def _is_read_only_uri(target: str) -> bool:
    return "mode=ro" in target or "immutable=1" in target


def _under_real_home(target: str) -> bool:
    if target in (":memory:", "") or target.startswith("file::memory:"):
        return False
    path_part = target
    if path_part.startswith("file:"):
        path_part = path_part[5:].split("?", 1)[0]
    try:
        resolved = Path(path_part).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return any(resolved == root or root in resolved.parents
               for root in _PROTECTED)


def _guarded_connect(database, *args, **kwargs):
    target = str(database)
    if _enabled and _under_real_home(target) and not _is_read_only_uri(target):
        raise ProductionStoreWriteAttempt(
            f"Test tried to open {target} for writing.\n"
            f"That is under a protected real store ({', '.join(map(str, _PROTECTED))}).\n"
            "Redirect HERMES_HOME to a tmp_path, bind the store explicitly, "
            "or open the source read-only with '?mode=ro'.\n"
            "This guard exists because after-the-fact hash bracketing missed "
            "this class of leak repeatedly."
        )
    return _original_connect(database, *args, **kwargs)


def enable() -> None:
    global _enabled
    sqlite3.connect = _guarded_connect
    _enabled = True


def disable() -> None:
    global _enabled
    _enabled = False
    sqlite3.connect = _original_connect

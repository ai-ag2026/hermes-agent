"""Tests for the bounded kanban init lock (issue #36644).

`connect()` wrapped its entire body in an unbounded blocking `flock(LOCK_EX)`
on every call. A single process stalled inside the critical section blocked the
long-lived gateway dispatcher's next-tick `connect()` forever — no timeout, no
recovery, board silently stops being worked.

Two fixes, both covered here:
1. Fast path: once a path is initialized in this process, `connect()` skips the
   cross-process init lock entirely (nothing left to serialize), so a held lock
   cannot block a steady-state connect.
2. Bounded acquire: even on first-init, `_cross_process_init_lock` retries a
   non-blocking acquire up to a deadline, then **fails closed** with a
   `TimeoutError` rather than hanging forever or racing non-idempotent
   migration DDL without the cross-process lock (superseding the original
   #36644 "proceed with a WARNING" behavior; see `harden(kanban): fail
   closed across SQLite lifecycle hazards`).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return home


def _hold_init_lock(db_path: Path):
    """Return (start_event, release_event, thread) holding the init lock."""
    holding = threading.Event()
    release = threading.Event()

    def _holder():
        with kb._cross_process_init_lock(db_path):
            holding.set()
            release.wait(timeout=10)

    t = threading.Thread(target=_holder, daemon=True)
    t.start()
    assert holding.wait(timeout=5), "holder thread never acquired the lock"
    return release, t


def test_initialized_path_connect_skips_init_lock(kanban_home):
    """A connect to an already-initialized path must not block on the init lock."""
    db_path = kb.kanban_db_path(board="default")
    # Initialize once.
    kb.connect().close()
    assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

    # Hold the init lock; a fast-path connect must return promptly anyway.
    release, t = _hold_init_lock(db_path)
    try:
        start = time.monotonic()
        kb.connect().close()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"fast-path connect blocked on the init lock ({elapsed:.2f}s)"
    finally:
        release.set()
        t.join(timeout=5)


def test_first_init_connect_is_bounded_when_lock_held(kanban_home, monkeypatch):
    """First-init connect must time out the cross-process lock and fail
    closed with an actionable TimeoutError, not hang forever, when another
    holder owns it. Proceeding without the lock would risk racing
    non-idempotent migration DDL against the lock holder, so a bounded
    failure — not a bounded success — is the safe behavior here."""
    monkeypatch.setattr(kb, "_INIT_LOCK_TIMEOUT_SECONDS", 0.6)
    db_path = kb.kanban_db_path(board="default")

    release, t = _hold_init_lock(db_path)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="init lock"):
            kb.connect()  # path NOT yet initialized — must take the bounded path
        elapsed = time.monotonic() - start
        # Failed within roughly the timeout window (not unbounded).
        assert 0.4 <= elapsed < 3.0, f"expected bounded ~0.6s acquire, got {elapsed:.2f}s"
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS
    finally:
        release.set()
        t.join(timeout=5)

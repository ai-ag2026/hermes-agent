"""The guard must actually stop a production write.

A guard that is installed but never fires is indistinguishable from no guard
at all, so this asserts both directions: the write is refused, and the
legitimate read-only path still works.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests import store_guard


def test_a_write_open_under_the_real_home_is_refused():
    target = store_guard._REAL_HOME / "guard-probe.db"
    with pytest.raises(store_guard.ProductionStoreWriteAttempt):
        sqlite3.connect(str(target))
    assert not target.exists(), "the guard fired but the file was created anyway"


def test_a_read_only_uri_under_the_real_home_is_allowed():
    """Probes legitimately read production data."""
    live = store_guard._REAL_HOME / "state.db"
    if not live.exists():
        pytest.skip("no live state.db in this environment")
    conn = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
    conn.close()


def test_tmp_paths_are_unaffected(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "fine.db"))
    conn.execute("CREATE TABLE t (x)")
    conn.close()


def test_memory_databases_are_unaffected():
    sqlite3.connect(":memory:").close()

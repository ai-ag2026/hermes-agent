"""Step② (2026-07-14) — mutation budget: rate ceiling + scope-bound manifest.

Autonomy middle-way, Säule 2: destructive/authority mutations (archive, gate_off)
are bounded STRUCTURALLY so the 2026-07-13 incident (46 archives in ~40s) is
impossible WITHOUT a per-action human gate. Ad-hoc bursts trip a rate ceiling; a
declared scope-bound manifest lets legit bulk through but rejects scope-creep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Deterministic, low ceiling; budget explicitly on.
    monkeypatch.setenv("HERMES_KANBAN_MUTATION_BUDGET", "on")
    monkeypatch.setenv("HERMES_KANBAN_MUTATION_RATE_LIMIT", "5")
    monkeypatch.setenv("HERMES_KANBAN_MUTATION_RATE_WINDOW", "60")
    kb.init_db()
    return home


def _mk(conn, n, prefix="c"):
    return [kb.create_task(conn, title=f"{prefix}{i}") for i in range(n)]


# --- WP②.1 rate ceiling (ad-hoc) ------------------------------------------------

def test_adhoc_archive_burst_trips_rate_ceiling(kanban_home):
    with kb.connect() as conn:
        ids = _mk(conn, 10)
        for t in ids[:5]:
            assert kb.archive_task(conn, t) is True          # 5 allowed
        with pytest.raises(kb.MutationBudgetError) as exc:
            kb.archive_task(conn, ids[5])                    # 6th trips
        assert "rate ceiling" in str(exc.value)
        # The rejected archive rolled back — the card is NOT archived.
        assert kb.get_task(conn, ids[5]).status != "archived"


def test_disabled_budget_allows_unbounded(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_MUTATION_BUDGET", "off")
    with kb.connect() as conn:
        ids = _mk(conn, 12)
        for t in ids:
            assert kb.archive_task(conn, t) is True          # no ceiling when off


# --- WP②.2 scope-bound manifest -------------------------------------------------

def test_manifest_bypasses_ceiling_for_in_scope(kanban_home):
    with kb.connect() as conn:
        ids = _mk(conn, 8)                                   # 8 > ceiling of 5
        kb.open_mutation_manifest(conn, op="archive", task_ids=ids, actor="curator",
                                  rationale="audited cleanup")
        for t in ids:
            assert kb.archive_task(conn, t) is True          # all 8 pass under manifest


def test_manifest_rejects_scope_creep(kanban_home):
    with kb.connect() as conn:
        bound = _mk(conn, 2, prefix="in")
        outside = kb.create_task(conn, title="outside")
        kb.open_mutation_manifest(conn, op="archive", task_ids=bound, actor="curator")
        assert kb.archive_task(conn, bound[0]) is True
        with pytest.raises(kb.MutationBudgetError) as exc:
            kb.archive_task(conn, outside)                   # not in the manifest set
        assert "scope-creep" in str(exc.value)
        assert kb.get_task(conn, outside).status != "archived"


def test_manifest_exhaustion_rejected_direct(kanban_home):
    # Direct enforce: consuming more than max_size (same id re-counted) is refused.
    with kb.connect() as conn:
        mid = kb.open_mutation_manifest(conn, op="gate_off", task_ids=["t_a"], actor="x")
        with kb.write_txn(conn):
            kb._enforce_mutation_budget(conn, "t_a", "gate_off")   # consumes 1/1
        with pytest.raises(kb.MutationBudgetError) as exc:
            with kb.write_txn(conn):
                kb._enforce_mutation_budget(conn, "t_a", "gate_off")  # 2nd > max_size
        assert "exhausted" in str(exc.value)
        assert kb.get_active_manifest(conn, "gate_off")["consumed"] == 1


def test_close_manifest_falls_back_to_ceiling(kanban_home):
    with kb.connect() as conn:
        ids = _mk(conn, 8)
        mid = kb.open_mutation_manifest(conn, op="archive", task_ids=ids)
        assert kb.close_mutation_manifest(conn, mid) is True
        # Manifest closed -> ad-hoc ceiling applies again (5).
        for t in ids[:5]:
            assert kb.archive_task(conn, t) is True
        with pytest.raises(kb.MutationBudgetError):
            kb.archive_task(conn, ids[5])


# --- Incident replay ------------------------------------------------------------

def test_incident_replay_46_vs_30_manifest(kanban_home):
    """The 2026-07-13 incident under Step②: a curation declares a manifest over the
    30 AUDITED cards; those archive freely, the 16 others are rejected as scope-creep.
    Net: exactly 30 archived, 16 refused, and NO operator prompt anywhere."""
    with kb.connect() as conn:
        audited = _mk(conn, 30, prefix="aud")
        extra = _mk(conn, 16, prefix="extra")
        kb.open_mutation_manifest(conn, op="archive", task_ids=audited,
                                  actor="tars", rationale="curation audit 30")
        archived = 0
        for t in audited:
            if kb.archive_task(conn, t):
                archived += 1
        refused = 0
        for t in extra:
            try:
                kb.archive_task(conn, t)
            except kb.MutationBudgetError:
                refused += 1
        assert archived == 30
        assert refused == 16
        assert len(kb.list_tasks(conn, include_archived=True)) == 46
        # exactly the audited 30 are archived
        arch_ids = {r.id for r in kb.list_tasks(conn, include_archived=True)
                    if r.status == "archived"}
        assert arch_ids == set(audited)


# --- Robustness -----------------------------------------------------------------

def test_open_manifest_validates_op_and_ids(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(ValueError):
            kb.open_mutation_manifest(conn, op="delete", task_ids=["t"])   # not enforced
        with pytest.raises(ValueError):
            kb.open_mutation_manifest(conn, op="archive", task_ids=[])     # empty


# --- Step⑤ WP⑤.1 (2026-07-14): operator digest (Säule 3) -----------------------

def test_digest_summarizes_adhoc_and_manifest_mutations(kanban_home):
    with kb.connect() as conn:
        # 2 ad-hoc archives.
        for t in _mk(conn, 2, prefix="adhoc"):
            kb.archive_task(conn, t)
        # 3 manifest-bound archives.
        bulk = _mk(conn, 3, prefix="bulk")
        kb.open_mutation_manifest(conn, op="archive", task_ids=bulk, actor="curator",
                                  rationale="cleanup")
        for t in bulk:
            kb.archive_task(conn, t)
        d = kb.board_mutation_digest(conn, since_seconds=3600)
        arch = d["mutations"]["archive"]
        assert arch["total"] == 5 and arch["ad_hoc"] == 2 and arch["manifest"] == 3
        assert len(d["manifests"]) == 1 and d["manifests"][0]["consumed"] == 3
        assert d["events"].get("archived") == 5
        text = kb.format_mutation_digest(d)
        assert "archive: 5" in text and "bulk manifests: 1" in text


def test_digest_empty_window(kanban_home):
    with kb.connect() as conn:
        d = kb.board_mutation_digest(conn, since_seconds=3600)
        assert d["mutations"] == {}
        assert "no rate-limited mutations" in kb.format_mutation_digest(d)


def test_enforce_fails_open_when_tables_absent(kanban_home):
    # Drop the Step② tables to simulate an un-migrated DB: enforce must NOT brick.
    with kb.connect() as conn:
        with kb.write_txn(conn):
            conn.execute("DROP TABLE mutation_log")
            conn.execute("DROP TABLE mutation_manifest")
        t = kb.create_task(conn, title="x")
        # Would trip the ceiling if active; with tables gone it fails open (allows).
        for _ in range(3):
            with kb.write_txn(conn):
                kb._enforce_mutation_budget(conn, t, "archive")

"""Step③ (2026-07-14) — reversibility (autonomy middle-way, Säule 1).

Archiving is soft and fully recoverable: unarchive_task restores a card without raw
board SQL, and a card archived MID-RUN preserves its session/workspace/history so the
work can CONTINUE (one-shot resume) instead of being lost. This is the invariant that
lets Step④ drop the H2 archive-running gate — nothing is irreversibly destroyed.
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
    monkeypatch.setenv("HERMES_KANBAN_RESUME_ON_RECLAIM", "true")
    kb.init_db()
    return home


# --- WP③.1 unarchive restores ---------------------------------------------------

def test_unarchive_restores_archived_card(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        assert kb.archive_task(conn, t) is True
        assert kb.get_task(conn, t).status == "archived"
        assert kb.unarchive_task(conn, t) is True
        assert kb.get_task(conn, t).status in ("todo", "ready")


def test_unarchive_completed_card_restores_to_done(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.claim_task(conn, t)
        kb.complete_task(conn, t, result="shipped")
        assert kb.archive_task(conn, t) is True
        assert kb.unarchive_task(conn, t) is True
        assert kb.get_task(conn, t).status == "done"


def test_unarchive_explicit_status(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.archive_task(conn, t)
        assert kb.unarchive_task(conn, t, to_status="ready") is True
        assert kb.get_task(conn, t).status == "ready"


def test_unarchive_non_archived_is_noop(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        assert kb.unarchive_task(conn, t) is False


def test_unarchive_bad_status_rejected(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.archive_task(conn, t)
        with pytest.raises(ValueError):
            kb.unarchive_task(conn, t, to_status="archived")


# --- WP③.2 non-destruction invariant (archive-of-running) -----------------------

def test_archive_of_running_preserves_session_and_history(kanban_home):
    """The precondition for dropping the H2 gate: archiving a running card keeps the
    run row (outcome=reclaimed), its session_id, and the workspace — nothing lost but
    the unflushed tail."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="live", workspace_kind="dir",
                           workspace_path=str(kanban_home))
        kb.claim_task(conn, t)
        # Pin a session id on the live run (as a spawned worker would).
        run_id = kb.get_task(conn, t).current_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET session_id='kbwrk_live_1' WHERE id=?", (run_id,))
        # Archive the RUNNING card (grant machinery is exercised elsewhere; here we
        # archive a card whose run we reclaim directly to assert preservation).
        with kb.write_txn(conn):
            kb._enforce_mutation_budget(conn, t, "archive")  # no-op path fine
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (t,))
            kb._end_run(conn, t, outcome="reclaimed", status="reclaimed",
                        summary="archived mid-run")
        run = conn.execute("SELECT outcome, session_id, ended_at FROM task_runs WHERE id=?",
                           (run_id,)).fetchone()
        assert run["outcome"] == "reclaimed"
        assert run["session_id"] == "kbwrk_live_1"    # session PRESERVED
        # Workspace pointer preserved on the task.
        assert kb.get_task(conn, t).workspace_path == str(kanban_home)


def test_evidence_is_append_only_until_permanent_delete(kanban_home):
    """Comments/events survive archive; only delete_archived_task (a deliberate second
    step on an already-archived card) hard-removes them."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x")
        kb.add_comment(conn, t, "user", "keep me")
        kb.archive_task(conn, t)
        comments = conn.execute("SELECT COUNT(*) AS n FROM task_comments WHERE task_id=?",
                                (t,)).fetchone()["n"]
        assert comments >= 1                          # evidence survives archive
        assert kb.delete_archived_task(conn, t) is True
        gone = conn.execute("SELECT COUNT(*) AS n FROM task_comments WHERE task_id=?",
                            (t,)).fetchone()["n"]
        assert gone == 0                              # only permanent delete removes it


# --- WP③.3 one-shot resume on unarchive -----------------------------------------

def test_unarchive_arms_one_shot_resume_for_midrun_card(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="live")
        kb.claim_task(conn, t)
        run_id = kb.get_task(conn, t).current_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET session_id='kbwrk_resume_me' WHERE id=?", (run_id,))
        # Archive mid-run (reclaim the run), then unarchive.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (t,))
            kb._end_run(conn, t, outcome="reclaimed", status="reclaimed")
        assert kb.unarchive_task(conn, t) is True
        # Hint armed on the card.
        hint = conn.execute("SELECT resume_session_hint FROM tasks WHERE id=?", (t,)).fetchone()
        assert hint["resume_session_hint"] == "kbwrk_resume_me"
        # The next claim resolves to the pinned session AND clears the hint (one-shot).
        with kb.write_txn(conn):
            sess, is_resume = kb._resolve_worker_session(conn, t, run_id + 1)
        assert (sess, is_resume) == ("kbwrk_resume_me", True)
        cleared = conn.execute("SELECT resume_session_hint FROM tasks WHERE id=?", (t,)).fetchone()
        assert cleared["resume_session_hint"] is None


def test_unarchive_no_resume_flag_skips_hint(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="live")
        kb.claim_task(conn, t)
        run_id = kb.get_task(conn, t).current_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET session_id='s' WHERE id=?", (run_id,))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (t,))
            kb._end_run(conn, t, outcome="reclaimed", status="reclaimed")
        assert kb.unarchive_task(conn, t, resume=False) is True
        hint = conn.execute("SELECT resume_session_hint FROM tasks WHERE id=?", (t,)).fetchone()
        assert hint["resume_session_hint"] is None


def test_ttl_stale_reclaim_does_not_auto_resume(kanban_home):
    """A TTL-stale reclaim sets NO hint, so it must not resume — the DEVCHAIN crash
    invariant stays intact (only an explicit unarchive arms resume)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="stale")
        kb.claim_task(conn, t)
        run_id = kb.get_task(conn, t).current_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET session_id='s_stale' WHERE id=?", (run_id,))
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (t,))
            kb._end_run(conn, t, outcome="reclaimed", status="reclaimed")
        with kb.write_txn(conn):
            sess, is_resume = kb._resolve_worker_session(conn, t, run_id + 1)
        assert is_resume is False          # reclaimed alone never resumes

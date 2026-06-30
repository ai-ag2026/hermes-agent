import asyncio
import logging
from pathlib import Path

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_state import SessionDB


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = ({Platform.TELEGRAM: adapter} if adapter is not None else {})
    runner._kanban_sub_fail_counts = {}
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _init_isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_session_subscription_delivers_blocker_to_origin_session(tmp_path, monkeypatch):
    home = _init_isolated_home(tmp_path, monkeypatch)
    db = SessionDB(home / "state.db")
    db.create_session("origin-session", source="webui")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="needs input",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="__session__",
            chat_id="origin-session",
        )
        kb.block_task(conn, tid, reason="Which path should I use?", kind="needs_input")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))

    rows = db.get_messages("origin-session")
    assert len(rows) == 1
    assert rows[0]["role"] == "assistant"
    assert "blocked" in rows[0]["content"]
    assert "Which path should I use?" in rows[0]["content"]


def test_needs_input_blocker_notification_is_decision_ready_and_deduped(
    tmp_path, monkeypatch,
):
    home = _init_isolated_home(tmp_path, monkeypatch)
    db = SessionDB(home / "state.db")
    db.create_session("origin-session", source="webui")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="choose API route",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="__session__",
            chat_id="origin-session",
        )
        kb.add_comment(
            conn,
            tid,
            author="worker",
            body="Evidence: current v1 clients still call /api/v1/tasks",
        )
        kb.block_task(
            conn,
            tid,
            reason="Which route should own updates? Recommended default: keep /api/v1/tasks until v2 ships.",
            kind="needs_input",
        )
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))

    rows = db.get_messages("origin-session")
    assert len(rows) == 1
    text = rows[0]["content"]
    assert tid in text
    assert "choose API route" in text
    assert "blocked (needs_input)" in text
    assert "Which route should own updates?" in text
    assert "Decision needed: answer the worker question" in text
    assert "Recommended default: keep /api/v1/tasks until v2 ships" in text
    assert "current v1 clients still call /api/v1/tasks" in text
    assert "Unblock after resolving" in text


def test_capability_blocker_notification_includes_remediation_hint(
    tmp_path, monkeypatch,
):
    _init_isolated_home(tmp_path, monkeypatch)
    adapter = RecordingAdapter()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="review implementation",
            assignee="reviewer",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
        )
        kb.add_comment(
            conn,
            tid,
            author="reviewer",
            body="Spawn failed before tests could run",
        )
        kb.block_task(
            conn,
            tid,
            reason="reviewer profile lacked terminal tool for running tests",
            kind="capability",
        )
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert tid in text
    assert "blocked (capability)" in text
    assert "reviewer profile lacked terminal tool" in text
    assert "Decision needed: remediate the missing tool/capability" in text
    assert "hermes -p reviewer tools enable terminal" in text
    assert "Spawn failed before tests could run" in text


def test_review_required_blocker_notification_includes_files_and_tests(
    tmp_path, monkeypatch,
):
    _init_isolated_home(tmp_path, monkeypatch)
    adapter = RecordingAdapter()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="improve notifier",
            assignee="backend-eng",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
        )
        kb.add_comment(
            conn,
            tid,
            author="backend-eng",
            body=(
                "changed_files:\n"
                "- gateway/kanban_watchers.py\n"
                "- tests/gateway/test_kanban_workflow_delivery.py\n"
                "\n"
                "tests_run:\n"
                "- pytest tests/gateway/test_kanban_workflow_delivery.py -q -> 9 passed"
            ),
        )
        kb.block_task(
            conn,
            tid,
            reason="review-required: notification renderer changed",
            kind="needs_input",
        )
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert tid in text
    assert "review-required: notification renderer changed" in text
    assert "Decision needed: review the listed changes/tests" in text
    assert "Recommended default: approve if the comment's changed files" in text
    assert "Changed files: gateway/kanban_watchers.py" in text
    assert "tests/gateway/test_kanban_workflow_delivery.py" in text
    assert "Tests: pytest tests/gateway/test_kanban_workflow_delivery.py -q -> 9 passed" in text
    assert "Changed files: " in text and "Changed files: \n" not in text


def test_session_subscription_delivers_worker_attention_events(tmp_path, monkeypatch):
    home = _init_isolated_home(tmp_path, monkeypatch)
    db = SessionDB(home / "state.db")
    db.create_session("origin-session", source="webui")

    conn = kb.connect()
    try:
        tids = {}
        for kind in ("gave_up", "crashed", "timed_out"):
            tid = kb.create_task(
                conn,
                title=f"phase {kind}",
                assignee="worker",
                session_id="origin-session",
            )
            tids[kind] = tid
            kb.add_notify_sub(
                conn,
                task_id=tid,
                platform="__session__",
                chat_id="origin-session",
            )
        kb._append_event(conn, tids["gave_up"], "gave_up", {"error": "spawn failed"})
        kb._append_event(conn, tids["crashed"], "crashed")
        kb._append_event(
            conn,
            tids["timed_out"],
            "timed_out",
            {"limit_seconds": 12, "elapsed_seconds": 99},
        )
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))

    text = "\n".join(row["content"] for row in db.get_messages("origin-session"))
    assert "gave up" in text
    assert "spawn failed" in text
    assert "worker crashed" in text
    assert "timed out" in text
    assert "max_runtime=12s" in text


def test_workflow_final_completion_sends_one_aggregate_not_phase_spam(tmp_path, monkeypatch):
    _init_isolated_home(tmp_path, monkeypatch)
    adapter = RecordingAdapter()

    conn = kb.connect()
    try:
        first = kb.create_task(
            conn,
            title="Demo workflow — 1/2: build",
            assignee="worker",
            session_id="origin-session",
        )
        second = kb.create_task(
            conn,
            title="Demo workflow — 2/2: verify",
            assignee="worker",
            parents=[first],
            session_id="origin-session",
        )
        conn.execute(
            "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? WHERE id = ?",
            ("demo-wf", "build", first),
        )
        conn.execute(
            "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? WHERE id = ?",
            ("demo-wf", "verify", second),
        )
        conn.commit()
        for tid in (first, second):
            kb.add_notify_sub(
                conn,
                task_id=tid,
                platform="telegram",
                chat_id="chat-1",
            )
        kb.complete_task(conn, first, summary="Build phase finished")
        kb._append_event(conn, first, "completed", {"summary": "Build phase emitted again"})
        kb.complete_task(conn, second, summary="Verify phase finished")
    finally:
        conn.close()

    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "Workflow demo-wf complete" in text
    assert first in text and second in text
    assert "Build phase finished" in text
    assert "Verify phase finished" in text
    assert "Build phase emitted again" not in text
    assert "Kanban " not in text


def test_workflow_final_completion_appends_one_aggregate_to_origin_session(
    tmp_path, monkeypatch,
):
    home = _init_isolated_home(tmp_path, monkeypatch)
    db = SessionDB(home / "state.db")
    db.create_session("origin-session", source="webui")

    conn = kb.connect()
    try:
        first = kb.create_task(
            conn,
            title="Demo workflow — 1/2: build",
            assignee="worker",
            session_id="origin-session",
        )
        second = kb.create_task(
            conn,
            title="Demo workflow — 2/2: verify",
            assignee="worker",
            parents=[first],
            session_id="origin-session",
        )
        conn.execute(
            "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? WHERE id = ?",
            ("demo-wf", "build", first),
        )
        conn.execute(
            "UPDATE tasks SET workflow_template_id = ?, current_step_key = ? WHERE id = ?",
            ("demo-wf", "verify", second),
        )
        conn.commit()
        for tid in (first, second):
            kb.add_notify_sub(
                conn,
                task_id=tid,
                platform="__session__",
                chat_id="origin-session",
            )
        kb.complete_task(conn, first, summary="Build phase finished")
        kb._append_event(conn, first, "completed", {"summary": "Build phase emitted again"})
        kb.complete_task(conn, second, summary="Verify phase finished")
    finally:
        conn.close()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))

    rows = db.get_messages("origin-session")
    assert len(rows) == 1
    text = rows[0]["content"]
    assert "Workflow demo-wf complete" in text
    assert first in text and second in text
    assert "Build phase finished" in text
    assert "Verify phase finished" in text
    assert "Build phase emitted again" not in text
    assert "Kanban " not in text

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))
    assert len(db.get_messages("origin-session")) == 1


def test_session_subscription_with_missing_bound_session_logs_warning(
    tmp_path, monkeypatch, caplog,
):
    _init_isolated_home(tmp_path, monkeypatch)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="legacy unbound workflow phase",
            assignee="worker",
            session_id="missing-origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="__session__",
            chat_id="missing-origin-session",
        )
        kb.block_task(conn, tid, reason="Need a human", kind="needs_input")
    finally:
        conn.close()

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner()))

    assert "missing-origin-session" in caplog.text
    assert "session delivery" in caplog.text

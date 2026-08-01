import asyncio
import json
import pytest

from pathlib import Path
from types import SimpleNamespace
from hermes_cli import kanban_db as kb
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Allow the kanban notifier path-validator to upload artifacts the
    # tests write under ``tmp_path``. Without this, every artifact-delivery
    # test silently drops files because ``tmp_path`` isn't inside the
    # default ``MEDIA_DELIVERY_SAFE_ROOTS`` cache dirs.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()
    return home


def _assert_inherited_notify_sub(subs: list[dict]) -> None:
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "topic1"
    assert subs[0]["user_id"] == "user1"
    assert subs[0]["notifier_profile"] == "default"










# ---------------------------------------------------------------------------
# Regression: gateway watchers must not double-init the kanban DB.
#
# Both the notifier watcher (`_kanban_notifier_watcher`) and the dispatcher
# tick (`_tick_once_for_board`) used to call `_kb.connect(board=slug)`
# immediately followed by `_kb.init_db(board=slug)`. Since `connect()`
# already runs the schema + idempotent migration on first open per process,
# the explicit `init_db()` was redundant — and worse, `init_db()`
# deliberately busts the per-process cache and re-runs the migration on a
# *second* connection, which races the first.  On legacy DBs this surfaced
# as `duplicate column name: <col>` (now tolerated by
# `_add_column_if_missing`) and intermittent `database is locked` errors
# (issue #21378).
#
# The fix removes the `init_db()` calls in both watchers; this regression
# test pins that behaviour so we don't reintroduce them.
# ---------------------------------------------------------------------------




@pytest.mark.asyncio
async def test_gateway_create_autosubscribes_on_explicit_board(kanban_home):
    """`/kanban --board <slug> create ...` must subscribe on that board.

    The gateway handler currently auto-subscribes after `/kanban create`,
    but the create detection must still work when the shared `--board`
    flag appears before the subcommand, and the subscription must land in
    that board's DB rather than the ambient/default board.
    """
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    kb.create_board("projx")

    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        chat_type="dm",
        thread_id="20197",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban --board projx create "hello" --assignee alice',
        source=source,
        message_id="462",
        reply_to_message_id=None,
    )

    out = await GatewayRunner._handle_kanban_command(runner, event)

    assert "subscribed" in out.lower()

    conn = kb.connect(board="projx")
    try:
        subs = kb.list_notify_subs(conn)
        tasks = kb.list_tasks(conn)
    finally:
        conn.close()

    assert [t.title for t in tasks] == ["hello"]
    assert len(subs) == 1
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "20197"
    assert subs[0]["delivery_metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }

    conn = kb.connect(board="default")
    try:
        assert kb.list_notify_subs(conn) == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_notifier_uploads_artifacts_on_completion(kanban_home, tmp_path, monkeypatch):
    """When a completed event carries ``artifacts`` in its payload, the
    notifier uploads each file to the subscribed chat as a native
    attachment. Images batch through send_multiple_images; documents
    route through send_document. See the artifacts wiring in
    gateway/run.py._deliver_kanban_artifacts.
    """
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from tools import kanban_tools as kt

    # Completion promotes artifacts under the board's durable root; allow that
    # root through the gateway's independent media-delivery safety filter.
    monkeypatch.setenv(
        "HERMES_MEDIA_ALLOW_DIRS", str(kb.completion_artifacts_root())
    )

    # Materialize real files so os.path.isfile passes inside the helper.
    chart_path = tmp_path / "q3-revenue.png"
    chart_path.write_bytes(b"PNG-fake-bytes")
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-fake")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="render q3 chart",
            assignee="worker1",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        # Kernel worker gates (C1): mirror a real dispatcher-spawned worker —
        # claimed card + matching run identity in the process env.
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
    finally:
        conn.close()

    # Use the production handler so we exercise the full path: tool args
    # → metadata.artifacts → event payload promotion.
    import os
    os.environ["HERMES_KANBAN_TASK"] = tid
    os.environ["HERMES_KANBAN_RUN_ID"] = str(run_id)
    try:
        out = kt._handle_complete({
            "summary": "rendered the chart",
            "artifacts": [str(chart_path), str(report_path)],
        })
    finally:
        os.environ.pop("HERMES_KANBAN_TASK", None)
        os.environ.pop("HERMES_KANBAN_RUN_ID", None)
    import json as _json
    assert _json.loads(out)["ok"] is True

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()

    fake_adapter = MagicMock()
    fake_adapter.name = "telegram"

    sends: list = []
    images_uploaded: list = []
    documents_uploaded: list = []

    async def _send(chat_id, msg, metadata=None):
        sends.append((chat_id, msg))
        runner._running = False

    async def _send_images(chat_id, images, metadata=None, **_kw):
        images_uploaded.extend(p for p, _ in images)

    async def _send_document(chat_id, file_path, metadata=None, **_kw):
        documents_uploaded.append(file_path)

    fake_adapter.send = AsyncMock(side_effect=_send)
    fake_adapter.send_multiple_images = AsyncMock(side_effect=_send_images)
    fake_adapter.send_document = AsyncMock(side_effect=_send_document)
    # extract_local_files is used internally for legacy path fallback;
    # the real BasePlatformAdapter implementation lives there, so wire it.
    from gateway.platforms.base import BasePlatformAdapter
    fake_adapter.extract_local_files = BasePlatformAdapter.extract_local_files

    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # The text completion notification fired.
    assert len(sends) == 1
    # The PNG rode the image-batch path.
    assert any("q3-revenue.png" in p for p in images_uploaded), images_uploaded
    # The PDF rode the document path.
    assert any("report.pdf" in p for p in documents_uploaded), documents_uploaded


@pytest.mark.asyncio
async def test_missing_artifact_blocks_completion_before_notifier(
    kanban_home, tmp_path, monkeypatch
):
    """A missing artifact is rejected before a completed event exists."""
    import hermes_cli.kanban_db as kb
    from tools import kanban_tools as kt

    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(b"%PDF-fake")

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="t",
            assignee="worker1",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
        )
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        # Kernel worker gates (C1): mirror a real worker (claim + run id).
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
    finally:
        conn.close()

    import os
    os.environ["HERMES_KANBAN_TASK"] = tid
    os.environ["HERMES_KANBAN_RUN_ID"] = str(run_id)
    try:
        out = kt._handle_complete({
            "summary": "one real, one ghost",
            "artifacts": [str(real_pdf), "/tmp/definitely-does-not-exist.pdf"],
        })
    finally:
        os.environ.pop("HERMES_KANBAN_TASK", None)
        os.environ.pop("HERMES_KANBAN_RUN_ID", None)

    rejected = json.loads(out)
    assert rejected["success"] is False
    assert rejected["kind"] == "evidence_missing"
    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        # Rejection happens BEFORE any state change: the claimed card is
        # still running (pre-C1 this test used an unclaimed card → 'ready').
        assert task.status == "running"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (tid,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


# fork(tars) 31.07.2026: upstreams
# test_notifier_artifact_delivery_skips_missing_files ist hier ENTFALLEN.
# Es kodiert upstreams Vertrag „fehlende Artefaktpfade beim Zustellen still
# überspringen". Unser Fork zieht die Prüfung bewusst VOR: ein fehlendes
# Artefakt blockiert bereits kanban_complete (siehe
# test_missing_artifact_blocks_completion_before_notifier direkt darüber) —
# das Completed-Event entsteht dann gar nicht erst, und upstreams Test wartet
# bei uns zwangsläufig in einen Timeout. Beide Verträge können nicht
# gleichzeitig gelten; unserer ist der strengere und der gewollte.

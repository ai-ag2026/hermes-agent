"""Stage-1 transport/framing tests for the Reachy platform adapter.

Offline proof (no live gateway): the adapter's WebSocket server accepts a mock
client, turns an inbound ``stt`` frame into a MessageEvent dispatched via
``handle_message`` (patched to capture), and pushes ``send`` / ``edit_message``
output back to the client as ``say`` frames. This exercises the whole transport
contract a Reachy adapter must honour, without running an agent turn.
"""

import asyncio
import json
import os
import socket

import pytest

os.environ.setdefault("REACHY_WS_PORT", "8770")

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.reachy.adapter import (  # noqa: E402
    ReachyAdapter,
    check_requirements,
)

try:
    from websockets.asyncio.client import connect as ws_connect

    WS_CLIENT = True
except Exception:  # pragma: no cover
    WS_CLIENT = False


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_adapter(port: int) -> ReachyAdapter:
    cfg = PlatformConfig()
    cfg.extra = {"host": "127.0.0.1", "port": port}
    return ReachyAdapter(cfg)


# ── unit-level ──────────────────────────────────────────────────────────────
def test_flags_and_platform():
    a = _make_adapter(_free_port())
    assert a.platform.value == "reachy"
    assert a.SUPPORTS_MESSAGE_EDITING is True
    assert a.supports_async_delivery is True


def test_robot_id_from_path():
    assert ReachyAdapter._robot_id_from_path("/robot/lab-1") == "lab-1"
    assert ReachyAdapter._robot_id_from_path("/robot/lab-1?x=1") == "lab-1"
    assert ReachyAdapter._robot_id_from_path("") == "reachy"
    assert ReachyAdapter._robot_id_from_path("/") == "reachy"


def test_check_requirements(monkeypatch):
    monkeypatch.setenv("REACHY_WS_PORT", "8770")
    assert check_requirements() is True
    monkeypatch.delenv("REACHY_WS_PORT", raising=False)
    # websockets is installed in this env, but no port => not configured
    assert check_requirements() is False


# ── end-to-end transport round trip ─────────────────────────────────────────
@pytest.mark.skipif(not WS_CLIENT, reason="websockets client unavailable")
def test_transport_round_trip():
    async def scenario():
        port = _free_port()
        adapter = _make_adapter(port)

        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture  # type: ignore[assignment]

        assert await adapter.connect() is True
        try:
            async with ws_connect(f"ws://127.0.0.1:{port}/robot/reachy") as client:
                # inbound: hello then stt
                await client.send(json.dumps({"type": "hello", "robot_id": "reachy"}))
                await client.send(json.dumps({"type": "stt", "text": "hallo tars"}))

                # wait for the event to be dispatched
                for _ in range(50):
                    if captured:
                        break
                    await asyncio.sleep(0.02)

                assert len(captured) == 1
                ev = captured[0]
                assert ev.text == "hallo tars"
                assert ev.source.platform.value == "reachy"
                assert ev.source.chat_id == "reachy"

                # outbound: send() -> a standalone final "say" frame (kind=message)
                res = await adapter.send("reachy", "Hallo Manfred.")
                assert res.success is True
                frame = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert frame["type"] == "say"
                assert frame["kind"] == "message"
                assert frame["content"] == "Hallo Manfred."
                assert frame["final"] is True

                # outbound: streamed edit_message deltas (growing accumulated text, kind=stream)
                mid = "m-stream-1"
                r1 = await adapter.edit_message("reachy", mid, "Hallo", finalize=False)
                assert r1.success is True
                f1 = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert f1["kind"] == "stream" and f1["message_id"] == mid
                assert f1["content"] == "Hallo" and f1["final"] is False

                r2 = await adapter.edit_message("reachy", mid, "Hallo Manfred, wie geht", finalize=True)
                assert r2.success is True
                f2 = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert f2["kind"] == "stream" and f2["content"] == "Hallo Manfred, wie geht"
                assert f2["final"] is True

                # turn boundary: on_processing_complete -> turn_end frame
                from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome

                src = adapter.build_source(chat_id="reachy", user_id="reachy")
                ev = MessageEvent(text="x", message_type=MessageType.TEXT, source=src)
                await adapter.on_processing_complete(ev, ProcessingOutcome.SUCCESS)
                te = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert te["type"] == "turn_end" and te["outcome"] == "success"
        finally:
            await adapter.disconnect()

    asyncio.run(scenario())


@pytest.mark.skipif(not WS_CLIENT, reason="websockets client unavailable")
def test_send_to_absent_robot_fails_cleanly():
    async def scenario():
        adapter = _make_adapter(_free_port())
        assert await adapter.connect() is True
        try:
            res = await adapter.send("ghost", "nobody home")
            assert res.success is False
            assert "not connected" in (res.error or "")
        finally:
            await adapter.disconnect()

    asyncio.run(scenario())


@pytest.mark.skipif(not WS_CLIENT, reason="websockets client unavailable")
def test_turn_id_echoed_on_interactive_frames_and_null_for_proactive():
    """Client-supplied turn_id round-trips onto say/turn_end during a turn; a send with no
    active turn is tagged origin=proactive/turn_id=None (audit 2026-07-02, V5c correlation)."""

    async def scenario():
        port = _free_port()
        adapter = _make_adapter(port)

        # Emulate the turn machinery: handle_message stamps the contextvar-scoped turn (the real
        # gateway inherits it via the detached task). Here we call the outbound methods INSIDE the
        # _dispatch_text scope by patching handle_message to emit frames synchronously.
        from plugins.platforms.reachy import adapter as adapter_mod

        async def _fake_handle(event):
            # runs while _CURRENT_TURN_ID is set to the client's id
            await adapter.edit_message("reachy", "m1", "Ein schwarzes Loch,", finalize=False)
            await adapter.on_processing_complete(event, _Outcome())
            assert event.metadata.get("reachy_turn_id") == "t-123"

        adapter.handle_message = _fake_handle  # type: ignore[assignment]

        assert await adapter.connect() is True
        try:
            async with ws_connect(f"ws://127.0.0.1:{port}/robot/reachy") as client:
                await client.send(json.dumps({"type": "stt", "text": "was ist ein schwarzes loch", "turn_id": "t-123"}))
                say = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert say["type"] == "say" and say["turn_id"] == "t-123" and say["origin"] == "turn"
                te = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert te["type"] == "turn_end" and te["turn_id"] == "t-123" and te["origin"] == "turn"

                # proactive: a send OUTSIDE any turn scope -> turn_id None, origin proactive
                res = await adapter.send("reachy", "Recherche fertig: Wellington.")
                assert res.success is True
                pro = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert pro["type"] == "say" and pro["turn_id"] is None and pro["origin"] == "proactive"
        finally:
            await adapter.disconnect()

    asyncio.run(scenario())


class _Outcome:
    value = "success"


@pytest.mark.skipif(not WS_CLIENT, reason="websockets client unavailable")
def test_interrupt_frame_dispatches_like_stt():
    async def scenario():
        port = _free_port()
        adapter = _make_adapter(port)
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture  # type: ignore[assignment]
        assert await adapter.connect() is True
        try:
            async with ws_connect(f"ws://127.0.0.1:{port}/robot/reachy") as client:
                await client.send(json.dumps({"type": "interrupt", "text": "stopp mal"}))
                for _ in range(50):
                    if captured:
                        break
                    await asyncio.sleep(0.02)
                assert len(captured) == 1
                assert captured[0].text == "stopp mal"
        finally:
            await adapter.disconnect()

    asyncio.run(scenario())


@pytest.mark.skipif(not WS_CLIENT, reason="websockets client unavailable")
def test_pending_drain_frames_carry_latest_turn_id():
    """Busy-mode interrupt: the drain task inherits the OLD turn's ContextVar, but frames must
    be stamped with the robot's LATEST dispatched turn_id or the client drops the interrupt
    turn's answer (review 2026-07-02 round 2, P1-7)."""

    async def scenario():
        port = _free_port()
        adapter = _make_adapter(port)

        calls = []

        async def _fake_handle(event):
            calls.append(event)
            if len(calls) == 1:
                # Turn 1 dispatched (ContextVar=t-old). Simulate the client interrupting:
                # a second stt arrives and is dispatched (updates _turn_ids to t-new), then
                # the gateway core drains the pending event FROM THE OLD TASK'S CONTEXT —
                # emulated here by emitting the answer while ContextVar is still t-old.
                await adapter._dispatch_text("reachy", "neue frage", turn_id="t-new")
                # back in turn-1 context (ContextVar t-old): emit as the drain task would
                await adapter.edit_message("reachy", "m-drain", "Antwort auf die neue Frage.", finalize=True)
            # second (nested) dispatch: no emission — the answer above stands in for it

        adapter.handle_message = _fake_handle  # type: ignore[assignment]

        assert await adapter.connect() is True
        try:
            async with ws_connect(f"ws://127.0.0.1:{port}/robot/reachy") as client:
                await client.send(json.dumps({"type": "stt", "text": "alte frage", "turn_id": "t-old"}))
                say = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                # emitted from the OLD context AFTER t-new was dispatched -> must carry t-new
                assert say["type"] == "say" and say["turn_id"] == "t-new" and say["origin"] == "turn"
        finally:
            await adapter.disconnect()

    asyncio.run(scenario())


@pytest.mark.skipif(not WS_CLIENT, reason="websockets client unavailable")
def test_typing_frames_are_turn_tagged():
    """Untagged typing fell back to the client's temporal rule and reset the active turn's
    timeout even for foreign turns (review 2026-07-02 round 2, P2)."""

    async def scenario():
        port = _free_port()
        adapter = _make_adapter(port)

        async def _fake_handle(event):
            await adapter.send_typing("reachy")

        adapter.handle_message = _fake_handle  # type: ignore[assignment]

        assert await adapter.connect() is True
        try:
            async with ws_connect(f"ws://127.0.0.1:{port}/robot/reachy") as client:
                await client.send(json.dumps({"type": "stt", "text": "hi", "turn_id": "t-9"}))
                ty = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert ty["type"] == "typing" and ty["turn_id"] == "t-9" and ty["origin"] == "turn"
        finally:
            await adapter.disconnect()

    asyncio.run(scenario())

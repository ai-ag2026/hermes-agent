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

                # outbound: send() -> a final "say" frame
                res = await adapter.send("reachy", "Hallo Manfred.")
                assert res.success is True
                frame = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert frame["type"] == "say"
                assert frame["content"] == "Hallo Manfred."
                assert frame["final"] is True

                # outbound: streamed edit_message deltas (growing accumulated text)
                mid = "m-stream-1"
                r1 = await adapter.edit_message("reachy", mid, "Hallo", finalize=False)
                assert r1.success is True
                f1 = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert f1["message_id"] == mid and f1["content"] == "Hallo" and f1["final"] is False

                r2 = await adapter.edit_message("reachy", mid, "Hallo Manfred, wie geht", finalize=True)
                assert r2.success is True
                f2 = json.loads(await asyncio.wait_for(client.recv(), timeout=2))
                assert f2["message_id"] == mid and f2["content"] == "Hallo Manfred, wie geht"
                assert f2["final"] is True
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

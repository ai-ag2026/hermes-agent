import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _rabbit_adapter(*, source_ips=None, wait_seconds=0.01):
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "rabbit_fast_response": {
                    "enabled": True,
                    "source_ips": source_ips or ["127.0.0.1"],
                    "user_agent_prefixes": ["rabbit-test/"],
                    "sync_wait_seconds": wait_seconds,
                    "ack_text": "ACK",
                    "pending_text": "PENDING",
                    "ready_text": "READY",
                    "empty_text": "EMPTY",
                }
            },
        )
    )


def _app(adapter):
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _body(message):
    return {"model": "TARS", "messages": [{"role": "user", "content": message}]}


def _content(payload):
    return payload["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_rabbit_slow_turn_acknowledges_then_status_delivers_result():
    adapter = _rabbit_adapter()
    release = asyncio.Event()

    async def slow_run(**kwargs):
        await release.wait()
        return ({"final_response": "FERTIG", "completed": True}, {"total_tokens": 3})

    adapter._run_agent = slow_run
    headers = {"User-Agent": "rabbit-test/1.0"}

    async with TestClient(TestServer(_app(adapter))) as client:
        first = await client.post("/v1/chat/completions", json=_body("lange Aufgabe"), headers=headers)
        assert first.status == 200
        assert _content(await first.json()) == "ACK"
        assert first.headers["X-Hermes-Rabbit-Async"] == "accepted"

        pending = await client.post("/v1/chat/completions", json=_body("Status?"), headers=headers)
        assert pending.status == 200
        assert _content(await pending.json()) == "PENDING"
        assert pending.headers["X-Hermes-Rabbit-Async"] == "running"

        release.set()
        await asyncio.sleep(0)

        result = await client.post("/v1/chat/completions", json=_body("Ergebnis"), headers=headers)
        assert result.status == 200
        assert _content(await result.json()) == "FERTIG"
        assert result.headers["X-Hermes-Rabbit-Async"] == "completed"

        empty = await client.post("/v1/chat/completions", json=_body("Status"), headers=headers)
        assert empty.status == 200
        assert _content(await empty.json()) == "EMPTY"
        assert empty.headers["X-Hermes-Rabbit-Async"] == "empty"


@pytest.mark.asyncio
async def test_rabbit_fast_turn_stays_single_response_without_mailbox():
    adapter = _rabbit_adapter(wait_seconds=0.2)
    async def fast_run(**kwargs):
        return ({"final_response": "SOFORT", "completed": True}, {})

    adapter._run_agent = fast_run
    headers = {"User-Agent": "rabbit-test/1.0"}
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post("/v1/chat/completions", json=_body("kurz"), headers=headers)
        assert response.status == 200
        assert _content(await response.json()) == "SOFORT"
        assert response.headers["X-Hermes-Rabbit-Async"] == "immediate"
        assert not adapter._rabbit_fast_mailboxes


@pytest.mark.asyncio
async def test_non_rabbit_api_client_keeps_normal_synchronous_path():
    adapter = _rabbit_adapter(source_ips=["203.0.113.99"])
    calls = 0

    async def normal_run(**kwargs):
        nonlocal calls
        calls += 1
        return ({"final_response": "NORMAL", "completed": True}, {})

    adapter._run_agent = normal_run
    headers = {"User-Agent": "rabbit-test/1.0"}
    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post("/v1/chat/completions", json=_body("normal"), headers=headers)
        assert response.status == 200
        assert _content(await response.json()) == "NORMAL"
        assert "X-Hermes-Rabbit-Async" not in response.headers
        assert calls == 1
        assert not adapter._rabbit_fast_mailboxes


def test_rabbit_detection_requires_both_configured_ip_and_user_agent_prefix():
    adapter = _rabbit_adapter(source_ips=["192.0.2.10"])

    class Request:
        remote = "192.0.2.10"
        headers = {"User-Agent": "rabbit-test/2.0"}
        transport = None
        method = "POST"
        path_qs = "/v1/chat/completions"

    assert adapter._rabbit_fast_request_key(Request(), "session-a") == "192.0.2.10"

    Request.headers = {"User-Agent": "other-client/1.0"}
    assert adapter._rabbit_fast_request_key(Request(), "session-a") is None

    Request.remote = "192.0.2.11"
    Request.headers = {"User-Agent": "rabbit-test/2.0"}
    assert adapter._rabbit_fast_request_key(Request(), "session-a") is None

"""Route-level streaming TTS credential-routing regressions."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.audio_key_guard import AudioEndpointCredentialPolicyError


def test_elevenlabs_streaming_noncanonical_endpoints_never_resolve_or_build_sdk():
    import tools.tts_streaming as ts

    fallback = MagicMock(return_value="broad-eleven-key")
    sdk = MagicMock()
    streamer = ts.ElevenLabsStreamer(
        {}, {"base_url": "https://capture.example", "wss_url": "wss://capture.example"}
    )
    with patch.object(ts, "_resolve_key", fallback), patch(
        "tools.tts_tool._import_elevenlabs", return_value=sdk
    ), patch("tools.tts_tool._elevenlabs_environment_kwargs", return_value={}):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.elevenlabs"):
            list(streamer.stream("hi"))

    fallback.assert_not_called()
    sdk.assert_not_called()


def test_openai_streaming_noncanonical_base_never_resolves_or_builds_sdk():
    import tools.tts_streaming as ts

    fallback = MagicMock(return_value="broad-openai-key")
    streamer = ts.OpenAIStreamer({}, {"base_url": "https://capture.example/v1"})
    with patch.object(ts, "resolve_openai_audio_api_key", fallback):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.openai.api_key"):
            list(streamer.stream("hi"))

    fallback.assert_not_called()


def test_gemini_streaming_noncanonical_base_never_resolves_or_posts():
    import tools.tts_streaming as ts

    fallback = MagicMock(return_value="broad-gemini-key")
    streamer = ts.GeminiStreamer({}, {"base_url": "https://capture.example/v1"})
    with patch.object(ts, "_resolve_key", fallback), patch("requests.post") as post:
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.gemini.api_key"):
            list(streamer.stream("hi"))

    fallback.assert_not_called()
    post.assert_not_called()


def test_xai_streaming_noncanonical_wss_never_resolves_or_connects():
    import tools.tts_streaming as ts

    fake_xai_http = MagicMock()
    fake_xai_http.resolve_xai_http_credentials.return_value = {
        "provider": "xai",
        "api_key": "broad-xai-key",
        "base_url": "https://api.x.ai/v1",
    }
    class _DoneSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, _payload):
            return None

        async def recv(self):
            return "done"

    fake_websockets = MagicMock()
    fake_websockets.connect.return_value = _DoneSocket()
    streamer: ts.XAIStreamer = ts.XAIStreamer(
        {}, {"streaming_url": "wss://capture.example/v1/tts"}
    )

    async def consume():
        async for _frame in streamer._async_frames("hi"):
            pass

    with patch.dict(sys.modules, {"tools.xai_http": fake_xai_http, "websockets": fake_websockets}):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.xai.api_key"):
            asyncio.run(consume())

    fake_websockets.connect.assert_not_called()


def test_openai_streaming_noncanonical_base_is_unavailable_without_lookup():
    import tools.tts_streaming as ts

    fallback = MagicMock(return_value="broad-openai-key")
    with patch.object(ts, "resolve_openai_audio_api_key", fallback):
        assert ts.resolve_streaming_provider({
            "provider": "openai",
            "openai": {"base_url": "https://capture.example/v1"},
        }) is None

    fallback.assert_not_called()


def test_elevenlabs_streaming_noncanonical_base_is_unavailable_without_lookup():
    import tools.tts_streaming as ts

    fallback = MagicMock(return_value="broad-eleven-key")
    with patch.object(ts, "_resolve_key", fallback):
        assert ts.resolve_streaming_provider({
            "provider": "elevenlabs",
            "elevenlabs": {"base_url": "https://capture.example"},
        }) is None

    fallback.assert_not_called()


def test_gemini_streaming_noncanonical_base_is_unavailable_without_lookup():
    import tools.tts_streaming as ts

    fallback = MagicMock(return_value="broad-gemini-key")
    with patch.object(ts, "_resolve_key", fallback):
        assert ts.resolve_streaming_provider({
            "provider": "gemini",
            "gemini": {"base_url": "https://capture.example/v1"},
        }) is None

    fallback.assert_not_called()


def test_xai_streaming_noncanonical_wss_is_unavailable_without_lookup():
    import tools.tts_streaming as ts

    fake_xai_http = MagicMock()
    with patch.dict(sys.modules, {"tools.xai_http": fake_xai_http}):
        assert ts.resolve_streaming_provider({
            "provider": "xai",
            "xai": {"streaming_url": "wss://capture.example/v1/tts"},
        }) is None

    fake_xai_http.resolve_xai_http_credentials.assert_not_called()

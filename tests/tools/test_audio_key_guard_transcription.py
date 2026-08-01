"""Route-level STT credential-routing regressions.

These tests prove the guard sits before a native SDK/HTTP transport, rather
than merely sanitizing a resolved key after it has already escaped.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.audio_key_guard import AudioEndpointCredentialPolicyError


def test_groq_noncanonical_base_never_resolves_fallback_or_builds_client(monkeypatch):
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-groq-key")
    with patch.object(
        tt,
        "_load_stt_config",
        return_value={"groq": {"base_url": "https://capture.example/v1"}},
    ), patch.object(tt, "_resolve_provider_key", fallback), patch.object(
        tt, "_HAS_OPENAI", True
    ), patch("openai.OpenAI") as client:
        result = tt._transcribe_groq("/tmp/voice.wav", "whisper-large-v3-turbo")

    assert result["success"] is False
    assert "stt.groq.api_key" in result["error"]
    fallback.assert_not_called()
    client.assert_not_called()


def test_elevenlabs_noncanonical_base_never_resolves_fallback_or_posts(monkeypatch):
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-eleven-key")
    with patch.object(
        tt,
        "_load_stt_config",
        return_value={"elevenlabs": {"base_url": "https://capture.example/v1"}},
    ), patch.object(tt, "_resolve_provider_key", fallback), patch("requests.post") as post:
        result = tt._transcribe_elevenlabs("/tmp/voice.wav", "scribe_v2")

    assert result["success"] is False
    assert "stt.elevenlabs.api_key" in result["error"]
    fallback.assert_not_called()
    post.assert_not_called()


def test_deepinfra_noncanonical_base_never_resolves_fallback_or_delegates(monkeypatch):
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-deepinfra-key")
    with patch.object(
        tt,
        "_load_stt_config",
        return_value={"deepinfra": {"base_url": "https://capture.example/v1"}},
    ), patch.object(tt, "_resolve_provider_key", fallback), patch(
        "hermes_cli.models.deepinfra_base_url", return_value="https://capture.example/v1"
    ), patch.object(tt, "_transcribe_openai") as transcribe:
        result = tt._transcribe_deepinfra("/tmp/voice.wav", "vendor/stt")

    assert result["success"] is False
    assert "stt.deepinfra.api_key" in result["error"]
    fallback.assert_not_called()
    transcribe.assert_not_called()


def test_openai_noncanonical_local_base_requires_explicit_config_key():
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-openai-key")
    with patch.object(
        tt,
        "_load_stt_config",
        return_value={"openai": {"base_url": "http://localhost:8504/v1"}},
    ), patch.object(tt, "resolve_openai_audio_api_key", fallback):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="stt.openai.api_key"):
            tt._resolve_openai_audio_client_config()

    fallback.assert_not_called()


def test_openai_noncanonical_base_uses_explicit_config_key_without_fallback():
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-openai-key")
    with patch.object(
        tt,
        "_load_stt_config",
        return_value={
            "openai": {
                "api_key": "local-openai-key",
                "base_url": "http://localhost:8504/v1",
            }
        },
    ), patch.object(tt, "resolve_openai_audio_api_key", fallback):
        api_key, base_url = tt._resolve_openai_audio_client_config()
    assert (api_key, base_url) == ("local-openai-key", "http://localhost:8504/v1")
    fallback.assert_not_called()


def test_stt_provider_selection_rejects_noncanonical_groq_without_lookup():
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-groq-key")
    with (
        patch.object(tt, "_HAS_OPENAI", True),
        patch.object(tt, "_resolve_provider_key", fallback),
    ):
        assert tt._get_provider({
            "provider": "groq",
            "groq": {"base_url": "https://capture.example/openai/v1"},
        }) == "none"

    fallback.assert_not_called()


def test_xai_noncanonical_base_with_env_key_never_posts(tmp_path):
    from tools import transcription_tools as tt

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    fake_xai_http = MagicMock()
    fake_xai_http.resolve_xai_oauth_credentials.return_value = None

    requested_env_names = []

    def fake_get_env_value(name, default=None):
        requested_env_names.append(name)
        return {
            "XAI_API_KEY": "broad-xai-key",
            "XAI_STT_BASE_URL": "https://capture.example/v1",
        }.get(name, default)

    with patch.dict(sys.modules, {"tools.xai_http": fake_xai_http}), patch.object(
        tt, "get_env_value", side_effect=fake_get_env_value
    ), patch("requests.post") as post:
        result = tt._transcribe_xai(str(audio), "grok-stt")

    assert result["success"] is False
    assert "stt.xai.api_key" in result["error"]
    post.assert_not_called()
    fake_xai_http.resolve_xai_api_key_credentials.assert_not_called()
    assert "XAI_API_KEY" not in requested_env_names


@pytest.mark.parametrize(
    "direct_endpoint",
    [
        "https://alternate.x.ai/v1",
        "https://api.x.ai:444/v1",
    ],
)
def test_xai_stt_noncanonical_base_uses_pinned_oauth_without_direct_key_lookup(
    tmp_path, direct_endpoint
):
    from tools import transcription_tools as tt

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    oauth = MagicMock(return_value={
        "provider": "xai-oauth",
        "api_key": "broad-oauth-token",
        "base_url": "https://api.x.ai/v1",
    })
    direct_fallback = MagicMock()
    requested_env_names = []

    def fake_get_env_value(name, default=None):
        requested_env_names.append(name)
        return {
            "XAI_STT_BASE_URL": direct_endpoint,
        }.get(name, default)

    response = MagicMock(status_code=200)
    response.json.return_value = {"text": "ok"}
    with (
        patch.object(tt, "get_env_value", side_effect=fake_get_env_value),
        patch("tools.xai_http.resolve_xai_oauth_credentials", oauth),
        patch("tools.xai_http.resolve_xai_api_key_credentials", direct_fallback),
        patch("requests.post", return_value=response) as post,
    ):
        result = tt._transcribe_xai(str(audio), "grok-stt")

    assert result["success"] is True
    assert post.call_args.args[0] == "https://api.x.ai/v1/stt"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer broad-oauth-token"
    direct_fallback.assert_not_called()
    assert "XAI_API_KEY" not in requested_env_names


def test_xai_stt_direct_resolver_precedes_oauth_at_canonical_endpoint(tmp_path):
    from tools import transcription_tools as tt

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    direct = MagicMock(return_value={
        "provider": "xai",
        "api_key": "profile-or-pool-direct-key",
        "base_url": "https://api.x.ai/v1",
    })
    oauth = MagicMock(return_value={
        "provider": "xai-oauth",
        "api_key": "oauth-token",
        "base_url": "https://api.x.ai/v1",
    })
    response = MagicMock(status_code=200)
    response.json.return_value = {"text": "ok"}

    with (
        patch.object(tt, "get_env_value", return_value=None),
        patch("tools.xai_http.resolve_xai_api_key_credentials", direct),
        patch("tools.xai_http.resolve_xai_oauth_credentials", oauth),
        patch("requests.post", return_value=response) as post,
    ):
        result = tt._transcribe_xai(str(audio), "grok-stt")

    assert result["success"] is True
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer profile-or-pool-direct-key"
    oauth.assert_not_called()


def test_xai_stt_selection_direct_resolver_precedes_oauth_at_canonical_endpoint():
    from tools import transcription_tools as tt

    direct = MagicMock(return_value={
        "provider": "xai",
        "api_key": "profile-or-pool-direct-key",
        "base_url": "https://api.x.ai/v1",
    })
    oauth = MagicMock(return_value={
        "provider": "xai-oauth",
        "api_key": "oauth-token",
        "base_url": "https://api.x.ai/v1",
    })
    with (
        patch.object(tt, "get_env_value", return_value=None),
        patch("tools.xai_http.resolve_xai_api_key_credentials", direct),
        patch("tools.xai_http.resolve_xai_oauth_credentials", oauth),
    ):
        assert tt._get_provider({"provider": "xai"}) == "xai"

    oauth.assert_not_called()


def test_xai_noncanonical_base_uses_explicit_config_key(tmp_path):
    from tools import transcription_tools as tt

    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    fake_xai_http = MagicMock()
    fake_xai_http.hermes_xai_user_agent.return_value = "test-agent"
    response = MagicMock(status_code=200)
    response.json.return_value = {"text": "ok"}

    with patch.dict(sys.modules, {"tools.xai_http": fake_xai_http}), patch.object(
        tt,
        "_load_stt_config",
        return_value={
            "xai": {
                "api_key": "local-xai-key",
                "base_url": "https://selfhosted.example/v1",
            }
        },
    ), patch("requests.post", return_value=response) as post:
        result = tt._transcribe_xai(str(audio), "grok-stt")

    assert result["success"] is True
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer local-xai-key"
    fake_xai_http.resolve_xai_http_credentials.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "has_openai"),
    [
        ("groq", True),
        ("openai", True),
        ("xai", False),
        ("elevenlabs", False),
        ("deepinfra", True),
    ],
)
def test_stt_provider_selection_allows_explicit_custom_key_without_fallback(
    provider, has_openai
):
    from tools import transcription_tools as tt

    fallback = MagicMock(return_value="broad-fallback-key")
    config = {
        "provider": provider,
        provider: {
            "api_key": f"local-{provider}-key",
            "base_url": "https://selfhosted.example/v1",
        },
    }
    with (
        patch.object(tt, "_HAS_OPENAI", has_openai),
        patch.object(tt, "_resolve_provider_key", fallback),
        patch.object(tt, "resolve_openai_audio_api_key", fallback),
    ):
        assert tt._get_provider(config) == provider

    fallback.assert_not_called()


@pytest.mark.parametrize(
    "direct_endpoint",
    [
        "https://alternate.x.ai/v1",
        "https://api.x.ai:444/v1",
    ],
)
def test_stt_selection_accepts_pinned_xai_oauth_without_direct_fallback(direct_endpoint):
    from tools import transcription_tools as tt

    oauth = MagicMock(return_value={
        "provider": "xai-oauth",
        "api_key": "oauth-token",
        "base_url": "https://api.x.ai/v1",
    })
    direct_fallback = MagicMock()
    config = {
        "provider": "xai",
        "xai": {"base_url": direct_endpoint},
    }
    with (
        patch("tools.xai_http.resolve_xai_oauth_credentials", oauth),
        patch("tools.xai_http.resolve_xai_api_key_credentials", direct_fallback),
    ):
        assert tt._get_provider(config) == "xai"

    direct_fallback.assert_not_called()

"""Route-level synchronous TTS credential-routing regressions."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.audio_key_guard import AudioEndpointCredentialPolicyError


def test_openai_tts_noncanonical_base_never_resolves_fallback():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-openai-key")
    with patch.object(
        tt,
        "_load_tts_config",
        return_value={"openai": {"base_url": "https://capture.example/v1"}},
    ), patch.object(tt, "resolve_openai_audio_api_key", fallback):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.openai.api_key"):
            tt._resolve_openai_audio_client_config()

    fallback.assert_not_called()


def test_elevenlabs_tts_noncanonical_base_never_resolves_fallback_or_builds_sdk():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-eleven-key")
    sdk = MagicMock()
    with patch.object(tt, "_resolve_provider_key", fallback), patch.object(
        tt, "_import_elevenlabs", return_value=sdk
    ), patch.object(tt, "_elevenlabs_environment_kwargs", return_value={}):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.elevenlabs.api_key"):
            tt._generate_elevenlabs(
                "hi", "/tmp/voice.mp3", {"elevenlabs": {"base_url": "https://capture.example"}}
            )

    fallback.assert_not_called()
    sdk.assert_not_called()


def test_deepinfra_tts_noncanonical_base_never_resolves_fallback_or_delegates():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-deepinfra-key")
    with patch.object(tt, "_resolve_provider_key", fallback), patch(
        "hermes_cli.models.deepinfra_base_url", return_value="https://capture.example/v1"
    ), patch.object(tt, "_generate_openai_tts") as generate:
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.deepinfra.api_key"):
            tt._generate_deepinfra_tts(
                "hi",
                "/tmp/voice.mp3",
                {"deepinfra": {"base_url": "https://capture.example/v1", "model": "vendor/tts"}},
            )

    fallback.assert_not_called()
    generate.assert_not_called()


def test_xai_tts_noncanonical_base_never_resolves_fallback_or_posts():
    from tools import tts_tool as tt

    oauth = MagicMock(return_value=None)
    direct_fallback = MagicMock(return_value={
        "provider": "xai",
        "api_key": "broad-xai-key",
        "base_url": "https://api.x.ai/v1",
    })
    with patch(
        "tools.xai_http.resolve_xai_oauth_credentials", oauth
    ), patch(
        "tools.xai_http.resolve_xai_api_key_credentials", direct_fallback
    ), patch("requests.post") as post:
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.xai.api_key"):
            tt._generate_xai_tts(
                "hi", "/tmp/voice.mp3", {"xai": {"base_url": "https://capture.example/v1"}}
            )

    post.assert_not_called()
    direct_fallback.assert_not_called()


def test_minimax_noncanonical_base_never_resolves_fallback():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-minimax-key")
    with patch.object(tt, "_resolve_provider_key", fallback):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.minimax.api_key"):
            tt._resolve_minimax_tts_runtime(
                {"minimax": {"base_url": "https://capture.example/v1/t2a_v2"}}
            )

    fallback.assert_not_called()


def test_mistral_tts_noncanonical_base_never_resolves_fallback_or_builds_sdk():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-mistral-key")
    sdk = MagicMock()
    with patch.object(tt, "_resolve_provider_key", fallback), patch.object(
        tt, "_import_mistral_client", return_value=sdk
    ):
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.mistral.api_key"):
            tt._generate_mistral_tts(
                "hi", "/tmp/voice.mp3", {"mistral": {"base_url": "https://capture.example/v1"}}
            )

    fallback.assert_not_called()
    sdk.assert_not_called()


def test_gemini_tts_noncanonical_base_never_resolves_fallback_or_posts():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-gemini-key")
    with patch.object(tt, "_resolve_provider_key", fallback), patch("requests.post") as post:
        with pytest.raises(AudioEndpointCredentialPolicyError, match="tts.gemini.api_key"):
            tt._generate_gemini_tts(
                "hi", "/tmp/voice.wav", {"gemini": {"base_url": "https://capture.example/v1"}}
            )

    fallback.assert_not_called()
    post.assert_not_called()


def test_tts_requirements_rejects_noncanonical_gemini_without_lookup():
    from tools import tts_tool as tt

    fallback = MagicMock(return_value="broad-gemini-key")
    with (
        patch.object(tt, "_load_tts_config", return_value={
            "provider": "gemini",
            "gemini": {"base_url": "https://capture.example/v1"},
        }),
        patch.object(tt, "_resolve_provider_key", fallback),
    ):
        assert tt.check_tts_requirements() is False

    fallback.assert_not_called()


def test_tts_requirements_accepts_pinned_xai_oauth_without_direct_fallback():
    from tools import tts_tool as tt

    oauth = MagicMock(return_value={
        "provider": "xai-oauth",
        "api_key": "oauth-token",
        "base_url": "https://api.x.ai/v1",
    })
    direct_fallback = MagicMock()
    with (
        patch.object(
            tt,
            "_load_tts_config",
            return_value={
                "provider": "xai",
                "xai": {"base_url": "https://capture.example/v1"},
            },
        ),
        patch.object(tt, "_get_provider", return_value="xai"),
        patch("tools.xai_http.resolve_xai_oauth_credentials", oauth),
        patch("tools.xai_http.resolve_xai_api_key_credentials", direct_fallback),
    ):
        assert tt.check_tts_requirements() is True

    direct_fallback.assert_not_called()

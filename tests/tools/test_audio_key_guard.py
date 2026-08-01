"""Unit tests for fail-closed audio credential routing.

Fallback credentials come from profile secrets / credential pools and are
therefore broader than an endpoint-specific config key. They may leave the
process only for a provider's canonical TLS origin; endpoint-specific config
keys are an explicit self-hosted/proxy opt-in.
"""

from __future__ import annotations

import socket
from unittest.mock import Mock

import pytest

from tools.audio_key_guard import (
    CANONICAL_AUDIO_HOSTS,
    AudioEndpointCredentialPolicyError,
    select_audio_provider_key,
)


def test_canonical_audio_host_matrix_is_explicit_and_narrow():
    assert CANONICAL_AUDIO_HOSTS == {
        "deepinfra": frozenset({"api.deepinfra.com"}),
        "elevenlabs": frozenset({"api.elevenlabs.io"}),
        "gemini": frozenset({"generativelanguage.googleapis.com"}),
        "groq": frozenset({"api.groq.com"}),
        "minimax-cn": frozenset({"api.minimaxi.com"}),
        "minimax-global": frozenset({"api.minimax.io", "api-uw.minimax.io"}),
        "mistral": frozenset({"api.mistral.ai"}),
        "openai": frozenset({"api.openai.com"}),
        "xai": frozenset({"api.x.ai"}),
    }


@pytest.mark.parametrize(
    ("endpoint", "schemes"),
    [
        ("https://api.openai.com/v1", {"https"}),
        ("https://api.openai.com:443/v1", {"https"}),
        ("wss://api.x.ai/v1/tts", {"wss"}),
        ("wss://api.x.ai:443/v1/tts", {"wss"}),
    ],
)
def test_canonical_tls_endpoint_resolves_fallback_key(endpoint, schemes):
    resolve_fallback = Mock(return_value="broad-secret")

    assert (
        select_audio_provider_key(
            configured_key="",
            resolve_fallback=resolve_fallback,
            endpoint=endpoint,
            canonical_hosts={"api.openai.com", "api.x.ai"},
            allowed_schemes=schemes,
            endpoint_setting="tts.example.base_url",
            key_setting="tts.example.api_key",
        )
        == "broad-secret"
    )
    resolve_fallback.assert_called_once_with()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://api.openai.com/v1",
        "https://api.openai.com:444/v1",
        "https://api.openai.com.attacker.invalid/v1",
        "https://***@attacker.invalid/v1",
        "https://127.0.0.1/v1",
        "https://[::1]/v1",
        "https://proxy.example/v1",
        "api.openai.com/v1",
        "",
    ],
)
def test_noncanonical_endpoint_refuses_without_reading_fallback_secret(endpoint):
    resolve_fallback = Mock(return_value="broad-secret")

    with pytest.raises(AudioEndpointCredentialPolicyError) as exc:
        select_audio_provider_key(
            configured_key="",
            resolve_fallback=resolve_fallback,
            endpoint=endpoint,
            canonical_hosts={"api.openai.com"},
            allowed_schemes={"https"},
            endpoint_setting="tts.openai.base_url",
            key_setting="tts.openai.api_key",
        )

    assert str(exc.value) == (
        "tts.openai.base_url is noncanonical; set tts.openai.api_key "
        "for that endpoint"
    )
    resolve_fallback.assert_not_called()


def test_configured_provider_key_is_explicit_opt_in_for_custom_endpoint():
    resolve_fallback = Mock(return_value="broad-secret")

    assert (
        select_audio_provider_key(
            configured_key="endpoint-specific-secret",
            resolve_fallback=resolve_fallback,
            endpoint="http://127.0.0.1:8080/v1",
            canonical_hosts={"api.openai.com"},
            allowed_schemes={"https"},
            endpoint_setting="tts.openai.base_url",
            key_setting="tts.openai.api_key",
        )
        == "endpoint-specific-secret"
    )
    resolve_fallback.assert_not_called()


def test_canonical_endpoint_without_any_key_stays_empty():
    assert (
        select_audio_provider_key(
            configured_key="",
            resolve_fallback=lambda: "",
            endpoint="https://api.openai.com/v1",
            canonical_hosts={"api.openai.com"},
            allowed_schemes={"https"},
            endpoint_setting="tts.openai.base_url",
            key_setting="tts.openai.api_key",
        )
        == ""
    )


def test_guard_does_not_use_dns_to_authorize_credential_routing(monkeypatch):
    def _unexpected_dns(*_args, **_kwargs):
        raise AssertionError("credential routing must not authorize via DNS")

    monkeypatch.setattr(socket, "getaddrinfo", _unexpected_dns)
    resolve_fallback = Mock(return_value="broad-secret")

    with pytest.raises(AudioEndpointCredentialPolicyError):
        select_audio_provider_key(
            configured_key="",
            resolve_fallback=resolve_fallback,
            endpoint="https://proxy.example/v1",
            canonical_hosts={"api.openai.com"},
            allowed_schemes={"https"},
            endpoint_setting="tts.openai.base_url",
            key_setting="tts.openai.api_key",
        )
    resolve_fallback.assert_not_called()

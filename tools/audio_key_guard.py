"""Fail-closed credential selection for configurable native audio endpoints.

Audio provider credentials resolved from the profile environment or credential
pool are broad fallback secrets. A configurable endpoint is not enough reason
to forward them: fallback credentials are sent only to a provider's canonical
TLS origin. A provider-local config key is an explicit opt-in for a custom or
self-hosted endpoint.

This module intentionally performs no DNS lookups. DNS answers and failures
are transport facts, not credential-routing authority; exact scheme, hostname,
and port are the pre-connect boundary here, while TLS authenticates the remote
canonical origin during connection.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from urllib.parse import urlsplit


_DEFAULT_PORTS = {"https": 443, "wss": 443}

# Exact hosts, derived from provider defaults/official API documentation. Keep
# this list intentionally narrow: a public endpoint is not trusted merely for
# having a plausible provider suffix.
CANONICAL_AUDIO_HOSTS = {
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


class AudioEndpointCredentialPolicyError(ValueError):
    """A broad credential was not authorized for a configured endpoint."""

    def __init__(self, endpoint_setting: str, key_setting: str) -> None:
        self.endpoint_setting = endpoint_setting
        self.key_setting = key_setting
        super().__init__(
            f"{endpoint_setting} is noncanonical; set {key_setting} for that endpoint"
        )


def require_canonical_audio_endpoint(
    endpoint: object,
    canonical_hosts: Collection[str],
    allowed_schemes: Collection[str],
    *,
    endpoint_setting: str,
    key_setting: str,
) -> None:
    """Fail closed unless an endpoint is an exact canonical TLS origin."""
    if not _is_canonical_tls_endpoint(endpoint, canonical_hosts, allowed_schemes):
        raise AudioEndpointCredentialPolicyError(endpoint_setting, key_setting)


def select_audio_provider_key(
    configured_key: object,
    resolve_fallback: Callable[[], object],
    endpoint: object,
    canonical_hosts: Collection[str],
    allowed_schemes: Collection[str],
    *,
    endpoint_setting: str,
    key_setting: str,
) -> str:
    """Return an endpoint-authorized audio credential without DNS.

    ``configured_key`` is directly paired with the provider's configured
    endpoint and therefore explicitly authorizes custom/self-hosted routes.
    ``resolve_fallback`` may obtain an environment, profile-scope or credential
    pool secret. It is invoked only for an exact canonical TLS endpoint; an
    untrusted endpoint is rejected before any broad secret lookup occurs.
    """
    explicit = _nonempty(configured_key)
    if explicit:
        return explicit

    require_canonical_audio_endpoint(
        endpoint,
        canonical_hosts,
        allowed_schemes,
        endpoint_setting=endpoint_setting,
        key_setting=key_setting,
    )

    return _nonempty(resolve_fallback())


def _is_canonical_tls_endpoint(
    endpoint: object,
    canonical_hosts: Collection[str],
    allowed_schemes: Collection[str],
) -> bool:
    """Return whether *endpoint* is an exact allowed TLS origin, without DNS."""
    raw_endpoint = _nonempty(endpoint)
    if not raw_endpoint:
        return False

    try:
        parsed = urlsplit(raw_endpoint)
        port = parsed.port
    except ValueError:
        return False

    scheme = parsed.scheme.lower()
    allowed = {str(value).lower() for value in allowed_schemes}
    if scheme not in allowed:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False

    hostname = (parsed.hostname or "").lower()
    canonical = {str(value).lower() for value in canonical_hosts}
    if hostname not in canonical:
        return False

    return port in {None, _DEFAULT_PORTS.get(scheme)}


def _nonempty(value: object) -> str:
    return str(value or "").strip()

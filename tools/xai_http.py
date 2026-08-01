"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 30 * 24 * 60 * 60
SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 2 * 24 * 60 * 60


def has_xai_credentials() -> bool:
    """Cheap probe — return True when xAI credentials are *likely* usable.

    Deliberately avoids :func:`resolve_xai_http_credentials` so callers in
    hot-paint paths (``hermes tools`` repaint, tool-registration scans,
    ``WebSearchProvider.is_available()``) don't incur disk locks or — in
    the OAuth path — a network token refresh. The ABC contract on
    :meth:`agent.web_search_provider.WebSearchProvider.is_available`
    explicitly forbids network calls for exactly this reason.

    Resolution order, fast-to-slow:

    1. ``XAI_API_KEY`` env var (cheapest; covers explicit-key users).
    2. ``~/.hermes/auth.json`` has a non-empty ``providers.xai-oauth.tokens.access_token``
       (single file read, no expiry check, no refresh).
    3. ``credential_pool.xai-oauth`` has any entry with a non-empty
       ``access_token`` (covers multi-account ``hermes auth add xai-oauth``
       grants that are pool-only / ``manual:device_code``).

    Returns False on any exception so a corrupted auth store can't block
    other availability scans. Truthful refresh + expiry handling happens
    in ``search()`` (or whichever caller actually makes the request).
    """
    if os.environ.get("XAI_API_KEY", "").strip():
        return True
    try:
        from hermes_constants import get_hermes_home

        auth_path = get_hermes_home() / "auth.json"
        if not auth_path.exists():
            return False
        store = json.loads(auth_path.read_text(encoding="utf-8"))
        providers = store.get("providers") if isinstance(store, dict) else None
        xai_state = providers.get("xai-oauth") if isinstance(providers, dict) else None
        tokens = xai_state.get("tokens") if isinstance(xai_state, dict) else None
        access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if str(access_token or "").strip():
            return True
        # Pool-only grants (multi-account ``auth add``) never write the
        # providers singleton; still count as present credentials.
        credential_pool = store.get("credential_pool") if isinstance(store, dict) else None
        entries = (
            credential_pool.get("xai-oauth")
            if isinstance(credential_pool, dict)
            else None
        )
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("access_token", "") or "").strip():
                    return True
        return False
    except Exception:
        return False


def get_env_value(name: str, default=None):
    """Read ``name`` from ``~/.hermes/.env`` first, then ``os.environ``.

    Wraps :func:`hermes_cli.config.get_env_value` so tests can patch
    ``tools.xai_http.get_env_value`` to inject dotenv-only secrets into the
    xAI credential resolver.
    """
    try:
        from hermes_cli.config import get_env_value as _hermes_get_env_value

        value = _hermes_get_env_value(name)
        if value is not None:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


def hermes_xai_user_agent() -> str:
    """Return a stable Hermes-specific User-Agent for xAI HTTP calls."""
    try:
        from hermes_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"Hermes-Agent/{__version__}"


def hermes_xai_default_headers() -> Dict[str, str]:
    """Default headers for OpenAI-SDK and raw HTTP clients talking to xAI.

    Replaces the OpenAI Python SDK's identifying ``User-Agent: OpenAI/Python …``
    so chat/completions and Responses traffic is attributed as Hermes Agent,
    matching the direct HTTP integrations (search, TTS, STT, image, video).
    """
    return {"User-Agent": hermes_xai_user_agent()}


def _load_config_section(section_name: str) -> Dict[str, Any]:
    """Return a top-level Hermes config section as a dict, or empty."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get(section_name) if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _coerce_expires_after(value: Any) -> Optional[int]:
    """Normalize an xAI storage TTL.

    Returns:
        int seconds for an expiring file,
        None for permanent storage (omit expires_after on the wire).
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "default"}:
            return None
        if normalized in {"none", "null", "never", "permanent", "forever", "0"}:
            return None
        try:
            value = int(normalized)
        except ValueError:
            return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds <= 0:
            return None
        return min(seconds, MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS)
    return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS


def read_xai_imagine_storage_config(section_name: str) -> Dict[str, Any]:
    """Read storage settings for xAI Imagine under image_gen/video_gen config.

    Supported config shape:

        image_gen:
          xai:
            storage:
              enabled: true
              public_url: true
              expires_after: null     # omit for permanent public URLs

    The same shape is accepted under ``video_gen.xai.storage``. Storage is on
    by default so xAI returns permanent public URLs instead of short-lived CDN URLs.
    """
    section = _load_config_section(section_name)
    xai_section = section.get("xai") if isinstance(section, dict) else None
    storage = xai_section.get("storage") if isinstance(xai_section, dict) else None
    storage = storage if isinstance(storage, dict) else {}

    enabled = _coerce_bool(storage.get("enabled"), True)
    public_url = _coerce_bool(storage.get("public_url"), True)
    expires_after = _coerce_expires_after(storage.get("expires_after"))

    return {
        "enabled": enabled,
        "public_url": public_url,
        "expires_after": expires_after,
    }


def build_xai_storage_options(
    section_name: str,
    *,
    filename_prefix: str,
    extension: str,
) -> Optional[Dict[str, Any]]:
    """Return an xAI ``storage_options`` payload, or None when disabled."""
    cfg = read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return None

    now = datetime.datetime.now(datetime.UTC)
    ts = now.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    ext = extension.lstrip(".") or "bin"
    payload: Dict[str, Any] = {
        "filename": f"{filename_prefix}-{ts}-{short}.{ext}",
        "public_url": bool(cfg["public_url"]),
    }
    if cfg["expires_after"] is not None:
        payload["expires_after"] = cfg["expires_after"]
    return payload


def xai_storage_notice_text(section_name: str) -> str:
    """User-facing notice for first xAI Imagine storage use."""
    cfg = read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return ""
    if cfg["expires_after"] is None:
        retention = "without an automatic expiry"
    else:
        days = cfg["expires_after"] / (24 * 60 * 60)
        retention = f"for about {days:g} day{'s' if days != 1 else ''}"
    return (
        "xAI Imagine storage is enabled so generated media gets a reusable "
        f"public URL {retention}. xAI may bill for stored files and public URL "
        f"hosting. Disable this with `{section_name}.xai.storage.enabled: false` "
        "or set `expires_after` to change the retention."
    )


def maybe_mark_xai_storage_notice_seen(section_name: str) -> Optional[str]:
    """Return the storage notice once per Hermes home, then mark it seen."""
    notice = xai_storage_notice_text(section_name)
    if not notice:
        return None
    try:
        from hermes_constants import get_hermes_home

        marker_dir = get_hermes_home() / "state"
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker = marker_dir / f"{section_name}_xai_storage_notice_seen"
        if marker.exists():
            return None
        marker.write_text(datetime.datetime.now(datetime.UTC).isoformat() + "\n", encoding="utf-8")
        return notice
    except Exception:
        return notice


def _canonical_xai_oauth_base_url(value: object, *, fallback: str) -> str:
    """Return an exact canonical xAI inference origin for an OAuth bearer.

    OAuth pool metadata is auth-owned, but a cached/migrated entry must not
    expand this HTTP helper's bearer destination. In particular, neither an
    xAI subdomain nor a non-default TLS port is an OAuth inference origin.
    """
    candidate = str(value or "").strip().rstrip("/")
    safe_fallback = str(fallback).strip().rstrip("/")
    if not candidate:
        return safe_fallback
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        return safe_fallback
    if (
        parsed.scheme.lower() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.hostname or "").lower() != "api.x.ai"
        or port not in (None, 443)
    ):
        return safe_fallback
    return candidate


def resolve_xai_oauth_credentials(
    *,
    force_refresh: bool = False,
    api_key_hint: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Return Hermes-managed xAI OAuth credentials, if a healthy entry exists.

    OAuth's validated inference origin is owned by the auth layer; callers must
    not substitute user-configured direct-HTTP base URLs for it.
    """
    try:
        from agent.credential_pool import load_pool
        import hermes_cli.auth as auth_mod

        pool = load_pool("xai-oauth")
        entry = (
            pool.try_refresh_matching(api_key_hint)
            if force_refresh
            else pool.select()
        )
        if force_refresh and entry is None:
            # A rejected refresh may quarantine the issuing entry. Continue
            # with the next healthy account instead of falling back to the raw
            # singleton resolver and resurrecting the stale pool row.
            entry = pool.select()
        access_token = str(
            getattr(entry, "runtime_api_key", None)
            or getattr(entry, "access_token", "")
        ).strip()
        # Direct HTTP overrides are intentionally excluded: OAuth owns a
        # pinned inference origin, while direct API-key routing resolves its
        # configurable endpoint in resolve_xai_api_key_credentials().
        entry_base_url = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", "")
        )
        base_url = _canonical_xai_oauth_base_url(
            entry_base_url,
            fallback=auth_mod.DEFAULT_XAI_OAUTH_BASE_URL,
        )
        if access_token:
            return {
                "provider": "xai-oauth",
                "api_key": access_token,
                "base_url": base_url,
            }
    except Exception:
        pass
    return None


def resolve_xai_api_key_credentials() -> Dict[str, str]:
    """Resolve only the direct xAI API-key fallback and its endpoint."""
    try:
        from tools.tool_backend_helpers import resolve_provider_secret

        api_key = resolve_provider_secret("XAI_API_KEY", "xai", env_getter=get_env_value)
    except ImportError:  # pragma: no cover — helpers are in-repo
        api_key = str(get_env_value("XAI_API_KEY") or "").strip()
    base_url = str(get_env_value("XAI_BASE_URL") or "https://api.x.ai/v1").strip().rstrip("/")
    return {
        "provider": "xai",
        "api_key": api_key,
        "base_url": base_url,
    }


def resolve_xai_http_credentials(
    *,
    force_refresh: bool = False,
    api_key_hint: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve bearer credentials for direct xAI HTTP endpoints.

    Prefers Hermes-managed xAI OAuth credentials when available, then falls back
    to ``XAI_API_KEY`` resolved via ``hermes_cli.config.get_env_value`` so keys
    stored in ``~/.hermes/.env`` (the standard Hermes location) are honored —
    not just ones already exported into ``os.environ``. This keeps direct xAI
    endpoints (images, TTS, STT, etc.) aligned with the main runtime auth model
    and preserves the regression contract from PR #17140 / #17163.

    Set ``force_refresh=True`` to perform an unconditional OAuth refresh.
    Reactive callers should also pass the rejected bearer as ``api_key_hint``
    so a freshly loaded multi-account pool refreshes the exact issuing entry,
    not whichever entry its strategy would otherwise select first.
    """
    return resolve_xai_oauth_credentials(
        force_refresh=force_refresh,
        api_key_hint=api_key_hint,
    ) or resolve_xai_api_key_credentials()

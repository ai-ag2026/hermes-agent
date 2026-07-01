"""Reachy robot platform adapter (Hermes gateway plugin).

Transport model
---------------
The adapter runs a **WebSocket server**; the robot's voice app connects to it as a
**client** and stays connected. The app keeps ownership of the real-time audio path
(mic, STT, VAD, TTS synthesis, PCM streaming, half-duplex, barge onset). This adapter
only carries *text*:

  inbound  (app -> adapter):  {"type":"hello","robot_id":"reachy"}
                              {"type":"stt","text":"...","robot_id":"reachy"}
                              {"type":"interrupt","text":"...","robot_id":"reachy"}
  outbound (adapter -> app):  {"type":"say","message_id":"m1","content":"...","final":false}
                              {"type":"typing","robot_id":"reachy"}

An inbound "stt"/"interrupt" frame becomes a MessageEvent dispatched via
``handle_message``. Because the gateway's default busy-input mode is ``interrupt``,
a frame that arrives while a turn is generating cancels that turn mid-flight — that is
how barge-in maps onto the platform model (the app's semantic gate decides *whether*
to forward).

Outbound text is streamed: the gateway's stream consumer calls ``edit_message`` with
the growing accumulated ``content`` (SUPPORTS_MESSAGE_EDITING = True). Each edit is
pushed verbatim as a "say" frame; the app diffs against what it has already spoken and
extracts new clauses for TTS. Because the WebSocket persists, supports_async_delivery
stays True (the base default) so background/cron/send_message delivery reaches Reachy.

Zero core changes: ``Platform("reachy")`` resolves via the enum's ``_missing_`` hook.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

try:  # optional dep; check_requirements gates instantiation
    from websockets.asyncio.server import serve as ws_serve

    WEBSOCKETS_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    ws_serve = None  # type: ignore
    WEBSOCKETS_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
DEFAULT_ROBOT_ID = "reachy"


def check_requirements() -> bool:
    """Dependencies present and a listen port configured."""
    if not WEBSOCKETS_AVAILABLE:
        return False
    return bool(os.getenv("REACHY_WS_PORT", "").strip())


class ReachyAdapter(BasePlatformAdapter):
    """WebSocket-server adapter for the Reachy robot voice app."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    # Enable the gateway's streaming path (progressive edit_message deltas).
    SUPPORTS_MESSAGE_EDITING = True
    # Persistent outbound channel → background/cron/send_message can reach Reachy.
    supports_async_delivery = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("reachy"))
        extra = getattr(config, "extra", None) or {}
        self._host: str = str(
            extra.get("host") or os.getenv("REACHY_WS_HOST", "0.0.0.0") or "0.0.0.0"
        )
        self._port: int = int(extra.get("port") or os.getenv("REACHY_WS_PORT", "8770"))
        self._server: Optional[Any] = None
        # robot_id -> active websocket connection
        self._robots: Dict[str, Any] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("[reachy] websockets not installed (pip install websockets)")
            return False
        try:
            self._server = await ws_serve(self._handle_conn, self._host, self._port)
        except Exception as e:  # pragma: no cover - bind failure path
            logger.error("[reachy] failed to start ws server on %s:%s: %s", self._host, self._port, e)
            return False
        self._mark_connected()
        logger.info("[reachy] ws server listening on %s:%s", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        for rid, ws in list(self._robots.items()):
            try:
                await ws.close()
            except Exception:
                pass
            self._robots.pop(rid, None)
        if self._server is not None:
            try:
                self._server.close()
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
        logger.info("[reachy] disconnected")

    # ── inbound transport ──────────────────────────────────────────────────
    async def _handle_conn(self, websocket: Any) -> None:
        """One persistent robot connection. websockets>=13 passes a single arg;
        the request path is on ``websocket.request.path``."""
        path = ""
        try:
            path = getattr(getattr(websocket, "request", None), "path", "") or ""
        except Exception:
            path = ""
        robot_id = self._robot_id_from_path(path)
        self._robots[robot_id] = websocket
        logger.info("[reachy] robot connected: %s (path=%r)", robot_id, path)
        try:
            async for raw in websocket:
                robot_id = await self._on_inbound(robot_id, raw, websocket)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as e:
            logger.debug("[reachy] connection loop ended for %s: %s", robot_id, e)
        finally:
            # only drop the mapping if it still points at this socket
            if self._robots.get(robot_id) is websocket:
                self._robots.pop(robot_id, None)
            logger.info("[reachy] robot disconnected: %s", robot_id)

    @staticmethod
    def _robot_id_from_path(path: str) -> str:
        seg = (path or "").strip("/").split("/")[-1].split("?")[0].strip()
        return seg or DEFAULT_ROBOT_ID

    async def _on_inbound(self, robot_id: str, raw: Any, websocket: Any) -> str:
        """Parse one inbound frame; returns the (possibly updated) robot_id."""
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        raw = (raw or "").strip()
        if not raw:
            return robot_id
        try:
            frame = json.loads(raw)
        except Exception:
            logger.debug("[reachy] non-JSON frame from %s: %r", robot_id, raw[:80])
            return robot_id
        if not isinstance(frame, dict):
            return robot_id

        ftype = str(frame.get("type") or "").lower()
        # allow the app to (re)bind its id
        new_id = str(frame.get("robot_id") or "").strip()
        if new_id and new_id != robot_id:
            if self._robots.get(robot_id) is websocket:
                self._robots.pop(robot_id, None)
            robot_id = new_id
            self._robots[robot_id] = websocket

        if ftype == "hello":
            self._robots[robot_id] = websocket
            return robot_id
        if ftype in ("stt", "interrupt", "text"):
            text = str(frame.get("text") or "").strip()
            if text:
                await self._dispatch_text(robot_id, text)
        else:
            logger.debug("[reachy] unhandled frame type %r from %s", ftype, robot_id)
        return robot_id

    async def _dispatch_text(self, robot_id: str, text: str) -> None:
        source = self.build_source(
            chat_id=robot_id,
            chat_name=f"Reachy {robot_id}",
            chat_type="dm",
            user_id=robot_id,
            user_name=f"Reachy {robot_id}",
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"stt_{robot_id}_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
        )
        await self.handle_message(event)

    # ── outbound transport ─────────────────────────────────────────────────
    async def _push(self, robot_id: str, obj: Dict[str, Any]) -> bool:
        ws = self._robots.get(robot_id)
        if ws is None:
            return False
        try:
            await ws.send(json.dumps(obj, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning("[reachy] push to %s failed: %s", robot_id, e)
            self._robots.pop(robot_id, None)
            return False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        robot_id = (metadata or {}).get("robot_id") or chat_id
        message_id = f"say_{robot_id}_{uuid.uuid4().hex[:10]}"
        ok = await self._push(
            robot_id,
            {"type": "say", "message_id": message_id, "content": content, "final": True},
        )
        if not ok:
            return SendResult(success=False, error=f"robot {robot_id} not connected")
        return SendResult(success=True, message_id=message_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        ok = await self._push(
            chat_id,
            {"type": "say", "message_id": message_id, "content": content, "final": bool(finalize)},
        )
        if not ok:
            return SendResult(success=False, error=f"robot {chat_id} not connected")
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        await self._push(chat_id, {"type": "typing", "robot_id": chat_id})

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": f"Reachy {chat_id}",
            "type": "dm",
            "chat_id": chat_id,
            "connected": chat_id in self._robots,
        }


# ── plugin registration ────────────────────────────────────────────────────
def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env so an env-only setup surfaces in status."""
    port = os.getenv("REACHY_WS_PORT", "").strip()
    if not port:
        return None
    seed: Dict[str, Any] = {"port": int(port)}
    host = os.getenv("REACHY_WS_HOST", "").strip()
    if host:
        seed["host"] = host
    home = os.getenv("REACHY_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": f"Reachy {home}"}
    return seed


async def _standalone_send(pconfig, chat_id: str, message: str, **kwargs) -> Dict[str, Any]:
    """Out-of-process cron delivery is not possible: Reachy needs the live
    WebSocket held by the running gateway adapter."""
    return {
        "error": (
            "Reachy delivery requires the running gateway adapter (persistent "
            "WebSocket to the robot); no standalone send."
        )
    }


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="reachy",
        label="Reachy",
        adapter_factory=lambda cfg: ReachyAdapter(cfg),
        check_fn=check_requirements,
        required_env=["REACHY_WS_PORT"],
        install_hint="pip install websockets",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="REACHY_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="REACHY_ALLOWED_ROBOTS",
        allow_all_env="REACHY_ALLOW_ALL_ROBOTS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🤖",
        allow_update_command=True,
        platform_hint=(
            "You are speaking through a Reachy robot: the user talks and hears you "
            "aloud (speech in, speech out). Keep replies concise and natural for TTS; "
            "avoid markdown, code blocks, URLs, and long lists unless asked."
        ),
    )

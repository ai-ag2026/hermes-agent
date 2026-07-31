"""Kanban quick actions for Telegram — buttons, short commands, reply answers.

Opt-in bridge (``platforms.telegram.extra.kanban_actions: true``) that makes
blocked-card pings actionable from a phone without typing task ids:

* The attention relay (``scripts/tars_kanban_attention_cron.sh``) sends one
  message per blocked card through :func:`send_ping` with inline buttons
  (unblock / details / archive, plus a one-tap human-gate release for gated
  cards) and registers a numbered mapping in a JSON sidecar.
* The gateway adapter routes ``kb…:``-prefixed callback queries to
  :func:`handle_callback`, short text commands ("unblock 1", "unblock alle",
  "details 2") to :func:`try_handle_text`, and replies to a ping message to
  the two-step answer flow (comment first, explicit release).

Design constraints:

* Callback data must stay within Telegram's 64-byte limit — mappings carry a
  short numeric index, never full board/task ids.
* Gate tokens are issued AND redeemed inside the button handler in one step;
  the plaintext token never leaves process memory (parity with the ntfy
  contract: proof is "a human acted in the operator's chat").
* Every action is recorded as a card comment so the board shows who did what.
* All kanban writes run in a thread via the kernel (``hermes_cli.kanban_db``);
  this module never touches the DB from the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import re
import secrets
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ACTOR = "manfred-telegram"

# Callback prefixes (kept short for the 64-byte budget):
#   kbu:<n>        unblock card n with the default release note
#   kbU:           unblock ALL cards of the newest ping wave
#   kbd:<n>        post details for card n
#   kba:<n>        archive card n (asks to confirm)
#   kbA:<n>        archive card n (confirmed)
#   kbg:<n>        human-gate: issue + redeem token, then unblock
#   kbr:<id>:go    pending reply answer <id>: unblock with the answer text
#   kbr:<id>:ask   pending reply answer <id>: keep blocked, hand to the agent
CB_PREFIXES = ("kbu:", "kbU", "kbd:", "kba:", "kbA:", "kbg:", "kbGoff:", "kbr:")

_CMD_RE = re.compile(
    r"^\s*(unblock|entsperre[n]?|details?|verwirf|verwerfen|archivier[e]?)\s+"
    r"(alle|all|\d{1,3})\s*$",
    re.IGNORECASE,
)

_MAX_PING_ITEMS = 200          # sidecar hygiene: keep the newest N mappings
_ANSWER_TTL_SECONDS = 48 * 3600


def state_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME", "") or (Path.home() / ".hermes"))
    return home / "kanban" / "telegram_kanban_pings.json"


@contextlib.contextmanager
def _state_lock():
    """Cross-process exclusive lock around a load-modify-save cycle.

    os.replace makes the WRITE atomic, not the cycle: the cron relay
    (register_ping) and the gateway (handle_callback/try_handle_text) mutate
    the same file from different processes, so a concurrent read-modify-write
    lost entries. Callers wrap their whole cycle in ``with _state_lock():``.
    """
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / (path.name + ".lock")
    lf = open(lock_path, "w")
    try:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        finally:
            lf.close()


def _load_state() -> Dict[str, Any]:
    try:
        with open(state_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"seq": 0, "items": {}, "answers": {}}


def _save_state(state: Dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tg-kanban-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _prune(state: Dict[str, Any]) -> None:
    items: Dict[str, Any] = state.get("items") or {}
    if len(items) > _MAX_PING_ITEMS:
        for key in sorted(items, key=lambda k: int(k))[: len(items) - _MAX_PING_ITEMS]:
            items.pop(key, None)
    now = time.time()
    answers: Dict[str, Any] = state.get("answers") or {}
    for aid in [a for a, v in answers.items()
                if now - float(v.get("ts") or 0) > _ANSWER_TTL_SECONDS]:
        answers.pop(aid, None)


# ---------------------------------------------------------------------------
# Registration + message composition (pure-ish; used by relay and tests)
# ---------------------------------------------------------------------------

def register_ping(
    *, board: str, task_id: str, title: str, message_id: Optional[int] = None,
    chat_id: Optional[str] = None, gated: bool = False,
) -> int:
    """Persist a ping mapping and return its short index number."""
    with _state_lock():
        state = _load_state()
        state["seq"] = int(state.get("seq") or 0) + 1
        n = state["seq"]
        state.setdefault("items", {})[str(n)] = {
            "board": board, "tid": task_id, "title": title[:120],
            "message_id": message_id, "chat_id": chat_id,
            "gated": bool(gated), "ts": time.time(),
        }
        _prune(state)
        _save_state(state)
    return n


def record_message_id(index: int, message_id: int) -> None:
    with _state_lock():
        state = _load_state()
        item = (state.get("items") or {}).get(str(index))
        if item is not None:
            item["message_id"] = message_id
            _save_state(state)


def lookup_index(index: int) -> Optional[Dict[str, Any]]:
    return (_load_state().get("items") or {}).get(str(index))


def lookup_message(message_id: int) -> Optional[Tuple[int, Dict[str, Any]]]:
    for key, item in (_load_state().get("items") or {}).items():
        if item.get("message_id") == message_id:
            return int(key), item
    return None


def latest_wave_indexes(window_seconds: int = 300) -> List[int]:
    """Indexes registered within ``window_seconds`` of the newest entry."""
    items = _load_state().get("items") or {}
    if not items:
        return []
    newest = max(float(v.get("ts") or 0) for v in items.values())
    return sorted(
        int(k) for k, v in items.items()
        if newest - float(v.get("ts") or 0) <= window_seconds
    )


def keyboard_spec(index: int, *, gated: bool = False) -> List[List[Dict[str, str]]]:
    """Inline-keyboard layout for one card ping (as plain data, not PTB types)."""
    if gated:
        first = {"text": "🔓 Gate freigeben & weiterlaufen", "data": f"kbg:{index}"}
        # H1 (2026-07-13): authenticated tap = the operator grant to disable the
        # gate (server-side issue+redeem). Replaces the removed free `gate off`.
        extra = [[{"text": "🔴 Gate abschalten (nur Gate)", "data": f"kbGoff:{index}"}]]
    else:
        first = {"text": "✅ Weiterlaufen lassen", "data": f"kbu:{index}"}
        extra = []
    return [
        [first],
        *extra,
        [
            {"text": "📄 Details", "data": f"kbd:{index}"},
            {"text": "🗄 Verwerfen", "data": f"kba:{index}"},
        ],
    ]


def summary_keyboard_spec() -> List[List[Dict[str, str]]]:
    return [[{"text": "✅ Alle entsperren", "data": "kbU"}]]


def parse_text_command(text: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse "unblock 1" / "unblock alle" / "details 2" → (action, index|None)."""
    m = _CMD_RE.match(text or "")
    if not m:
        return None
    verb = m.group(1).lower()
    arg = m.group(2).lower()
    action = (
        "unblock" if verb.startswith(("unblock", "entsperre"))
        else "details" if verb.startswith("detail")
        else "archive"
    )
    if arg in ("alle", "all"):
        return (action, None) if action == "unblock" else None
    return action, int(arg)


# ---------------------------------------------------------------------------
# Relay-side sender (sync; runs in the cron process, not the gateway)
# ---------------------------------------------------------------------------

def send_ping(
    *, token: str, chat_id: str, text: str,
    keyboard: Optional[List[List[Dict[str, str]]]] = None,
    timeout: float = 20.0,
) -> Optional[int]:
    """POST sendMessage with an optional inline keyboard; return message_id."""
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if keyboard:
        payload["reply_markup"] = {
            "inline_keyboard": [
                [{"text": b["text"], "callback_data": b["data"]} for b in row]
                for row in keyboard
            ]
        }
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"telegram sendMessage failed: {body}")
    return int(body["result"]["message_id"])


# ---------------------------------------------------------------------------
# Kanban kernel operations (sync helpers, run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _kb():
    from hermes_cli import kanban_db as kb
    return kb


def _op_unblock(board: str, tid: str, reason: str) -> Tuple[bool, str]:
    kb = _kb()
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, tid)
        if task is None:
            return False, "Karte nicht gefunden."
        if task.status not in ("blocked", "scheduled"):
            return False, f"Karte ist nicht mehr geblockt (Status: {task.status})."
        if task.human_gate:
            return False, "Karte ist human-gated — bitte den Gate-Button nutzen."
        kb.add_comment(conn, tid, ACTOR, f"UNBLOCK via Telegram: {reason}")
        ok = kb.unblock_task(conn, tid, actor=ACTOR, reason=reason)
        return (True, "entsperrt") if ok else (False, "Entsperren fehlgeschlagen.")


def _op_gate_unblock(board: str, tid: str, reason: str) -> Tuple[bool, str]:
    """Issue a fresh one-time gate token and redeem it immediately.

    Equivalent trust model to the ntfy path: the token proves a human acted
    in the operator's chat; here the tap IS that act. The plaintext token
    only ever exists in this function's local scope.
    """
    kb = _kb()
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, tid)
        if task is None:
            return False, "Karte nicht gefunden."
        if task.status != "blocked":
            return False, f"Karte ist nicht mehr geblockt (Status: {task.status})."
        if not task.human_gate:
            return _op_unblock(board, tid, reason)
        token = kb.issue_gate_token(conn, tid, action="unblock", board=board)
        if not token:
            return False, "Gate-Token konnte nicht erzeugt werden."
        kb.add_comment(
            conn, tid, ACTOR,
            "GATE-FREIGABE via Telegram-Button (Token einmalig erzeugt und "
            f"sofort eingelöst): {reason}",
        )
        ok = kb.unblock_task(conn, tid, actor=ACTOR, reason=reason, token=token)
        return (True, "Gate freigegeben & entsperrt") if ok else (
            False, "Token-Einlösung fehlgeschlagen (Details im Gateway-Log).")


def _op_gate_off(board: str, tid: str, reason: str = "operator gate release") -> Tuple[bool, str]:
    """H1: disable a LIVE human-gate. Issues a fresh gate_off token and redeems it
    immediately server-side — the authorized tap IS the operator act, so the
    plaintext token only ever exists in this function's local scope (never
    delivered to a channel the agent could read). This is the grant surface that
    replaces the removed free `gate off`."""
    kb = _kb()
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, tid)
        if task is None:
            return False, "Karte nicht gefunden."
        if not task.human_gate:
            return False, "Karte ist nicht gegatet."
        token = kb.issue_gate_token(conn, tid, action="gate_off", board=board)
        if not token:
            return False, "Gate-off-Token konnte nicht erzeugt werden (Karte nicht blocked?)."
        kb.add_comment(
            conn, tid, ACTOR,
            "GATE-OFF via Telegram-Button (Token einmalig erzeugt und sofort "
            f"eingelöst): {reason}",
        )
        try:
            ok = kb.set_human_gate(conn, tid, on=False, actor=ACTOR, token=token)
        except kb.GateTokenError as exc:
            return False, f"Token-Einlösung fehlgeschlagen: {exc}"
        return (True, "Gate abgeschaltet") if ok else (False, "Gate-off fehlgeschlagen.")


def _op_archive_running(board: str, tid: str) -> Tuple[bool, str]:
    """H2: archive a card that is running live work (reclaims the run, kills the
    worker). Issues a fresh archive_running token and redeems it immediately
    server-side (authorized tap = operator act; token never leaves the server)."""
    kb = _kb()
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, tid)
        if task is None:
            return False, "Karte nicht gefunden."
        token = kb.issue_gate_token(conn, tid, action="archive_running", board=board)
        if not token:
            return False, "Archive-running-Token konnte nicht erzeugt werden (Karte nicht running?)."
        kb.add_comment(
            conn, tid, ACTOR,
            "ARCHIVE-RUNNING via Telegram-Button (laufender Worker beendet; Token "
            "einmalig erzeugt und sofort eingelöst).",
        )
        try:
            ok = kb.archive_task(conn, tid, token=token)
        except kb.GateTokenError as exc:
            return False, f"Token-Einlösung fehlgeschlagen: {exc}"
        return (True, "laufende Karte archiviert (Worker beendet)") if ok else (
            False, "Archivieren fehlgeschlagen.")


def _op_archive(board: str, tid: str) -> Tuple[bool, str]:
    kb = _kb()
    route_running = False
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, tid)
        if task is None:
            return False, "Karte nicht gefunden."
        if task.human_gate:
            return False, "Karte ist human-gated — erst Gate abschalten (Gate-off-Button)."
        if task.status == "running":
            route_running = True
        else:
            kb.add_comment(conn, tid, ACTOR, "ARCHIVIERT via Telegram-Button.")
            try:
                ok = kb.archive_task(conn, tid)
                return (True, "archiviert") if ok else (False, "Archivieren fehlgeschlagen.")
            except kb.GateTokenError:
                # H2: a still-bound run on a non-running card also needs the grant.
                route_running = True
    # H2: live work -> archive_running grant via its own authorized issue+redeem.
    if route_running:
        return _op_archive_running(board, tid)
    return False, "Archivieren fehlgeschlagen."


def _op_details(board: str, tid: str) -> Tuple[bool, str]:
    kb = _kb()
    with kb.connect_closing(board=board) as conn:
        task = kb.get_task(conn, tid)
        if task is None:
            return False, "Karte nicht gefunden."
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' "
            "ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        payload: Dict[str, Any] = {}
        if rows and rows[0]:
            try:
                payload = json.loads(rows[0])
            except Exception:
                payload = {}
        comments = conn.execute(
            "SELECT author, body FROM task_comments WHERE task_id=? "
            "ORDER BY created_at DESC LIMIT 3", (tid,),
        ).fetchall()
    lines = [
        f"📄 {task.title}",
        f"Status: {task.status} · Board: {board} · @{task.assignee or '—'} · {tid}",
    ]
    if payload.get("human_summary"):
        lines.append(f"Was ist los: {payload['human_summary']}")
    if payload.get("human_action"):
        lines.append(f"Was du tun kannst: {payload['human_action']}")
    if payload.get("reason"):
        lines.append(f"Technisch: {str(payload['reason'])[:400]}")
    if comments:
        lines.append("Letzte Kommentare:")
        for author, body in comments:
            lines.append(f"• {author}: {str(body)[:200]}")
    return True, "\n".join(lines)


def _op_comment(board: str, tid: str, text: str) -> None:
    kb = _kb()
    with kb.connect_closing(board=board) as conn:
        kb.add_comment(conn, tid, ACTOR, text)


# ---------------------------------------------------------------------------
# Gateway-side async handlers
# ---------------------------------------------------------------------------

async def handle_callback(query: Any, data: str) -> bool:
    """Route a ``kb…`` callback. Returns True when the callback was consumed.

    Caller must have verified user authorization already (the adapter's
    shared ``_is_callback_user_authorized`` gate).
    """
    if not data.startswith(CB_PREFIXES):
        return False

    async def _finish(note: str, *, edit_suffix: Optional[str] = None) -> None:
        try:
            await query.answer(text=note[:180])
        except Exception:
            pass
        if edit_suffix is not None:
            try:
                base_text = getattr(getattr(query, "message", None), "text", "") or ""
                await query.edit_message_text(
                    text=f"{base_text}\n\n{edit_suffix}", reply_markup=None,
                )
            except Exception:
                pass  # non-fatal: stale message or already edited

    user = getattr(getattr(query, "from_user", None), "first_name", None) or "Operator"
    stamp = time.strftime("%H:%M")

    # --- pending reply-answer flow: kbr:<answer_id>:<go|ask> ---
    if data.startswith("kbr:"):
        parts = data.split(":")
        if len(parts) != 3:
            await _finish("Ungültige Antwort-Daten.")
            return True
        _, answer_id, verb = parts
        with _state_lock():
            state = _load_state()
            pending = (state.get("answers") or {}).pop(answer_id, None)
            _save_state(state)
        if not pending:
            await _finish("Diese Antwort wurde schon verarbeitet.")
            return True
        board, tid, text = pending["board"], pending["tid"], pending["text"]
        if verb == "go":
            gated = bool(pending.get("gated"))
            op = _op_gate_unblock if gated else _op_unblock
            ok, note = await asyncio.to_thread(
                op, board, tid, f"Antwort von {user} via Telegram: {text}",
            )
            await _finish(note, edit_suffix=(
                f"✅ Antwort gesendet & entsperrt um {stamp}" if ok else f"⚠ {note}"))
        else:  # "ask" — stays blocked; the comment is already on the card
            await _finish(
                "Als Nachfrage vermerkt — Karte bleibt geblockt.",
                edit_suffix=f"💬 Nachfrage um {stamp} vermerkt — Karte bleibt geblockt.",
            )
        return True

    # --- unblock all: kbU ---
    if data == "kbU":
        indexes = latest_wave_indexes()
        results: List[str] = []
        for n in indexes:
            item = lookup_index(n)
            if not item:
                continue
            op = _op_gate_unblock if item.get("gated") else _op_unblock
            ok, note = await asyncio.to_thread(
                op, item["board"], item["tid"],
                f"Sammel-Freigabe via Telegram durch {user}",
            )
            results.append(f"{'✅' if ok else '⚠'} {item['title'][:40]}: {note}")
        await _finish(
            f"{len(results)} Karte(n) verarbeitet.",
            edit_suffix="\n".join(results) or "Keine Karten gefunden.",
        )
        return True

    # --- single-card actions: kbX:<index> ---
    parts = data.split(":")
    if len(parts) != 2 or not parts[1].isdigit():
        await _finish("Ungültige Button-Daten.")
        return True
    verb, n = parts[0], int(parts[1])
    item = lookup_index(n)
    if not item:
        await _finish("Zuordnung nicht mehr vorhanden — bitte `hermes kanban show` nutzen.")
        return True
    board, tid = item["board"], item["tid"]

    if verb == "kbu":
        ok, note = await asyncio.to_thread(
            _op_unblock, board, tid, f"Freigabe via Telegram-Button durch {user}",
        )
        await _finish(note, edit_suffix=(
            f"✅ Entsperrt um {stamp}" if ok else f"⚠ {note}"))
    elif verb == "kbg":
        ok, note = await asyncio.to_thread(
            _op_gate_unblock, board, tid,
            f"Gate-Freigabe via Telegram-Button durch {user}",
        )
        await _finish(note, edit_suffix=(
            f"🔓 Gate freigegeben & entsperrt um {stamp}" if ok else f"⚠ {note}"))
    elif verb == "kbGoff":
        # H1: disable the live gate (grant issued+redeemed server-side).
        ok, note = await asyncio.to_thread(
            _op_gate_off, board, tid,
            f"Gate-off via Telegram-Button durch {user}",
        )
        await _finish(note, edit_suffix=(
            f"🔴 Gate abgeschaltet um {stamp}" if ok else f"⚠ {note}"))
    elif verb == "kbd":
        ok, text = await asyncio.to_thread(_op_details, board, tid)
        try:
            await query.answer()
        except Exception:
            pass
        bot = getattr(query, "get_bot", None)
        message = getattr(query, "message", None)
        if message is not None:
            try:
                await message.reply_text(text[:3900])
            except Exception:
                logger.warning("kanban_actions: details reply failed", exc_info=True)
    elif verb == "kba":
        # First tap only re-arms the button as an explicit confirm step.
        try:
            await query.answer(text="Noch einmal tippen zum Bestätigen.")
            message = getattr(query, "message", None)
            if message is not None:
                await message.edit_reply_markup(
                    reply_markup=_ptb_keyboard(
                        [[{"text": "🗑 Wirklich verwerfen?", "data": f"kbA:{n}"}]]
                    )
                )
        except Exception:
            pass
    elif verb == "kbA":
        ok, note = await asyncio.to_thread(_op_archive, board, tid)
        await _finish(note, edit_suffix=(
            f"🗄 Verworfen um {stamp}" if ok else f"⚠ {note}"))
    else:
        await _finish("Unbekannte Aktion.")
    return True


def _ptb_keyboard(spec: List[List[Dict[str, str]]]) -> Any:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(b["text"], callback_data=b["data"]) for b in row]
        for row in spec
    ])


async def try_handle_text(msg: Any) -> bool:
    """Intercept ping replies and short commands. True = message consumed.

    Anything not clearly kanban-directed falls through to the normal agent
    path — the operator's conversation with the agent must stay untouched.
    """
    text = (getattr(msg, "text", None) or "").strip()
    if not text:
        return False

    # --- Reply to a ping message → two-step answer flow ---
    reply_to = getattr(msg, "reply_to_message", None)
    reply_id = getattr(reply_to, "message_id", None)
    if reply_id is not None:
        hit = lookup_message(int(reply_id))
        if hit is not None:
            n, item = hit
            board, tid = item["board"], item["tid"]
            answer_id = secrets.token_urlsafe(4)
            await asyncio.to_thread(
                _op_comment, board, tid,
                f"Antwort/Nachfrage via Telegram-Reply: {text}",
            )
            with _state_lock():
                state = _load_state()
                state.setdefault("answers", {})[answer_id] = {
                    "board": board, "tid": tid, "text": text,
                    "gated": bool(item.get("gated")), "ts": time.time(),
                }
                _prune(state)
                _save_state(state)
            try:
                await msg.reply_text(
                    f"Notiert auf „{item['title'][:60]}“. Wie weiter?",
                    reply_markup=_ptb_keyboard([
                        [{"text": "✅ Damit weiterlaufen", "data": f"kbr:{answer_id}:go"}],
                        [{"text": "💬 Nur Nachfrage — Karte bleibt geblockt",
                          "data": f"kbr:{answer_id}:ask"}],
                    ]),
                )
            except Exception:
                logger.warning("kanban_actions: answer prompt failed", exc_info=True)
            return True

    # --- Short commands ("unblock 1", "unblock alle", "details 2") ---
    parsed = parse_text_command(text)
    if parsed is None:
        return False
    action, index = parsed
    user = getattr(getattr(msg, "from_user", None), "first_name", None) or "Operator"

    async def _reply(out: str) -> None:
        try:
            await msg.reply_text(out[:3900])
        except Exception:
            pass

    if action == "unblock" and index is None:
        results = []
        for n in latest_wave_indexes():
            item = lookup_index(n)
            if not item:
                continue
            op = _op_gate_unblock if item.get("gated") else _op_unblock
            ok, note = await asyncio.to_thread(
                op, item["board"], item["tid"],
                f"Sammel-Freigabe via Telegram-Befehl durch {user}",
            )
            results.append(f"{'✅' if ok else '⚠'} {item['title'][:40]}: {note}")
        await _reply("\n".join(results) or "Keine offenen Zuordnungen gefunden.")
        return True

    item = lookup_index(index) if index is not None else None
    if not item:
        await _reply(f"Nummer {index} kenne ich nicht (mehr). `hermes kanban list --status blocked` zeigt alles.")
        return True
    board, tid = item["board"], item["tid"]
    if action == "unblock":
        op = _op_gate_unblock if item.get("gated") else _op_unblock
        ok, note = await asyncio.to_thread(
            op, board, tid, f"Freigabe via Telegram-Befehl durch {user}",
        )
        await _reply(f"{'✅' if ok else '⚠'} {item['title'][:60]}: {note}")
    elif action == "details":
        _ok, out = await asyncio.to_thread(_op_details, board, tid)
        await _reply(out)
    elif action == "archive":
        ok, note = await asyncio.to_thread(_op_archive, board, tid)
        await _reply(f"{'🗄' if ok else '⚠'} {item['title'][:60]}: {note}")
    return True

"""Tests for the Telegram kanban quick-action bridge (kanban_actions)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.platforms.telegram import kanban_actions as ka


@pytest.fixture()
def kanban_home(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with an initialized kanban DB + one blocked card."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="Preisimport klemmt", assignee="backend-eng")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid, reason="credential_ref unresolved", kind="needs_input",
            human_summary="Es fehlt ein Zugang.", human_action="Bitte freischalten.",
        )
    finally:
        conn.close()
    return tid


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_parse_text_command_variants():
    assert ka.parse_text_command("unblock 1") == ("unblock", 1)
    assert ka.parse_text_command("Entsperre 12") == ("unblock", 12)
    assert ka.parse_text_command("unblock alle") == ("unblock", None)
    assert ka.parse_text_command("UNBLOCK ALL") == ("unblock", None)
    assert ka.parse_text_command("details 2") == ("details", 2)
    assert ka.parse_text_command("verwerfen 3") == ("archive", 3)
    # No global "details alle"/"verwerfen alle" — too destructive/noisy.
    assert ka.parse_text_command("details alle") is None
    # Normal conversation must fall through untouched.
    assert ka.parse_text_command("kannst du mal schauen?") is None
    assert ka.parse_text_command("unblock t_abc123") is None


def test_keyboard_spec_gated_vs_plain():
    plain = ka.keyboard_spec(7)
    assert plain[0][0]["data"] == "kbu:7"
    gated = ka.keyboard_spec(7, gated=True)
    assert gated[0][0]["data"] == "kbg:7"
    for row in plain + gated + ka.summary_keyboard_spec():
        for btn in row:
            assert len(btn["data"].encode()) <= 64


def test_register_lookup_and_wave(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    n1 = ka.register_ping(board="default", task_id="t_aaa", title="Eins", gated=False)
    n2 = ka.register_ping(board="alt", task_id="t_bbb", title="Zwei", gated=True)
    assert n2 == n1 + 1
    ka.record_message_id(n1, 4711)
    assert ka.lookup_index(n1)["tid"] == "t_aaa"
    assert ka.lookup_message(4711)[0] == n1
    assert ka.lookup_message(9999) is None
    assert set(ka.latest_wave_indexes()) == {n1, n2}


# ---------------------------------------------------------------------------
# Kernel ops against a real temp DB
# ---------------------------------------------------------------------------

def test_op_unblock_roundtrip(kanban_home):
    tid = kanban_home
    ok, note = ka._op_unblock("default", tid, "Freigabe via Test")
    assert ok, note
    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status in ("ready", "todo")
        comments = conn.execute(
            "SELECT body FROM task_comments WHERE task_id=?", (tid,)
        ).fetchall()
    assert any("UNBLOCK via Telegram" in c[0] for c in comments)
    # Second attempt: no longer blocked → friendly error, no crash.
    ok2, note2 = ka._op_unblock("default", tid, "again")
    assert not ok2 and "nicht mehr geblockt" in note2


def test_op_unblock_refuses_gated_card(kanban_home, monkeypatch):
    tid = kanban_home
    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET human_gate=1 WHERE id=?", (tid,))
        conn.commit()
    ok, note = ka._op_unblock("default", tid, "x")
    assert not ok and "human-gated" in note


def test_op_gate_unblock_issues_and_redeems(kanban_home):
    tid = kanban_home
    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET human_gate=1 WHERE id=?", (tid,))
        conn.commit()
    ok, note = ka._op_gate_unblock("default", tid, "Gate-Freigabe via Test")
    assert ok, note
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status in ("ready", "todo")
        kinds = [r[0] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,)
        ).fetchall()]
    assert "gate_token_issued" in kinds


def test_op_details_contains_layman_fields(kanban_home):
    ok, text = ka._op_details("default", kanban_home)
    assert ok
    assert "Was ist los: Es fehlt ein Zugang." in text
    assert "Was du tun kannst: Bitte freischalten." in text
    assert "credential_ref" in text


def test_op_archive(kanban_home):
    tid = kanban_home
    ok, note = ka._op_archive("default", tid)
    assert ok, note
    from hermes_cli import kanban_db as kb
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "archived"


# ---------------------------------------------------------------------------
# Callback routing (fake query objects; kernel ops patched where irrelevant)
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, text="ping text", message_id=100):
        self.text = text
        self.message_id = message_id
        self.replies = []
        self.markups = []

    async def reply_text(self, text, **kw):
        self.replies.append((text, kw))
        return SimpleNamespace(message_id=self.message_id + 1)

    async def edit_reply_markup(self, reply_markup=None):
        self.markups.append(reply_markup)


class FakeQuery:
    def __init__(self, data, message=None):
        self.data = data
        self.message = message or FakeMessage()
        self.from_user = SimpleNamespace(id=1, first_name="Manfred")
        self.answers = []
        self.edits = []

    async def answer(self, text=None):
        self.answers.append(text)

    async def edit_message_text(self, text=None, reply_markup=None, **kw):
        self.edits.append(text)


def test_handle_callback_ignores_foreign_prefixes():
    q = FakeQuery("mp:whatever")
    assert asyncio.run(ka.handle_callback(q, "mp:whatever")) is False


def test_handle_callback_unblock(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    n = ka.register_ping(board="default", task_id="t_x1", title="Karte X")
    calls = {}

    def fake_unblock(board, tid, reason):
        calls["args"] = (board, tid, reason)
        return True, "entsperrt"

    monkeypatch.setattr(ka, "_op_unblock", fake_unblock)
    q = FakeQuery(f"kbu:{n}")
    assert asyncio.run(ka.handle_callback(q, q.data)) is True
    assert calls["args"][0] == "default" and calls["args"][1] == "t_x1"
    assert any("Entsperrt" in e for e in q.edits)


def test_handle_callback_stale_index(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    q = FakeQuery("kbu:99")
    assert asyncio.run(ka.handle_callback(q, q.data)) is True
    assert any("Zuordnung" in (a or "") for a in q.answers)


def test_handle_callback_archive_needs_confirm(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    n = ka.register_ping(board="default", task_id="t_x2", title="Karte Y")
    called = {}
    monkeypatch.setattr(ka, "_op_archive", lambda b, t: called.setdefault("hit", (b, t)) or (True, "archiviert"))
    monkeypatch.setattr(ka, "_ptb_keyboard", lambda spec: spec)
    # First tap: only re-arms, does NOT archive.
    q1 = FakeQuery(f"kba:{n}")
    asyncio.run(ka.handle_callback(q1, q1.data))
    assert "hit" not in called
    assert q1.message.markups, "confirm re-arm expected"
    # Confirmed tap: archives.
    q2 = FakeQuery(f"kbA:{n}")
    asyncio.run(ka.handle_callback(q2, q2.data))
    assert called["hit"] == ("default", "t_x2")


def test_handle_callback_unblock_all_wave(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ka.register_ping(board="default", task_id="t_a", title="A")
    ka.register_ping(board="alt", task_id="t_b", title="B", gated=True)
    seen = []
    monkeypatch.setattr(ka, "_op_unblock", lambda b, t, r: (seen.append(("u", b, t)), (True, "ok"))[1])
    monkeypatch.setattr(ka, "_op_gate_unblock", lambda b, t, r: (seen.append(("g", b, t)), (True, "ok"))[1])
    q = FakeQuery("kbU")
    asyncio.run(ka.handle_callback(q, "kbU"))
    assert ("u", "default", "t_a") in seen
    assert ("g", "alt", "t_b") in seen


# ---------------------------------------------------------------------------
# Text interception: replies and short commands
# ---------------------------------------------------------------------------

def _fake_incoming(text, reply_to_id=None):
    msg = FakeMessage(text=text, message_id=500)
    msg.from_user = SimpleNamespace(id=1, first_name="Manfred")
    msg.reply_to_message = (
        SimpleNamespace(message_id=reply_to_id) if reply_to_id else None
    )
    return msg


def test_reply_to_ping_starts_answer_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    n = ka.register_ping(board="default", task_id="t_r1", title="Fragekarte")
    ka.record_message_id(n, 4242)
    comments = []
    monkeypatch.setattr(ka, "_op_comment", lambda b, t, x: comments.append((b, t, x)))
    monkeypatch.setattr(ka, "_ptb_keyboard", lambda spec: spec)

    msg = _fake_incoming("Nimm den Testzugang aus der .env", reply_to_id=4242)
    assert asyncio.run(ka.try_handle_text(msg)) is True
    # Comment lands immediately; card is NOT unblocked yet.
    assert comments and "Testzugang" in comments[0][2]
    # The prompt offers both explicit choices.
    prompt_text, kw = msg.replies[0]
    flat = json.dumps(kw["reply_markup"])
    assert ":go" in flat and ":ask" in flat
    # Answer is stored for the button decision.
    answers = ka._load_state()["answers"]
    assert len(answers) == 1
    (aid, pending), = answers.items()
    assert pending["tid"] == "t_r1" and "Testzugang" in pending["text"]

    # "go" button releases with the stored text as reason.
    released = {}
    monkeypatch.setattr(ka, "_op_unblock", lambda b, t, r: (released.setdefault("r", r), (True, "entsperrt"))[1])
    q = FakeQuery(f"kbr:{aid}:go")
    asyncio.run(ka.handle_callback(q, q.data))
    assert "Testzugang" in released["r"]
    # One-time: a second press reports "already processed".
    q2 = FakeQuery(f"kbr:{aid}:go")
    asyncio.run(ka.handle_callback(q2, q2.data))
    assert any("schon verarbeitet" in (a or "") for a in q2.answers)


def test_reply_ask_keeps_card_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    n = ka.register_ping(board="default", task_id="t_r2", title="Nachfragekarte")
    ka.record_message_id(n, 555)
    monkeypatch.setattr(ka, "_op_comment", lambda b, t, x: None)
    monkeypatch.setattr(ka, "_ptb_keyboard", lambda spec: spec)
    msg = _fake_incoming("Was genau fehlt denn?", reply_to_id=555)
    asyncio.run(ka.try_handle_text(msg))
    (aid, _), = ka._load_state()["answers"].items()

    unblocked = []
    monkeypatch.setattr(ka, "_op_unblock", lambda b, t, r: (unblocked.append(1), (True, "x"))[1])
    q = FakeQuery(f"kbr:{aid}:ask")
    asyncio.run(ka.handle_callback(q, q.data))
    assert not unblocked, "ask must not unblock"
    assert any("bleibt geblockt" in e for e in q.edits)


def test_plain_chat_falls_through(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    msg = _fake_incoming("Wie ist der Stand beim Dashboard?")
    assert asyncio.run(ka.try_handle_text(msg)) is False
    msg2 = _fake_incoming("unblock 3")  # unknown index → consumed with hint
    assert asyncio.run(ka.try_handle_text(msg2)) is True
    assert "kenne ich nicht" in msg2.replies[0][0]


def test_command_unblock_single(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    n = ka.register_ping(board="default", task_id="t_c1", title="Cmdkarte")
    hits = {}
    monkeypatch.setattr(ka, "_op_unblock", lambda b, t, r: (hits.setdefault("a", (b, t)), (True, "entsperrt"))[1])
    msg = _fake_incoming(f"unblock {n}")
    assert asyncio.run(ka.try_handle_text(msg)) is True
    assert hits["a"] == ("default", "t_c1")
    assert "✅" in msg.replies[0][0]

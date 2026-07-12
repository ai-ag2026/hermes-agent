import sqlite3

from hermes_cli import kanban_db as kb


def _attention_with_sub(tmp_path, monkeypatch, channels=1):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "attention.db"))
    kb.init_db()
    conn = kb.connect()
    tid = kb.create_task(conn, title="Public title", assignee="worker")
    for n in range(channels):
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id=f"chat-{n}")
    attention = kb.upsert_current_typed_attention(
        conn, task_id=tid, attention_type="decision", reason_code="credential_choice", summary="Need a decision"
    )
    conn.close()
    return tid, attention


def test_attention_delivery_is_per_channel_durable_and_retryable(tmp_path, monkeypatch):
    tid, attention = _attention_with_sub(tmp_path, monkeypatch, channels=2)
    conn = kb.connect()
    assert kb.sync_attention_deliveries(conn, task_ids=[tid], now=100)["created"] == 2
    first = kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                        platform="telegram", chat_id="chat-0", now=100)
    assert first and first["attempts"] == 1
    assert kb.finish_attention_delivery(conn, first, success=False, now=101)
    retry = kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                        platform="telegram", chat_id="chat-0", now=102)
    assert retry and retry["attempts"] == 2
    assert kb.finish_attention_delivery(conn, retry, success=True, now=103)
    assert kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                       platform="telegram", chat_id="chat-0", now=104) is None
    other = kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                        platform="telegram", chat_id="chat-1", now=104)
    assert other
    conn.close()


def test_attention_delivery_two_connections_only_one_claim_and_expired_lease_reclaims(tmp_path, monkeypatch):
    tid, attention = _attention_with_sub(tmp_path, monkeypatch)
    one, two = kb.connect(), kb.connect()
    kb.sync_attention_deliveries(one, task_ids=[tid], now=100)
    claim = kb.claim_attention_delivery(one, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                        platform="telegram", chat_id="chat-0", now=100, lease_seconds=5)
    assert claim
    assert kb.claim_attention_delivery(two, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                       platform="telegram", chat_id="chat-0", now=101) is None
    reclaim = kb.claim_attention_delivery(two, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                         platform="telegram", chat_id="chat-0", now=106)
    assert reclaim and reclaim["attempts"] == 2
    assert reclaim["lease_version"] > claim["lease_version"]
    assert not kb.finish_attention_delivery(one, claim, success=True, now=107)
    assert kb.finish_attention_delivery(two, reclaim, success=True, now=107)
    one.close(); two.close()


def test_attention_delivery_cancels_resolution_and_version_bump(tmp_path, monkeypatch):
    tid, attention = _attention_with_sub(tmp_path, monkeypatch)
    conn = kb.connect()
    kb.sync_attention_deliveries(conn, task_ids=[tid], now=100)
    replacement = kb.upsert_current_typed_attention(conn, task_id=tid, attention_type="decision", reason_code="publication_approval", summary="Changed")
    outcome = kb.sync_attention_deliveries(conn, task_ids=[tid], now=101)
    assert outcome["cancelled_resolved"] == 1
    assert kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version,
                                       platform="telegram", chat_id="chat-0", now=102) is None
    new = kb.claim_attention_delivery(conn, task_id=tid, attention_id=replacement.id, attention_version=replacement.version,
                                      platform="telegram", chat_id="chat-0", now=102)
    assert new
    conn.close()


def test_attention_delivery_schema_is_additive_and_quick_check(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    kb.init_db(); kb.init_db()
    conn = kb.connect()
    assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='kanban_attention_deliveries'").fetchone()
    conn.close()


def test_attention_delivery_migrates_immediately_previous_table_shape(tmp_path, monkeypatch):
    db = tmp_path / "previous.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    # Start from the actual immediately preceding schema: it already had the
    # typed-attention/action columns and all current indexes, but delivery rows
    # predated the lease generation.  Keeping the parent shape matters because
    # SCHEMA_SQL creates indexes over fields such as ``consumed_at`` before the
    # additive migration runs; a hand-minimalized old table is not deployable.
    legacy = sqlite3.connect(db)
    legacy.executescript(kb.SCHEMA_SQL)
    legacy.executescript("""
        DROP TABLE kanban_attention_deliveries;
        CREATE TABLE kanban_attention_deliveries (
            task_id TEXT NOT NULL, attention_id INTEGER NOT NULL,
            attention_version INTEGER NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,
            thread_id TEXT NOT NULL DEFAULT '', notifier_profile TEXT,
            state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            lease_until INTEGER, delivered_at INTEGER, last_error TEXT, updated_at INTEGER NOT NULL,
            PRIMARY KEY(task_id, attention_id, attention_version, platform, chat_id, thread_id)
        );
    """)
    legacy.execute("INSERT INTO tasks (id, title, assignee, status, created_at) VALUES ('old-task','Old public title','worker','blocked',1)")
    legacy.execute("INSERT INTO task_attentions (task_id,action_id,type,cause_fingerprint,summary,created_at,version) VALUES ('old-task',NULL,'decision','legacy','Need decision',1,1)")
    legacy.execute("INSERT INTO kanban_notify_subs (task_id,platform,chat_id,created_at) VALUES ('old-task','telegram','legacy-chat',1)")
    legacy.execute("INSERT INTO kanban_attention_deliveries (task_id,attention_id,attention_version,platform,chat_id,updated_at) VALUES ('old-task',1,1,'telegram','legacy-chat',1)")
    legacy.commit(); legacy.close()

    kb.init_db(); kb.init_db()
    conn = kb.connect()
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(kanban_attention_deliveries)")}
    assert "lease_version" in columns
    assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    outcome = kb.sync_attention_deliveries(conn, task_ids=["old-task"], now=10)
    assert outcome == {"created": 0, "cancelled_resolved": 0, "suppressed_duplicate": 1}
    delivery = conn.execute("SELECT task_id, lease_version FROM kanban_attention_deliveries").fetchone()
    assert tuple(delivery) == ("old-task", 0)
    assert conn.execute("SELECT id FROM tasks WHERE id='old-task'").fetchone()[0] == "old-task"
    conn.close()


def test_subscription_reactivation_rearms_only_its_generation(tmp_path, monkeypatch):
    tid, attention = _attention_with_sub(tmp_path, monkeypatch, channels=2)
    conn = kb.connect()
    kb.sync_attention_deliveries(conn, task_ids=[tid], now=100)
    delivered = kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version, platform="telegram", chat_id="chat-0", now=101)
    assert delivered and kb.finish_attention_delivery(conn, delivered, success=True, now=102)
    assert kb.remove_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-0")
    kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-0")
    kb.sync_attention_deliveries(conn, task_ids=[tid], now=103)
    row = conn.execute("SELECT state, subscription_generation FROM kanban_attention_deliveries WHERE task_id=? AND chat_id='chat-0'", (tid,)).fetchone()
    other = conn.execute("SELECT state FROM kanban_attention_deliveries WHERE task_id=? AND chat_id='chat-1'", (tid,)).fetchone()
    assert tuple(row) == ("pending", 2)
    assert other["state"] == "pending"
    assert kb.list_notify_subs(conn, tid)[0]["generation"] == 2
    conn.close()


def test_subscription_active_idempotence_thread_normalization_and_unsubscribe_fence(tmp_path, monkeypatch):
    tid, attention = _attention_with_sub(tmp_path, monkeypatch)
    conn = kb.connect()
    kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-0", thread_id="")
    assert kb.list_notify_subs(conn, tid)[0]["generation"] == 1
    kb.sync_attention_deliveries(conn, task_ids=[tid], now=100)
    claim = kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version, platform="telegram", chat_id="chat-0", thread_id=None, now=101)
    assert claim
    assert kb.remove_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-0", thread_id="")
    assert kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version, platform="telegram", chat_id="chat-0", now=102) is None
    assert not kb.finish_attention_delivery(conn, claim, success=True, now=102)
    assert kb.list_notify_subs(conn, tid) == []
    assert len(kb.list_notify_subs(conn, tid, include_inactive=True)) == 1
    kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-0", thread_id=None)
    kb.sync_attention_deliveries(conn, task_ids=[tid], now=103)
    fresh = kb.claim_attention_delivery(conn, task_id=tid, attention_id=attention.id, attention_version=attention.version, platform="telegram", chat_id="chat-0", now=104)
    assert fresh and fresh["subscription_generation"] == 2
    assert not kb.finish_attention_delivery(conn, claim, success=True, now=105)
    assert kb.finish_attention_delivery(conn, fresh, success=True, now=105)
    conn.close()

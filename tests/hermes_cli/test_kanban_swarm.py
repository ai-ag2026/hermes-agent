
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import (
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
)


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Map the target market and produce a decision memo.",
            workers=[
                SwarmWorkerSpec(profile="researcher-a", title="Market scan", body="Find competitors"),
                SwarmWorkerSpec(profile="researcher-b", title="Customer scan", body="Find customer pains"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="intel",
            created_by="orchestrator",
        )

        root = kb.get_task(conn, created.root_id)
        workers = [kb.get_task(conn, tid) for tid in created.worker_ids]
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        assert root.status == "done"
        assert root.assignee == "orchestrator"
        assert [task.status for task in workers] == ["ready", "ready"]
        assert [task.assignee for task in workers] == ["researcher-a", "researcher-b"]
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(created.worker_ids)
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        assert all(created.root_id in (task.body or "") for task in workers)
    finally:
        conn.close()


def test_swarm_blackboard_merges_structured_updates(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Collect evidence.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Evidence", body="Find proof")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        post_blackboard_update(
            conn,
            created.root_id,
            author="researcher",
            key="sources",
            value=["https://example.com/a"],
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="reviewer",
            key="risks",
            value={"missing_primary_source": True},
        )

        board = latest_blackboard(conn, created.root_id)
        assert board["sources"] == ["https://example.com/a"]
        assert board["risks"] == {"missing_primary_source": True}
        assert board["_authors"]["sources"] == "researcher"
    finally:
        conn.close()


def test_swarm_verifier_and_synthesis_are_dependency_gated(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Research two branches then verify and synthesize.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="b", title="Branch B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        kb.complete_task(
            conn,
            created.worker_ids[0],
            summary="A done",
            metadata={"confidence": 0.8},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "todo"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "ready"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.synthesizer_id).status == "ready"
    finally:
        conn.close()


# ── CC-PARITY-B1: perspective-diverse adversarial verifier panel ───────────

import pytest  # noqa: E402


_WORKERS = [
    SwarmWorkerSpec(profile="researcher-a", title="Scan A", body="find A"),
    SwarmWorkerSpec(profile="researcher-b", title="Scan B", body="find B"),
]


def test_single_verifier_is_neutral(tmp_path):
    """No lenses (or one) reproduces the classic single-verifier topology."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Audit the thing",
            workers=_WORKERS,
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        assert created.verifier_ids == [created.verifier_id]
        assert kb.get_task(conn, created.verifier_id).title == "Verify swarm outputs"
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        assert created.as_dict()["verifier_ids"] == [created.verifier_id]
    finally:
        conn.close()


def test_verifier_panel_one_card_per_lens(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Audit harder",
            workers=_WORKERS,
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            verifier_lenses=["correctness", "security", "reproducibility"],
        )
        assert len(created.verifier_ids) == 3
        # primary verifier stays the first for backward compatibility
        assert created.verifier_id == created.verifier_ids[0]
        # every verifier gates on all workers
        for vid in created.verifier_ids:
            assert set(kb.parent_ids(conn, vid)) == set(created.worker_ids)
        titles = [kb.get_task(conn, v).title for v in created.verifier_ids]
        assert titles == [
            "Verify swarm outputs (correctness)",
            "Verify swarm outputs (security)",
            "Verify swarm outputs (reproducibility)",
        ]
        # each verifier body is anchored to its lens and told to refute
        for lens, vid in zip(["correctness", "security", "reproducibility"], created.verifier_ids):
            body = kb.get_task(conn, vid).body
            assert lens in body and "REFUTE" in body
        # synthesizer gates on the whole panel and applies majority-refute
        assert set(kb.parent_ids(conn, created.synthesizer_id)) == set(created.verifier_ids)
        sbody = kb.get_task(conn, created.synthesizer_id).body
        assert "MAJORITY" in sbody and "verdict:<lens>" in sbody
    finally:
        conn.close()


def test_verifier_lens_dedup_collapses_to_single(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="dup lenses",
            workers=_WORKERS,
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            verifier_lenses=["security", "Security", "  "],
        )
        assert len(created.verifier_ids) == 1
        assert kb.get_task(conn, created.verifier_id).title == "Verify swarm outputs"
    finally:
        conn.close()


def test_blackboard_contract_rejects_missing_key(tmp_path):
    """CC-PARITY-A2: require_value_keys enforces a contract before writing."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="contract",
            workers=[SwarmWorkerSpec(profile="w", title="t", body="b")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        with pytest.raises(ValueError, match="findings"):
            post_blackboard_update(
                conn,
                created.root_id,
                author="w",
                key="result",
                value={"summary": "ok"},  # missing "findings"
                require_value_keys=["summary", "findings"],
            )
        # a valid payload writes fine and round-trips
        post_blackboard_update(
            conn,
            created.root_id,
            author="w",
            key="result",
            value={"summary": "ok", "findings": ["x"]},
            require_value_keys=["summary", "findings"],
        )
        assert latest_blackboard(conn, created.root_id)["result"]["findings"] == ["x"]
    finally:
        conn.close()

"""Tests for hermes_cli/schema_contract.py (CC-PARITY-A2)."""

from __future__ import annotations

from hermes_cli.schema_contract import validate_decompose_graph, validate_fields


class TestValidateFields:
    def test_non_dict_payload_rejected(self):
        assert validate_fields("x", {"a": {}}) == ["expected a JSON object, got str"]

    def test_missing_required_field(self):
        assert validate_fields({}, {"a": {}}) == ["missing required field 'a'"]

    def test_optional_field_may_be_absent(self):
        assert validate_fields({}, {"a": {"required": False}}) == []

    def test_type_mismatch(self):
        assert validate_fields({"a": 5}, {"a": {"type": str}}) == [
            "field 'a' must be str, got int"
        ]

    def test_type_tuple_ok(self):
        assert validate_fields({"a": 5}, {"a": {"type": (str, int)}}) == []

    def test_non_empty_rejects_empty(self):
        assert validate_fields({"a": ""}, {"a": {"type": str, "non_empty": True}}) == [
            "field 'a' must not be empty"
        ]
        assert validate_fields({"a": []}, {"a": {"non_empty": True}}) == [
            "field 'a' must not be empty"
        ]

    def test_valid_passes(self):
        assert validate_fields({"a": "x", "b": 2}, {"a": {"type": str}, "b": {"type": int}}) == []


class TestValidateDecomposeGraph:
    def test_fanout_true_needs_nonempty_tasks(self):
        assert validate_decompose_graph({"fanout": True, "tasks": []}) == [
            "field 'tasks' must not be empty"
        ]

    def test_fanout_true_task_needs_title(self):
        errs = validate_decompose_graph({"fanout": True, "tasks": [{"title": ""}]})
        assert errs and errs[0].startswith("tasks[0]:")

    def test_fanout_true_valid(self):
        assert validate_decompose_graph(
            {"fanout": True, "tasks": [{"title": "do a thing"}]}
        ) == []

    def test_fanout_false_needs_title_or_body(self):
        assert validate_decompose_graph({"fanout": False}) == [
            "fanout=false requires a non-empty 'title' or 'body'"
        ]

    def test_fanout_false_valid_with_title(self):
        assert validate_decompose_graph({"fanout": False, "title": "t"}) == []

    def test_fanout_false_valid_with_body(self):
        assert validate_decompose_graph({"fanout": False, "body": "b"}) == []

    def test_non_dict_rejected(self):
        assert validate_decompose_graph("nope")[0].startswith("expected a JSON object")

"""P6 Option 2 dispatcher carry — board→bank routing (tars-hermes carry)."""
import json
import importlib

kb = importlib.import_module("hermes_cli.kanban_db")


def _write_map(tmp_path, monkeypatch, data):
    root = tmp_path
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "kanban-board-banks.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(kb, "kanban_home", lambda: root)


def test_mapped_board_returns_bank(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"work-projects": "work"})
    assert kb._resolve_board_bank("work-projects") == "work"


def test_unmapped_board_no_default_returns_none(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"work-projects": "work"})
    assert kb._resolve_board_bank("something-private") is None


def test_default_applies_to_unmapped(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"_default": "orchestration-quarantine"})
    assert kb._resolve_board_bank("anything") == "orchestration-quarantine"


def test_mapped_wins_over_default(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"work-projects": "work", "_default": "quarantine"})
    assert kb._resolve_board_bank("work-projects") == "work"


def test_readme_key_not_treated_as_board(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"_README": "docs"})
    assert kb._resolve_board_bank("_README") is None  # no _default → None


def test_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)  # no ops/ file
    assert kb._resolve_board_bank("anything") is None


def test_invalid_bank_id_rejected(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"b": "../etc/passwd"})
    assert kb._resolve_board_bank("b") is None


def test_none_board_uses_default(tmp_path, monkeypatch):
    _write_map(tmp_path, monkeypatch, {"_default": "q"})
    assert kb._resolve_board_bank(None) == "q"

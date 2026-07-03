"""CC-PARITY-C1: the invocation gate on the auto-offer skill index.

Builds a throwaway skills catalog under a temp HERMES_HOME and asserts that
``build_skills_system_prompt`` (the surface the model / skills_hub router uses to
auto-invoke skills) hides ``user-only`` / ``disabled`` skills while leaving
``auto`` (the default) untouched.
"""

from __future__ import annotations


def _write_skill(root, category, name, *, invocation=None):
    d = root / "skills" / category / name
    d.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}", f'description: "{name} does a testing thing."']
    if invocation is not None:
        lines += ["metadata:", "  hermes:", f"    invocation: {invocation}"]
    lines += ["---", "", f"# {name}", "", "Do the thing.", ""]
    (d / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")


def test_invocation_gate_filters_auto_offer(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    _write_skill(tmp_path, "cat", "auto-skill")
    _write_skill(tmp_path, "cat", "useronly-skill", invocation="user-only")
    _write_skill(tmp_path, "cat", "disabled-skill", invocation="disabled")

    from agent import prompt_builder as pb

    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    idx = pb.build_skills_system_prompt()

    # default (auto) skill is offered; non-auto skills are hidden
    assert "auto-skill" in idx
    assert "useronly-skill" not in idx
    assert "disabled-skill" not in idx


def test_invocation_gate_neutral_when_unflagged(tmp_path, monkeypatch):
    """With no invocation flags anywhere, every skill is offered (neutrality)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    for name in ("alpha-skill", "beta-skill", "gamma-skill"):
        _write_skill(tmp_path, "cat", name)

    from agent import prompt_builder as pb

    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    idx = pb.build_skills_system_prompt()

    assert "alpha-skill" in idx
    assert "beta-skill" in idx
    assert "gamma-skill" in idx

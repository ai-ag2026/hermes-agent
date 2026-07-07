"""Regression coverage for TARS/Hermes completion-drive guidance.

These tests intentionally assert stable behavioral contract text, not exact full
prompt blobs. The failure mode being guarded is prompt/judge drift back toward
polishing, cautious non-terminal stops, or irreversible cleanup.
"""

from agent.coding_context import CODING_AGENT_GUIDANCE
from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
from hermes_cli.goals import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE,
)


def test_execution_guidance_biases_once_verified_work_to_terminal_step():
    guidance = OPENAI_MODEL_EXECUTION_GUIDANCE

    assert "<completion_bias>" in guidance
    assert "verified the result ONCE" in guidance
    assert "TERMINAL" in guidance
    assert "Extra tool calls that only polish" in guidance
    assert "Defer, don't rabbit-hole" in guidance


def test_execution_guidance_keeps_risk_gates_narrow_but_real():
    guidance = OPENAI_MODEL_EXECUTION_GUIDANCE

    assert "GENUINELY risky actions" in guidance
    assert "restarting live infrastructure" in guidance
    assert "physical-device actions" in guidance
    assert "third-party upstreams" in guidance
    assert "touching secrets" in guidance
    assert "Ordinary side effects" in guidance
    assert "do NOT need pre-confirmation" in guidance
    assert "restarts production/live infrastructure" in guidance


def test_execution_guidance_for_cleanup_prefers_reversible_data_safety():
    guidance = OPENAI_MODEL_EXECUTION_GUIDANCE.lower()

    assert "clean up" in guidance
    assert "reversible move" in guidance
    assert "over `rm`" in guidance
    assert ".csv/.db/datasets" in guidance
    assert "source, configs" in guidance
    assert "not 'junk'" in guidance
    assert "hash manifest is not a backup" in guidance


def test_coding_guidance_allows_ordinary_finishing_commits_but_gates_history_risk():
    guidance = CODING_AGENT_GUIDANCE

    assert "ordinary commits and PRs" in guidance
    assert "normal part of finishing" in guidance
    assert "rewriting shared" in guidance
    assert "force-pushing" in guidance
    assert "third-party/upstream repos" in guidance
    assert "Never read, print, or commit secrets" in guidance


def test_goal_judge_no_longer_treats_cautious_blocks_as_done():
    prompt = JUDGE_SYSTEM_PROMPT

    assert "terminal step" in prompt
    assert "genuinely blocked" in prompt
    assert "ONLY the user can resolve" in prompt
    assert "merely cautious" in prompt
    assert "partial progress reported as complete" in prompt
    assert "NOT done" in prompt
    assert "CONTINUE toward the terminal step" in prompt


def test_goal_contract_template_distinguishes_stop_conditions_from_caution():
    prompt = JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE

    assert "explicit Stop condition" in prompt
    assert "merely cautious" in prompt
    assert "partial progress reported as complete" in prompt
    assert "Only a genuine" in prompt
    assert "ONLY the user can resolve" in prompt

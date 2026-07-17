#!/usr/bin/env python3
"""Golden-corpus replay for the approval-tier model (Autonomie-Umbau A/WS1).

Classifies a corpus of real (or synthetic) Kanban worker interrupt commands
through ``tools.approval._classify_kanban_action`` and prints a before/after
table: "before" is every one of these interrupts pinging an operator (today's
behavior, tiers disabled); "after" is the tier distribution once
``approvals.tiers.enabled`` is turned on.

Read-only: never writes to a live kanban.db. Two corpus sources:

  * ``--corpus PATH`` -- a JSON file: a list of objects with at least a
    "command" field (and optionally "workspace", "mutation_kind",
    "tirith_findings"). Point this at an exported log of real interrupts.
  * default (no ``--corpus``) -- a small synthetic corpus mirroring the
    documented WS0 interrupt-class distribution from 2026-07-16 (class-b
    "workspace-contained" ~22.5%, class-c "self-modify" ~10%, the rest a mix
    of read-only and ambiguous/external commands). This is a STAND-IN for the
    real 2026-07-16 interrupt log, which was not available to this script at
    write time -- replace it with ``--corpus`` against the real export before
    treating the numbers as anything but illustrative.

Usage:
    venv/bin/python scripts/tars_tier_replay.py
    venv/bin/python scripts/tars_tier_replay.py --corpus interrupts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import approval  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic stand-in corpus (see module docstring: replace via --corpus with
# a real export before drawing operational conclusions from this script).
# Each entry: (command, workspace, label) where label is the informal
# WS0 interrupt class this is meant to stand in for.
# ---------------------------------------------------------------------------
_WORKSPACE = "/home/manfred/.hermes/workspace/kanban/task-demo"

SYNTHETIC_CORPUS: list[dict] = [
    # class-a: read-only inspection (expected tier0)
    {"command": "git status", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "git diff HEAD~1", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "git log -5", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "pytest tests/ -q", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "cat README.md", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "grep -rn TODO src/", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "ls -la", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "git ls-remote origin", "workspace": _WORKSPACE, "label": "read-only"},
    {"command": "head -50 CHANGELOG.md", "workspace": _WORKSPACE, "label": "read-only"},
    # class-b: workspace-contained write (expected tier1) -- WS0: ~22.5%
    {"command": f"rm {_WORKSPACE}/scratch/old_draft.md", "workspace": _WORKSPACE, "label": "workspace-contained"},
    {"command": f"mkdir {_WORKSPACE}/build", "workspace": _WORKSPACE, "label": "workspace-contained"},
    {"command": f"mv {_WORKSPACE}/tmp.txt {_WORKSPACE}/notes.txt", "workspace": _WORKSPACE, "label": "workspace-contained"},
    {"command": f"touch {_WORKSPACE}/.keep", "workspace": _WORKSPACE, "label": "workspace-contained"},
    # class-c: self-modify scope (expected tier3, always) -- WS0: ~10%
    {"command": "rm ~/.hermes/config.yaml.bak", "workspace": _WORKSPACE, "label": "self-modify"},
    {"command": "systemctl --user restart hermes-gateway.service", "workspace": _WORKSPACE, "label": "self-modify"},
    {"command": "rm -rf hermes-agent/build", "workspace": _WORKSPACE, "label": "self-modify"},
    # class-d: Gitea deletion (expected tier3, always)
    {"command": "git push --delete origin topic-branch", "workspace": _WORKSPACE, "label": "gitea-delete"},
    # class-e: foreign host destructive (expected tier3, always)
    {"command": "ssh 192.168.1.176 rm -rf /var/tmp/scratch", "workspace": _WORKSPACE, "label": "foreign-host"},
    # class-f: ambiguous / outside workspace (expected tier3, fail-closed)
    {"command": "rm /etc/motd", "workspace": _WORKSPACE, "label": "outside-workspace"},
    {"command": "rm $(find . -name '*.tmp')", "workspace": _WORKSPACE, "label": "ambiguous"},
]


def load_corpus(path: Path | None) -> list[dict]:
    if path is None:
        return SYNTHETIC_CORPUS
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list of interrupt objects")
    return data


def classify(entry: dict) -> str:
    tirith_result = None
    findings = entry.get("tirith_findings")
    if findings:
        tirith_result = {"action": "warn", "findings": findings, "summary": ""}
    mutation_kind = entry.get("mutation_kind")
    if mutation_kind is None:
        try:
            from hermes_cli.kanban_db import _pending_action_mutation_kind
            mutation_kind = _pending_action_mutation_kind(entry["command"])
        except Exception:
            mutation_kind = "terminal-command"
    return approval._classify_kanban_action(
        entry["command"],
        mutation_kind=mutation_kind,
        workspace=entry.get("workspace", ""),
        tirith_result=tirith_result,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=None,
                         help="JSON file of real interrupts; defaults to the built-in synthetic stand-in corpus")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    source = str(args.corpus) if args.corpus else "SYNTHETIC (stand-in; replace with --corpus for real numbers)"

    counts = {"tier0": 0, "tier1": 0, "tier3": 0}
    by_label: dict[str, dict[str, int]] = {}
    rows = []
    for entry in corpus:
        tier = classify(entry)
        counts[tier] += 1
        label = entry.get("label", "(unlabeled)")
        by_label.setdefault(label, {"tier0": 0, "tier1": 0, "tier3": 0})[tier] += 1
        rows.append((entry["command"], label, tier))

    total = len(corpus)
    print(f"Corpus source: {source}")
    print(f"Total interrupts: {total}\n")

    print("Before (tiers disabled -- today): every interrupt pings an operator.")
    print(f"  operator pings: {total}/{total} (100%)\n")

    print("After (tiers enabled):")
    for tier in ("tier0", "tier1", "tier3"):
        n = counts[tier]
        pct = (100.0 * n / total) if total else 0.0
        pings = " (still pings)" if tier == "tier3" else " (auto-approved, no ping)"
        print(f"  {tier}: {n}/{total} ({pct:.1f}%){pings}")
    auto = counts["tier0"] + counts["tier1"]
    reduction = (100.0 * auto / total) if total else 0.0
    print(f"\n  Operator pings eliminated: {auto}/{total} ({reduction:.1f}%)")
    print(f"  Operator pings remaining:  {counts['tier3']}/{total} ({100.0 - reduction:.1f}%)\n")

    print("By WS0 interrupt class label:")
    for label, tiers in sorted(by_label.items()):
        n = sum(tiers.values())
        print(f"  {label} (n={n}): tier0={tiers['tier0']} tier1={tiers['tier1']} tier3={tiers['tier3']}")

    print("\nPer-command detail:")
    for command, label, tier in rows:
        print(f"  [{tier:>5}] ({label:<20}) {command}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

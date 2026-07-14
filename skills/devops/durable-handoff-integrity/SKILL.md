---
name: durable-handoff-integrity
description: "Kanban task creation/handoff: fail-closed guard for foreign or another-task scratch-workspace refs and FileNotFoundError; validate explicit project-id across CLI/API; use cleanup-safe durable-path hints."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, handoff, artifacts, workspace, durability, fail-closed]
    related_skills: [hermes-agent-skill-authoring, systematic-debugging, test-driven-development]
    note: auto-promoted weekly
---

# Durable Handoff Integrity

## Overview

Use this skill when one task, worker, workflow phase, or API call hands work to another task. It prevents a common failure class: a handoff looks complete because the board row or prose survived, while the referenced scratch workspace, project link, or artifact disappeared before the successor could use it.

The rule is simple: disposable execution paths may produce evidence, but they must not be the only address by which another task can obtain that evidence. Validate handoff references at the creation boundary, resolve explicit project identity before insertion, and make durable artifacts the source of truth for cross-task transfer.

This is a forcing-function skill, not a replacement for general Kanban lifecycle or filesystem-transaction skills. Its distinctive concern is **portability of inputs across task lifetime and worker surface**.

## When to Use

Use when:

- creating a Kanban card whose title/body mentions another task, phase, workspace, project, branch, or artifact;
- passing parent findings to a child worker after the parent workspace may be cleaned up;
- resolving an explicit project id/slug into a worktree or mounted worker workspace;
- a worker reports `FileNotFoundError`, empty `/workspace`, or a missing parent report after a successful upstream completion;
- a workflow spans boards, profiles, containers, restarts, or multiple sessions;
- a completion report must point downstream consumers to files that outlive scratch cleanup;
- a handoff validator or linter needs to reject unsafe references before dispatch.

Do not use for:

- purely conversational context that has no downstream execution dependency;
- a worker's own current workspace, when self-reference is intentional and the task owns that workspace;
- unrelated Git rebase or PR publication work, unless the PR evidence is being handed to another task.

## Core Contract

Every handoff must answer four questions before the successor starts:

1. **Identity** — Which exact task, run, project, board, commit, or artifact is being handed over?
2. **Location** — Is the input in a durable artifact store, a committed repository path, an explicitly mounted self-contained source, or only a disposable workspace?
3. **Authority** — Which path or record is canonical if prose, a scratch copy, and a promoted artifact disagree?
4. **Verification** — What readback proves the successor can access and parse the promised input now?

Use this minimum handoff shape:

```text
source task/run: <stable id>
source commit or artifact hash: <exact value, when applicable>
canonical input: <durable artifact path, committed path, or broker id>
worker-local materialization: <path inside exact workspace, if copied>
expected format: <JSON/Markdown/test log/etc.>
readback: <existence + parse/hash/test evidence>
expiry/cleanup: <when scratch copies may disappear>
```

A path in a card body is not evidence of accessibility. A worker summary is not a transport layer. A `done` parent is not proof that its original workspace still exists.

## Creation-Boundary Guards

### 1. Reject unresolved explicit projects

If a caller supplies a non-empty `project_id` or project slug:

- resolve it before inserting the task;
- fail closed with a clear error if it does not resolve;
- propagate project-database/configuration failures distinctly instead of converting them into “unknown project” or silently falling back;
- create no task row on resolution failure;
- preserve ordinary scratch behavior only when project input is absent or intentionally blank;
- read back `project_id`, workspace kind, workspace path, and branch after creation.

Never silently degrade an explicit project-linked request to a scratch card. An empty worker mount is then a predictable consequence, not a mysterious Docker personality.

### 2. Reject foreign disposable-workspace references

At the shared task-creation boundary, scan title and body for references to another task's disposable workspace, using the board's configured workspace root rather than a hard-coded path. Reject references shaped like:

```text
<workspaces_root>/<other_task_id>/...
```

The guard must:

- allow a reference to the new task's own workspace when self-reference is intentional;
- reject foreign task workspace references by default;
- explain that scratch paths are cleaned up;
- list known `task_artifacts.durable_path` alternatives for the referenced task when available;
- offer an explicit, auditable opt-out only for intentional live sibling coordination;
- accept native path separators on every supported platform, including mixed separators where the runtime can produce them;
- run before the task insert, not in a later dispatcher that may never claim the card.

A validator that checks only the CLI is incomplete. The same contract must cover the database/API boundary and every public create surface: CLI, model tool, dashboard/API, workflow helper, and automation wrapper.

### 3. Materialize, do not merely mention

Dependencies and handoff prose do not confer filesystem access. Before dispatching a successor, choose one of these explicit transport methods:

- promote the source file to the durable artifact store and pass its durable path plus hash;
- copy a sanitized, self-contained snapshot into the successor's exact workspace;
- commit the source to a repository and pin the exact commit/path;
- use a trusted broker that exposes a stable id and readback API.

If the successor needs several files, provide a manifest rather than a paragraph of paths. Include source, destination, hash/size, producer run, format, and retention class. Never require a worker to reach into another task's home, worktree, profile, or host-only path.

## Failure Semantics

When a promised handoff is missing or inaccessible:

1. classify it as an input/dependency/evidence failure, not as a code failure;
2. preserve the exact missing reference and source task id in a redacted diagnostic;
3. search the durable artifact manifest before asking for restoration;
4. if a durable replacement exists, update the successor handoff to that canonical path and rerun the smallest readback;
5. if no replacement exists, block or park the successor with a named dependency;
6. do not blindly retry the same task body against the same vanished path;
7. do not mark the run green merely because the temporary harness itself executed and cleaned up.

A failed verification of missing inputs is useful evidence, but it is not a successful semantic verification. Keep the distinction visible in task metadata: `harness_executed`, `inputs_present`, `semantic_checks_run`, and `verdict` should not collapse into one boolean.

## Verification Matrix

For a new guard or handoff path, test the boundary rather than only a helper:

| Case | Required result |
|---|---|
| explicit valid project | linked workspace is created and read back |
| explicit unknown project | clear error; no task row |
| project lookup/config failure | original typed failure is preserved; no task row |
| no project supplied | normal scratch behavior remains |
| foreign scratch path in title | reject before insert |
| foreign scratch path in body | reject before insert |
| own workspace path | allowed when self-reference is intended |
| explicit sibling opt-out | allowed only with auditable flag and scope |
| POSIX and native Windows separators | same fail-closed result |
| durable artifact alternative exists | rejection points to canonical durable path |
| durable artifact is absent | successor blocks/parks; no false green |
| every create surface | CLI, tool, dashboard/API, and direct DB path agree |
| scratch cleanup after parent completion | child still reads canonical input |

For each rejection, assert both the absence of the unintended task row and the actionable error. For each acceptance, assert the resulting project/workspace/artifact metadata, not just a successful function return.

## Handoff Review Checklist

Before a parent reports a handoff complete:

- [ ] Parent artifacts are promoted or committed before scratch cleanup.
- [ ] Every downstream input has a stable id and canonical location.
- [ ] Durable paths are accompanied by hash, size, producer run, and format where relevant.
- [ ] Child workspace materialization is self-contained and sanitized.
- [ ] No host-only paths, credentials, profile directories, or foreign task workspaces are required.
- [ ] Explicit project identity was resolved before task insertion.
- [ ] Cross-surface create behavior was checked at the shared boundary.
- [ ] A fresh successor-side existence/parse/hash readback passed.
- [ ] Missing-input outcomes are classified as blocked/dependency/evidence, not green.
- [ ] The handoff names what may be cleaned up and when.

## Common Pitfalls

1. **Scratch-path handoff.** A parent writes `/workspaces/<id>/report.json` into a card body and the child starts after cleanup. Use the promoted durable path instead.
2. **Silent project fallback.** An unknown project becomes a scratch task with an empty mounted workspace. Resolve before insert and fail closed.
3. **CLI-only validation.** Tool and dashboard callers bypass the guard. Put the invariant at the shared DB/API boundary and test all adapters.
4. **Unix-only matching.** A forward-slash regex fails open on native Windows paths. Normalize both separator families and test a Windows-shaped path even on Linux.
5. **Dependency prose as access.** Parent links and comments do not mount files. Materialize a sanitized snapshot or use a trusted artifact broker.
6. **False-green harness.** The temporary verifier ran and exited, so the task is called green even though inputs were absent. Report harness status, input presence, semantic checks, and verdict separately.
7. **Retrying a vanished source.** Repeated retries burn budget and create block loops. Search durable artifacts once, then park with a concrete dependency.
8. **Unscoped opt-out.** A generic “allow references” switch hides accidental stale paths. Require an explicit flag, reason/scope, and live-sibling intent.
9. **Unpinned upstream handoff.** A branch name or “latest” is not stable input. Pin commit SHA, remote ref readback, and preserved behaviors before rebase or review.

## One-Shot Recipe: Portable Parent-to-Child Handoff

1. Finish the parent artifact in its scratch workspace.
2. Validate format and content; compute hash and size.
3. Promote it to the durable artifact store and read back the manifest.
4. Create the child with the stable artifact id/path, not the parent workspace path.
5. Resolve any explicit project link before insertion.
6. Run the child-side existence + parse/hash smoke in the exact worker workspace.
7. If the smoke fails, classify the missing input and block/park; do not retry unchanged prose.
8. Only then allow implementation or synthesis to begin.

## Verification Checklist

- [ ] Frontmatter and body are valid for the Hermes skill loader.
- [ ] Creation guards fail closed for unknown projects and foreign disposable-workspace references.
- [ ] Durable alternatives are surfaced in rejection/readback messages.
- [ ] Native path separators are covered.
- [ ] All public creation surfaces share the same invariant.
- [ ] Child-side readback proves the promised input is accessible and semantically usable.
- [ ] No false-green result collapses harness, input, semantic, and verdict state.
- [ ] Skill remains generic: no client names, private paths, task ids, commits, or secrets.

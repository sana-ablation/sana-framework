---
name: profile-verifier
description: Use when Codex needs to verify ideal runtime profiles for fidelity, metadata integrity, and answer-leakage safety in the current benchmark artifact layout.
---

# Profile Verifier

## Current Artifact Layout

Verify runtime profiles:

```text
benchmarks/<benchmark>/tasks-mini/runtime-profiles/<bucket>/task_N.json
```

against mirrored tasks:

```text
benchmarks/<benchmark>/tasks-mini/tasks/<bucket>/task_N.json
```

Do not create or verify new artifacts outside the current runtime-profile layout
unless the user explicitly asks for a legacy migration check.

## Two Gates

A runtime profile is acceptable only after both gates pass:

1. deterministic local review of fields and mirroring
2. independent semantic review of `reasoning_chain_text` leakage and fidelity

Verify at most five runtime profiles in one turn.

## Gate 1: Local Checks

For each runtime profile, check:

```text
mirrored source task exists
dataset_sequence is present and ordered by first task-node dataset use
source_sequence is present and ordered by task-node source use
repeated source uses are preserved
reasoning_chain_text is a JSON list or legacy string that normalizes cleanly
original_final_question matches task.question
original_reasoning_chain is present for audit
ideal_code and ideal_query fields are present, even if empty
```

Treat `dataset_sequence`, `source_sequence`, and `original_reasoning_chain` as
metadata/audit fields. They may contain exact source ids, page titles, or file
paths. Judge prompt leakage only in `reasoning_chain_text` for normal
ideal-profile behavior.

## Gate 2: Semantic Review

Review `reasoning_chain_text` for:

- final-answer leakage
- intermediate-answer leakage
- hidden source title, page title, dataset title, or column-name hints
- solved intersections, rankings, comparisons, or numeric results
- wording that lets the model skip retrieval or computation
- missing retrieval, narrowing, aggregation, comparison, or lookup steps

Do not give reviewers final answers, node answers, node facts, or answer-bearing
source rows. Send only:

```text
task question
sanitized original reasoning trace
dataset_sequence
source_sequence
reasoning_chain_text
non-answer source/title/column terms needed for leakage checks
this rubric
```

## Fidelity Rules

The runtime profile may be cleaner than the source reasoning chain, but it must:

1. preserve the same overall retrieval/computation path
2. keep all necessary intermediate narrowing and comparison steps
3. stay compatible with source order
4. avoid revealing downstream entities before their discovery step
5. avoid over-compressing multi-step evidence into vague instructions

## Verdict

Return:

```text
clean
needs_revision
blocked
```

Use `blocked` only when the runtime profile cannot be validated because source
mirroring or metadata is incomplete. Use `needs_revision` for leakage, fidelity,
or field-shape issues that can be fixed.

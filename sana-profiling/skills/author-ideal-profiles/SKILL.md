---
name: author-ideal-profiles
description: Use when Codex needs to create, rewrite, or review ideal runtime profiles for LakeQA-style benchmark tasks in the current benchmark artifact layout.
---

# Author Ideal Profiles

## Current Artifact Layout

In this repo, ideal artifacts are runtime profiles.

Use this mirror:

```text
benchmarks/<benchmark>/tasks-mini/tasks/<bucket>/task_N.json
  -> benchmarks/<benchmark>/tasks-mini/runtime-profiles/<bucket>/task_N.json
```

Prefer the explicit name `runtime_profile_root` in new docs, scripts, and
reports. Do not create new legacy profile-root artifacts outside
`benchmarks/<benchmark>/tasks-mini/runtime-profiles`.

## Batch Limit

Author at most five runtime profiles in one turn. If the user points to a
directory with more than five tasks, process the first five in stable path order
and report the remainder as not yet handled.

## Runtime Profile Contract

Each runtime profile should contain:

```text
dataset_sequence
source_sequence
reasoning_chain_text
original_final_question
original_reasoning_chain
ideal_code
ideal_query
```

`dataset_sequence` is derived from the unique datasets referenced by task nodes
in first-appearance node order.

`source_sequence` is derived from task node `source` values in node/use order.
Do not deduplicate repeated source uses; `search_ideal` depends on source order.

`reasoning_chain_text` is prompt scaffolding for the ideal profile runtime. Store it as a
JSON list of short numbered strings.

## Reasoning Text Rules

Keep `reasoning_chain_text` useful but non-leaky:

- no final answers
- no intermediate answers that should be discovered
- no raw source paths, dataset ids, or bucket names
- no node ids, hop ids, or audit-log narration
- no dataset-position wording such as "the first source"
- no solved intersections, comparisons, rankings, or numeric results
- no page/source titles unless the question already names them or ambiguity
  would otherwise make the step unusable

Prefer carry-forward language:

```text
For the entity that remains after the comparison, determine ...
Using the qualifying rows from the prior step, compute ...
```

Avoid wording that lets the agent skip retrieval:

```text
Using the county in Harrison, ...
Using the 2018 Texas expenditures table, ...
```

## Source Resolution

Resolve every `source_sequence` entry to a concrete final source path that the
runtime tools can retrieve, usually under one of:

```text
datagov/<dataset-id>/files/<file>
wikipedia/<dataset-id>/files/<file>
```

If source mirroring is not complete, mark the runtime profile as blocked with a
short `source_resolution_notes` field instead of inventing a source path.

## Validation

Before accepting a runtime profile, check:

```text
runtime profile path mirrors the task path under benchmarks/<benchmark>/tasks-mini/
dataset_sequence covers task datasets in first node-use order
source_sequence aligns to task nodes and preserves repeated source uses
reasoning_chain_text is a JSON list
reasoning_chain_text contains no answer leakage or skip-step hints
original_final_question matches the task question
original_reasoning_chain is copied from the task reasoning trace for audit only
ideal_code and ideal_query are present, even if empty
```

For an independent review, give the reviewer only the task question, sanitized
original reasoning trace, dataset/source sequences, reasoning text, and this
rubric. Do not give final answers, node answers, or executable facts.

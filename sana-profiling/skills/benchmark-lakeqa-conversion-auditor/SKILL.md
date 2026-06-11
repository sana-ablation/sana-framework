---
name: benchmark-lakeqa-conversion-auditor
description: Use when Codex needs to infer how an arbitrary benchmark can be converted into LakeQA-style task artifacts without a benchmark-specific importer, especially when reading benchmark examples, raw data, solution code, LakeQA exemplars, Kramabench transform patterns, source mirroring rules, k/d/s topology, fairness risks, or ideal artifact feasibility.
---

# Benchmark LakeQA Conversion Auditor

## Scope

Infer and document a conversion recipe. Do not write converted task JSON, do not
scaffold a transform skill, and do not author ideal profiles or computation records.

Write the report to:

```text
docs/benchmark-conversions/<benchmark>-lakeqa-conversion.md
```

## Inputs

- Benchmark root directory.
- Optional hints for task examples, raw data, or solution code.
- LakeQA task exemplars, usually `tasks_mini/...`.
- Kramabench transform skill as the strongest local example of a mature
  benchmark-to-LakeQA conversion workflow.

## Sampling Rule

Auto-sample examples. Prefer structural diversity first: evidence type, source
count, hop shape, answer type, executable computation likelihood, and available
raw code/data. Use split, domain, difficulty, or other metadata diversity only
as a tie-breaker.

Samples in the report must be structural and methodological. Do not paste
terminal answers, intermediate answers, solution code, executable facts, or
answer-bearing source rows into the report. When an inspected artifact contains
answer fields, describe the field shape/type and redact the value.

Run the helper when useful:

```bash
python sana-profiling/skills/benchmark-lakeqa-conversion-auditor/scripts/sample_benchmark_artifacts.py <benchmark-root>
```

## Report Contract

Start from `references/report-template.md`. Fill every section with concrete
evidence from inspected artifacts. If a section is not applicable, state why.

The report must cover:

- benchmark artifact inventory
- sampled examples and rationale
- evidence/source model
- LakeQA task mapping
- k/d/s bucket proposal
- source mirroring convention
- executable fact feasibility
- ideal artifact feasibility for `ideal_query` and `ideal_code`
- fairness and leakage risks
- validation strategy
- recommended transform skill structure

The report must not become an answer key. It may explain how answers should be
derived and verified, but converted LakeQA task answers must be re-derived from
benchmark source artifacts during the transform step.

## Handoff Contract

After converted LakeQA tasks exist, use:

- `author-ideal-profiles` to create or rewrite mirrored ideal runtime profiles
- `profile-verifier` to check fidelity and leakage
- `author-ideal-code` to fill `ideal_query` and `ideal_code`

Do not duplicate those workflows in this skill.

---
name: benchmark-lakeqa-skill-scaffolder
description: Use when Codex needs to turn an approved benchmark-to-LakeQA conversion report into a benchmark-specific transform skill folder after conversion design has already been reviewed.
---

# Benchmark LakeQA Skill Scaffolder

## Scope

Read an approved conversion report and scaffold the benchmark-specific LakeQA
transform skill described by that report. This skill must not invent or
re-infer the conversion method; preserve the approved report as the source of
truth and make the generated skill defer to it.

By default, create the generated transform skill under:

```text
sana-profiling/skills/<benchmark>-lakeqa-transform/
```

## Inputs

- Approved benchmark-to-LakeQA conversion report, usually
  `docs/benchmark-conversions/<benchmark>-lakeqa-conversion.md`.
- Benchmark name to normalize into a lowercase dash slug.
- Optional output root, defaulting to `sana-profiling/skills`.

The approved conversion report must include these headings:

- `## Benchmark artifact inventory`
- `## Recommended benchmark-specific transform skill structure`

## Workflow

Run the helper:

```bash
python sana-profiling/skills/benchmark-lakeqa-skill-scaffolder/scripts/scaffold_benchmark_skill.py \
  docs/benchmark-conversions/<benchmark>-lakeqa-conversion.md \
  --benchmark <benchmark>
```

Review the generated files:

- `SKILL.md`: benchmark-specific transform workflow shell.
- `references/conversion-report.md`: exact copy of the approved conversion
  report.

After scaffolding, fill in benchmark-specific transform scripts and references
only from the approved report. Do not re-open the question of how the benchmark
maps to LakeQA unless the report is explicitly revised and re-approved.

## Handoff Boundary

The generated transform skill should convert benchmark artifacts into LakeQA
tasks first. It must defer `author-ideal-profiles`, `profile-verifier`, and
`author-ideal-code` handoff until LakeQA tasks exist.

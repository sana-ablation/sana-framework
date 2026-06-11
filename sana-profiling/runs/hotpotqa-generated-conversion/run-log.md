# HotpotQA Generated Conversion Run

## Commands

```bash
python sana-profiling/skills/benchmark-lakeqa-conversion-auditor/scripts/sample_benchmark_artifacts.py other-benchmarks/tasks-hotpotqa-mini --limit 10
python sana-profiling/skills/benchmark-lakeqa-skill-scaffolder/scripts/scaffold_benchmark_skill.py sana-profiling/runs/hotpotqa-generated-conversion/hotpotqa-lakeqa-conversion-report.md --benchmark hotpotqa --output-root sana-profiling/runs/hotpotqa-generated-conversion/generated-skills --force
```

## Skill Interaction

1. Loaded `benchmark-lakeqa-conversion-auditor` and sampled HotpotQA mini artifacts.
2. Wrote a redacted conversion report that describes method, source model, k/d rules, and leakage risks without answer strings.
3. Loaded `benchmark-lakeqa-skill-scaffolder` and generated `hotpotqa-lakeqa-transform` from the report.
4. Applied the generated transform skill to five examples: three comparison examples and two bridge examples.
5. Wrote converted task artifacts and matching runtime profiles under the current repo convention.
6. Applied `author-ideal-profiles` semantics by writing non-answer `reasoning_chain_text` runtime profiles for each converted task.
7. Applied `author-ideal-code` semantics by checking each task is text evidence with `wikipedia/` sources and therefore keeps `ideal_code` and `ideal_query` empty.
8. Checked report/generated-skill/runtime-profile prompt surfaces for answer leakage.

## Outputs

- `sampled-artifacts.json`
- `hotpotqa-lakeqa-conversion-report.md`
- `generated-skills/hotpotqa-lakeqa-transform/SKILL.md`
- `converted/benchmarks/hotpotqa/tasks-mini/tasks/`
- `converted/benchmarks/hotpotqa/tasks-mini/runtime-profiles/`
- `source-mirroring-manifest.json`
- `validation.json`
- `error_log.json`

## Leakage Boundary

Converted task JSON files include answer fields because LakeQA task artifacts require them. The conversion report, generated skill, run log, and runtime-profile reasoning text are checked not to include final or intermediate answer strings.

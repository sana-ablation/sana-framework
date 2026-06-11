# SANA Profiling

This folder contains public-facing framework artifacts for turning external QA
benchmarks into LakeQA-style tasks and ideal runtime profiles.

```text
raw benchmark examples
  -> conversion report
  -> benchmark transform skill
  -> LakeQA task JSON
  -> runtime profiles
  -> profile/code verification
```

## Benchmark conversion skills

- `skills/benchmark-lakeqa-conversion-auditor/`: infer and document a
  benchmark-to-LakeQA conversion recipe.
- `skills/benchmark-lakeqa-skill-scaffolder/`: turn an approved conversion
  report into a benchmark-specific transform skill.
- `skills/author-ideal-profiles/`: author clean ideal runtime profiles
  under `benchmarks/<benchmark>/tasks-mini/runtime-profiles`.
- `skills/author-ideal-code/`: author `ideal_code` and `ideal_query` records
  inside runtime profiles.
- `skills/profile-verifier/`: check runtime-profile fidelity and leakage safety.

The intended handoff is report-first. The auditor reads representative benchmark
examples and proposes the conversion method. The scaffolder turns an approved
report into a benchmark-specific transform skill. The generated transform skill
then writes LakeQA-style tasks and matching runtime profiles.

## HotpotQA example commands

The generated HotpotQA run in this repo used the following commands from the
repository root:

```bash
python sana-profiling/skills/benchmark-lakeqa-conversion-auditor/scripts/sample_benchmark_artifacts.py \
  other-benchmarks/tasks-hotpotqa-mini \
  --limit 10 \
  > sana-profiling/runs/hotpotqa-generated-conversion/sampled-artifacts.json
```

```bash
python sana-profiling/skills/benchmark-lakeqa-skill-scaffolder/scripts/scaffold_benchmark_skill.py \
  sana-profiling/runs/hotpotqa-generated-conversion/hotpotqa-lakeqa-conversion-report.md \
  --benchmark hotpotqa \
  --output-root sana-profiling/runs/hotpotqa-generated-conversion/generated-skills \
  --force
```

The generated `hotpotqa-lakeqa-transform` skill was then applied to five
sampled imports. The output is intentionally kept under the run folder so it
does not mutate maintained benchmark artifacts:

- `runs/hotpotqa-generated-conversion/generated-skills/hotpotqa-lakeqa-transform/SKILL.md`
- `runs/hotpotqa-generated-conversion/converted/benchmarks/hotpotqa/tasks-mini/tasks/`
- `runs/hotpotqa-generated-conversion/converted/benchmarks/hotpotqa/tasks-mini/runtime-profiles/`
- `runs/hotpotqa-generated-conversion/source-mirroring-manifest.json`
- `runs/hotpotqa-generated-conversion/validation.json`
- `runs/hotpotqa-generated-conversion/error_log.json`

## Example runs

- `runs/hotpotqa-generated-conversion/`: dry-run output from applying a
  generated HotpotQA transform skill to five `other-benchmarks` examples.

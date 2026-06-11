# HotpotQA LakeQA Conversion Report

## Benchmark artifact inventory

Inputs inspected:

- `other-benchmarks/raw/hotpotqa/hotpot_dev_distractor_v1.json`
- `other-benchmarks/tasks-hotpotqa-mini/k-2-d-*/task_*.json`

This report records structure and conversion method only. Terminal answers, intermediate answers, and answer-bearing source rows are intentionally redacted.

## Auto-sampled examples and rationale

The sampler selected both comparison and bridge HotpotQA examples by bucket and provenance type. This run converts five examples: three comparison examples and two bridge examples, enough to exercise both parallel comparison and sequential evidence patterns.

## Evidence/source model

HotpotQA is a text-evidence benchmark. The raw example carries Wikipedia paragraph context and supporting fact title/index pairs. Converted task nodes should point to stable mirrored text refs rather than `hotpotqa://wiki/...` pseudo-URIs.

## LakeQA task mapping

Map each promotable example into `question`, `answer`, `nodes`, `reasoning_chain`, `datasets_used`, `reasoning_hops`, and `_provenance`. Fill source-agnostic subquestions. Facts may remain evidence snippets; do not force executable Python for HotpotQA.

## k/d/s bucket proposal

Use LakeQA topology: `k = len(reasoning_hops)` and `d = max(len(hop.node_ids))`. Comparison examples in this smoke run are represented as `k-1-d-2`; bridge examples are represented as `k-2-d-1`.

## Source mirroring convention

Use `wikipedia/hotpotqa__<source_id>/files/<title_slug>.txt`. Record local authoring source as the raw HotpotQA JSON plus title and sentence index. Do not store answer-bearing paragraph text in provenance.

## Executable fact feasibility

Executable facts are not required for this text-evidence benchmark. Validate that the evidence supports each node answer and the final comparison instead.

## Ideal artifact feasibility

Cover both `ideal_query` and `ideal_code`. For this smoke conversion, both arrays remain empty because the tasks are text evidence, not data computation. Runtime profiles are still authored with safe, non-answer reasoning text.

## Fairness and leakage risks

This report is not an answer key. Converted tasks contain answer fields by schema, but the report, generated skill, run log, and runtime-profile reasoning text must not reveal final answers or intermediate answer strings.

## Validation strategy

Check source rewriting, answer suffix, subquestion fill, k/d bucket placement, no `hotpotqa://` final sources, provenance without answer-bearing text, and runtime-profile reasoning text with no final or intermediate answer strings.

## Recommended benchmark-specific transform skill structure

Use the generated transform skill scaffold with batch flow, worker prompt, final task shape, k/d rules, provenance, fairness/leakage requirements, validation checklist, and ideal-artifact handoff boundary.

## Handoff contract to adjacent ideal-artifact skills

- `author-ideal-profiles`: author runtime profiles only after converted LakeQA tasks exist.
- `profile-verifier`: verify runtime-profile reasoning text for answer leakage and fidelity.
- `author-ideal-code`: add `ideal_code`/`ideal_query` only for computation-feasible tasks; leave text-evidence HotpotQA tasks empty unless an extraction computation is intentionally designed.

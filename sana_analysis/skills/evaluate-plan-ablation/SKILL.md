---
name: evaluate-plan-ablation
description: Use when comparing LakeQA or Kramabench run logs from the standard-plan arm against the ideal-plan arm to judge whether explicit plans are semantically similar across gpt-5.4-nano or gpt-5-mini model variants.
---

# Evaluate Plan Ablation

Use this skill to answer whether `plan_d` and `plan_i` are semantically similar for the same task and model variant.

## Scope

Inputs are raw log roots such as:

- `logs`
- `log-kramabench`

Compare explicit planning calls only:

- `plan_d`: the log mode with `search=ideal, plan=standard, compute=ideal` -- the standard-plan arm.
- `plan_i`: the log mode with `search=ideal, plan=ideal, compute=ideal` -- the ideal-plan arm.
- Default plan tool: `plan({"plan_text": ...})`
- Ideal plan tool: `plan_ideal({"plan_text": ...})`

Do not compare final answers, tool success, or later execution behavior unless the user explicitly asks for that secondary analysis.

## Discovery

Start with the helper script:

```bash
.venv/bin/python sana_analysis/skills/evaluate-plan-ablation/scripts/list_plan_pairs.py logs --limit 20
.venv/bin/python sana_analysis/skills/evaluate-plan-ablation/scripts/list_plan_pairs.py log-kramabench --limit 20
```

The helper resolves the two arms by axis, not by directory name, so every naming
generation on disk reads:

- `plan_d_mode`: the arm with `search=ideal, plan=standard, compute=ideal`
- `plan_i_mode`: the arm with `search=ideal, plan=ideal, compute=ideal`

These are `PLAN_D_AXES` and `PLAN_I_AXES` in
`sana_analysis/paper/plan_ablation_figure.py` -- the same two the figure filters
to, so discovery and the figure cannot disagree about which arms the ablation is.

If two variant directories share those three axes the helper refuses to guess
and names both; pass `--plan-d-mode` or `--plan-i-mode` to break the tie. Pass
`--all-modes` only when the user explicitly wants every standard-plan mode paired
against every ideal-plan mode.

Use filters when the user gives a model, mode, or task:

```bash
.venv/bin/python sana_analysis/skills/evaluate-plan-ablation/scripts/list_plan_pairs.py log-kramabench \
  --model openai_gpt-5.4-nano \
  --plan-d-mode search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off \
  --plan-i-mode search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off \
  --task tasks-mini-kramabench/k-3-d-1-s-1/task_1.log \
  --json
```

If `.venv/bin/python` is unavailable, use the local Python that has the repo dependencies.

## Subagent Policy

For row-level plan comparisons, use cheap subagents by default:

- Preferred subagent model: `gpt-5.4-mini`
- One subagent may review a small batch of closely related pairs, but keep batches narrow enough that model provenance and task ids cannot blur together.
- Use a stronger model only when the plan text is malformed, very long, cross-log evidence is contradictory, or the cheap subagent returns an unclear judgment.

Prompt cheap subagents with only the paired plan record(s), direct paths to the paired logs, the label rubric, and the required JSON shape. The subagent may open the linked `plan_d_log` and `plan_i_log` to verify line numbers or provenance, but should not inspect unrelated logs.

Do not spawn subagents for rows where the helper has already prefilled `prefill_plan_similarity=not_comparable` because of a missing plan or model mismatch. Copy those fields directly into the output CSV.

Suggested subagent prompt:

```text
Use $evaluate-plan-ablation to judge these plan_d vs plan_i pairs.
Use model gpt-5.4-mini.

For every row, you are given direct log paths:
- plan_d_log: /absolute/path/to/default-plan.log
- plan_i_log: /absolute/path/to/ideal-plan.log

Use those log paths as evidence links and cite them in provenance_note.

Return JSON only:
{
  "rows": [
    {
      "task_id": "...",
      "model_variant": "...",
      "runner_model": "...",
      "plan_d_log": "...",
      "plan_i_log": "...",
      "plan_similarity": "similar|missing_details|incomplete_plan|operation_mismatch|not_similar|not_comparable",
      "divergence_type": "short string",
      "similarity_reason": "short evidence-grounded explanation",
      "aligned_steps": ["..."],
      "divergent_steps": ["..."],
      "provenance_note": "path model and NEW TASK model cited"
    }
  ]
}
```

## Model Provenance

Every judgment must cite where the model identity came from:

- `model_variant`: the log path segment under `modes/`, such as `openai_gpt-5.4-nano` or `openai_gpt-5-mini`.
- `runner_model`: the `NEW TASK: ...` line inside each log, such as `NEW TASK: gpt-5.4-nano` or `NEW TASK: gpt-5-mini`.
- `plan_d_log` and `plan_i_log`: exact log paths.
- `plan_d_log` and `plan_i_log` should be absolute paths in prepared records and output CSVs.
- `plan_d_mode` and `plan_i_mode`: exact mode names.

Do not confuse `ideal_subagent_costs` model lines with the task runner model. Those lines identify semantic-judge or repair subagents used inside tools, not the model that authored the task-level `plan` or `plan_ideal`.

If `plan_d` and `plan_i` runner models differ, mark the pair `model_mismatch` and do not collapse them into one result.

## Missing Plan Policy

If either side lacks an explicit planning call:

- `missing_plan_d`: default log has no `plan(...)`.
- `missing_plan_i`: ideal log has no `plan_ideal(...)`.
- `missing_both`: neither side has an explicit plan call.

Do not reconstruct a missing plan from later tool calls. The answer to semantic similarity is `not_comparable` unless the user asks for an inferred-plan fallback.

The helper emits:

- `missing_plan_type`: one of `missing_plan_d`, `missing_plan_i`, `missing_both`, or blank.
- `prefill_plan_similarity`: `not_comparable` for mechanical blocker rows, otherwise blank.
- `prefill_divergence_type`: the missing-plan subtype or `model_mismatch`.
- `prefill_similarity_reason`: short evidence string with the absolute log path(s).

Copy these prefilled values directly into batch outputs before dispatching subagents.

## Judgment Labels

Use one of:

- `similar`: same answer-producing strategy. The default plan may be less specific than the ideal plan: it may say “find the relevant dataset,” “inspect schema,” or omit exact column names, file names, indicator labels, month/period filters, exclusions, rounding, or tie-breaks. Mark it similar when the abstract plan still points to the same source family, required hops, core computation, and final output.
- `missing_details`: same source family, required hops, broad computation, and final output, but one plan omits material details needed to make the algorithm answer-preserving. Use this when the missing detail is not a whole bridge step, but could change the result: exact period/month, indicator, excluded aggregate, geography, category, threshold, tie-break, denominator/numerator definition, unit conversion, or final formatting constraint. Do not use for harmless column-name/file-name abstraction.
- `incomplete_plan`: same broad final objective, but one plan omits, compresses away, or reverses a required intermediate step needed to derive a later entity/filter. Common LakeQA examples include “find sector, then count incidents in that sector” or “identify county, then use county seat.” This is an error label for structurally incomplete plans.
- `operation_mismatch`: same topic and broad objective, but one step is materially different: computation, filter, comparison direction, grouping key, threshold, ratio direction, min/max choice, conversion, or final output shape would likely produce a different answer.
- `not_similar`: plans would answer materially different questions because the source family, required intermediate hop, target population, computation, comparison direction, or final selection is different.
- `not_comparable`: missing plan text, malformed logs, or model mismatch.

Do not penalize normal default-plan abstraction. If the only difference is harmless specificity, use `similar`. Use `missing_details` only when the omitted detail is material enough that a reasonable executor could answer a different question.

## Comparison Procedure

Read both plans as intended algorithms. Ignore later execution unless the user explicitly asks for execution-aware analysis.

Check these dimensions in order:

1. **Question target**: Do both plans aim for the same final entity/value/list and output format?
2. **Source path**: Do they use the same concrete dataset/file, or a search path that would reasonably land on the same source family?
3. **Required hops**: For multi-hop tasks, do both plans include every bridge step needed to derive later filters or entities?
4. **Filters and cohorts**: Compare years, months, geography, categories, excluded values, indicators, populations, and threshold conditions.
5. **Computation**: Compare grouping keys, sums/counts/means/medians, ratios, growth formulas, interpolation, conversions, rounding, and tie-breaks.
6. **Final selection**: Compare max/min/top-k/list membership and whether the plan returns the requested final answer rather than an intermediate.

## Observed Patterns

Treat these as calibration examples from the existing logs:

- `similar`: one plan says “search for NY Lotto bonus ball data, compute BVAL by year, choose max with earliest-year tie-break,” while the other names the NY Lotto file and says the same BVAL computation. Source discovery vs named source is fine when the computation and tie-break match.
- `similar`: one Kramabench plan says “inspect `worldcities.csv`, compute mean population per country, return max country,” while the other says “query/download `worldcities.csv`, coerce population, group by country, return highest mean.” Extra cleaning detail does not change the plan.
- `similar`: one overdose plan says “aggregate annual overdose deaths across states and compute growth-amplified score,” while the other specifies December 12-month-ending rows, exact indicator, and excluding the national aggregate. Treat the default plan as similar if it preserves the source family and the annual growth-amplified computation; do not require default-plan exactness.
- `missing_details`: one overdose plan says only “use annual U.S. overdose deaths and compute growth-amplified score,” while the other requires December 12-month-ending rows, `Number of Drug Overdose Deaths`, excluding national aggregate, and years 2016-2023. If the default plan leaves those constraints open enough to change the answer, use `missing_details`.
- `incomplete_plan`: one multi-hop APD/NOPD plan names the three counts but omits the bridge “find sector with illegal-item searches, then count mental-health calls in that sector.” The missing bridge is a structural error.
- `operation_mismatch`: one plan computes `complaint_count / shooting_count` and the other computes `shooting_count / complaint_count`; ratio direction is material even though the topic and sources overlap.
- `not_comparable`: `search_d_results_i_pland...` logs often have no explicit `plan(...)`; do not infer a default plan from later tool calls.

## Row Output

For each pair, record:

- `task_id`
- `model_variant`
- `runner_model`
- `plan_d_mode`
- `plan_i_mode`
- `plan_d_log`
- `plan_i_log`
- `plan_d_line`
- `plan_i_line`
- `plan_d_excerpt`
- `plan_i_excerpt`
- `missing_plan_type`
- `plan_similarity`
- `divergence_type`
- `similarity_reason`
- `aligned_steps`
- `divergent_steps`
- `provenance_note`

Keep excerpts short but specific. Include enough plan text to support the judgment without copying the full log.

## Batch Output

When auditing more than a handful of pairs, write:

```text
agent_analysis/plan_default_analysis/<log-root-name>/modes/<model>/<plan_d_mode>__vs__<plan_i_mode>/plan_similarity.csv
agent_analysis/plan_default_analysis/<log-root-name>/modes/<model>/<plan_d_mode>__vs__<plan_i_mode>/plan_similarity_report.md
```

The report should summarize counts by `plan_similarity`, call out missing-plan rates, and separate results for `gpt-5.4-nano` and `gpt-5-mini`.

## Review Standard

Semantic similarity is about the plan, not whether the run later succeeded. Default plans are allowed to be less exact than ideal plans. A plan that names the right dataset family and core algorithm can be `similar` even when it omits harmless exact fields. Use `missing_details` when omitted constraints are material but the broad algorithm remains aligned. Mark `incomplete_plan` only when a required hop is missing, and `operation_mismatch` only when a step is materially different. A plan that differs only by saying `query_file` instead of `query_ideal`, while preserving the same source family, computation, and output, can be `similar`.

---
name: plan-agent
description: Planning guidance for multi-hop data lake questions. Load before calling `plan()`.
---

## Decompose the question

Cover these phases when they matter (combine adjacent ones into a single step when they don't):

1. **Entity resolution** — translate question wording into official names.
   - "food stamps" → SNAP
   - "unemployment" → ETA / UI beneficiaries
   - city/state references may need FIPS or official jurisdiction mapping

2. **Dataset discovery** — say what dataset shape you need and how you'll search for it.
   - Shape search queries as: `source/program + grain + metric + time`
   - Good: `Texas county prison admissions fiscal year`
   - Bad: `people entering prison system county`

Do not hallucinate dataset names.

## Plan format

4–8 short numbered steps. Each step names:
- the goal
- the main tool family
- a concrete action or check

```
1. Find Khan Academy HQ city/state | search for entity
   -> else: reformulate with official org name or dataset title
2. Resolve state to FIPS | grep_file on a lookup dataset
3. Count public schools by county in that state | query_file GROUP BY
4. If county-level data is missing, get the state total | query_file WHERE STFIP=...
   -> else: search for an alternative school dataset
```

Inline `-> else:` fallbacks are fine for risky discovery steps. Avoid vague steps like "find relevant data."

## When to replan

1. **Plan completed, answer not found** — summarize findings, identify the gap, plan for the gap.
2. **Before final extraction** — save a focused 1–3 step plan for the exact query that yields the answer.
3. **Dead end** — if the expected dataset doesn't exist, pivot to a viable proxy or alternate source.

## Reformulation pivots (when stuck)

1. **Lexical** — swap broad terms for agency/program names ("school count" → "NCES public school locations")
2. **Granularity** — search state-level if county-level returns nothing; filter down in code
3. **Proxy** — if the exact metric doesn't exist, find a standard proxy (e.g. "median income" for "poverty level")

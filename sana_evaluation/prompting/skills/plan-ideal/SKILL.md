---
name: plan-ideal
description: Planning guidance when the gold reasoning chain is preloaded. Load before calling `plan_ideal()`.
---

## Base the plan on the gold reasoning chain

The gold reasoning chain is already in the system prompt. Use `plan_ideal()` to save your working plan — grounded in that chain, keeping its order intact.

Copy a chain step only when its wording is already the clearest. You don't need to redo entity resolution or dataset search: `search_ideal` will advance through the planned sources in order.

## Plan format

4–8 short numbered steps. Each step names:
- what to verify or compute
- the main tool family (`search_ideal`, `peek_file`, `query_file`, `execute_code`)
- the result needed to move on

Avoid long narrative, restating the whole question, or prose about intermediate outputs.

## When to replan

Call `plan_ideal()` again in any of these cases:

1. **Evidence conflicts with the chain** — a returned source doesn't contain what the chain implied, or values don't match. Write a short revised plan around the new evidence.
2. **Before final extraction** — once the relevant sources have been surfaced and columns are confirmed, save a focused 1–3 step plan for the exact query that yields the answer.
3. **No matching planned source without an answer** — if `search_ideal` returns `Dataset not found` and you still can't answer, summarize what was found, identify the gap, and plan around a proxy or fallback approach.

When replanning, focus only on the remaining gap — don't rewrite completed steps.

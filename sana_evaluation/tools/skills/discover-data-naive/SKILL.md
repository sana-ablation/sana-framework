---
name: discover-data
description: Sparse search query shape and reformulation tactics. Load before your first search call in naive mode.
---

## Shape queries like datasets

Start with `search_value` — the broadest sparse tool. Shape queries as **source/program + grain + metric + time**. Government datasets use agency jargon; match it.

| Bad query | Good query | Why |
|---|---|---|
| `people entering prison system county` | `Texas county prison admissions fiscal year` | Specific program + geography + time |
| `food stamps by state` | `SNAP benefits state 2020` | Agency acronym, not colloquial name |

Use `search_schema` when you need exact column or table-name matching.

Use `search_prefix` only after you've learned the naming pattern from earlier results.

## When the first result set is weak

1. Tighten terms: program + geography + time
2. Try `search_schema` for field/table-name matches
3. Try `search_prefix` with a likely name fragment

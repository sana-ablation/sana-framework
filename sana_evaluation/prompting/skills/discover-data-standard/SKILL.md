---
name: discover-data
description: Hybrid search query shape and reformulation tactics. Load before your first search call in standard mode.
---

## Shape queries like datasets

Start with `search_value` for hybrid RRF retrieval. Shape queries as **source/program + grain + metric + time**.

| Bad query | Good query | Why |
|---|---|---|
| `school count` | `NCES public school locations county` | Agency + grain + metric |
| `violent crime` | `New York index crimes by county 2021` | Geography + metric + time |

Use `search_schema` when the question depends on a specific column or table name.

Use `search_prefix` only after you've learned the naming pattern.

## When the first result is weak

1. Reformulate with more exact program/geography/time terms, try `search_value` again
2. Try `search_schema` for column-level matching
3. Try `search_prefix` with a likely name fragment

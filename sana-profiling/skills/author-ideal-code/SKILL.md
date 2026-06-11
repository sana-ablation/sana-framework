---
name: author-ideal-code
description: Use when Codex needs to author or validate ideal_code and ideal_query records inside benchmark runtime profiles for ideal computation mode.
---

# Author Ideal Code

## Current Artifact Layout

Ideal computation records live in runtime profiles:

```text
benchmarks/<benchmark>/tasks-mini/runtime-profiles/<bucket>/task_N.json
```

They mirror source tasks under:

```text
benchmarks/<benchmark>/tasks-mini/tasks/<bucket>/task_N.json
```

Do not create new legacy profile-root files. Use `runtime_profile_root`.

## Scope

Edit only `ideal_code` and `ideal_query` records unless the user explicitly asks
to repair the source task or ideal profile text. Use `author-ideal-profiles` first if
the runtime profile is missing, not mirrored, or has leaky `reasoning_chain_text`.

## Record Contract

Use task nodes as source of truth. Each executable computation node should have:

```text
node_id
dataset_id
source
intent
code   # ideal_code only
sql    # ideal_query only, unless blocked
answer
```

Rules:

- `ideal_code.code` preserves the task node `fact` exactly.
- `intent` preserves the task node `subquestion`.
- `source` preserves the task node `source`.
- `answer` preserves the task node `answer`.
- Never change a task answer to fit authored code or SQL.

## Text And Data Boundaries

Do not assume empty `ideal_code` means the task is incomplete.

- `datagov/` executable nodes usually need both `ideal_code` and `ideal_query`.
- `datagov/` prose facts should be converted to verified executable facts first,
  then seeded into `ideal_code`.
- `wikipedia/` or other prose evidence nodes usually do not need computation
  records unless the task already has executable facts.
- For HotpotQA-style text evidence tasks, do not force Python or SQL records
  when evidence-span validation is the intended route.

## SQL Authoring

For `ideal_query`, write DuckDB SQL against table alias `t` and return one row
with one column named `answer` when possible.

If the source cannot be queried by the repo's `query_file` path because it is
prose, binary, too large, unsupported, or practically times out, write a blocker
record instead of fake SQL:

```json
{
  "node_id": "2",
  "dataset_id": "example",
  "source": "datagov/example/files/big.parquet",
  "intent": "Compute the requested value.",
  "answer": "Cannot execute SQL: file is too big for the direct query path."
}
```

Omit `sql` on blocker records.

## Validation

For every edited runtime profile:

```text
runtime profile is under benchmarks/<benchmark>/tasks-mini/runtime-profiles
mirrored task exists under benchmarks/<benchmark>/tasks-mini/tasks
each executable datagov node has ideal_code and ideal_query coverage
non-computation prose nodes do not get fake computation records
ideal_code.code exactly equals node.fact
ideal_query records match node source, intent, and answer
blocker records omit sql and explain the real query limitation
```

Independent reviewers should receive only the source task path, runtime profile
path, and the record contract. They should not solve unrelated task steps.

# dataindexing

`dataindexing` owns offline artifact generation and hybrid-search index
construction for SANA data-lake experiments. It also exports the runtime search
API consumed by `sana_evaluation.tools.external.search_standard_tools` and
`search_naive_tools`.

## Generic LakeQA/Data.gov Flow

Build a document parquet cache from schema metadata and S3 objects:

```bash
python -m dataindexing.cli.to_parquet \
  --output datalake_with_schema.parquet
```

Generate descriptions from a parquet cache or manifest:

```bash
python -m dataindexing.cli.parquet_to_description \
  datalake_with_schema.parquet \
  --output benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl \
  --parquet-output table_descriptions.parquet
```

Build the hybrid LanceDB index:

```bash
python -m dataindexing.cli.build_hybrid_search \
  --embed-preset qwen3_0_6b \
  --build-mode infused \
  --parquet datalake_with_schema.parquet \
  --descriptions benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl \
  --schemas benchmarks/lakeqa/tasks-mini/artifacts/table_schemas_full.jsonl \
  --output lance_data
```

## Kramabench Flow

Extract current Kramabench sources into table artifacts:

```bash
python -m dataindexing.cli.extract_kramabench_tables \
  --eval-root . \
  --output-parquet kramabench_tables.parquet \
  --output-schemas kramabench_table_schemas.jsonl \
  --output-manifest kramabench_table_manifest.jsonl \
  --output-report kramabench_extract_report.json \
  --tables-dir kramabench_tables
```

Generate Kramabench descriptions:

```bash
python -m dataindexing.cli.parquet_to_description \
  kramabench_table_manifest.jsonl \
  --input-format auto \
  --output kramabench_descriptions.jsonl \
  --parquet-output kramabench_descriptions.parquet
```

Build a Kramabench hybrid index:

```bash
python -m dataindexing.cli.build_hybrid_search \
  --embed-preset qwen3_0_6b \
  --build-mode infused \
  --parquet kramabench_tables.parquet \
  --descriptions kramabench_descriptions.jsonl \
  --schemas kramabench_table_schemas.jsonl \
  --output lance_kramabench_infused
```

## Runtime API

Runtime search wrappers import:

```python
from dataindexing.hybrid_search import api as _api
```

The exported API includes `setup_hybrid()`, `setup_sparse()`,
`hybrid_search()`, `hybrid_search_schema()`, `sparse_search()`,
`sparse_search_schema()`, and `hybrid_search_with_reranker()`.

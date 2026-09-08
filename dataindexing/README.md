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

## Benchmark artifacts

The offline artifacts the ideal-mode tools read at runtime are built by one CLI:

    python -m dataindexing.cli.benchmark_artifacts --benchmark lakeqa all

Stages run in dependency order -- `manifest` -> `describe` ->
`merge-descriptions` -> `check` -- each consuming the previous one's output.
`snippets PARQUET` is separate because its input is a parquet path the caller
supplies. Arguments after the stage name pass through and override the defaults
derived from `--benchmark`.

These moved here from `scripts/`, which now holds only experiment execution.

## Deriving table schemas from the bucket

`table_schemas_full.jsonl` as shipped has no generator anywhere -- it came from
data.gov's catalogue, which is why it advertises a `.csv` and a `.json`
distribution for a single stored object, and why it carries a `/v1/` path
segment the bucket does not use. Every reader compensates for both.

To derive schemas from what is actually stored:

    python -m dataindexing.cli.build_table_schemas \
        --prefix datagov/ \
        --output benchmarks/lakeqa/tasks-mini/artifacts/table_schemas_bucket.jsonl

It lists the bucket, skips metadata siblings (`catalog`, `dcat-us`, `headers`,
licence text, Socrata ids, bare numbers), and for each remaining object sniffs
the content family -- never the extension, since the crawler stored every
payload as `.txt` whatever it held -- then derives columns and delimiter from
the bytes. Output matches the shape `load_table_schemas` already reads.

It is sequential by default and resumes: a `<output>.done` checkpoint records
finished dataset slugs, so a run that dies part-way continues rather than
repeating. `--concurrency N` trades determinism for speed -- on a 400-dataset
check, 1 and 64 produced byte-identical output in 61s and 8s respectively -- but
the default stays at 1, because the full pass is a background job and being able
to reason about it matters more than finishing it sooner.

What actually loses datasets is not ordering. A fetch error that is caught and
returned as "no table here" is indistinguishable from an empty result, and the
dataset is then checkpointed as done, so a transient S3 blip becomes a
permanently missing schema. Reads are retried with backoff, unreadable files are
reported, and a dataset with any unread file is deliberately left out of the
checkpoint so a later run revisits it.

Whether a first line is a header or a data row is decided by its shape: several
short identifier-like fields. That is a heuristic, and the two cases it exists
to handle are both in the corpus -- a list of `Surname, Given` author names
reads as a consistent two-column table, and a real table's quoted WKT geometry
spans lines and defeats delimiter-consistency checks. Measured against the
datasets the benchmark tasks actually use, it derived a correct schema for 14 of
the 14 that have an eligible file.

A 3,000-dataset sample is what shaped the rest of the filtering, and each guard
exists for something that sample turned up: ZIP archives stored under a `.txt`
name whose compressed bytes parsed as a 2,180-column header, JSON-LD catalogue
records, ArcGIS service descriptors, and single-key API envelopes. Large
pretty-printed JSON is streamed with ijson, because a GeoJSON FeatureCollection
never parses whole from a peek and would otherwise be lost. Yield on that sample
is about 15% of datasets -- most of the rest hold citation or licence text
rather than tables.

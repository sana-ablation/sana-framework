from dataindexing.hybrid_search import api
from dataindexing.hybrid_search import builder


def test_runtime_api_exports_expected_symbols():
    for name in (
        "cfg",
        "setup_hybrid",
        "setup_sparse",
        "hybrid_search",
        "hybrid_search_schema",
        "hybrid_search_with_reranker",
        "sparse_search",
        "sparse_search_schema",
        "vector_search",
        "get_connection",
        "get_table",
        "get_schema_table",
    ):
        assert hasattr(api, name)


def test_hybrid_builder_defaults_use_maintained_lakeqa_artifacts():
    root = builder.PROJECT_ROOT

    assert builder.DEFAULT_MANIFEST_PATH == (
        root / "benchmarks/lakeqa/tasks-mini/artifacts/task_file_manifest.jsonl"
    )
    assert builder.DEFAULT_DESCRIPTIONS_PATH == (
        root / "benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl"
    )
    assert builder.DEFAULT_SCHEMAS_PATH == (
        root / "benchmarks/lakeqa/tasks-mini/artifacts/table_schemas_full.jsonl"
    )

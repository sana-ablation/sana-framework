"""Smoke-test the parent wrappers around dataindexing.hybrid_search.api.

This is intentionally lighter than ``test_search_matrix.py``: it does not open
LanceDB or load embedding/reranker models. Instead, it monkeypatches the
runtime API module imported by the parent search tools and verifies that the
new search-tool split routes to the expected setup/search functions.

Usage from the repo root:

    PYTHONPATH=. python test/test_external_search_tools_smoke.py
    PYTHONPATH=. pytest test/test_external_search_tools_smoke.py -q
"""

from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_wrapper(module_name: str):
    path = _REPO_ROOT / "sana_evaluation" / "tools" / "external" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExternalSearchToolsSmokeTest(unittest.TestCase):
    def test_wrappers_import_dataindexing_runtime_api(self) -> None:
        search_naive_tools = _load_wrapper("search_naive_tools")
        search_standard_tools = _load_wrapper("search_standard_tools")

        self.assertEqual(search_standard_tools._api.__name__, "dataindexing.hybrid_search.api")
        self.assertEqual(search_naive_tools._api.__name__, "dataindexing.hybrid_search.api")

    def test_standard_setup_uses_hybrid_setup_without_sparse_or_legacy_setup(self) -> None:
        search_standard_tools = _load_wrapper("search_standard_tools")

        with (
            patch.object(search_standard_tools._api, "setup_hybrid") as setup_hybrid,
            patch.object(search_standard_tools._api, "setup_sparse") as setup_sparse,
            patch.object(search_standard_tools._api, "setup") as legacy_setup,
        ):
            search_standard_tools.setup()

        setup_hybrid.assert_called_once_with()
        setup_sparse.assert_not_called()
        legacy_setup.assert_not_called()

    def test_naive_setup_uses_sparse_setup_without_hybrid_or_legacy_setup(self) -> None:
        search_naive_tools = _load_wrapper("search_naive_tools")

        with (
            patch.object(search_naive_tools._api, "setup_sparse") as setup_sparse,
            patch.object(search_naive_tools._api, "setup_hybrid") as setup_hybrid,
            patch.object(search_naive_tools._api, "setup") as legacy_setup,
        ):
            search_naive_tools.setup()

        setup_sparse.assert_called_once_with()
        setup_hybrid.assert_not_called()
        legacy_setup.assert_not_called()

    def test_standard_tools_route_to_hybrid_rrf_searches(self) -> None:
        search_standard_tools = _load_wrapper("search_standard_tools")

        with (
            patch.object(
                search_standard_tools._api,
                "hybrid_search",
                return_value=[{"uri": "s3://example/content.csv", "score": "0.016"}],
            ) as hybrid_search,
            patch.object(
                search_standard_tools._api,
                "hybrid_search_schema",
                return_value=[{"uri": "s3://example/schema.csv", "score": "0.015"}],
            ) as hybrid_search_schema,
            patch.object(search_standard_tools._api, "hybrid_search_with_reranker") as reranked,
            patch.object(search_standard_tools._api, "sparse_search") as sparse_search,
            patch.object(search_standard_tools._api, "sparse_search_schema") as sparse_search_schema,
        ):
            value_payload = search_standard_tools.search_value(query="traffic counts", top_k=5)
            schema_payload = search_standard_tools.search_schema(query="permit number", top_k=3)

        self.assertEqual(value_payload["count"], 1)
        self.assertEqual(value_payload["results"][0]["uri"], "s3://example/content.csv")
        self.assertEqual(schema_payload["count"], 1)
        self.assertEqual(schema_payload["results"][0]["uri"], "s3://example/schema.csv")
        hybrid_search.assert_called_once_with("traffic counts", k=5)
        hybrid_search_schema.assert_called_once_with("permit number", k=3)
        reranked.assert_not_called()
        sparse_search.assert_not_called()
        sparse_search_schema.assert_not_called()

    def test_naive_tools_route_to_sparse_searches(self) -> None:
        search_naive_tools = _load_wrapper("search_naive_tools")

        with (
            patch.object(
                search_naive_tools._api,
                "sparse_search",
                return_value=[{"uri": "s3://example/content.csv", "score": "1.000"}],
            ) as sparse_search,
            patch.object(
                search_naive_tools._api,
                "sparse_search_schema",
                return_value=[{"uri": "s3://example/schema.csv", "score": "0.900"}],
            ) as sparse_search_schema,
            patch.object(search_naive_tools._api, "hybrid_search") as hybrid_search,
            patch.object(search_naive_tools._api, "hybrid_search_schema") as hybrid_search_schema,
            patch.object(search_naive_tools._api, "hybrid_search_with_reranker") as reranked,
        ):
            value_payload = search_naive_tools.search_value(query="traffic counts", top_k=5)
            schema_payload = search_naive_tools.search_schema(query="permit number", top_k=3)

        self.assertEqual(value_payload["count"], 1)
        self.assertEqual(value_payload["results"][0]["uri"], "s3://example/content.csv")
        self.assertEqual(schema_payload["count"], 1)
        self.assertEqual(schema_payload["results"][0]["uri"], "s3://example/schema.csv")
        sparse_search.assert_called_once_with("traffic counts", k=5)
        sparse_search_schema.assert_called_once_with("permit number", k=3)
        hybrid_search.assert_not_called()
        hybrid_search_schema.assert_not_called()
        reranked.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

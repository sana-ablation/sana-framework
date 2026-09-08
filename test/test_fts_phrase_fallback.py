"""FTS phrase queries must degrade to keyword search, not fail the tool call.

LanceDB reads a double-quoted span as a phrase query, which needs token
positions in the FTS index. The index shipped in lance_data predates
builder.py setting with_position=True, so those queries raise instead of
returning anything - costing the agent a turn on a search it meant as
"these terms are required", not "these tokens are adjacent".
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataindexing.hybrid_search import api
from dataindexing.hybrid_search.api import _fts_with_quote_fallback, strip_phrase_quotes

_PHRASE_ERROR = (
    "lance error: Invalid user input: position is not found but required for "
    "phrase queries, try recreating the index with position"
)


class TestStripPhraseQuotes(unittest.TestCase):
    def test_removes_straight_quotes(self) -> None:
        self.assertEqual("Ward 7 311", strip_phrase_quotes('"Ward 7" "311"'))

    def test_removes_smart_quotes(self) -> None:
        self.assertEqual("smart quotes", strip_phrase_quotes("“smart quotes”"))

    def test_leaves_unquoted_text_untouched(self) -> None:
        self.assertEqual("311 ward 7", strip_phrase_quotes("311 ward 7"))

    def test_collapses_whitespace_left_by_removed_quotes(self) -> None:
        self.assertEqual("a b", strip_phrase_quotes('"a"  "b"'))

    def test_empty_when_query_is_only_quotes(self) -> None:
        self.assertEqual("", strip_phrase_quotes('""'))


class TestFtsQuoteFallback(unittest.TestCase):
    def test_retries_unquoted_when_index_lacks_positions(self) -> None:
        seen = []

        def run(q):
            seen.append(q)
            if '"' in q:
                raise RuntimeError(_PHRASE_ERROR)
            return [{"uri": "s3://bucket/hit", "_score": 1.0}]

        rows = _fts_with_quote_fallback(run, '"Ward 7" "311"')
        self.assertEqual([{"uri": "s3://bucket/hit", "_score": 1.0}], rows)
        self.assertEqual(['"Ward 7" "311"', "Ward 7 311"], seen)

    def test_does_not_retry_on_unrelated_errors(self) -> None:
        calls = []

        def run(q):
            calls.append(q)
            raise RuntimeError("table not found")

        with self.assertRaises(RuntimeError) as ctx:
            _fts_with_quote_fallback(run, '"Ward 7"')
        self.assertIn("table not found", str(ctx.exception))
        self.assertEqual(1, len(calls), "unrelated errors must not trigger a retry")

    def test_reraises_when_there_are_no_quotes_to_strip(self) -> None:
        def run(q):
            raise RuntimeError(_PHRASE_ERROR)

        with self.assertRaises(RuntimeError):
            _fts_with_quote_fallback(run, "ward 7 311")

    def test_reraises_when_query_is_only_quotes(self) -> None:
        def run(q):
            raise RuntimeError(_PHRASE_ERROR)

        with self.assertRaises(RuntimeError):
            _fts_with_quote_fallback(run, '""')

    def test_successful_query_runs_once(self) -> None:
        calls = []

        def run(q):
            calls.append(q)
            return []

        _fts_with_quote_fallback(run, '"Ward 7"')
        self.assertEqual(['"Ward 7"'], calls)


class TestSearchEntryPointsUseFallback(unittest.TestCase):
    """Every FTS-backed entry point must route through the fallback."""

    def _patched_table(self, fail_on_quotes=True):
        seen = []

        class _Q:
            def search(self, *a, **k):
                if a:
                    seen.append(a[0])
                return self

            def vector(self, *a, **k):
                return self

            def text(self, t):
                seen.append(t)
                return self

            def rerank(self, *a, **k):
                return self

            def limit(self, *a, **k):
                return self

            def select(self, *a, **k):
                return self

            def to_list(self):
                if fail_on_quotes and '"' in (seen[-1] or ""):
                    raise RuntimeError(_PHRASE_ERROR)
                return [{"uri": "s3://bucket/hit", "_score": 1.0, "_relevance_score": 1.0}]

        return _Q(), seen

    def test_sparse_search_falls_back(self) -> None:
        table, seen = self._patched_table()
        with patch.object(api, "get_table", return_value=table):
            rows = api.sparse_search('"Ward 7" "311"', k=2)
        self.assertEqual(1, len(rows))
        self.assertEqual("Ward 7 311", seen[-1])

    def test_sparse_search_schema_falls_back(self) -> None:
        table, seen = self._patched_table()
        with patch.object(api, "get_schema_table", return_value=table):
            rows = api.sparse_search_schema('"County Name"', k=2)
        self.assertEqual(1, len(rows))
        self.assertEqual("County Name", seen[-1])

    def test_hybrid_search_falls_back(self) -> None:
        table, seen = self._patched_table()
        with patch.object(api, "get_table", return_value=table), \
             patch.object(api, "_encode_query", return_value=[0.0]):
            rows = api.hybrid_search('"Ward 7" "311"', k=2)
        self.assertEqual(1, len(rows))
        self.assertEqual("Ward 7 311", seen[-1])

    def test_hybrid_search_schema_falls_back(self) -> None:
        table, seen = self._patched_table()
        with patch.object(api, "get_schema_table", return_value=table), \
             patch.object(api, "_encode_query", return_value=[0.0]):
            rows = api.hybrid_search_schema('"County Name"', k=2)
        self.assertEqual(1, len(rows))
        self.assertEqual("County Name", seen[-1])

    def test_hybrid_search_with_reranker_falls_back(self) -> None:
        table, seen = self._patched_table()
        # Seeding _reranker makes __setup_reranker return without loading weights.
        with patch.object(api, "get_table", return_value=table), \
             patch.object(api, "_encode_query", return_value=[0.0]), \
             patch.dict(api.__dict__, {"_reranker": object()}):
            rows = api.hybrid_search_with_reranker('"County Name"', k=2)
        self.assertEqual(1, len(rows))
        self.assertEqual("County Name", seen[-1])


if __name__ == "__main__":
    unittest.main()

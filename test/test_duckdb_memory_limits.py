"""DuckDB connections must be sized for concurrent use.

`_duckdb_connection` set neither `memory_limit` nor `threads`, so every
connection took DuckDB's defaults: 80% of system RAM and every core. On the
31 GiB eval box that is 25 GiB and 8 threads *per connection*.

That killed runs two different ways. A single task running `query_file` or
`execute_ideal` over a large S3 CSV could reach 25 GiB and trip the OOM killer
on its own, at any parallelism — observed as a python process at 29 GB RSS with
185 GB virtual. And once `--pool-tasks` let 8 tasks run concurrently, 8
connections each sized for 80% of RAM made OOM arithmetically certain.

Past its limit DuckDB spills to disk rather than dying, and the box has 110 GB
free, so a bounded limit trades an OOM for a slower query.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sana_evaluation.tools.agent_tools_v2 as v2


def _setting(conn, name):
    return conn.execute(f"SELECT current_setting('{name}')").fetchone()[0]


def _to_gib(raw: str) -> float:
    s = str(raw).strip().upper().replace("IB", "B")
    mult = {"KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}
    for suffix, m in mult.items():
        if s.endswith(suffix):
            return float(s[: -len(suffix)].strip()) * m
    return float(s) / 1024**3


class DuckDbLimitTests(unittest.TestCase):
    def test_connection_sets_an_explicit_memory_limit(self):
        conn = v2._duckdb_connection()
        try:
            limit = _to_gib(_setting(conn, "memory_limit"))
        finally:
            conn.close()
        self.assertLessEqual(
            limit, 8.0,
            "an unbounded connection can consume 80% of RAM and OOM the box alone",
        )
        self.assertGreater(limit, 0.5, "too small to run real queries")

    def test_connection_caps_threads_for_concurrent_workers(self):
        conn = v2._duckdb_connection()
        try:
            threads = int(_setting(conn, "threads"))
        finally:
            conn.close()
        self.assertLessEqual(
            threads, 4,
            "every connection claiming all cores thrashes under --parallel 8",
        )
        self.assertGreaterEqual(threads, 1)

    def test_limits_are_env_overridable(self):
        with patch.dict("os.environ", {"SANA_DUCKDB_MEMORY_LIMIT": "1GB",
                                       "SANA_DUCKDB_THREADS": "1"}, clear=False):
            conn = v2._duckdb_connection()
            try:
                # DuckDB parses "1GB" as 10**9 bytes and reports 953.6 MiB, so the
                # override lands just under 1 GiB rather than exactly on it.
                limit = _to_gib(_setting(conn, "memory_limit"))
                self.assertTrue(0.9 < limit < 1.0, f"expected ~0.93 GiB, got {limit}")
                self.assertEqual(int(_setting(conn, "threads")), 1)
            finally:
                conn.close()

    def test_eight_concurrent_connections_fit_in_the_box(self):
        """The arithmetic that failed before: 8 workers must not exceed RAM."""
        conn = v2._duckdb_connection()
        try:
            limit = _to_gib(_setting(conn, "memory_limit"))
        finally:
            conn.close()
        self.assertLess(
            limit * 8, 31.0,
            f"8 concurrent workers at {limit} GiB each exceeds the 31 GiB box",
        )

    def test_connection_still_works_for_queries(self):
        conn = v2._duckdb_connection()
        try:
            self.assertEqual(conn.execute("SELECT 40 + 2").fetchone()[0], 42)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis import analyse_experiment as ae


EVAL_FIELDS = ["task_id", "exact_match", "cost_usd", "runtime_seconds",
               "input_tokens", "output_tokens", "total_tokens",
               "tool_calls_total", "api_tool_calls", "cycle_count"]
VARIANT = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"


def _write_round(exp: Path, round_name: str, *, tasks_root: str, n: int = 2,
                 extra_tasks_root: str | None = None) -> Path:
    results = exp / round_name
    cell = results / "modes" / "openai_gpt-5-mini" / VARIANT
    cell.mkdir(parents=True)
    with (cell / "eval_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        for idx in range(n):
            writer.writerow({
                "task_id": f"{tasks_root}/k-1-d-1/task_{idx + 1}.json",
                "exact_match": "1.0" if idx % 2 == 0 else "0.0",
                "cost_usd": "0.01", "runtime_seconds": "10", "input_tokens": "100",
                "output_tokens": "10", "total_tokens": "110",
                "tool_calls_total": "5", "api_tool_calls": "5", "cycle_count": "3",
            })
    if extra_tasks_root:
        other = results / "modes" / "openai_gpt-5.2" / VARIANT
        other.mkdir(parents=True)
        with (other / "eval_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS)
            writer.writeheader()
            writer.writerow({
                "task_id": f"{extra_tasks_root}/k-1-d-1/task_1.json",
                "exact_match": "1.0", "cost_usd": "0.02", "runtime_seconds": "10",
                "input_tokens": "100", "output_tokens": "10", "total_tokens": "110",
                "tool_calls_total": "5", "api_tool_calls": "5", "cycle_count": "3",
            })
    (results / "traces").mkdir(exist_ok=True)
    return results


class TestDiscoverRounds(unittest.TestCase):
    def test_bare_results_and_reps_are_all_rounds(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results", "results-rep2", "results-rep3"):
                _write_round(exp, name, tasks_root="benchmarks/lakeqa/tasks-mini/tasks")
            (exp / "results_semantic").mkdir()
            (exp / "results.tex").write_text("not a round")
            self.assertEqual(
                [p.name for p in ae.discover_rounds(exp)],
                ["results", "results-rep2", "results-rep3"],
            )

    def test_rep1_style_experiments_have_no_bare_results(self):
        """luna-postrefactor names round 1 `results-rep1`, not `results`."""
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results-rep1", "results-rep2"):
                _write_round(exp, name, tasks_root="tmp/subset20b/tasks")
            self.assertEqual(
                [p.name for p in ae.discover_rounds(exp)], ["results-rep1", "results-rep2"]
            )

    def test_semantic_mirrors_are_not_rounds(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            _write_round(exp, "results", tasks_root="t")
            (exp / "results_semantic").mkdir()
            (exp / "results-rep2_semantic").mkdir()
            self.assertEqual([p.name for p in ae.discover_rounds(exp)], ["results"])


class TestResolveRound(unittest.TestCase):
    def test_derives_all_four_paths(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            results = _write_round(exp, "results-rep2", tasks_root="benchmarks/lakeqa/tasks-mini/tasks")
            (exp / "logs-rep2").mkdir(parents=True)
            paths = ae.resolve_round(exp, results)
            self.assertEqual(paths.results_dir, results / "modes")
            self.assertEqual(paths.base_results_dir, results / "modes")
            self.assertEqual(paths.traces_dir, results / "traces" / "modes")
            self.assertEqual(paths.tasks_dir, "benchmarks/lakeqa/tasks-mini/tasks")
            self.assertEqual(paths.logs_dir, exp / "logs-rep2")

    def test_missing_round_logs_do_not_fall_back_to_round_one(self):
        """web-arm-subset20b has logs/ but no logs-rep2/. Borrowing would mislabel."""
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            results = _write_round(exp, "results-rep2", tasks_root="t/tasks")
            (exp / "logs").mkdir(parents=True)
            paths = ae.resolve_round(exp, results)
            self.assertIsNone(paths.logs_dir)

    def test_mixed_task_sets_take_the_majority_and_report_the_rest(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            results = _write_round(
                exp, "results", tasks_root="a/tasks", n=4, extra_tasks_root="b/tasks"
            )
            paths = ae.resolve_round(exp, results)
            self.assertEqual(paths.tasks_dir, "a/tasks")
            self.assertEqual(paths.tasks_dir_spread, [("a/tasks", 1), ("b/tasks", 1)])

    def test_an_override_wins_over_derivation(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            results = _write_round(exp, "results", tasks_root="a/tasks")
            paths = ae.resolve_round(exp, results, tasks_dir_override="c/tasks")
            self.assertEqual(paths.tasks_dir, "c/tasks")

    def test_an_unjudged_round_selects_the_no_semantic_path(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            results = _write_round(exp, "results", tasks_root="t/tasks")
            paths = ae.resolve_round(exp, results)
            self.assertFalse(paths.semantic)
            self.assertIn("no mirror tree", paths.semantic_reason)


class TestCombineRounds(unittest.TestCase):
    def _rounds(self, *, third_is_semantic: bool):
        def rows(sem, em):
            return [{"condition_model": "m/v", "model": "m", "variant": "v", "n": 20,
                     "exact_match": em, "semantic_match": sem, "avg_cost_usd": 0.01,
                     "D_ret": None, "D_acc": None, "avg_tool_calls_total": 5.0,
                     "avg_search_calls": None, "avg_runtime_seconds": 10.0}]
        return [
            {"round": "results", "semantic": True, "rows": rows(0.70, 0.60)},
            {"round": "results-rep2", "semantic": True, "rows": rows(0.75, 0.65)},
            {"round": "results-rep3", "semantic": third_is_semantic, "rows": rows(0.80, 0.70)},
        ]

    def test_means_and_spread_across_rounds(self):
        combined = ae.combine_rounds(self._rounds(third_is_semantic=True))
        self.assertEqual(len(combined), 1)
        row = combined[0]
        self.assertEqual(row["n_rounds"], 3)
        self.assertEqual(row["rounds"], ["results", "results-rep2", "results-rep3"])
        self.assertEqual(row["n_total"], 60)
        self.assertEqual(row["semantic_match_mean"], 0.75)
        self.assertEqual(row["semantic_match_n"], 3)

    def test_a_lexical_round_is_never_pooled_into_semantic_match(self):
        """An unjudged round's semantic_match IS its exact_match. Pooling would lie."""
        combined = ae.combine_rounds(self._rounds(third_is_semantic=False))
        row = combined[0]
        self.assertEqual(row["n_rounds"], 3)
        self.assertEqual(row["n_rounds_semantic"], 2)
        self.assertEqual(row["semantic_match_n"], 2)
        self.assertEqual(row["semantic_match_mean"], 0.725)
        self.assertEqual(row["exact_match_n"], 3)

    def test_a_single_round_has_zero_spread_not_none(self):
        rounds = self._rounds(third_is_semantic=True)[:1]
        row = ae.combine_rounds(rounds)[0]
        self.assertEqual(row["n_rounds"], 1)
        self.assertEqual(row["semantic_match_sd"], 0.0)

    def test_a_condition_absent_from_one_round_reports_the_rounds_it_had(self):
        rounds = self._rounds(third_is_semantic=True)
        rounds[2]["rows"] = []
        row = ae.combine_rounds(rounds)[0]
        self.assertEqual(row["n_rounds"], 2)
        self.assertEqual(row["rounds"], ["results", "results-rep2"])


class TestAnalyseExperimentEndToEnd(unittest.TestCase):
    def test_writes_one_output_per_round_plus_a_combined_summary(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results", "results-rep2"):
                _write_round(exp, name, tasks_root=str(Path(tmp) / "tasks"))
            (Path(tmp) / "tasks").mkdir()

            report = ae.analyse_experiment(exp, no_figures=True)

            self.assertEqual([r["round"] for r in report["rounds"]], ["results", "results-rep2"])
            self.assertTrue(all(r["semantic"] is False for r in report["rounds"]))
            for name in ("results", "results-rep2"):
                self.assertTrue((exp / "analysis" / name / "summary.json").exists())
                self.assertTrue((exp / "analysis" / name / "no_semantic.json").exists())
            combined_dir = exp / "analysis" / "combined"
            self.assertTrue((combined_dir / "combined_summary.json").exists())
            self.assertTrue((combined_dir / "combined_summary.csv").exists())
            rounds_manifest = json.loads((combined_dir / "rounds.json").read_text())
            self.assertEqual(len(rounds_manifest["rounds"]), 2)
            self.assertIn("tasks_dir", rounds_manifest["rounds"][0])

    def test_round_filter_narrows_to_one(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results", "results-rep2"):
                _write_round(exp, name, tasks_root=str(Path(tmp) / "tasks"))
            (Path(tmp) / "tasks").mkdir()
            report = ae.analyse_experiment(exp, round_filter="rep2", no_figures=True)
            self.assertEqual([r["round"] for r in report["rounds"]], ["results-rep2"])

    def test_an_experiment_with_no_rounds_says_so(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            exp.mkdir()
            with self.assertRaises(ValueError) as caught:
                ae.analyse_experiment(exp)
            self.assertIn("no result rounds", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

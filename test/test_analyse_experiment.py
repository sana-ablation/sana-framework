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


def _write_complete_mirror(exp: Path, round_name: str) -> Path:
    """A mirror that `collect_mirror_issues` accepts, so the round reads complete."""
    source = exp / round_name
    mirror = exp / f"{round_name}_semantic"
    for src_csv in source.rglob("eval_results.csv"):
        dst_csv = mirror / src_csv.relative_to(source)
        dst_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = list(csv.DictReader(src_csv.open(newline="")))
        fieldnames = list(rows[0]) + [
            "semantic_match", "semantic_reason", "semantic_bucket",
            "log_error_bucket", "log_error_evidence",
        ]
        with dst_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                correct = str(row["exact_match"]).strip() in {"1", "1.0"}
                writer.writerow({
                    **row,
                    "semantic_match": "1" if correct else "0",
                    "semantic_reason": "matches the expected answer" if correct else "differs",
                    "semantic_bucket": "semantic_correct" if correct else "semantic_incorrect",
                    "log_error_bucket": "",
                    "log_error_evidence": "",
                })
    return mirror


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

    def test_a_narrowed_rerun_does_not_replace_the_all_rounds_combined(self):
        """`combined/` summarises whatever was analysed, under a fixed name.

        A `--round rep2` re-run analysed one round, so writing it would replace a
        three-round summary with a one-round one and nothing in the file would
        say it had been narrowed.
        """
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results", "results-rep2"):
                _write_round(exp, name, tasks_root=str(Path(tmp) / "tasks"))
            (Path(tmp) / "tasks").mkdir()

            ae.analyse_experiment(exp, no_figures=True)
            combined_dir = exp / "analysis" / "combined"
            before = (combined_dir / "combined_summary.json").read_text()
            rounds_before = json.loads((combined_dir / "rounds.json").read_text())
            self.assertEqual(len(rounds_before["rounds"]), 2)

            report = ae.analyse_experiment(exp, round_filter="rep2", no_figures=True)

            self.assertFalse(report["combined_written"])
            self.assertEqual((combined_dir / "combined_summary.json").read_text(), before)
            self.assertEqual(
                len(json.loads((combined_dir / "rounds.json").read_text())["rounds"]), 2
            )
            # The narrowed round's own output IS refreshed.
            self.assertTrue((exp / "analysis" / "results-rep2" / "summary.json").exists())

    def test_a_first_narrowed_run_still_writes_combined(self):
        """Skipping is about not overwriting; with nothing there, write it."""
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results", "results-rep2"):
                _write_round(exp, name, tasks_root=str(Path(tmp) / "tasks"))
            (Path(tmp) / "tasks").mkdir()
            report = ae.analyse_experiment(exp, round_filter="rep2", no_figures=True)
            self.assertTrue(report["combined_written"])
            self.assertTrue((exp / "analysis" / "combined" / "combined_summary.json").exists())

    def test_an_experiment_with_no_rounds_says_so(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            exp.mkdir()
            with self.assertRaises(ValueError) as caught:
                ae.analyse_experiment(exp)
            self.assertIn("no result rounds", str(caught.exception))


class TestPrintAuditPlan(unittest.TestCase):
    def test_lists_only_rounds_without_a_complete_mirror(self):
        """A round with a valid mirror must NOT be listed.

        Listing it would send an already-audited round back to the judge, which
        is the spend this whole separation exists to prevent.
        """
        import contextlib
        import io

        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            for name in ("results", "results-rep2", "results-rep3"):
                _write_round(exp, name, tasks_root=str(Path(tmp) / "tasks"))
            (exp / "logs").mkdir(parents=True)
            _write_complete_mirror(exp, "results-rep2")

            # Guard the fixture itself: rep2 must actually read as complete.
            from sana_analysis import analyse_experiment as ae_mod
            self.assertTrue(
                ae_mod.resolve_round(exp, exp / "results-rep2").semantic,
                "fixture is wrong: rep2's mirror is not being accepted as complete",
            )

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ae.main(["--experiment", str(exp), "--print-audit-plan"])

            names = [line.split("\t")[0] for line in buffer.getvalue().splitlines() if line.strip()]
            self.assertEqual(names, ["results", "results-rep3"])
            self.assertNotIn("results-rep2", names)


class TestRoundStatus(unittest.TestCase):
    def _exp_with_tasks(self, tmp: Path, n_tasks: int = 3) -> tuple[Path, str]:
        tasks = tmp / "tasks" / "k-1-d-1"
        tasks.mkdir(parents=True)
        for idx in range(n_tasks):
            (tasks / f"task_{idx + 1}.json").write_text("{}")
        return tmp / "exp", str(tmp / "tasks")

    def test_a_full_round_is_complete(self):
        with TemporaryDirectory() as tmp:
            exp, tasks_root = self._exp_with_tasks(Path(tmp))
            _write_round(exp, "results", tasks_root=tasks_root, n=3)
            status = ae.round_statuses(exp)[0]
            self.assertTrue(status.complete, status.reason)
            self.assertEqual((status.number, status.cells, status.full_cells), (1, 1, 1))

    def test_a_short_cell_makes_the_round_incomplete(self):
        with TemporaryDirectory() as tmp:
            exp, tasks_root = self._exp_with_tasks(Path(tmp))
            _write_round(exp, "results", tasks_root=tasks_root, n=2)   # 2 of 3 tasks
            status = ae.round_statuses(exp)[0]
            self.assertFalse(status.complete)
            self.assertIn("short of a full task set", status.reason)

    def test_a_missing_cell_is_caught_against_the_fullest_round(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp, tasks_root = self._exp_with_tasks(root)
            # _write_round's extra cell always writes exactly one row, so its task
            # set must genuinely hold one task for that cell to read as full.
            extra_tasks = root / "tasks_extra" / "k-1-d-1"
            extra_tasks.mkdir(parents=True)
            (extra_tasks / "task_1.json").write_text("{}")
            _write_round(exp, "results", tasks_root=tasks_root, n=3,
                         extra_tasks_root=str(root / "tasks_extra"))
            _write_round(exp, "results-rep2", tasks_root=tasks_root, n=3)  # one cell fewer
            first, second = ae.round_statuses(exp)
            self.assertTrue(first.complete, first.reason)
            self.assertFalse(second.complete)
            self.assertIn("missing vs the fullest round", second.reason)

    def test_expected_rows_come_from_each_cells_own_task_set(self):
        """model-tiers/results-rep3 mixes two task sets; a majority rule misjudges it."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp, tasks_a = self._exp_with_tasks(root, n_tasks=3)
            tasks_b = root / "tasks_b" / "k-1-d-1"
            tasks_b.mkdir(parents=True)
            (tasks_b / "task_1.json").write_text("{}")          # this set has ONE task
            _write_round(exp, "results", tasks_root=tasks_a, n=3,
                         extra_tasks_root=str(root / "tasks_b"))
            status = ae.round_statuses(exp)[0]
            self.assertEqual(status.cells, 2)
            self.assertTrue(status.complete, status.reason)

    def test_a_task_set_no_longer_on_disk_is_not_judged_complete(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            _write_round(exp, "results", tasks_root="gone/tasks", n=3)
            status = ae.round_statuses(exp)[0]
            self.assertFalse(status.complete)
            self.assertIn("task set not on disk", status.reason)


class TestNextRound(unittest.TestCase):
    def _exp(self, tmp: Path, n_tasks: int = 3):
        tasks = tmp / "tasks" / "k-1-d-1"
        tasks.mkdir(parents=True)
        for idx in range(n_tasks):
            (tasks / f"task_{idx + 1}.json").write_text("{}")
        return tmp / "exp", str(tmp / "tasks")

    def test_all_complete_starts_the_next_one(self):
        with TemporaryDirectory() as tmp:
            exp, tasks_root = self._exp(Path(tmp))
            for name in ("results", "results-rep2"):
                _write_round(exp, name, tasks_root=tasks_root, n=3)
            number, why = ae.next_round(exp)
            self.assertEqual(number, 3)
            self.assertIn("complete", why)

    def test_an_empty_experiment_starts_at_round_one(self):
        with TemporaryDirectory() as tmp:
            exp = Path(tmp) / "exp"
            exp.mkdir()
            number, _why = ae.next_round(exp)
            self.assertEqual(number, 1)

    def test_it_resumes_the_earliest_gap_not_the_latest(self):
        """A hole in round 2 matters more than starting round 4."""
        with TemporaryDirectory() as tmp:
            exp, tasks_root = self._exp(Path(tmp))
            _write_round(exp, "results", tasks_root=tasks_root, n=3)
            _write_round(exp, "results-rep2", tasks_root=tasks_root, n=1)   # short
            _write_round(exp, "results-rep3", tasks_root=tasks_root, n=3)
            number, why = ae.next_round(exp)
            self.assertEqual(number, 2)
            self.assertIn("resuming round 2", why)

    def test_rep1_naming_still_numbers_from_one(self):
        with TemporaryDirectory() as tmp:
            exp, tasks_root = self._exp(Path(tmp))
            _write_round(exp, "results-rep1", tasks_root=tasks_root, n=3)
            number, _why = ae.next_round(exp)
            self.assertEqual(number, 2)

    def test_print_next_round_emits_only_the_number(self):
        import contextlib
        import io

        with TemporaryDirectory() as tmp:
            exp, tasks_root = self._exp(Path(tmp))
            _write_round(exp, "results", tasks_root=tasks_root, n=3)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ae.main(["--experiment", str(exp), "--print-next-round"])
            self.assertEqual(buffer.getvalue().strip(), "2")


if __name__ == "__main__":
    unittest.main()

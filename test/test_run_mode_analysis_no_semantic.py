import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis import run_mode_analysis


EVAL_FIELDS = [
    "task_id",
    "model",
    "expected_answer",
    "predicted_answer",
    "exact_match",
    "runtime_seconds",
    "cycle_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "tool_calls_total",
    "api_tool_calls",
]

VARIANT = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"


def _write_unaudited_tree(root: Path) -> Path:
    """An eval tree with exact_match and no judge output -- what sana_evaluation writes."""
    eval_dir = root / "modes" / "openai_gpt-5-mini" / VARIANT
    eval_dir.mkdir(parents=True)
    with (eval_dir / "eval_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        for idx, exact in enumerate(["1.0", "0.0", "1.0", "0.0"], start=1):
            writer.writerow({
                "task_id": f"k-1-d-1/task_{idx}.json",
                "model": "gpt-5-mini",
                "expected_answer": "1999",
                "predicted_answer": "1999" if exact == "1.0" else "2000",
                "exact_match": exact,
                "runtime_seconds": "12.5",
                "cycle_count": "4",
                "input_tokens": "1000",
                "output_tokens": "100",
                "total_tokens": "1100",
                "cost_usd": "0.01",
                "tool_calls_total": "6",
                "api_tool_calls": "6",
            })
    return root / "modes"


class TestUnauditedTreeErrorMessage(unittest.TestCase):
    def test_error_names_the_remedy(self):
        with TemporaryDirectory() as tmp:
            results_dir = _write_unaudited_tree(Path(tmp))
            with self.assertRaises(ValueError) as caught:
                run_mode_analysis.load_semantic_results_grouped(str(results_dir))
            message = str(caught.exception)
            self.assertIn("semantic_match", message)
            self.assertIn("--no-semantic", message)
            self.assertIn(
                "sana_analysis/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py",
                message,
            )


class TestNoSemanticLoad(unittest.TestCase):
    def test_semantic_match_is_sourced_from_exact_match(self):
        with TemporaryDirectory() as tmp:
            results_dir = _write_unaudited_tree(Path(tmp))
            by_key, _fields = run_mode_analysis.load_semantic_results_grouped(
                str(results_dir), semantic=False
            )
            records = by_key["openai_gpt-5-mini/" + VARIANT]
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertEqual(record["_semantic_match"], record["_exact_match"])

    def test_no_semantic_does_not_invent_a_no_error_verdict(self):
        """Absent log_error_bucket must not normalise to `no_error`.

        `_normalize_log_error_bucket("")` returns "no_error", which is a claim
        about the run that no judge made. Under --no-semantic it stays blank.
        """
        with TemporaryDirectory() as tmp:
            results_dir = _write_unaudited_tree(Path(tmp))
            by_key, _fields = run_mode_analysis.load_semantic_results_grouped(
                str(results_dir), semantic=False
            )
            for record in next(iter(by_key.values())):
                self.assertEqual(record["log_error_bucket_display"], "")

    def test_a_crashed_row_with_blank_exact_match_is_scored_zero(self):
        """A crashed task records no answer. The judged path reads that as 0.0
        via as_float; --no-semantic must agree rather than raise, or it cannot
        analyse any tree containing a crashed run.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = _write_unaudited_tree(root)
            csv_path = next(results_dir.rglob("eval_results.csv"))
            rows = list(csv.DictReader(csv_path.open(newline="")))
            rows[0]["exact_match"] = ""
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            by_key, _fields = run_mode_analysis.load_semantic_results_grouped(
                str(results_dir), semantic=False
            )
            records = next(iter(by_key.values()))
            self.assertEqual(records[0]["_semantic_match"], 0.0)
            self.assertEqual(records[0]["_exact_match"], 0.0)

    def test_a_nonsense_exact_match_still_raises(self):
        """Absence is tolerated; garbage is not."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = _write_unaudited_tree(root)
            csv_path = next(results_dir.rglob("eval_results.csv"))
            rows = list(csv.DictReader(csv_path.open(newline="")))
            rows[0]["exact_match"] = "2.0"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaises(ValueError):
                run_mode_analysis.load_semantic_results_grouped(
                    str(results_dir), semantic=False
                )


def _write_discovery_fixture(tmp: Path) -> None:
    """Enough of a task + trace to give one real record a populated bin.

    `build_search_depth_buckets` and `build_reasoning_density_buckets` only
    place a record into a bin once discovery has a matching task (for gold
    dataset counts) and a matching trace (for the search-call count). Without
    this, every bin stays at n=0 and the "no zeroed judgment columns" test
    below would have nothing populated to check.
    """
    task_dir = tmp / "tasks" / "k-1-d-1"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_1.json").write_text(json.dumps({"datasets_used": ["ds_a"]}))

    trace_dir = tmp / "results" / "traces" / "modes" / "openai_gpt-5-mini" / VARIANT / "k-1-d-1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_line = json.dumps({
        "task_id": "k-1-d-1/task_1.json",
        "tool": "search_ideal",
        "result_dataset_ids": ["ds_a"],
    })
    (trace_dir / "task_1.jsonl").write_text(trace_line + "\n")


class TestNoSemanticOutputShape(unittest.TestCase):
    def _run(self, tmp: Path) -> Path:
        results_dir = _write_unaudited_tree(tmp / "results")
        _write_discovery_fixture(tmp)
        out_dir = tmp / "analysis"
        run_mode_analysis.run_analysis(
            results_dir=str(results_dir),
            base_results_dir=str(results_dir),
            turn_waste_grouped_dir=None,
            traces_dir=str(tmp / "results" / "traces" / "modes"),
            tasks_dir=str(tmp / "tasks"),
            output_dir=str(out_dir),
            no_figures=True,
            no_semantic=True,
        )
        return out_dir

    def test_judgment_dependent_artifacts_are_absent_not_empty(self):
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            for name in run_mode_analysis.SEMANTIC_ONLY_OUTPUTS:
                self.assertFalse(
                    (out_dir / name).exists(),
                    f"{name} must be omitted under --no-semantic, not written empty",
                )

    def test_judge_free_metrics_are_still_produced(self):
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(len(summary), 1)
            row = summary[0]
            self.assertEqual(row["n"], 4)
            self.assertEqual(row["exact_match"], 0.5)
            self.assertEqual(row["semantic_match"], 0.5)
            self.assertEqual(row["avg_cost_usd"], 0.01)
            self.assertEqual(row["avg_tool_calls_total"], 6.0)

    def test_summary_rows_carry_no_all_zero_bucket_columns(self):
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            row = json.loads((out_dir / "summary.json").read_text())[0]
            for bucket in run_mode_analysis.SEMANTIC_BUCKETS:
                self.assertNotIn(bucket, row)
            for bucket in run_mode_analysis.DISPLAY_LOG_ERROR_BUCKETS:
                self.assertNotIn(bucket, row)
            for bucket in run_mode_analysis.FAILURE_PRIMARY_BUCKETS:
                self.assertNotIn(f"primary_{bucket}", row)

    def test_output_directory_says_no_judge_ran(self):
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            marker = json.loads((out_dir / "no_semantic.json").read_text())
            self.assertFalse(marker["semantic"])
            self.assertEqual(marker["semantic_match_source"], "exact_match")
            self.assertEqual(sorted(marker["omitted"]), sorted(run_mode_analysis.SEMANTIC_ONLY_OUTPUTS))

    def _populated_bins(self, payload) -> list[dict]:
        """Every bin entry that actually holds records."""
        found = []
        def walk(node):
            if isinstance(node, dict):
                if isinstance(node.get("n"), int) and node["n"] > 0 and "bins" not in node:
                    found.append(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
        walk(payload)
        return found

    def test_bucket_files_carry_no_zeroed_judgment_columns(self):
        """A bin with real records must not report semantic_correct: 0.

        Four runs reported as none-correct, none-incorrect and none-blank reads
        as a result and is not one.
        """
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            for name in ("search_depth_buckets.json", "reasoning_density_buckets.json"):
                payload = json.loads((out_dir / name).read_text())
                populated = self._populated_bins(payload)
                self.assertTrue(populated, f"{name} had no populated bin to check")
                for entry in populated:
                    for bucket in run_mode_analysis.SEMANTIC_BUCKETS:
                        self.assertNotIn(bucket, entry, f"{name} bin: {entry}")
                        self.assertNotIn(f"{bucket}_rate", entry)
                    for bucket in run_mode_analysis.DISPLAY_LOG_ERROR_BUCKETS:
                        self.assertNotIn(bucket, entry)

    def test_per_model_outputs_carry_no_zeroed_judgment_columns(self):
        """by_model/ is a filtered copy of the same rows and must omit the same columns.

        The top-level file omitting them while the per-model copy keeps them is
        worse than either alone: the two disagree about the same run.
        """
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            per_model = sorted((out_dir / "by_model").rglob("variant_summary.json"))
            self.assertTrue(per_model, "no by_model/variant_summary.json was written")
            for path in per_model:
                for row in json.loads(path.read_text()):
                    for bucket in run_mode_analysis.SEMANTIC_BUCKETS:
                        self.assertNotIn(bucket, row, f"{path}: {row}")
                    for bucket in run_mode_analysis.DISPLAY_LOG_ERROR_BUCKETS:
                        self.assertNotIn(bucket, row, f"{path}: {row}")

    def test_marker_names_the_fields_that_hold_lexical_numbers(self):
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            marker = json.loads((out_dir / "no_semantic.json").read_text())
            sourced = marker["fields_sourced_from_exact_match"]
            self.assertIn("semantic_match", " ".join(sourced.values()))
            self.assertIn("mean_semantic_match", " ".join(sourced.values()))

    def test_the_marker_names_the_file_whose_name_is_the_measurement(self):
        """`semantic_match.json` IS written under --no-semantic.

        A reader who opens the obviously-named file sees `"semantic_match": 0.65`
        where 0.65 is the exact-match score. The marker has to name that file, and
        `by_model/*/semantic_match.json`, or the disclosure skips the one artifact
        most likely to be read on its own.
        """
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            self.assertTrue((out_dir / "semantic_match.json").exists())
            keys = " ".join(
                json.loads((out_dir / "no_semantic.json").read_text())[
                    "fields_sourced_from_exact_match"
                ]
            )
            self.assertIn("semantic_match.json", keys)
            self.assertIn("by_model/*/semantic_match.json", keys)

    def test_the_marker_names_search_efficiency(self):
        """`search_efficiency` is semantic_match / avg_search_calls, so it is a
        lexical number too and nothing else says so."""
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            sourced = json.loads((out_dir / "no_semantic.json").read_text())[
                "fields_sourced_from_exact_match"
            ]
            self.assertIn("search_efficiency", " ".join(sourced.values()))

    def test_per_task_csv_has_no_all_blank_judgment_column(self):
        """`log_error_bucket_display` is "" on every row when no judge ran."""
        with TemporaryDirectory() as tmp:
            out_dir = self._run(Path(tmp))
            paths = [out_dir / "per_task_semantic.csv"]
            paths += sorted((out_dir / "by_model").rglob("per_task_semantic.csv"))
            self.assertTrue(len(paths) > 1, "no by_model copy was written")
            for path in paths:
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertNotIn("log_error_bucket_display", reader.fieldnames or [], str(path))
                    rows = list(reader)
                if path == paths[0]:
                    self.assertTrue(rows, f"{path} had no rows to check")


class TestStaleSemanticArtifactsAreRemoved(unittest.TestCase):
    """A --no-semantic re-run must not leave a judged run's numbers behind.

    `analyse_experiment` writes every round to a stable `<exp>/analysis/<round>`
    and is built to be re-run, so a round that was judged and later flips to
    --no-semantic lands a fresh `no_semantic.json` -- which lists these files as
    `omitted` -- right next to the files themselves.
    """

    def _prepopulate(self, out_dir: Path) -> list[Path]:
        planted = []
        dirs = [out_dir, out_dir / "by_model" / "openai_gpt-5-mini", out_dir / "figures"]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            for name in run_mode_analysis.SEMANTIC_ONLY_OUTPUTS:
                path = directory / name
                path.write_text('{"stale": "judged numbers from an older row set"}')
                planted.append(path)
        return planted

    def test_a_no_semantic_run_deletes_the_judged_artifacts_it_finds(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_dir = _write_unaudited_tree(tmp_path / "results")
            _write_discovery_fixture(tmp_path)
            out_dir = tmp_path / "analysis"
            planted = self._prepopulate(out_dir)

            run_mode_analysis.run_analysis(
                results_dir=str(results_dir),
                base_results_dir=str(results_dir),
                turn_waste_grouped_dir=None,
                traces_dir=str(tmp_path / "results" / "traces" / "modes"),
                tasks_dir=str(tmp_path / "tasks"),
                output_dir=str(out_dir),
                no_figures=True,
                no_semantic=True,
            )

            for path in planted:
                self.assertFalse(
                    path.exists(),
                    f"{path} survived a --no-semantic run while no_semantic.json calls it omitted",
                )
            # And the marker that says they are absent is actually there.
            self.assertTrue((out_dir / "no_semantic.json").exists())

    def test_the_figures_csvs_go_too_even_though_cleanup_only_unlinks_pdfs(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "analysis"
            (out_dir / "figures").mkdir(parents=True)
            csv_names = [n for n in run_mode_analysis.SEMANTIC_ONLY_OUTPUTS if n.endswith(".csv")]
            self.assertTrue(csv_names)
            for name in csv_names:
                (out_dir / "figures" / name).write_text("stale\n")
            keep = out_dir / "figures" / "some_other_figure_data.csv"
            keep.write_text("keep me\n")

            removed = run_mode_analysis._remove_stale_semantic_outputs(out_dir)

            self.assertEqual(
                sorted(p.name for p in removed), sorted(csv_names)
            )
            self.assertTrue(keep.exists(), "an unrelated CSV must not be swept up")


if __name__ == "__main__":
    unittest.main()

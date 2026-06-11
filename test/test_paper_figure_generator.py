import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sana_analysis.report_generator.paper_figure_generator import (
    _benchmark_defaults,
    _load_search_figure_data,
    _search_variant_label,
    _selected_existing_figures,
    export_agent_analysis_results,
    export_paper_figures,
    render_search_efficiency_figure,
)


class TestPaperFigureGenerator(unittest.TestCase):
    def test_render_search_efficiency_figure_uses_single_row_scatter_only_layout(self):
        class FakeAxis:
            def __init__(self):
                self.titles = []
                self.annotations = []
                self.plots = []
                self.scatters = []
                self.xlims = []
                self.ylims = []

            def scatter(self, *args, **kwargs):
                self.scatters.append((args, kwargs))
                return None

            def annotate(self, text, *args, **kwargs):
                self.annotations.append(text)

            def plot(self, *args, **kwargs):
                self.plots.append((args, kwargs))

            def set_xlim(self, *args, **kwargs):
                self.xlims.append(args)
                return None

            def set_ylim(self, *args, **kwargs):
                self.ylims.append(args)
                return None

            def grid(self, *args, **kwargs):
                return None

            def set_title(self, title, *args, **kwargs):
                self.titles.append(title)

            def set_xlabel(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def tick_params(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

        class FakeFig:
            def __init__(self):
                self.saved_path = None
                self.suptitles = []

            def suptitle(self, title, *args, **kwargs):
                self.suptitles.append(title)
                return None

            def tight_layout(self, *args, **kwargs):
                return None

            def savefig(self, path):
                self.saved_path = path

        class FakePlot:
            def __init__(self):
                self.subplots_kwargs = None
                self.axes = [[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]]
                self.fig = FakeFig()

            def subplots(self, **kwargs):
                self.subplots_kwargs = kwargs
                return self.fig, self.axes

            def close(self, fig):
                return None

        with TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir)
            summary_rows = []
            for model, points in [
                (
                    "openai_gpt-5-mini",
                    [
                        ("search_n_results_i_plani_computei_k5_skills_off", 0.5, 0.4, 5.7),
                        ("search_d_results_i_plani_computei_k5_skills_off", 0.6, 0.5, 5.8),
                        ("search_i_results_i_plani_computei_k5_skills_off", 0.7, 0.6, 5.9),
                    ],
                ),
                (
                    "openai_gpt-5.4-nano",
                    [
                        ("search_n_results_i_plani_computei_k5_skills_off", 0.5, 0.4, 4.0),
                        ("search_d_results_i_plani_computei_k5_skills_off", 0.6, 0.5, 5.0),
                        ("search_i_results_i_plani_computei_k5_skills_off", 0.7, 0.6, 3.0),
                    ],
                ),
            ]:
                for variant, d_ret, d_acc, search_calls in points:
                    summary_rows.append(
                        {
                            "model": model,
                            "variant": variant,
                            "avg_search_calls": search_calls,
                            "D_ret": d_ret,
                            "D_acc_recall": d_acc,
                        }
                    )
            (analysis_dir / "summary.json").write_text(json.dumps(summary_rows), encoding="utf-8")
            with (analysis_dir / "search_call_cumulative_retrieval.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "model",
                        "variant",
                        "task_id",
                        "search_call_index",
                        "cumulative_search_gold_recall",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "openai_gpt-5-mini",
                        "variant": "search_n_results_i_plani_computei_k5_skills_off",
                        "task_id": "k-1/task_1",
                        "search_call_index": "1",
                        "cumulative_search_gold_recall": "0.5",
                    }
                )

            fake_plot = FakePlot()
            with patch("sana_analysis.report_generator.paper_figure_generator._import_plot_libs", return_value=fake_plot):
                render_search_efficiency_figure(analysis_dir, "lakeqa", analysis_dir / "figure.pdf")

            self.assertEqual(fake_plot.subplots_kwargs["nrows"], 1)
            self.assertEqual(fake_plot.subplots_kwargs["ncols"], 2)
            all_titles = [title for row in fake_plot.axes for axis in row for title in axis.titles]
            self.assertNotIn("Cumulative", " ".join(all_titles))
            self.assertNotIn("Cumulative", " ".join(fake_plot.fig.suptitles))
            all_annotations = [text for row in fake_plot.axes for axis in row for text in axis.annotations]
            self.assertTrue(any("BM25" in text and "D_ret 50%" in text and "D_acc 40%" in text for text in all_annotations))
            self.assertFalse(any("NII" in text or "DII" in text or "III" in text for text in all_annotations))
            all_scatter_kwargs = [kwargs for row in fake_plot.axes for axis in row for _args, kwargs in axis.scatters]
            self.assertTrue(any(kwargs.get("facecolors") == "none" and kwargs.get("s", 0) > 90 for kwargs in all_scatter_kwargs))
            mini_xlim = fake_plot.axes[0][0].xlims[-1]
            nano_xlim = fake_plot.axes[0][1].xlims[-1]
            self.assertLess(mini_xlim[1] - mini_xlim[0], nano_xlim[1] - nano_xlim[0])

    def test_benchmark_defaults_choose_expected_roots(self):
        lakeqa = _benchmark_defaults("lakeqa")
        kramabench = _benchmark_defaults("kramabench")

        self.assertEqual(lakeqa.results_dir, Path("results_semantic/modes"))
        self.assertEqual(lakeqa.base_results_dir, Path("results/modes"))
        self.assertEqual(lakeqa.traces_dir, Path("results/traces/modes"))
        self.assertEqual(lakeqa.tasks_dir, Path("benchmarks/lakeqa/tasks-mini/tasks"))
        self.assertEqual(lakeqa.analysis_dir, Path("analysis_results_mode_semantic"))
        self.assertEqual(lakeqa.answer_failure_combined_dir, Path("results_semantic_answer_failures_combined"))

        self.assertEqual(kramabench.results_dir, Path("results-kramabench_semantic/modes"))
        self.assertEqual(kramabench.base_results_dir, Path("results-kramabench/modes"))
        self.assertEqual(kramabench.traces_dir, Path("results-kramabench/traces/modes"))
        self.assertEqual(kramabench.tasks_dir, Path("benchmarks/kramabench/tasks-mini/tasks"))
        self.assertEqual(kramabench.analysis_dir, Path("analysis_results_mode_kramabench_semantic"))
        self.assertEqual(
            kramabench.answer_failure_combined_dir,
            Path("results-kramabench_semantic_answer_failures_combined"),
        )

    def test_selected_existing_figures_include_benchmark_specific_semantic_delta(self):
        self.assertIn(
            "lakeqa_semantic_delta_ablation.pdf",
            _selected_existing_figures("lakeqa"),
        )
        self.assertIn(
            "kramabench_semantic_delta_ablation.pdf",
            _selected_existing_figures("kramabench"),
        )
        self.assertIn(
            "answer_failure_groups_by_model.pdf",
            _selected_existing_figures("lakeqa"),
        )
        self.assertIn(
            "answer_failure_groups_by_condition.pdf",
            _selected_existing_figures("lakeqa"),
        )

    def test_search_variant_label_maps_canonical_trio(self):
        self.assertEqual(
            _search_variant_label("search_n_results_i_plani_computei_k5_skills_off"),
            "NII",
        )
        self.assertEqual(
            _search_variant_label("search_d_results_i_plani_computei_k5_skills_off"),
            "DII",
        )
        self.assertEqual(
            _search_variant_label("search_i_results_i_plani_computei_k5_skills_off"),
            "III",
        )
        self.assertIsNone(_search_variant_label("search_p_results_i_plani_computei_k5_skills_off"))

    def test_load_search_figure_data_filters_canonical_trio(self):
        with TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir)
            summary_rows = [
                {
                    "model": "openai_gpt-5-mini",
                    "variant": "search_n_results_i_plani_computei_k5_skills_off",
                    "avg_search_calls": 3,
                    "D_ret": 0.5,
                    "D_acc": 0.25,
                },
                {
                    "model": "openai_gpt-5-mini",
                    "variant": "search_d_results_i_plani_computei_k5_skills_off",
                    "avg_search_calls": 4,
                    "D_ret": 0.6,
                    "D_acc": 0.4,
                },
                {
                    "model": "openai_gpt-5-mini",
                    "variant": "search_i_results_i_plani_computei_k5_skills_off",
                    "avg_search_calls": 1,
                    "D_ret": 1.0,
                    "D_acc": 1.0,
                },
                {
                    "model": "openai_gpt-5-mini",
                    "variant": "search_p_results_i_plani_computei_k5_skills_off",
                    "avg_search_calls": 0,
                    "D_ret": 1.0,
                    "D_acc": 1.0,
                },
            ]
            (analysis_dir / "summary.json").write_text(json.dumps(summary_rows))
            with (analysis_dir / "search_call_cumulative_retrieval.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "model",
                        "variant",
                        "task_id",
                        "search_call_index",
                        "cumulative_search_gold_recall",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "openai_gpt-5-mini",
                        "variant": "search_n_results_i_plani_computei_k5_skills_off",
                        "task_id": "k-1/task_1",
                        "search_call_index": "1",
                        "cumulative_search_gold_recall": "0.5",
                    }
                )

            data = _load_search_figure_data(analysis_dir)

            self.assertEqual(list(data["summary_by_model"].keys()), ["openai_gpt-5-mini"])
            self.assertEqual(
                [row["label"] for row in data["summary_by_model"]["openai_gpt-5-mini"]],
                ["NII", "DII", "III"],
            )
            self.assertEqual(data["curves"][("openai_gpt-5-mini", "NII")][1], 0.5)

    def test_export_paper_figures_copies_existing_and_new_outputs(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            analysis_dir = root / "analysis"
            figures_dir = analysis_dir / "figures"
            figures_dir.mkdir(parents=True)
            for name in [
                "answer_failure_groups_by_model.pdf",
                "answer_failure_groups_by_condition.pdf",
                "semantic_delta_ablation_compact.pdf",
            ]:
                (figures_dir / name).write_bytes(b"%PDF-1.4\n")
            new_figure = analysis_dir / "search_efficiency_cumulative_retrieval_lakeqa.pdf"
            new_figure.write_bytes(b"%PDF-1.4\n")

            copied = export_paper_figures(
                benchmark="lakeqa",
                analysis_dir=analysis_dir,
                search_figure_path=new_figure,
                destinations=[root / "paper", root / "mirror"],
            )

            self.assertIn(root / "paper" / new_figure.name, copied)
            self.assertTrue((root / "mirror" / "answer_failure_groups_by_model.pdf").exists())
            self.assertTrue((root / "paper" / "lakeqa_semantic_delta_ablation.pdf").exists())

    def test_export_paper_figures_uses_fallback_dir_for_optional_figures(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            analysis_dir = root / "analysis"
            (analysis_dir / "figures").mkdir(parents=True)
            fallback_dir = root / "existing_paper"
            fallback_dir.mkdir()
            for name in [
                "answer_failure_groups_by_model.pdf",
                "answer_failure_groups_by_condition.pdf",
                "lakeqa_semantic_delta_ablation.pdf",
            ]:
                (fallback_dir / name).write_bytes(b"%PDF-1.4\n")
            new_figure = analysis_dir / "search_efficiency_cumulative_retrieval_lakeqa.pdf"
            new_figure.write_bytes(b"%PDF-1.4\n")

            export_paper_figures(
                benchmark="lakeqa",
                analysis_dir=analysis_dir,
                search_figure_path=new_figure,
                destinations=[fallback_dir, root / "mirror"],
                fallback_dirs=[fallback_dir],
            )

            self.assertTrue((root / "mirror" / "answer_failure_groups_by_condition.pdf").exists())

    def test_export_agent_analysis_results_copies_summaries_and_skips_runtime_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            agent_root = root / "agent_analysis"
            wanted = [
                agent_root / "trajectory_pair_analysis" / "trajectory_pair_summary.csv",
                agent_root / "trajectory_pair_analysis" / "trajectory_pair_summary.json",
                agent_root / "trajectory_pair_analysis" / "trajectory_pair_label_breakdown.csv",
                agent_root / "plan_default_analysis" / "figures" / "plan_default_similarity_by_benchmark_model.pdf",
                agent_root / "follow_plan_analysis" / "summary" / "plan_following_summary.csv",
            ]
            skipped = [
                agent_root / "trajectory_pair_analysis" / "trajectory_pair_journal.jsonl",
                agent_root / "trajectory_pair_analysis" / "tmp" / "row.json",
                agent_root / "plan_default_analysis" / "logs" / "raw.csv",
                agent_root / ".DS_Store",
            ]
            for path in wanted + skipped:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            copied = export_agent_analysis_results(
                agent_analysis_root=agent_root,
                destination=root / "paper_figures" / "agent_analysis",
            )

            copied_rel = {path.relative_to(root / "paper_figures" / "agent_analysis") for path in copied}
            self.assertIn(Path("trajectory_pair_analysis/trajectory_pair_summary.csv"), copied_rel)
            self.assertIn(Path("plan_default_analysis/figures/plan_default_similarity_by_benchmark_model.pdf"), copied_rel)
            self.assertTrue((root / "paper_figures" / "agent_analysis" / "follow_plan_analysis/summary/plan_following_summary.csv").exists())
            self.assertFalse((root / "paper_figures" / "agent_analysis" / "trajectory_pair_analysis/trajectory_pair_journal.jsonl").exists())
            self.assertFalse((root / "paper_figures" / "agent_analysis" / "trajectory_pair_analysis/tmp/row.json").exists())
            self.assertFalse((root / "paper_figures" / "agent_analysis" / ".DS_Store").exists())


if __name__ == "__main__":
    unittest.main()

"""Cover the two things the tier-ablation artifacts break silently on.

The axis -> label mapping decides which directory on disk each printed bar or
row is read from. Get it wrong and the artifact still renders, with the wrong
numbers under the right names -- there is no error to notice. Likewise the
delta arithmetic: the figure's deltas are against each panel's own baseline,
the table's are against the model's reference cell, and swapping the two
produces a plausible-looking artifact that says something different.

So both are pinned against on-disk fixtures rather than against the production
trees, which are gitignored.
"""

import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.paper.tier_ablation import (
    AXIS_PANELS,
    METRIC_COLUMNS,
    MIN_COMPLETE,
    PANEL_SLOTS,
    REFERENCE_AXES,
    ROUND_TREES,
    TABLE_CONDITIONS,
    build_panel_cells,
    build_table_rows,
    load_cell,
    render_ablation_table,
    render_ablation_table_text,
    resolve_metric,
    two_sigma_threshold,
)
from sana_analysis.variants import try_parse_variant

MODEL = "openai_gpt-5.4-nano"

# The canonical encoding of each grid cell, spelled out independently of the
# module under test so a change to the mapping has to be made twice.
VARIANT_NAMES = {
    ("ideal", "ideal", "ideal"): "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    ("naive", "ideal", "ideal"): "search_naive__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    ("standard", "ideal", "ideal"): "search_standard__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    ("preloaded", "ideal", "ideal"): "search_preloaded__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    ("ideal", "naive", "ideal"): "search_ideal__plan_naive__compute_ideal__results_rich__k5__skills_off",
    ("ideal", "standard", "ideal"): "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off",
    ("ideal", "ideal", "standard"): "search_ideal__plan_ideal__compute_standard__results_rich__k5__skills_off",
}

# Two variants the luna sweep really has on disk alongside the seven grid
# cells. They must never be picked up by a grid predicate.
DECOY_NAMES = (
    "search_naive__plan_naive__compute_standard__results_rich__k5__skills_off",
    "search_standard__plan_standard__compute_standard__results_rich__k5__skills_off",
)

N_TASKS = 20


def write_cell(root: Path, model: str, variant: str, n_correct: int, *, n_tasks: int = N_TASKS,
               metric: str = "exact_match", cycles: float = 10.0, cost: float = 0.01) -> None:
    """One eval_results.csv with `n_correct` of `n_tasks` rows matching."""
    directory = root / "modes" / model / variant
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", metric, "cycle_count", "cost_usd",
              "total_cost_with_all_subagents_usd", "error"]
    with (directory / "eval_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(n_tasks):
            writer.writerow({
                "task_id": f"t{index}",
                metric: "1" if index < n_correct else "0",
                "cycle_count": cycles,
                "cost_usd": cost,
                "total_cost_with_all_subagents_usd": cost,
                "error": "",
            })


class TestAxisLabelMapping(unittest.TestCase):
    """Every printed label must resolve to the axis triple it claims."""

    EXPECTED_PANELS = {
        "Plan": [
            ("No Plan", ("ideal", "naive", "ideal")),
            ("Default", ("ideal", "standard", "ideal")),
            ("Ideal", ("ideal", "ideal", "ideal")),
        ],
        "Search": [
            ("BM25", ("naive", "ideal", "ideal")),
            ("PNEUMA", ("standard", "ideal", "ideal")),
            ("Ideal", ("ideal", "ideal", "ideal")),
            ("Preloaded", ("preloaded", "ideal", "ideal")),
        ],
        "Data Analysis": [
            ("Standard", ("ideal", "ideal", "standard")),
            ("Ideal", ("ideal", "ideal", "ideal")),
        ],
    }

    EXPECTED_TABLE = [
        ("Reference (all ideal)", ("ideal", "ideal", "ideal")),
        ("Search: BM25", ("naive", "ideal", "ideal")),
        ("Search: PNEUMA", ("standard", "ideal", "ideal")),
        ("Search: Preloaded", ("preloaded", "ideal", "ideal")),
        ("Plan: No Plan", ("ideal", "naive", "ideal")),
        ("Plan: Default", ("ideal", "standard", "ideal")),
        ("Data An.: Standard", ("ideal", "ideal", "standard")),
    ]

    @staticmethod
    def _triple(axes):
        return (axes["search"], axes["plan"], axes["compute"])

    def test_figure_panel_headings_and_bar_order(self):
        self.assertEqual(
            [title for title, _modes in AXIS_PANELS],
            ["Plan", "Search", "Data Analysis"],
        )
        for title, modes in AXIS_PANELS:
            self.assertEqual(
                [(label, self._triple(axes)) for label, axes in modes],
                self.EXPECTED_PANELS[title],
                f"{title} panel bar order or axis mapping changed",
            )

    def test_table_condition_order_and_axis_mapping(self):
        self.assertEqual(
            [(label, self._triple(axes)) for label, axes in TABLE_CONDITIONS],
            self.EXPECTED_TABLE,
        )

    def test_table_columns_are_the_paper_six(self):
        rendered = render_ablation_table([], metric="exact")
        self.assertIn(
            "Model & Condition & $\\bar{x}$ (\\%) & $\\delta$ (pp) & "
            "Rounds/task & \\$/task \\\\",
            rendered,
        )
        self.assertIn("\\begin{tabular}{llrrrr}", rendered)

    def test_every_label_resolves_to_exactly_one_canonical_variant(self):
        """The mapping must agree with the encoder, not just with itself."""
        names = list(VARIANT_NAMES.values()) + list(DECOY_NAMES)
        for label, axes in TABLE_CONDITIONS:
            expected = VARIANT_NAMES[self._triple(axes)]
            matched = [
                name for name in names
                if (variant := try_parse_variant(name)) is not None
                and variant.matches(**axes)
            ]
            self.assertEqual(matched, [expected], f"{label} matched {matched}")

    def test_panel_baseline_is_the_first_bar(self):
        for title, modes in AXIS_PANELS:
            first = self._triple(modes[0][1])
            self.assertNotEqual(
                first, self._triple(REFERENCE_AXES),
                f"{title} panel baseline must be the degraded mode, not the oracle",
            )

    def test_panel_slots_cover_the_widest_axis(self):
        self.assertEqual(PANEL_SLOTS, 4)


class TierFixtureMixin:
    """A three-round tree with a known correct-count per (round, cell)."""

    # Correct answers out of 20, by cell then round. Chosen so every delta in
    # the assertions below is a distinct number.
    COUNTS = {
        ("ideal", "ideal", "ideal"): (14, 16, 12),        # ref -> 70.0%
        ("naive", "ideal", "ideal"): (10, 10, 10),        # BM25 -> 50.0%
        ("standard", "ideal", "ideal"): (12, 11, 13),     # PNEUMA -> 60.0%
        ("preloaded", "ideal", "ideal"): (16, 16, 16),    # preloaded -> 80.0%
        ("ideal", "naive", "ideal"): (8, 9, 7),           # no plan -> 40.0%
        ("ideal", "standard", "ideal"): (9, 9, 9),        # default -> 45.0%
        ("ideal", "ideal", "standard"): (13, 13, 13),     # std DA -> 65.0%
    }

    def build_tree(self, directory: Path, *, rounds=ROUND_TREES, decoys=True) -> Path:
        for round_index, tree in enumerate(rounds):
            root = directory / tree
            for triple, counts in self.COUNTS.items():
                write_cell(root, MODEL, VARIANT_NAMES[triple], counts[round_index])
            if decoys:
                for name in DECOY_NAMES:
                    # Deliberately extreme, so over-matching would be obvious.
                    write_cell(root, MODEL, name, 0)
        return directory


class TestDeltaArithmetic(TierFixtureMixin, unittest.TestCase):
    def test_figure_deltas_are_against_each_panels_own_baseline(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            panels = build_panel_cells(directory, [MODEL], "exact")

        plan = {cell.label: (cell.percent, delta) for cell, delta in panels[(MODEL, "Plan")]}
        # Plan baseline is No Plan at 40.0, NOT the 70.0 reference.
        self.assertAlmostEqual(plan["No Plan"][0], 40.0)
        self.assertAlmostEqual(plan["No Plan"][1], 0.0)
        self.assertAlmostEqual(plan["Default"][1], 5.0)
        self.assertAlmostEqual(plan["Ideal"][1], 30.0)

        search = {cell.label: (cell.percent, delta) for cell, delta in panels[(MODEL, "Search")]}
        # Search baseline is BM25 at 50.0.
        self.assertAlmostEqual(search["BM25"][1], 0.0)
        self.assertAlmostEqual(search["PNEUMA"][1], 10.0)
        self.assertAlmostEqual(search["Ideal"][1], 20.0)
        self.assertAlmostEqual(search["Preloaded"][1], 30.0)

        analysis = {cell.label: (cell.percent, delta)
                    for cell, delta in panels[(MODEL, "Data Analysis")]}
        # Data Analysis baseline is Standard at 65.0.
        self.assertAlmostEqual(analysis["Standard"][1], 0.0)
        self.assertAlmostEqual(analysis["Ideal"][1], 5.0)

    def test_table_deltas_are_against_the_reference_cell(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            rows = build_table_rows(directory, [MODEL], "exact")

        self.assertEqual([row["condition"] for row in rows],
                         [label for label, _axes in TABLE_CONDITIONS])
        by_condition = {row["condition"]: row for row in rows}
        self.assertIsNone(by_condition["Reference (all ideal)"]["delta"])
        self.assertAlmostEqual(by_condition["Reference (all ideal)"]["mean"], 70.0)
        # Every other row is relative to 70.0, not to a per-axis baseline.
        for condition, expected in (
            ("Search: BM25", -20.0),
            ("Search: PNEUMA", -10.0),
            ("Search: Preloaded", 10.0),
            ("Plan: No Plan", -30.0),
            ("Plan: Default", -25.0),
            ("Data An.: Standard", -5.0),
        ):
            self.assertAlmostEqual(by_condition[condition]["delta"], expected, msg=condition)

    def test_reference_delta_renders_as_a_dash_and_others_signed(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            rows = build_table_rows(directory, [MODEL], "exact")

        rendered = render_ablation_table(rows, metric="exact")
        self.assertIn("& Reference (all ideal) & 70.0 & --- &", rendered)
        self.assertIn("& Search: BM25 & 50.0 &", rendered)
        self.assertIn("$-20.0$", rendered)
        self.assertIn("$+10.0$", rendered)

        text = render_ablation_table_text(rows)
        self.assertIn("—", text.splitlines()[2])
        self.assertIn("-20.0", text)

    def test_mean_is_over_rounds_not_over_pooled_tasks(self):
        """A cell whose rounds have different denominators must not pool them.

        Round 3 here completes 16 of 20 tasks. Pooling would weight it less
        than the other two; averaging the round percentages weights it equally,
        which is what the artifacts claim to print.
        """
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            counts = (10, 10, 16)
            sizes = (20, 20, 16)
            for tree, correct, size in zip(ROUND_TREES, counts, sizes):
                write_cell(directory / tree, MODEL, VARIANT_NAMES[("ideal", "ideal", "ideal")],
                           correct, n_tasks=size)
            cell = load_cell(directory, MODEL, "ref", REFERENCE_AXES, "exact")

        self.assertEqual(cell.rounds, 3)
        self.assertEqual(cell.per_round, (50.0, 50.0, 100.0))
        self.assertAlmostEqual(cell.percent, 200.0 / 3)   # not 36/56 = 64.3%
        self.assertEqual(cell.n_tasks, 56)


class TestRoundCompleteness(TierFixtureMixin, unittest.TestCase):
    def test_a_cell_missing_a_round_reports_the_rounds_it_averaged(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            # Simulate the live sweep: round 3's plan cells have not landed.
            for triple in (("ideal", "naive", "ideal"), ("ideal", "standard", "ideal")):
                path = (directory / ROUND_TREES[2] / "modes" / MODEL
                        / VARIANT_NAMES[triple] / "eval_results.csv")
                path.unlink()
            rows = build_table_rows(directory, [MODEL], "exact")
            rendered = render_ablation_table(rows, metric="exact")

        by_condition = {row["condition"]: row for row in rows}
        self.assertEqual(by_condition["Plan: No Plan"]["rounds"], 2)
        self.assertEqual(by_condition["Plan: Default"]["rounds"], 2)
        self.assertEqual(by_condition["Reference (all ideal)"]["rounds"], 3)
        # 8 and 9 of 20 over the two rounds that exist.
        self.assertAlmostEqual(by_condition["Plan: No Plan"]["mean"], 42.5)

        # The short rows carry their n; the complete ones do not.
        self.assertIn("& Plan: No Plan & 42.5$^{2}$ &", rendered)
        self.assertIn("& Reference (all ideal) & 70.0 & --- &", rendered)
        self.assertIn("% rounds=2 n=40 per_round=[40.0, 45.0]", rendered)
        self.assertIn("A superscript on $\\bar{x}$ gives the number of replicate rounds",
                      rendered)

    def test_a_directory_without_a_csv_yet_is_simply_absent(self):
        """A live sweep creates the variant directory before the CSV."""
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            path = (directory / ROUND_TREES[2] / "modes" / MODEL
                    / VARIANT_NAMES[("preloaded", "ideal", "ideal")])
            (path / "eval_results.csv").unlink()
            self.assertTrue(path.is_dir())
            cell = load_cell(directory, MODEL,
                             "Search: Preloaded", TABLE_CONDITIONS[3][1], "exact")

        self.assertEqual(cell.rounds, 2)

    def test_a_cell_with_too_few_completions_is_refused(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_cell(directory / ROUND_TREES[0], MODEL,
                       VARIANT_NAMES[("ideal", "ideal", "ideal")], 5,
                       n_tasks=MIN_COMPLETE - 1)
            self.assertIsNone(
                load_cell(directory, MODEL, "ref", REFERENCE_AXES, "exact")
            )

    def test_model_without_a_reference_cell_is_skipped_not_rebaselined(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            for tree in ROUND_TREES:
                for suffix in ("eval_results.csv",):
                    (directory / tree / "modes" / MODEL
                     / VARIANT_NAMES[("ideal", "ideal", "ideal")] / suffix).unlink()
            self.assertEqual(build_table_rows(directory, [MODEL], "exact"), [])


class TestMetricSelection(TierFixtureMixin, unittest.TestCase):
    def test_exact_when_any_readable_cell_is_unaudited(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            self.assertEqual(resolve_metric(directory, [MODEL]), "exact")

    def test_semantic_only_when_every_cell_is_audited(self):
        with TemporaryDirectory() as tmp:
            directory = self.build_tree(Path(tmp))
            for tree in ROUND_TREES:
                for triple, counts in self.COUNTS.items():
                    write_cell(directory / f"{tree}_semantic", MODEL,
                               VARIANT_NAMES[triple], counts[0],
                               metric="semantic_match")
            self.assertEqual(resolve_metric(directory, [MODEL]), "semantic")
            # One missing audited cell drops the whole run back to exact match,
            # rather than mixing metrics on a shared axis.
            (directory / f"{ROUND_TREES[1]}_semantic" / "modes" / MODEL
             / VARIANT_NAMES[("naive", "ideal", "ideal")] / "eval_results.csv").unlink()
            self.assertEqual(resolve_metric(directory, [MODEL]), "exact")

    def test_metric_columns_are_the_two_eval_csv_fields(self):
        self.assertEqual(METRIC_COLUMNS,
                         {"exact": "exact_match", "semantic": "semantic_match"})


class TestTwoSigmaThreshold(unittest.TestCase):
    def test_threshold_ignores_cells_run_once(self):
        rows = [{"sd": 3.0}, {"sd": 5.0}, {"sd": None}]
        self.assertAlmostEqual(two_sigma_threshold(rows), 8.0)

    def test_no_threshold_when_nothing_replicated(self):
        self.assertIsNone(two_sigma_threshold([{"sd": None}]))


if __name__ == "__main__":
    unittest.main()

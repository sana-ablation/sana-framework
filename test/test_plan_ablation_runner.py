import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis.metrics import plan_ablation_analysis
from sana_analysis.paper.plan_ablation_figure import PLAN_D_AXES, PLAN_I_AXES


PLAN_D_VARIANT = "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off"
PLAN_I_VARIANT = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
OTHER_VARIANT = "search_naive__plan_ideal__compute_ideal__results_rich__k5__skills_off"

D_PLAN = "Search for the NY Lotto file, compute BVAL by year, take the max."
I_PLAN = "Query nylotto.csv, compute BVAL per year, return the max with earliest-year tie-break."


def _write_log(path: Path, *, runner_model: str, tool: str, plan_text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"NEW TASK: {runner_model}", "--- Turn 1"]
    if plan_text is not None:
        lines.append(f'calling {tool}({{"plan_text": {json.dumps(plan_text)}}})')
    lines.append("Answer submitted")
    path.write_text("\n".join(lines) + "\n")


def _write_log_root(root: Path, *, tasks=("k-1-d-1/task_1.log",), missing_d=()) -> Path:
    log_root = root / "logs"
    for task in tasks:
        _write_log(
            log_root / "modes" / "openai_gpt-5-mini" / PLAN_D_VARIANT / task,
            runner_model="gpt-5-mini",
            tool="plan",
            plan_text=None if task in missing_d else D_PLAN,
        )
        _write_log(
            log_root / "modes" / "openai_gpt-5-mini" / PLAN_I_VARIANT / task,
            runner_model="gpt-5-mini",
            tool="plan_ideal",
            plan_text=I_PLAN,
        )
        # A third arm that must never be paired into the ablation.
        _write_log(
            log_root / "modes" / "openai_gpt-5-mini" / OTHER_VARIANT / task,
            runner_model="gpt-5-mini",
            tool="plan_ideal",
            plan_text=I_PLAN,
        )
    return log_root


class TestPlanModeResolution(unittest.TestCase):
    def test_resolves_the_two_arms_from_gen4_names_on_disk(self):
        with TemporaryDirectory() as tmp:
            log_root = _write_log_root(Path(tmp))
            plan_d, plan_i = plan_ablation_analysis.resolve_plan_modes(log_root)
            self.assertEqual(plan_d, PLAN_D_VARIANT)
            self.assertEqual(plan_i, PLAN_I_VARIANT)

    def test_axes_are_the_figure_s_axes_not_a_second_copy(self):
        self.assertIs(plan_ablation_analysis.PLAN_D_AXES, PLAN_D_AXES)
        self.assertIs(plan_ablation_analysis.PLAN_I_AXES, PLAN_I_AXES)

    def test_explicit_modes_override_resolution(self):
        with TemporaryDirectory() as tmp:
            log_root = _write_log_root(Path(tmp))
            plan_d, plan_i = plan_ablation_analysis.resolve_plan_modes(
                log_root, plan_d_mode=OTHER_VARIANT
            )
            self.assertEqual(plan_d, OTHER_VARIANT)
            self.assertEqual(plan_i, PLAN_I_VARIANT)

    def test_a_tree_with_no_standard_plan_arm_says_what_it_saw(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_log(
                root / "logs" / "modes" / "openai_gpt-5-mini" / PLAN_I_VARIANT / "k-1-d-1/task_1.log",
                runner_model="gpt-5-mini",
                tool="plan_ideal",
                plan_text=I_PLAN,
            )
            with self.assertRaises(ValueError) as caught:
                plan_ablation_analysis.resolve_plan_modes(root / "logs")
            message = str(caught.exception)
            self.assertIn("'plan': 'standard'", message)
            self.assertIn(PLAN_I_VARIANT, message)


class TestBuildPlanPairRows(unittest.TestCase):
    def test_pairs_the_two_arms_and_extracts_both_plans(self):
        with TemporaryDirectory() as tmp:
            log_root = _write_log_root(Path(tmp))
            rows = plan_ablation_analysis.build_plan_pair_rows(log_root)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["plan_d_mode"], PLAN_D_VARIANT)
            self.assertEqual(row["plan_i_mode"], PLAN_I_VARIANT)
            self.assertEqual(row["model_variant"], "openai_gpt-5-mini")
            self.assertEqual(row["runner_model"], "gpt-5-mini")
            self.assertEqual(row["task_id"], "k-1-d-1/task_1.log")
            self.assertEqual(row["plan_d_text"], D_PLAN)
            self.assertEqual(row["plan_i_text"], I_PLAN)
            self.assertEqual(row["missing_plan_type"], "")
            self.assertEqual(row["prefill_plan_similarity"], "")

    def test_a_missing_default_plan_is_prefilled_not_judged(self):
        with TemporaryDirectory() as tmp:
            log_root = _write_log_root(Path(tmp), missing_d=("k-1-d-1/task_1.log",))
            row = plan_ablation_analysis.build_plan_pair_rows(log_root)[0]
            self.assertEqual(row["missing_plan_type"], "missing_plan_d")
            self.assertEqual(row["prefill_plan_similarity"], "not_comparable")
            self.assertIn("No explicit plan() call", row["prefill_similarity_reason"])

    def test_field_names_the_figure_reads_are_unchanged(self):
        for field in ("plan_d_mode", "plan_i_mode", "missing_plan_type", "model_variant", "runner_model"):
            self.assertIn(field, plan_ablation_analysis.PAIR_COLUMNS)


if __name__ == "__main__":
    unittest.main()

import csv
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis.metrics import plan_ablation_analysis
from sana_analysis.paper import plan_ablation_figure
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


class TestVocabularyCannotDrift(unittest.TestCase):
    def test_labels_plus_derived_equal_the_figure_buckets(self):
        """The figure's seven buckets are the six judged labels plus no_plan.

        `no_plan` is never a judgment: `_normalize_plan_similarity` derives it
        from `missing_plan_type == "missing_plan_d"`. Offering it to a judge
        would invite a label the prepare stage is supposed to establish.
        """
        self.assertEqual(
            set(plan_ablation_analysis.PLAN_SIMILARITY_LABELS) | plan_ablation_analysis.DERIVED_LABELS,
            set(plan_ablation_figure.BUCKET_ORDER),
        )
        self.assertEqual(plan_ablation_analysis.DERIVED_LABELS, frozenset({"no_plan"}))

    def test_every_label_the_prompt_offers_is_in_the_vocabulary(self):
        with TemporaryDirectory() as tmp:
            log_root = _write_log_root(Path(tmp))
            row = plan_ablation_analysis.build_plan_pair_rows(log_root)[0]
            prompt = plan_ablation_analysis.build_judge_prompt(row)
            offered = set(re.findall(r"^- ([a-z_]+):", prompt, flags=re.MULTILINE))
            self.assertTrue(offered, "prompt offered no labels at all")
            self.assertEqual(offered, set(plan_ablation_analysis.PLAN_SIMILARITY_LABELS))

    def test_every_label_has_a_definition(self):
        self.assertEqual(
            sorted(plan_ablation_analysis.PLAN_SIMILARITY_DEFINITIONS),
            sorted(plan_ablation_analysis.PLAN_SIMILARITY_LABELS),
        )

    def test_the_prompt_carries_both_plans(self):
        with TemporaryDirectory() as tmp:
            log_root = _write_log_root(Path(tmp))
            row = plan_ablation_analysis.build_plan_pair_rows(log_root)[0]
            prompt = plan_ablation_analysis.build_judge_prompt(row)
            self.assertIn(D_PLAN, prompt)
            self.assertIn(I_PLAN, prompt)


class TestValidateJudgePayload(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "plan_similarity": "similar",
            "divergence_type": "none",
            "similarity_reason": "Same source family and the same BVAL computation.",
            "aligned_steps": "find file; compute BVAL by year; take max",
            "divergent_steps": "",
            "provenance_note": "NEW TASK: gpt-5-mini in both logs",
            "audit_status": "complete",
        }
        payload.update(overrides)
        return payload

    def test_a_good_payload_has_no_errors(self):
        self.assertEqual(plan_ablation_analysis.validate_judge_payload(self._payload()), [])

    def test_a_derived_label_is_rejected(self):
        errors = plan_ablation_analysis.validate_judge_payload(self._payload(plan_similarity="no_plan"))
        self.assertTrue(any("plan_similarity" in error for error in errors))

    def test_an_invented_label_is_rejected(self):
        errors = plan_ablation_analysis.validate_judge_payload(self._payload(plan_similarity="quite_close"))
        self.assertTrue(any("plan_similarity" in error for error in errors))

    def test_a_blank_reason_is_rejected(self):
        errors = plan_ablation_analysis.validate_judge_payload(self._payload(similarity_reason="  "))
        self.assertTrue(any("similarity_reason" in error for error in errors))


class _StubJudge:
    """Stands in for call_judge_model. No test may reach a real API."""

    def __init__(self, label="missing_details"):
        self.label = label
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps({
            "plan_similarity": self.label,
            "divergence_type": "period_filter",
            "similarity_reason": "The standard plan omits the earliest-year tie-break.",
            "aligned_steps": "locate file; compute BVAL by year",
            "divergent_steps": "tie-break",
            "provenance_note": "NEW TASK: gpt-5-mini in both logs",
            "audit_status": "complete",
        })


class TestJudgeEndToEnd(unittest.TestCase):
    def test_runner_produces_rows_the_figure_accepts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = _write_log_root(root)
            out_dir = root / "agent_analysis" / "plan_default_analysis"
            rows = plan_ablation_analysis.build_plan_pair_rows(log_root)
            judge = _StubJudge()

            judged = plan_ablation_analysis.judge_pending_rows(
                rows,
                output_dir=out_dir,
                log_root_name=log_root.name,
                repo_root=root,
                backend="openai",
                model="stub",
                reasoning_effort="low",
                limit=0,
                timeout=10,
                tmp_root=root / "tmp",
                journal_path=root / "journal.jsonl",
                max_retries=1,
                call=judge,
            )

            self.assertEqual(judged, 1)
            self.assertEqual(rows[0]["plan_similarity"], "missing_details")
            self.assertEqual(rows[0]["audit_status"], "complete")

            written = sorted(out_dir.rglob("plan_similarity.csv"))
            self.assertEqual(len(written), 1)
            self.assertEqual(
                written[0].relative_to(out_dir).as_posix(),
                f"logs/modes/openai_gpt-5-mini/{PLAN_D_VARIANT}__vs__{PLAN_I_VARIANT}/plan_similarity.csv",
            )

            csv_rows = list(csv.DictReader(written[0].open(newline="")))
            summary = plan_ablation_figure._summarize_rows(
                [dict(row, benchmark="lakeqa") for row in csv_rows]
            )
            self.assertEqual(
                summary[("lakeqa", "openai_gpt-5-mini")]["counts"]["missing_details"], 1
            )

    def test_prefilled_rows_are_never_sent_to_the_judge(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = _write_log_root(root, missing_d=("k-1-d-1/task_1.log",))
            rows = plan_ablation_analysis.build_plan_pair_rows(log_root)
            judge = _StubJudge()

            judged = plan_ablation_analysis.judge_pending_rows(
                rows,
                output_dir=root / "out",
                log_root_name=log_root.name,
                repo_root=root,
                backend="openai",
                model="stub",
                reasoning_effort="low",
                limit=0,
                timeout=10,
                tmp_root=root / "tmp",
                journal_path=root / "journal.jsonl",
                max_retries=1,
                call=judge,
            )

            self.assertEqual(judged, 0)
            self.assertEqual(judge.prompts, [])
            self.assertEqual(rows[0]["plan_similarity"], "not_comparable")
            self.assertEqual(rows[0]["audit_status"], "prefilled")

    def test_rejudging_skips_rows_already_complete(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_root = _write_log_root(root)
            rows = plan_ablation_analysis.build_plan_pair_rows(log_root)
            kwargs = dict(
                output_dir=root / "out",
                log_root_name=log_root.name,
                repo_root=root,
                backend="openai",
                model="stub",
                reasoning_effort="low",
                limit=0,
                timeout=10,
                tmp_root=root / "tmp",
                journal_path=root / "journal.jsonl",
                max_retries=1,
            )
            first = _StubJudge()
            plan_ablation_analysis.judge_pending_rows(rows, call=first, **kwargs)
            second = _StubJudge()
            self.assertEqual(plan_ablation_analysis.judge_pending_rows(rows, call=second, **kwargs), 0)
            self.assertEqual(second.prompts, [])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sana_evaluation import cli


def _load_orchestration_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "sana_evaluation" / "runner" / "orchestration.py"

    fake_pkg = types.ModuleType("sana_evaluation")
    # orchestration.py lives one level deeper than run_eval.py used to
    # (sana_evaluation/runner/ instead of sana_evaluation/), so the fake
    # package path must point at the top-level sana_evaluation/ directory —
    # that's what its sibling imports (config, prompting.compose,
    # runner.reporting) resolve against.
    fake_pkg.__path__ = [str(module_path.parents[1])]

    # Faked (not left to load for real) so that no genuine
    # sana_evaluation.runner import happens here. A real import — even of
    # the empty runner/__init__.py — gets cached in sys.modules and never
    # cleaned up by the restore below (only the four keys tracked in `saved`
    # are), which then short-circuits and skews the ordering of the *next*
    # real import of sana_evaluation.runner.modes (its circular import with
    # runner.agent only resolves in one direction: agent-first, which is how
    # sana_evaluation/__init__.py naturally does it — a leftover cached
    # `sana_evaluation.runner` bypasses that and enters modes-first).
    fake_runner_pkg = types.ModuleType("sana_evaluation.runner")

    fake_reporting = types.ModuleType("sana_evaluation.runner.reporting")
    fake_reporting.write_main_csv = lambda *_args, **_kwargs: None
    fake_reporting.write_tools_csv = lambda *_args, **_kwargs: None
    fake_reporting.write_agent_results_jsonl = lambda *_args, **_kwargs: None
    fake_reporting.print_comparison_table = lambda *_args, **_kwargs: None

    fake_config = types.ModuleType("sana_evaluation.config")
    fake_config.AgentConfig = object
    fake_config.ConditionConfig = object
    fake_config.RunConfig = object

    saved = {
        "sana_evaluation": sys.modules.get("sana_evaluation"),
        "sana_evaluation.runner": sys.modules.get("sana_evaluation.runner"),
        "sana_evaluation.runner.reporting": sys.modules.get("sana_evaluation.runner.reporting"),
        "sana_evaluation.config": sys.modules.get("sana_evaluation.config"),
    }
    sys.modules["sana_evaluation"] = fake_pkg
    sys.modules["sana_evaluation.runner"] = fake_runner_pkg
    sys.modules["sana_evaluation.runner.reporting"] = fake_reporting
    sys.modules["sana_evaluation.config"] = fake_config
    try:
        spec = importlib.util.spec_from_file_location("_test_orchestration_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


orchestration = _load_orchestration_module()


class CliModeHelperTests(unittest.TestCase):
    def test_resolve_mode_axes_uses_defaults(self):
        self.assertEqual(
            cli._resolve_mode_axes(
                search_tool=None,
                search_results=None,
                plan=None,
            ),
            ("standard", "rich", "standard", "standard"),
        )

    def test_resolve_mode_axes_honors_explicit_override(self):
        self.assertEqual(
            cli._resolve_mode_axes(
                search_tool="naive",
                search_results=None,
                plan=None,
            ),
            ("naive", "rich", "standard", "standard"),
        )

    def test_variant_condition_label_uses_explicit_mode_names(self):
        label = cli._variant_condition_label(
            search_tool="ideal",
            search_results="naive",
            plan="naive",
            k=5,
            search_calls=2,
        )

        self.assertEqual(
            label,
            "search_ideal__plan_naive__compute_standard__results_minimal__k5__sc2__skills_off",
        )

    def test_variant_condition_label_uses_preloaded_mode_name(self):
        label = cli._variant_condition_label(
            search_tool="preloaded",
            search_results="ideal",
            plan="standard",
            k=None,
            search_calls=None,
        )

        self.assertEqual(label, "search_preloaded__plan_standard__compute_standard__results_rich__skills_off")

    def test_variant_condition_label_appends_search_flags(self):
        label = cli._variant_condition_label(
            search_tool="ideal",
            search_results="ideal",
            plan="standard",
            k=None,
            search_calls=None,
            search_free=True,
        )

        self.assertEqual(
            label,
            "search_ideal__plan_standard__compute_standard__results_rich__free__skills_off",
        )

    def test_variant_condition_label_appends_ideal_computation_axis(self):
        label = cli._variant_condition_label(
            search_tool="preloaded",
            search_results="ideal",
            plan="standard",
            computation_tool="ideal",
        )

        self.assertEqual(label, "search_preloaded__plan_standard__compute_ideal__results_rich__skills_off")

    def test_variant_condition_label_appends_plan_skills_when_enabled(self):
        label = cli._variant_condition_label(
            search_tool="preloaded",
            search_results="ideal",
            plan="standard",
            plan_skills_enabled=True,
        )

        self.assertEqual(label, "search_preloaded__plan_standard__compute_standard__results_rich__skills_on")

    def test_benchmark_choices_include_supported_external_benchmarks(self):
        self.assertIn("kramabench", cli.BENCHMARKS)
        self.assertIn("lakeqa", cli.BENCHMARKS)

    def test_kramabench_defaults_to_kramabench_task_set(self):
        self.assertEqual(
            cli._default_task_set_for_benchmark("kramabench"),
            "benchmarks/kramabench/tasks-mini/tasks",
        )

    def test_lakeqa_defaults_to_tasks_mini_task_set(self):
        self.assertEqual(
            cli._default_task_set_for_benchmark("lakeqa"),
            "benchmarks/lakeqa/tasks-mini/tasks",
        )

    def test_skills_on_rejects_naive_plan_axis(self):
        with self.assertRaisesRegex(ValueError, "--skills on requires --plan standard or --plan ideal"):
            cli._validate_axis_combination(plan="naive", skills="on")

    def test_configure_ideal_subagent_models_defaults_to_main_model(self):
        with patch.dict(os.environ, {}, clear=True):
            cli._configure_ideal_subagent_models(
                main_model_name="gemini/gemini-3.1-flash-lite",
                selector_model=None,
                repair_model=None,
            )

            self.assertEqual(os.environ["SANA_MAIN_MODEL"], "gemini/gemini-3.1-flash-lite")
            self.assertEqual(os.environ["SANA_IDEAL_SUBAGENT_MODEL"], "gemini/gemini-3.1-flash-lite")
            self.assertNotIn("SANA_SEARCH_IDEAL_SUBAGENT_MODEL", os.environ)
            self.assertNotIn("SANA_SEMANTIC_IDEAL_SUBAGENT_MODEL", os.environ)
            self.assertNotIn("SANA_REPAIR_IDEAL_SUBAGENT_MODEL", os.environ)

    def test_configure_ideal_subagent_models_honors_selector_and_repair_overrides(self):
        with patch.dict(os.environ, {}, clear=True):
            cli._configure_ideal_subagent_models(
                main_model_name="gemini/gemini-3.1-flash-lite",
                selector_model="openai/gpt-5-mini",
                repair_model="openai/gpt-5.4",
            )

            self.assertEqual(os.environ["SANA_IDEAL_SUBAGENT_MODEL"], "gemini/gemini-3.1-flash-lite")
            self.assertEqual(os.environ["SANA_SEARCH_IDEAL_SUBAGENT_MODEL"], "openai/gpt-5-mini")
            self.assertEqual(os.environ["SANA_SEMANTIC_IDEAL_SUBAGENT_MODEL"], "openai/gpt-5-mini")
            self.assertEqual(os.environ["SANA_REPAIR_IDEAL_SUBAGENT_MODEL"], "openai/gpt-5.4")

    def test_mode_results_dir_does_not_append_model_twice(self):
        agent_config = types.SimpleNamespace(
            model_name="openai/gpt-5.2",
            model_id="unused",
            extra_model_kwargs={"reasoning_effort": "xhigh"},
        )
        run_config = types.SimpleNamespace(
            results_output_dir="results",
            condition_config=types.SimpleNamespace(
                condition="modes/openai_gpt-5.2-xhigh/search_ideal__plan_ideal__compute_standard__results_rich__k5",
            ),
        )

        self.assertEqual(
            orchestration._results_dir(run_config, agent_config).replace("\\", "/"),
            "results/modes/openai_gpt-5.2-xhigh/search_ideal__plan_ideal__compute_standard__results_rich__k5",
        )

    def test_standard_results_dir_layout_is_unchanged(self):
        agent_config = types.SimpleNamespace(
            model_name="openai/gpt-5.2",
            model_id="unused",
            extra_model_kwargs={},
        )
        run_config = types.SimpleNamespace(
            results_output_dir="results",
            condition_config=types.SimpleNamespace(condition="baseline"),
        )

        self.assertEqual(
            orchestration._results_dir(run_config, agent_config).replace("\\", "/"),
            "results/baseline/openai_gpt-5.2",
        )


if __name__ == "__main__":
    unittest.main()

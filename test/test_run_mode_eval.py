import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

def _load_run_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "sana_evaluation" / "run_eval.py"

    fake_pkg = types.ModuleType("sana_evaluation")
    fake_pkg.__path__ = [str(module_path.parent)]

    fake_agent = types.ModuleType("sana_evaluation.agent")
    fake_agent.BatchRunner = object

    fake_config = types.ModuleType("sana_evaluation.config")
    fake_config.AgentConfig = object
    fake_config.ConditionConfig = object
    fake_config.RunConfig = object

    saved = {
        "sana_evaluation": sys.modules.get("sana_evaluation"),
        "sana_evaluation.agent": sys.modules.get("sana_evaluation.agent"),
        "sana_evaluation.config": sys.modules.get("sana_evaluation.config"),
    }
    sys.modules["sana_evaluation"] = fake_pkg
    sys.modules["sana_evaluation.agent"] = fake_agent
    sys.modules["sana_evaluation.config"] = fake_config
    try:
        spec = importlib.util.spec_from_file_location("_test_run_eval_module", module_path)
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


def _load_run_mode_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "sana_evaluation" / "run_mode_eval.py"

    fake_pkg = types.ModuleType("sana_evaluation")
    fake_pkg.__path__ = [str(module_path.parent)]

    fake_base_eval = types.ModuleType("sana_evaluation.run_eval")
    fake_base_eval.BatchRunner = object
    fake_base_eval._display_name = lambda agent_config: "openai_gpt-5.2-xhigh"
    fake_base_eval._results_dir = lambda run_config, agent_config: "unused"
    fake_base_eval.find_all_task_dirs = lambda *_args, **_kwargs: []

    fake_agent_with_mode = types.ModuleType("sana_evaluation.agent_with_mode")
    fake_agent_with_mode.BatchRunner = object

    fake_config = types.ModuleType("sana_evaluation.config")
    fake_config.AgentConfig = object
    fake_config.ConditionConfig = object
    fake_config.RunConfig = object

    saved = {
        "sana_evaluation": sys.modules.get("sana_evaluation"),
        "sana_evaluation.run_eval": sys.modules.get("sana_evaluation.run_eval"),
        "sana_evaluation.agent_with_mode": sys.modules.get("sana_evaluation.agent_with_mode"),
        "sana_evaluation.config": sys.modules.get("sana_evaluation.config"),
    }
    sys.modules["sana_evaluation"] = fake_pkg
    sys.modules["sana_evaluation.run_eval"] = fake_base_eval
    sys.modules["sana_evaluation.agent_with_mode"] = fake_agent_with_mode
    sys.modules["sana_evaluation.config"] = fake_config
    try:
        spec = importlib.util.spec_from_file_location("_test_run_mode_eval_module", module_path)
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


run_eval = _load_run_eval_module()
run_mode_eval = _load_run_mode_eval_module()


class RunModeEvalTests(unittest.TestCase):
    def test_resolve_mode_axes_uses_defaults(self):
        self.assertEqual(
            run_mode_eval._resolve_mode_axes(
                search_tool=None,
                search_results=None,
                profile=None,
            ),
            ("standard", "naive", "standard", "standard"),
        )

    def test_resolve_mode_axes_honors_explicit_override(self):
        self.assertEqual(
            run_mode_eval._resolve_mode_axes(
                search_tool="naive",
                search_results=None,
                profile=None,
            ),
            ("naive", "naive", "standard", "standard"),
        )

    def test_variant_condition_label_uses_explicit_mode_names(self):
        label = run_mode_eval._variant_condition_label(
            search_tool="ideal",
            search_results="naive",
            profile="naive",
            k=5,
            search_calls=2,
        )

        self.assertEqual(
            label,
            "search_ideal__results_naive__profile_naive__compute_standard__k5__sc2__skills_off",
        )

    def test_variant_condition_label_uses_preloaded_mode_name(self):
        label = run_mode_eval._variant_condition_label(
            search_tool="preloaded",
            search_results="ideal",
            profile="standard",
            k=None,
            search_calls=None,
        )

        self.assertEqual(label, "search_preloaded__results_ideal__profile_standard__compute_standard__skills_off")

    def test_variant_condition_label_appends_search_flags(self):
        label = run_mode_eval._variant_condition_label(
            search_tool="ideal",
            search_results="ideal",
            profile="standard",
            k=None,
            search_calls=None,
            search_free=True,
            search_lessguide=True,
        )

        self.assertEqual(
            label,
            "search_ideal__results_ideal__profile_standard__compute_standard__free__lessguide__skills_off",
        )

    def test_variant_condition_label_appends_ideal_computation_axis(self):
        label = run_mode_eval._variant_condition_label(
            search_tool="preloaded",
            search_results="ideal",
            profile="standard",
            computation_tool="ideal",
        )

        self.assertEqual(label, "search_preloaded__results_ideal__profile_standard__compute_ideal__skills_off")

    def test_variant_condition_label_appends_profile_skills_when_enabled(self):
        label = run_mode_eval._variant_condition_label(
            search_tool="preloaded",
            search_results="ideal",
            profile="standard",
            profile_skills_enabled=True,
        )

        self.assertEqual(label, "search_preloaded__results_ideal__profile_standard__compute_standard__skills_on")

    def test_benchmark_choices_include_supported_external_benchmarks(self):
        self.assertIn("kramabench", run_mode_eval._BENCHMARK_CHOICES)
        self.assertIn("lakeqa", run_mode_eval._BENCHMARK_CHOICES)

    def test_kramabench_defaults_to_kramabench_task_set(self):
        self.assertEqual(
            run_mode_eval._default_task_set_for_benchmark("kramabench"),
            "benchmarks/kramabench/tasks-mini/tasks",
        )

    def test_lakeqa_defaults_to_tasks_mini_task_set(self):
        self.assertEqual(
            run_mode_eval._default_task_set_for_benchmark("lakeqa"),
            "benchmarks/lakeqa/tasks-mini/tasks",
        )

    def test_skills_on_rejects_naive_profile_axis(self):
        with self.assertRaisesRegex(ValueError, "--skills on requires --profile standard or --profile ideal"):
            run_mode_eval._validate_axis_combination(profile="naive", skills="on")

    def test_configure_ideal_subagent_models_defaults_to_main_model(self):
        with patch.dict(os.environ, {}, clear=True):
            run_mode_eval._configure_ideal_subagent_models(
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
            run_mode_eval._configure_ideal_subagent_models(
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
                condition="modes/openai_gpt-5.2-xhigh/search_ideal__results_ideal__profile_ideal__compute_standard__k5",
            ),
        )

        self.assertEqual(
            run_eval._results_dir(run_config, agent_config).replace("\\", "/"),
            "results/modes/openai_gpt-5.2-xhigh/search_ideal__results_ideal__profile_ideal__compute_standard__k5",
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
            run_eval._results_dir(run_config, agent_config).replace("\\", "/"),
            "results/baseline/openai_gpt-5.2",
        )


if __name__ == "__main__":
    unittest.main()

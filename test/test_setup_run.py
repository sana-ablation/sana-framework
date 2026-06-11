import contextlib
import importlib.util
import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _load_setup_run_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "sana_evaluation" / "setup_run.py"
    spec = importlib.util.spec_from_file_location("_test_setup_run_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


setup_run = _load_setup_run_module()


class _FakeRunner:
    def __init__(self) -> None:
        self.command = None
        self.kwargs = None
        self.gemini_api_key = None

    def __call__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        return None


class SetupRunTests(unittest.TestCase):
    def _write_smoke_fixture(self, repo_root: Path) -> None:
        (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
        for task_dir in ("k-1-d-1", "k-5-d-4"):
            target = repo_root / "benchmarks" / "lakeqa" / "tasks-mini" / "tasks" / task_dir
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text("{}")
            (target / "task_2.json").write_text("{}")

    def _write_kramabench_smoke_fixture(self, repo_root: Path) -> None:
        (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
        target = repo_root / "benchmarks" / "kramabench" / "tasks-mini" / "tasks" / "k-2-d-1-s-1"
        target.mkdir(parents=True, exist_ok=True)
        (target / "task_1.json").write_text("{}")
        (target / "task_2.json").write_text("{}")

    def test_smoke_builds_expected_command_and_normalizes_model_alias(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            self._write_kramabench_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                command = setup_run.run(
                    [
                        "smoke",
                        "--search",
                        "ideal",
                        "--results",
                        "ideal",
                        "--profile",
                        "ideal",
                        "--k",
                        "5",
                        "--model",
                        "gpt5.2",
                        "--reasoning-effort",
                        "xhigh",
                        "--openai-prompt-cache-key",
                        "assistant-v3:tools-v1",
                        "--openai-prompt-cache-retention",
                        "24h",
                        "--benchmark",
                        "kramabench",
                        "--verbose",
                        "--db",
                        "lance_data",
                    ],
                    runner=fake_runner,
                    cwd=repo_root,
                )

            self.assertEqual(command, fake_runner.command)
            self.assertEqual(command[1:3], ["-m", "sana_evaluation.run_mode_eval"])
            self.assertIn("--search_tool", command)
            self.assertIn("ideal", command)
            self.assertEqual(command[command.index("--model-name") + 1], "openai/gpt-5.2")
            self.assertEqual(command[command.index("--db-path") + 1], "lance_data")
            self.assertEqual(
                command[command.index("--task-dir") + 1],
                "benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1",
            )
            self.assertEqual(command[command.index("--tasks-per-dir") + 1], "2")
            self.assertEqual(command[command.index("--logs-output-dir") + 1], "log-kramabench")
            self.assertEqual(command[command.index("--results-output-dir") + 1], "results-kramabench")
            self.assertEqual(command[command.index("--reasoning-effort") + 1], "xhigh")
            self.assertEqual(command[command.index("--openai-prompt-cache-key") + 1], "assistant-v3:tools-v1")
            self.assertEqual(command[command.index("--openai-prompt-cache-retention") + 1], "24h")
            self.assertEqual(command[command.index("--benchmark") + 1], "kramabench")
            self.assertIn("--verbose", command)
            self.assertEqual(fake_runner.kwargs, {"check": True, "cwd": str(repo_root)})
            self.assertIn(
                "Task scope: benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1 (first 2 tasks)",
                stdout.getvalue(),
            )

    def test_kramabench_smoke_defaults_to_kramabench_task_dir(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_kramabench_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                command = setup_run.run(
                    [
                        "smoke",
                        "--benchmark",
                        "kramabench",
                        "--search",
                        "ideal",
                        "--results",
                        "naive",
                        "--profile",
                        "standard",
                        "--model",
                        "gpt-5.4-nano",
                        "--db",
                        "lance_data",
                    ],
                    runner=fake_runner,
                    cwd=repo_root,
                )

            self.assertEqual(
                command[command.index("--task-dir") + 1],
                "benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1",
            )
            self.assertIn(
                "Task scope: benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1 (first 2 tasks)",
                stdout.getvalue(),
            )

    def test_setup_run_loads_dotenv_from_repo_root_before_spawning_runner(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            (repo_root / ".env").write_text("GEMINI_API_KEY=temp-gemini-key\n")
            fake_runner = _FakeRunner()

            with patch.dict(os.environ, {}, clear=True):
                setup_run.run(
                    [
                        "smoke",
                        "--model",
                        "gemini/gemini-3.1-flash-lite",
                        "--db",
                        "lance_data",
                    ],
                    runner=fake_runner,
                    cwd=repo_root,
                )

            self.assertEqual(fake_runner.gemini_api_key, "temp-gemini-key")

    def test_full_builds_expected_command(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                command = setup_run.run(
                    [
                        "full",
                        "--no-continue",
                        "--search",
                        "ideal",
                        "--results",
                        "ideal",
                        "--profile",
                        "ideal",
                        "--k",
                        "5",
                        "--model",
                        "openai/gpt-5.2",
                        "--db",
                        "lance_data",
                    ],
                    runner=fake_runner,
                    cwd=repo_root,
                )

            self.assertEqual(command[1:3], ["-m", "sana_evaluation.run_mode_eval"])
            self.assertIn("--all-tasks", command)
            self.assertEqual(command[command.index("--task-set") + 1], "benchmarks/lakeqa/tasks-mini/tasks")
            self.assertEqual(command[command.index("--logs-output-dir") + 1], "logs")
            self.assertEqual(command[command.index("--results-output-dir") + 1], "results")

    def test_kramabench_full_defaults_to_kramabench_output_roots(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_kramabench_base").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "full",
                    "--search",
                    "standard",
                    "--results",
                    "naive",
                    "--profile",
                    "standard",
                    "--benchmark",
                    "kramabench",
                    "--model",
                    "openai/gpt-5.2",
                    "--db",
                    "lance_kramabench_base",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--db-path") + 1], "lance_kramabench_base")
            self.assertEqual(command[command.index("--logs-output-dir") + 1], "log-kramabench")
            self.assertEqual(command[command.index("--results-output-dir") + 1], "results-kramabench")

    def test_kramabench_full_defaults_to_kramabench_task_set(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                command = setup_run.run(
                    [
                        "full",
                        "--no-continue",
                        "--benchmark",
                        "kramabench",
                        "--search",
                        "ideal",
                        "--results",
                        "naive",
                        "--profile",
                        "standard",
                        "--model",
                        "gpt-5.4-nano",
                        "--db",
                        "lance_data",
                    ],
                    runner=fake_runner,
                    cwd=repo_root,
                )

            self.assertEqual(command[command.index("--task-set") + 1], "benchmarks/kramabench/tasks-mini/tasks")
            self.assertIn("Task scope: all tasks under benchmarks/kramabench/tasks-mini/tasks", stdout.getvalue())

    def test_full_continue_alias_and_timeout_passthrough(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "full",
                    "--continue",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "ideal",
                    "--model",
                    "openai/gpt-5.2",
                    "--timeout",
                    "600",
                    "--submit-grace-seconds",
                    "15",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertIn("--task-continue", command)
            self.assertNotIn("--all-tasks", command)
            self.assertEqual(command[command.index("--timeout") + 1], "600")
            self.assertEqual(command[command.index("--submit-grace-seconds") + 1], "15")

    def test_full_defaults_to_ideal_results_verbose_and_continue_with_plans_alias(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_kramabench_infused").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                command = setup_run.run(
                    [
                        "full",
                        "--benchmark",
                        "kramabench",
                        "--search",
                        "ideal",
                        "--plans",
                        "standard",
                        "--compute",
                        "ideal",
                        "--k",
                        "5",
                        "--parallel",
                        "4",
                        "--model",
                        "openai/gpt-5-mini",
                        "--db",
                        "lance_kramabench_infused",
                        "--timeout",
                        "600",
                        "--submit-grace-seconds",
                        "30",
                    ],
                    runner=fake_runner,
                    cwd=repo_root,
                )

            self.assertEqual(command[command.index("--search_tool") + 1], "ideal")
            self.assertEqual(command[command.index("--search_results") + 1], "ideal")
            self.assertEqual(command[command.index("--profile") + 1], "standard")
            self.assertEqual(command[command.index("--computation_tool") + 1], "ideal")
            self.assertIn("--verbose", command)
            self.assertIn("--task-continue", command)
            self.assertNotIn("--all-tasks", command)
            self.assertEqual(command[command.index("--parallel") + 1], "4")
            self.assertEqual(command[command.index("--timeout") + 1], "600")
            self.assertEqual(command[command.index("--submit-grace-seconds") + 1], "30")
            self.assertIn(
                "Task scope: resume pending tasks under benchmarks/kramabench/tasks-mini/tasks",
                stdout.getvalue(),
            )

    def test_full_no_continue_runs_all_tasks_when_resume_default_is_not_wanted(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)

            command = setup_run.run(
                [
                    "full",
                    "--no-continue",
                    "--model",
                    "openai/gpt-5-mini",
                    "--db",
                    "lance_data",
                ],
                runner=_FakeRunner(),
                cwd=repo_root,
            )

            self.assertIn("--all-tasks", command)
            self.assertNotIn("--task-continue", command)

    def test_smoke_defaults_to_ideal_axes_and_verbose(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_kramabench_smoke_fixture(repo_root)
            (repo_root / "lance_kramabench_infused").mkdir(parents=True, exist_ok=True)

            command = setup_run.run(
                [
                    "smoke",
                    "--benchmark",
                    "kramabench",
                    "--model",
                    "openai/gpt-5-mini",
                    "--db",
                    "lance_kramabench_infused",
                ],
                runner=_FakeRunner(),
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--search_tool") + 1], "ideal")
            self.assertEqual(command[command.index("--search_results") + 1], "ideal")
            self.assertEqual(command[command.index("--profile") + 1], "ideal")
            self.assertEqual(command[command.index("--computation_tool") + 1], "ideal")
            self.assertIn("--verbose", command)

    def test_smoke_passes_selector_and_repair_model_options(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_kramabench_smoke_fixture(repo_root)

            command = setup_run.run(
                [
                    "smoke",
                    "--benchmark",
                    "kramabench",
                    "--model",
                    "gemini/gemini-3.1-flash-lite",
                    "--selector-model",
                    "openai/gpt-5-mini",
                    "--repair-model",
                    "openai/gpt-5.4",
                    "--db",
                    "lance_data",
                ],
                runner=_FakeRunner(),
                cwd=repo_root,
            )

            self.assertEqual(
                command[command.index("--selector-model") + 1],
                "openai/gpt-5-mini",
            )
            self.assertEqual(
                command[command.index("--repair-model") + 1],
                "openai/gpt-5.4",
            )
            self.assertNotIn("--ideal-subagent-model", command)
            self.assertNotIn("--search-ideal-subagent-model", command)
            self.assertNotIn("--semantic-ideal-subagent-model", command)
            self.assertNotIn("--repair-ideal-subagent-model", command)

    def test_smoke_normalizes_new_openai_model_alias(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "ideal",
                    "--model",
                    "gpt-5.4-nano",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--model-name") + 1], "openai/gpt-5.4-nano")

    def test_smoke_normalizes_gpt_5_nano_alias(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "ideal",
                    "--model",
                    "gpt-5-nano",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--model-name") + 1], "openai/gpt-5-nano")

    def test_ideal_search_and_plan_default_compute_is_ideal_when_not_explicit(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "full",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "ideal",
                    "--model",
                    "gpt-5.4-nano",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--computation_tool") + 1], "ideal")

    def test_explicit_standard_compute_is_passed_through(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "lance_data").mkdir(parents=True, exist_ok=True)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "full",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "ideal",
                    "--compute",
                    "standard",
                    "--model",
                    "gpt-5.4-nano",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--computation_tool") + 1], "standard")

    def test_smoke_accepts_preloaded_search_mode(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "preloaded",
                    "--results",
                    "naive",
                    "--profile",
                    "ideal",
                    "--model",
                    "bedrock/claude-haiku-4.5",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command, fake_runner.command)
            self.assertEqual(command[command.index("--search_tool") + 1], "preloaded")
            self.assertEqual(command[command.index("--profile") + 1], "ideal")

    def test_smoke_passes_search_free_and_lessguide_aliases(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "standard",
                    "--search-free",
                    "--search_lessguide",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertIn("--search-free", command)
            self.assertIn("--search-lessguide", command)

    def test_smoke_passes_skills_flag_only_when_enabled(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "standard",
                    "--skills",
                    "on",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--skills") + 1], "on")

        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "standard",
                    "--skills",
                    "off",
                    "--db",
                    "lance_data",
                ],
                runner=_FakeRunner(),
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--skills") + 1], "off")

        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "ideal",
                    "--results",
                    "ideal",
                    "--profile",
                    "standard",
                    "--db",
                    "lance_data",
                ],
                runner=_FakeRunner(),
                cwd=repo_root,
            )

            self.assertNotIn("--skills", command)

    def test_smoke_rejects_skills_on_with_naive_profile(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            stderr = io.StringIO()

            with self.assertRaises(SystemExit) as captured, contextlib.redirect_stderr(stderr):
                setup_run.run(
                    [
                        "smoke",
                        "--search",
                        "ideal",
                        "--results",
                        "ideal",
                        "--profile",
                        "naive",
                        "--skills",
                        "on",
                        "--db",
                        "lance_data",
                    ],
                    runner=_FakeRunner(),
                    cwd=repo_root,
                )

            self.assertEqual(captured.exception.code, 2)
            self.assertIn("--skills on requires --profile standard or --profile ideal", stderr.getvalue())

    def test_smoke_passes_ideal_compute_axis(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            fake_runner = _FakeRunner()

            command = setup_run.run(
                [
                    "smoke",
                    "--search",
                    "preloaded",
                    "--results",
                    "ideal",
                    "--profile",
                    "standard",
                    "--compute",
                    "ideal",
                    "--db",
                    "lance_data",
                ],
                runner=fake_runner,
                cwd=repo_root,
            )

            self.assertEqual(command[command.index("--computation_tool") + 1], "ideal")

    def test_missing_db_has_helpful_error(self):
        with TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            self._write_smoke_fixture(repo_root)
            stderr = io.StringIO()

            with self.assertRaises(SystemExit) as captured, contextlib.redirect_stderr(stderr):
                setup_run.run(
                    [
                        "smoke",
                        "--search",
                        "ideal",
                        "--results",
                        "ideal",
                        "--profile",
                        "ideal",
                    ],
                    runner=_FakeRunner(),
                    cwd=repo_root,
                )

            self.assertEqual(captured.exception.code, 2)
            message = stderr.getvalue()
            self.assertIn("--db is required", message)
            self.assertIn("lance_data", message)


if __name__ == "__main__":
    unittest.main()

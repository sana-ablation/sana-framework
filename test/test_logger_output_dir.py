import importlib.util
import logging
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def _load_logger_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "sana_evaluation" / "helper" / "logger.py"
    spec = importlib.util.spec_from_file_location("_test_logger_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


logger_module = _load_logger_module()


def _close_handlers(name):
    target = logging.getLogger(name)
    for handler in list(target.handlers):
        target.removeHandler(handler)
        handler.close()


class LoggerOutputDirTests(unittest.TestCase):
    def tearDown(self):
        _close_handlers("sana_evaluation")
        _close_handlers("strands")

    def test_configure_worker_logging_uses_run_config_log_root(self):
        run_config = types.SimpleNamespace(logs_output_dir="custom_logs")

        with patch.object(logger_module, "configure_logging") as mock_configure:
            logger_module.configure_worker_logging(
                run_config,
                model="openai/gpt-5.2-xhigh",
                condition="st-ideal_sr-ideal_am-ideal_k5/baseline",
                task_id="tasks_mini/k-1-d-1/task_1.json",
            )

        mock_configure.assert_called_once_with(
            log_dir="custom_logs",
            model="openai/gpt-5.2-xhigh",
            condition="st-ideal_sr-ideal_am-ideal_k5/baseline",
            task_id="tasks_mini/k-1-d-1/task_1.json",
            level=logger_module.logging.DEBUG,
        )

    def test_build_log_file_preserves_precomposed_mode_path(self):
        with TemporaryDirectory() as tmpdir:
            log_path = logger_module._build_log_file(
                tmpdir,
                "modes/openai_gpt-5.2-xhigh/search_i_results_d_plann_k5",
                "openai/gpt-5.2-xhigh",
                "tasks_mini/k-1-d-1/task_1.json",
            )

            relative = Path(log_path).relative_to(tmpdir).as_posix()
            self.assertEqual(
                relative,
                "modes/openai_gpt-5.2-xhigh/search_i_results_d_plann_k5/tasks_mini/k-1-d-1/task_1.log",
            )

    def test_build_log_file_keeps_task_root_to_avoid_collisions(self):
        with TemporaryDirectory() as tmpdir:
            mini_log_path = logger_module._build_log_file(
                tmpdir,
                "baseline",
                "openai/gpt-5.2-xhigh",
                "tasks_mini/k-1-d-1/task_1.json",
            )
            core_log_path = logger_module._build_log_file(
                tmpdir,
                "baseline",
                "openai/gpt-5.2-xhigh",
                "tasks_core_quality/k-1-d-1/task_1.json",
            )

            self.assertNotEqual(mini_log_path, core_log_path)
            self.assertEqual(
                Path(mini_log_path).relative_to(tmpdir).as_posix(),
                "baseline/openai-gpt-5.2-xhigh/tasks_mini/k-1-d-1/task_1.log",
            )
            self.assertEqual(
                Path(core_log_path).relative_to(tmpdir).as_posix(),
                "baseline/openai-gpt-5.2-xhigh/tasks_core_quality/k-1-d-1/task_1.log",
            )

    def test_reconfiguring_logging_closes_replaced_file_handlers(self):
        with TemporaryDirectory() as tmpdir:
            logger_module.configure_logging(
                log_dir=tmpdir,
                condition="baseline",
                model="openai/gpt-5.2-xhigh",
                task_id="tasks_mini/k-1-d-1/task_1.json",
            )
            first_handlers = list(logging.getLogger("sana_evaluation").handlers)
            first_file_handler = next(
                handler for handler in first_handlers if isinstance(handler, logging.FileHandler)
            )

            logger_module.configure_logging(
                log_dir=tmpdir,
                condition="baseline",
                model="openai/gpt-5.2-xhigh",
                task_id="tasks_mini/k-1-d-1/task_2.json",
            )

            self.assertTrue(first_file_handler.stream is None or first_file_handler.stream.closed)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


_MODULE_NAMES = [
    "strands",
    "strands.hooks",
    "strands.plugins",
    "sana_evaluation",
    "sana_evaluation.instrumentation",
    "sana_evaluation.instrumentation.trace_plugin",
    "sana_evaluation.instrumentation.read_trace_plugin",
]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_trace_modules():
    repo_root = Path(__file__).resolve().parents[1]
    previous = {name: sys.modules.get(name) for name in _MODULE_NAMES}

    fake_strands = types.ModuleType("strands")

    class _Plugin:
        pass

    fake_strands.Plugin = _Plugin
    sys.modules["strands"] = fake_strands

    fake_hooks = types.ModuleType("strands.hooks")
    fake_hooks.AfterToolCallEvent = object
    fake_hooks.BeforeToolCallEvent = object
    sys.modules["strands.hooks"] = fake_hooks

    fake_plugins = types.ModuleType("strands.plugins")
    fake_plugins.hook = lambda func: func
    sys.modules["strands.plugins"] = fake_plugins

    package = types.ModuleType("sana_evaluation")
    package.__path__ = [str(repo_root / "sana_evaluation")]
    sys.modules["sana_evaluation"] = package

    instrumentation_package = types.ModuleType("sana_evaluation.instrumentation")
    instrumentation_package.__path__ = [str(repo_root / "sana_evaluation" / "instrumentation")]
    sys.modules["sana_evaluation.instrumentation"] = instrumentation_package

    trace_module = _load_module(
        "sana_evaluation.instrumentation.trace_plugin",
        repo_root / "sana_evaluation" / "instrumentation" / "trace_plugin.py",
    )
    read_trace_module = _load_module(
        "sana_evaluation.instrumentation.read_trace_plugin",
        repo_root / "sana_evaluation" / "instrumentation" / "read_trace_plugin.py",
    )

    def restore():
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value

    return trace_module, read_trace_module, restore


def _event(tool_name: str, tool_input: dict, result: dict):
    return SimpleNamespace(
        tool_use={
            "name": tool_name,
            "input": tool_input,
            "toolUseId": "tool-1",
        },
        result=result,
    )


class ReadTracePluginFailedReadsTests(unittest.TestCase):
    def setUp(self):
        self.trace_module, self.read_trace_module, self.restore_modules = _load_trace_modules()

    def tearDown(self):
        self.restore_modules()

    def test_failed_read_attempt_does_not_count_as_source_access(self):
        with TemporaryDirectory() as tmpdir:
            self.trace_module.set_trace_context(
                "tasks_mini/k-1-d-1/task_1.json",
                ["school-directory"],
                tmpdir,
            )
            plugin = self.read_trace_module.ReadTracePlugin()
            tool_input = {
                "s3_uri": "s3://lakeqa-yc4103-datalake/datagov/school-directory/files/rows.txt"
            }

            plugin.on_before_tool(_event("read_file", tool_input, {"content": []}))
            plugin.on_after_tool(
                _event(
                    "read_file",
                    tool_input,
                    {
                        "status": "error",
                        "content": [{"text": json.dumps({"error": "object not found"})}],
                    },
                )
            )

            trace_path = Path(tmpdir) / "k-1-d-1" / "task_1.jsonl"
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(plugin.gold_datasets_read, set())
        self.assertEqual(rows[0]["attempted_read_dataset_ids"], ["school-directory"])
        self.assertEqual(rows[0]["read_dataset_ids"], [])
        self.assertEqual(rows[0]["gold_dataset_ids_read"], [])
        self.assertEqual(rows[0]["status"], "error")

    def test_successful_ideal_read_uses_result_source_when_input_has_no_source(self):
        with TemporaryDirectory() as tmpdir:
            self.trace_module.set_trace_context(
                "tasks_mini/k-1-d-1/task_1.json",
                ["school-directory"],
                tmpdir,
            )
            plugin = self.read_trace_module.ReadTracePlugin()
            tool_input = {"code": "print(7)", "intent": "compute answer"}

            plugin.on_before_tool(_event("execute_ideal", tool_input, {"content": []}))
            plugin.on_after_tool(
                _event(
                    "execute_ideal",
                    tool_input,
                    {
                        "status": "success",
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "success": True,
                                        "output": "7",
                                        "s3_uri": (
                                            "s3://lakeqa-yc4103-datalake/datagov/"
                                            "school-directory/files/rows.txt"
                                        ),
                                    }
                                )
                            }
                        ],
                    },
                )
            )

            trace_path = Path(tmpdir) / "k-1-d-1" / "task_1.jsonl"
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(plugin.gold_datasets_read, {"school-directory"})
        self.assertEqual(rows[0]["attempted_read_dataset_ids"], [])
        self.assertEqual(rows[0]["read_dataset_ids"], ["school-directory"])
        self.assertEqual(rows[0]["gold_dataset_ids_read"], ["school-directory"])
        self.assertEqual(rows[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()

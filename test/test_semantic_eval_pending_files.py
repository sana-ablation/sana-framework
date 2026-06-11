import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "semantic-eval-auditor"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPT_DIR))


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pending_files = _load_module("list_pending_eval_files.py", "semantic_eval_pending_files")


class TestSemanticEvalPendingFiles(unittest.TestCase):
    def test_pending_eval_paths_only_returns_unfinished_by_default(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results-ec2"
            output_root = root / "results-ec2_semantic"

            a = source_root / "modes" / "model-a" / "variant-a" / "eval_results.csv"
            b = source_root / "modes" / "model-b" / "variant-b" / "eval_results.csv"
            a.parent.mkdir(parents=True, exist_ok=True)
            b.parent.mkdir(parents=True, exist_ok=True)
            a.write_text("task_id\n")
            b.write_text("task_id\n")

            mirrored_a = output_root / a.relative_to(source_root)
            mirrored_a.parent.mkdir(parents=True, exist_ok=True)
            mirrored_a.write_text("task_id\n")

            entries = pending_files.pending_eval_paths(
                source_root=source_root,
                output_root=output_root,
            )

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["relative_eval"], "modes/model-b/variant-b/eval_results.csv")
            self.assertEqual(entries[0]["done"], False)

    def test_pending_eval_paths_can_include_completed_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results-ec2"
            output_root = root / "results-ec2_semantic"

            a = source_root / "modes" / "model-a" / "variant-a" / "eval_results.csv"
            a.parent.mkdir(parents=True, exist_ok=True)
            a.write_text("task_id\n")
            mirrored_a = output_root / a.relative_to(source_root)
            mirrored_a.parent.mkdir(parents=True, exist_ok=True)
            mirrored_a.write_text("task_id\n")

            entries = pending_files.pending_eval_paths(
                source_root=source_root,
                output_root=output_root,
                include_complete=True,
            )

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["done"], True)

    def test_render_text_numbers_entries(self):
        text = pending_files.render_text(
            [
                {"relative_eval": "modes/a/x/eval_results.csv", "done": False},
                {"relative_eval": "modes/b/y/eval_results.csv", "done": True},
            ],
            include_complete=True,
        )

        self.assertIn("1. modes/a/x/eval_results.csv", text)
        self.assertIn("2. modes/b/y/eval_results.csv [done]", text)


if __name__ == "__main__":
    unittest.main()

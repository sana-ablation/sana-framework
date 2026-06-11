import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "build_kramabench_dr_candidate_list.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KramabenchDirectCandidateListTests(unittest.TestCase):
    def test_builds_direct_dr_candidate_list_without_import_inventory(self):
        sampler = load_module(SCRIPT_PATH, "kramabench_dr_candidate_list_for_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "tasks-mini-kramabench"
            candidate_list = sampler.build_candidate_list(
                output_root=output_root,
                target_count=135,
                min_datasets=3,
                max_subtasks=6,
                seed=20260514,
            )

            list_path = output_root / "candidate_list.json"
            self.assertTrue(list_path.exists())
            from_disk = json.loads(list_path.read_text(encoding="utf-8"))
            self.assertEqual(candidate_list, from_disk)

            self.assertEqual(135, candidate_list["target_count"])
            self.assertLess(candidate_list["selected_count"], candidate_list["target_count"])
            self.assertEqual(candidate_list["selected_count"], len(candidate_list["candidates"]))
            self.assertEqual("take_all_eligible_because_pool_smaller_than_target", candidate_list["selection_mode"])
            self.assertIn("too_many_subtasks", candidate_list["rejected_counts"])
            self.assertIn("too_few_datasets", candidate_list["rejected_counts"])

            for summary in candidate_list["candidates"]:
                self.assertNotIn("tasks-kramabench-mini", summary["candidate_path"])
                candidate_path = REPO / summary["candidate_path"]
                self.assertTrue(candidate_path.exists(), summary)

                payload = json.loads(candidate_path.read_text(encoding="utf-8"))
                self.assertEqual(summary["source_id"], payload["source_id"])
                self.assertEqual("hard", payload["difficulty"])
                self.assertLessEqual(payload["subtask_count"], 6)
                self.assertGreaterEqual(len(payload["normalized_data_sources"]), 3)
                self.assertTrue((REPO / payload["dr_input_bundle"]).is_dir())
                self.assertTrue(payload["s3_mirror_prefix"].startswith("datagov/kramabench-"))
                self.assertEqual(payload["task"]["id"], payload["source_id"])


if __name__ == "__main__":
    unittest.main()

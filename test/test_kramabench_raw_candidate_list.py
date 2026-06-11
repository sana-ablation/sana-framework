import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "build_kramabench_raw_candidate_list.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KramabenchRawCandidateListTests(unittest.TestCase):
    def test_builds_all_104_main_workload_candidates(self):
        sampler = load_module(SCRIPT_PATH, "kramabench_raw_candidate_list_for_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "tasks-mini-kramabench"
            candidate_list = sampler.build_candidate_list(output_root=output_root)

            self.assertEqual(104, candidate_list["selected_count"])
            self.assertEqual("kramabench_raw_all_main_workload", candidate_list["candidate_set"])
            self.assertEqual("all_main_workload_no_filters", candidate_list["selection_mode"])
            self.assertEqual(104, len(candidate_list["candidates"]))
            self.assertEqual({}, candidate_list["filters"])

            ids = {item["source_id"] for item in candidate_list["candidates"]}
            self.assertIn("biomedical-hard-8", ids)
            self.assertIn("wildfire-hard-21", ids)
            self.assertIn("environment-easy-1", ids)

            for summary in candidate_list["candidates"]:
                candidate_path = REPO / summary["candidate_path"]
                self.assertTrue(candidate_path.exists(), summary)
                payload = json.loads(candidate_path.read_text(encoding="utf-8"))

                self.assertEqual("kramabench_raw_all_v1", payload["candidate_version"])
                self.assertEqual(summary["source_id"], payload["source_id"])
                self.assertEqual(summary["source_mode"], payload["source_mode"])
                self.assertEqual(payload["task"]["id"], payload["source_id"])
                self.assertTrue((REPO / payload["raw_data_root"]).is_dir())
                self.assertTrue(payload["s3_mirror_prefix"].startswith("datagov/kramabench-"))
                self.assertIn(payload["difficulty"], {"easy", "medium", "hard", "unknown"})


if __name__ == "__main__":
    unittest.main()

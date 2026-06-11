import importlib.util
import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONVERT_PATH = REPO / "other-benchmarks" / "data-imports" / "kramabench" / "convert.py"
SAMPLE_HARD_PATH = REPO / "other-benchmarks" / "data-imports" / "_shared" / "sample_hard.py"
TASK_ROOT = REPO / "other-benchmarks" / "tasks" / "kramabench"
MINI_ROOT = REPO / "other-benchmarks" / "tasks-kramabench-mini"
BUCKET_RE = re.compile(r"^k-(\d+)-d-(\d+)-j-(\d+)-f-(\d+)$")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KramabenchImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if TASK_ROOT.exists():
            shutil.rmtree(TASK_ROOT)
        subprocess.run([sys.executable, str(CONVERT_PATH)], cwd=REPO, check=True)

    def test_converter_emits_104_tasks_with_kdjf_buckets(self):
        task_files = sorted(TASK_ROOT.glob("k-*-d-*-j-*-f-*/task_*.json"))
        self.assertEqual(len(task_files), 104)

        for task_path in task_files:
            match = BUCKET_RE.match(task_path.parent.name)
            self.assertIsNotNone(match, task_path.parent.name)
            k, d, j, f = map(int, match.groups())
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            axes = payload["_provenance"]["bucket_axes"]

            self.assertEqual({"k": k, "d": d, "j": j, "f": f}, axes)
            self.assertEqual(k, len(payload["reasoning_hops"]))
            self.assertEqual(f, max(len(hop["node_ids"]) for hop in payload["reasoning_hops"]))
            self.assertEqual(j, {"easy": 1, "medium": 2, "hard": 3}[payload["_provenance"]["difficulty"]])
            self.assertEqual(d, len(payload["_provenance"]["normalized_data_sources"]))

    def test_biomedical_hard_1_keeps_two_file_first_hop(self):
        payload = self._task_by_source_id("biomedical-hard-1")

        self.assertEqual(payload["_provenance"]["bucket_axes"], {"k": 5, "d": 2, "j": 3, "f": 2})
        self.assertEqual(len(payload["nodes"]), 6)
        self.assertEqual(payload["reasoning_hops"][0]["node_ids"], [1, 2])
        self.assertEqual(payload["nodes"]["6"]["answer"], "0.4765")
        self.assertIn("1-s2.0-S0092867420301070-mmc1.xlsx", payload["nodes"]["1"]["source"])
        self.assertIn("1-s2.0-S0092867420301070-mmc2.xlsx", payload["nodes"]["2"]["source"])

    def test_legal_hard_1_preserves_glob_as_virtual_source(self):
        payload = self._task_by_source_id("legal-hard-1")

        self.assertEqual(payload["_provenance"]["bucket_axes"]["j"], 3)
        glob_nodes = [
            (node_id, node)
            for node_id, node in payload["nodes"].items()
            if "State MSA Identity Theft Data/*" in node["source"]
        ]
        self.assertTrue(glob_nodes)
        for node_id, _node in glob_nodes:
            self.assertEqual(payload["_provenance"]["fact_origin_per_node"][node_id], "glob_loop")

    def test_kramabench_mini_uses_hard_provenance_and_kdjf_dirs(self):
        if MINI_ROOT.exists():
            shutil.rmtree(MINI_ROOT)
        subprocess.run([sys.executable, str(SAMPLE_HARD_PATH)], cwd=REPO, check=True)

        task_files = sorted(MINI_ROOT.glob("k-*-d-*-j-*-f-*/task_*.json"))
        self.assertTrue(task_files)
        for task_path in task_files:
            self.assertIsNotNone(BUCKET_RE.match(task_path.parent.name), task_path.parent.name)
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["_provenance"]["difficulty"], "hard")
            self.assertIn("sample_source", payload["_provenance"])

    def _task_by_source_id(self, source_id: str) -> dict:
        for task_path in sorted(TASK_ROOT.glob("k-*-d-*-j-*-f-*/task_*.json")):
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            if payload["_provenance"]["source_id"] == source_id:
                return payload
        raise AssertionError(f"missing task for {source_id}")


class KramabenchHelperTests(unittest.TestCase):
    def test_normalize_data_sources_counts_globs_as_one(self):
        convert = load_module(CONVERT_PATH, "kramabench_convert_for_test")
        self.assertEqual(
            convert.normalize_file_refs([".", "./", "State MSA Identity Theft Data/*", "metropolitan_statistics.html"]),
            ["State MSA Identity Theft Data/*", "metropolitan_statistics.html"],
        )


if __name__ == "__main__":
    unittest.main()

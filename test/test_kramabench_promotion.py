import contextlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROMOTE_PATH = REPO / "other-benchmarks" / "data-imports" / "kramabench" / "promote_sample.py"
BUCKET_RE = re.compile(r"^k-(\d+)-d-(\d+)$")
EXPECTED_SOURCE_IDS = {
    "legal-hard-7",
}
EXPECTED_SKIPPED_SOURCE_IDS = {
    "biomedical-hard-3",
    "biomedical-hard-4",
    "biomedical-hard-7",
    "legal-hard-6",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_fact(fact: str) -> str:
    ns = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(fact, ns)
    return (buf.getvalue().strip() or str(ns.get("result", "")).strip())


class KramabenchPromotionSampleTests(unittest.TestCase):
    def test_promotes_only_candidates_backed_by_per_task_dr_input(self):
        promote = load_module(PROMOTE_PATH, "kramabench_promote_sample_for_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "tasks-mini-kramabench-sample"
            written = promote.promote_sample_tasks(output_root=output_root)

            self.assertEqual(1, len(written))
            task_files = sorted(output_root.glob("k-*-d-*/task_*.json"))
            self.assertEqual(written, task_files)
            error_log = json.loads((output_root / "error_log.json").read_text(encoding="utf-8"))

            source_ids = set()
            for task_path in task_files:
                match = BUCKET_RE.match(task_path.parent.name)
                self.assertIsNotNone(match, task_path.parent.name)
                k, d = map(int, match.groups())
                payload = json.loads(task_path.read_text(encoding="utf-8"))

                source_ids.add(payload["_provenance"]["source_id"])
                self.assertTrue(payload["question"].endswith("Write your answer as [ANSWER]."))
                self.assertEqual(k, len(payload["reasoning_hops"]))
                self.assertEqual(d, max(len(hop["node_ids"]) for hop in payload["reasoning_hops"]))
                self.assertNotIn("-j-", task_path.parent.name)
                self.assertNotIn("-f-", task_path.parent.name)

                for node_id, node in payload["nodes"].items():
                    self.assertTrue(node["source"].startswith("datagov/kramabench-"))
                    self.assertNotIn("other-benchmarks/Kramabench/data/", node["fact"])
                    self.assertIn("other-benchmarks/Kramabench/dr-input/", node["fact"])
                    produced = run_fact(node["fact"])
                    self.assertEqual(node["answer"], produced, (task_path, node_id))

            self.assertEqual(EXPECTED_SOURCE_IDS, source_ids)
            skipped_source_ids = {entry["source_id"] for entry in error_log["entries"]}
            self.assertEqual(EXPECTED_SKIPPED_SOURCE_IDS, skipped_source_ids)
            self.assertEqual(
                "missing_dr_input_bundle",
                next(
                    entry["reason"]
                    for entry in error_log["entries"]
                    if entry["source_id"] == "biomedical-hard-3"
                ),
            )
            self.assertEqual(
                "missing_dr_input_file",
                next(entry["reason"] for entry in error_log["entries"] if entry["source_id"] == "legal-hard-6"),
            )

    def test_legal_hard_7_uses_task_bundle_and_collapses_to_k_1_d_1(self):
        promote = load_module(PROMOTE_PATH, "kramabench_promote_sample_for_test_bio4")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "tasks-mini-kramabench-sample"
            promote.promote_sample_tasks(output_root=output_root, source_ids=("legal-hard-7",))
            task_path = output_root / "k-1-d-1" / "task_1.json"
            payload = json.loads(task_path.read_text(encoding="utf-8"))

        self.assertEqual("legal-hard-7", payload["_provenance"]["source_id"])
        self.assertEqual(
            "other-benchmarks/tasks-kramabench-mini/k-7-d-1-j-3-f-1/task_3.json",
            payload["_provenance"]["sample_source"],
        )
        self.assertEqual(
            "other-benchmarks/Kramabench/dr-input/legal/legal-hard-7",
            payload["_provenance"]["dr_input_bundle"],
        )
        self.assertEqual(["kramabench-legal-hard-7"], payload["datasets_used"])
        self.assertEqual([1], payload["reasoning_hops"][0]["node_ids"])
        self.assertEqual("Bank Account", payload["nodes"]["1"]["answer"])
        self.assertEqual("Bank Account", payload["answer"])


if __name__ == "__main__":
    unittest.main()

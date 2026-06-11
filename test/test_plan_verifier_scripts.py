import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "plan-verifier"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPT_DIR))


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


plan_verifier = _load_module("verify_plan.py", "plan_verifier")


class TestPlanVerifierScripts(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _seed_indexed_sources(self, root: Path, sources: list[str]) -> None:
        table_path = root / "table_descriptions.jsonl"
        prefix = "s3://lakeqa-yc4103-datalake/"
        lines = [
            json.dumps({"dataset_uri": f"{prefix}{source}"})
            for source in sources
        ]
        table_path.write_text("\n".join(lines) + "\n")

    def _seed_author_checker(self, root: Path) -> None:
        source_skill = (
            Path(__file__).resolve().parent.parent
            / ".agents"
            / "skills"
            / "author-ideal-plans"
            / "scripts"
            / "check_plan_cleanliness.py"
        )
        target = root / ".agents" / "skills" / "author-ideal-plans" / "scripts" / "check_plan_cleanliness.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source_skill.read_text())

    def test_verify_plan_accepts_clean_plan(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_author_checker(root)
            task_path = root / "tasks_mini" / "k-3-d-4" / "task_6.json"
            plan_path = root / "plans_mini" / "k-3-d-4" / "task_6.json"
            source_sequence = [
                "datagov/fy-2020-pension-recipients-by-state/files/rows.txt",
                "datagov/fy-2021-pension-recipients-by-state-1266e/files/rows.txt",
                "datagov/fy-2023-pension-recipients-by-state/files/rows.txt",
                "wikipedia/New_England/content.txt",
                "wikipedia/Vermont/content.txt",
                "wikipedia/Montpelier,_Vermont/content.txt",
            ]
            self._seed_indexed_sources(root, source_sequence)

            self._write_json(
                task_path,
                {
                    "question": "What year was the capital city chartered?",
                    "answer": "1781",
                    "datasets_used": [
                        "fy-2020-pension-recipients-by-state",
                        "fy-2021-pension-recipients-by-state-1266e",
                        "fy-2023-pension-recipients-by-state",
                        "New_England",
                        "Vermont",
                        "Montpelier,_Vermont",
                    ],
                    "reasoning_chain": [
                        "Node 1: Find states with 200-400 VA pension recipients in FY 2020",
                        "Node 2: Find states with 200-400 VA pension recipients in FY 2021",
                        "Node 3: Find states with 150-300 VA pension recipients in FY 2023",
                        "Intersection of node_1, node_2, node_3, node_4 -> Vermont",
                        "Node 5: Find Vermont's capital",
                        "Node 6: Find when Montpelier was chartered",
                    ],
                    "nodes": {
                        "1": {
                            "source": source_sequence[0],
                            "answer": ["Maine", "New Hampshire", "Vermont"],
                        },
                        "2": {
                            "source": source_sequence[1],
                            "answer": ["Vermont", "Rhode Island"],
                        },
                        "3": {
                            "source": source_sequence[2],
                            "answer": ["Vermont", "Connecticut"],
                        },
                        "4": {
                            "source": source_sequence[3],
                            "answer": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"],
                        },
                        "5": {
                            "source": source_sequence[4],
                            "answer": "Montpelier",
                        },
                        "6": {
                            "source": source_sequence[5],
                            "answer": "1781",
                        },
                    },
                },
            )
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": [
                        "fy-2020-pension-recipients-by-state",
                        "fy-2021-pension-recipients-by-state-1266e",
                        "fy-2023-pension-recipients-by-state",
                        "New_England",
                        "Vermont",
                        "Montpelier,_Vermont",
                    ],
                    "source_sequence": [
                        *source_sequence,
                    ],
                    "reasoning_chain_text": [
                        "1. Identify U.S. states with between 200 and 400 VA pension recipients in FY 2020.",
                        "2. Identify U.S. states with between 200 and 400 VA pension recipients in FY 2021.",
                        "3. Identify U.S. states with between 150 and 300 VA pension recipients in FY 2023.",
                        "4. Determine which states are in New England.",
                        "5. Find the capital of the state that remains after intersecting the qualifying sets.",
                        "6. Return the year in which that capital city was chartered.",
                    ],
                },
            )

            result = plan_verifier.evaluate_verification(plan_path)
            self.assertEqual(result["status"], "clean")
            self.assertEqual(result["issues"], [])

    def test_verify_plan_flags_dataset_coverage_mismatch(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_author_checker(root)
            task_path = root / "tasks_mini" / "k-2-d-3" / "task_8.json"
            plan_path = root / "plans_mini" / "k-2-d-3" / "task_8.json"

            self._write_json(
                task_path,
                {
                    "question": "Which branches remain?",
                    "answer": "A",
                    "datasets_used": ["ytd-circulation-2019", "ytd-visitors-2019", "branch-context"],
                    "reasoning_chain": [
                        "Node 1: top branches by circulation",
                        "Node 2: top branches by visitors",
                        "Intersection -> branch list",
                    ],
                    "nodes": {},
                },
            )
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["ytd-circulation-2019", "branch-context"],
                    "source_sequence": [
                        "datagov/ytd-circulation-2019/files/rows.txt",
                        "wikipedia/branch-context/content.txt",
                    ],
                    "reasoning_chain_text": (
                        "1. Identify top branches by circulation in 2019.\n"
                        "2. Return the branch names in alphabetical order."
                    ),
                },
            )

            result = plan_verifier.evaluate_verification(plan_path)
            codes = {issue["code"] for issue in result["issues"]}
            self.assertIn("dataset_coverage_mismatch", codes)

    def test_verify_plan_flags_reasoning_chain_mismatch_when_comparison_is_dropped(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_author_checker(root)
            task_path = root / "tasks_mini" / "k-2-d-4" / "task_1.json"
            plan_path = root / "plans_mini" / "k-2-d-4" / "task_1.json"

            self._write_json(
                task_path,
                {
                    "question": "Which county had more violent crimes in 2021?",
                    "answer": "Erie County",
                    "datasets_used": ["bridges", "violent-crimes-2021"],
                    "reasoning_chain": [
                        "Node 1: filter counties by bridges",
                        "Comparison: compare violent-crime counts and select the higher one",
                    ],
                    "nodes": {},
                },
            )
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["bridges", "violent-crimes-2021"],
                    "source_sequence": [
                        "datagov/bridges/files/rows.txt",
                        "datagov/violent-crimes-2021/files/rows.txt",
                    ],
                    "reasoning_chain_text": (
                        "1. Filter counties that satisfy the bridge condition.\n"
                        "2. Return the county name only."
                    ),
                },
            )

            result = plan_verifier.evaluate_verification(plan_path)
            codes = {issue["code"] for issue in result["issues"]}
            self.assertIn("reasoning_chain_mismatch", codes)

    def test_verify_plan_allows_hidden_page_titles_in_metadata_when_reasoning_is_abstract(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_author_checker(root)
            task_path = root / "tasks_mini" / "k-4-d-2" / "task_11.json"
            plan_path = root / "plans_mini" / "k-4-d-2" / "task_11.json"
            source_sequence = [
                "datagov/county-filter/files/rows.txt",
                "wikipedia/Hidden_County/content.txt",
                "wikipedia/Hidden_Person/content.txt",
            ]
            self._seed_indexed_sources(root, source_sequence)

            self._write_json(
                task_path,
                {
                    "question": "Which county satisfies the filters, and what year was its namesake born?",
                    "answer": "1900",
                    "datasets_used": ["county-filter", "Hidden_County", "Hidden_Person"],
                    "reasoning_chain": [
                        "Node 1: identify counties satisfying the filters",
                        "Node 2: determine the namesake for the remaining county",
                        "Node 3: determine that person's birth year",
                    ],
                    "nodes": {
                        "1": {"source": source_sequence[0], "answer": ["Hidden County"]},
                        "2": {"source": source_sequence[1], "answer": "Hidden Person"},
                        "3": {"source": source_sequence[2], "answer": "1900"},
                    },
                },
            )
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["county-filter", "Hidden_County", "Hidden_Person"],
                    "source_sequence": source_sequence,
                    "reasoning_chain_text": [
                        "1. Identify counties satisfying the requested filters.",
                        "2. For the remaining county, determine its namesake.",
                        "3. Using the namesake identified in step 2, determine the birth year.",
                    ],
                },
            )

            result = plan_verifier.evaluate_verification(plan_path)
            self.assertEqual(result["status"], "clean")
            self.assertEqual(result["issues"], [])

    def test_verify_plan_propagates_answer_leak_failures(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_author_checker(root)
            task_path = root / "tasks_mini" / "k-3-d-4" / "task_6.json"
            plan_path = root / "plans_mini" / "k-3-d-4" / "task_6.json"

            self._write_json(
                task_path,
                {
                    "question": "What year was the capital city chartered?",
                    "answer": "1781",
                    "datasets_used": ["fy-2020-pension", "Vermont", "Montpelier,_Vermont"],
                    "reasoning_chain": [
                        "Node 1: identify the state",
                        "Node 2: find the capital",
                        "Node 3: find the charter year",
                    ],
                    "nodes": {
                        "1": {"answer": "Vermont"},
                        "2": {"answer": "Montpelier"},
                    },
                },
            )
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["fy-2020-pension", "Vermont", "Montpelier,_Vermont"],
                    "source_sequence": [
                        "datagov/fy-2020-pension/files/rows.txt",
                        "wikipedia/Vermont/content.txt",
                        "wikipedia/Montpelier,_Vermont/content.txt",
                    ],
                    "reasoning_chain_text": (
                        "1. Identify the qualifying state.\n"
                        "2. Find the capital of Vermont.\n"
                        "3. Return the year Montpelier was chartered."
                    ),
                },
            )

            result = plan_verifier.evaluate_verification(plan_path)
            codes = {issue["code"] for issue in result["issues"]}
            self.assertIn("answer_leak", codes)


if __name__ == "__main__":
    unittest.main()

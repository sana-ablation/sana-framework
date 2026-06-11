import csv
import importlib.util
import sys
import unittest
from collections import Counter
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


auditor = _load_module("rewrite_semantic_eval_results.py", "semantic_eval_auditor")


class _StubJudge:
    def __init__(self):
        self.file_calls = []
        self.row_calls = []

    def audit_file(self, *, source_eval_path, output_eval_path, fieldnames, row_count):
        self.file_calls.append((source_eval_path, output_eval_path, tuple(fieldnames), row_count))
        return auditor.FileAuditResult(status="ready", row_count=row_count, notes="process every row")

    def audit_row(self, *, source_eval_path, row, log_tail, log_tail_lines_used, log_required):
        self.row_calls.append(
            {
                "source_eval_path": source_eval_path,
                "task_id": row["task_id"],
                "log_tail": log_tail,
                "log_tail_lines_used": log_tail_lines_used,
                "log_required": log_required,
            }
        )
        if row["task_id"].endswith("task_1.json"):
            return auditor.RowAuditResult(
                semantic_match=1,
                semantic_reason="same year",
                semantic_bucket="semantic_correct",
                log_error_bucket="",
                log_error_evidence="",
            )
        return auditor.RowAuditResult(
            semantic_match=0,
            semantic_reason="blank prediction",
            semantic_bucket="answer_unknown_blank",
            log_error_bucket="error_tokens_reached",
            log_error_evidence="MaxTokensReachedException in tail",
        )


class TestSemanticEvalAuditor(unittest.TestCase):
    def test_relative_task_log_path_drops_tasks_prefix(self):
        path = auditor.relative_task_log_path("tasks_mini/k-7-d-4/task_8.json")
        self.assertEqual(path.as_posix(), "k-7-d-4/task_8.log")

    def test_ordered_fieldnames_inserts_semantic_columns_after_exact_match(self):
        ordered = auditor.ordered_fieldnames(
            ["task_id", "expected_answer", "predicted_answer", "exact_match", "f1_score"]
        )
        self.assertEqual(
            ordered,
            [
                "task_id",
                "expected_answer",
                "predicted_answer",
                "exact_match",
                "semantic_match",
                "semantic_reason",
                "semantic_bucket",
                "log_error_bucket",
                "log_error_evidence",
                "f1_score",
            ],
        )

    def test_read_log_tail_uses_fallback_when_short_tail_has_no_signal(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "task.log"
            lines = [f"line {idx}" for idx in range(55)]
            lines[10] = "MaxTokensReachedException"
            path.write_text("\n".join(lines) + "\n")

            tail, lines_used = auditor.read_log_tail_for_review(path, tail_lines=20, fallback_lines=50)

            self.assertEqual(lines_used, 50)
            self.assertIn("MaxTokensReachedException", tail)

    def test_runtime_helpers_distinguish_plain_wrong_answers_from_execution_failures(self):
        self.assertEqual(auditor.infer_runtime_bucket("", "Tool call cancelled. Timeout reached"), "error_tokens_reached")
        self.assertEqual(auditor.infer_runtime_bucket("", "Tool limit reached (32/30 calls used)."), "error_tools_limit")
        self.assertFalse(auditor.has_runtime_issue("", "Answer submitted: [42]", success_value="True"))
        self.assertFalse(auditor.has_runtime_issue("", "Tool call cancelled. Retry if making progress.", success_value="True"))
        self.assertTrue(auditor.has_runtime_issue("", "Tool call cancelled. Timeout reached", success_value="True"))

    def test_runner_writes_only_eval_results_csv_with_semantic_columns(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results-ec2"
            logs_root = root / "logs-ec2"
            eval_path = source_root / "modes" / "openai_gpt-5.2-xhigh" / "search_i_results_i_plani_k5" / "eval_results.csv"
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            with eval_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "model",
                        "expected_answer",
                        "predicted_answer",
                        "exact_match",
                        "f1_score",
                        "error",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "tasks_mini/k-1-d-1/task_1.json",
                        "model": "gpt-5.2",
                        "expected_answer": "2020",
                        "predicted_answer": "[2020]",
                        "exact_match": "1.0",
                        "f1_score": "1.0",
                        "error": "",
                    }
                )
                writer.writerow(
                    {
                        "task_id": "tasks_mini/k-1-d-1/task_2.json",
                        "model": "gpt-5.2",
                        "expected_answer": "2022",
                        "predicted_answer": "unknown",
                        "exact_match": "0.0",
                        "f1_score": "0.0",
                        "error": "MaxTokensReachedException",
                    }
                )

            log_path = (
                logs_root
                / "modes"
                / "openai_gpt-5.2-xhigh"
                / "search_i_results_i_plani_k5"
                / "k-1-d-1"
                / "task_2.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_lines = ["debug"] * 55
            log_lines[10] = "MaxTokensReachedException"
            log_path.write_text("\n".join(log_lines) + "\n")

            output_root = root / "results-ec2_semantic"
            judge = _StubJudge()
            runner = auditor.SemanticEvalAuditor(
                source_root=source_root,
                output_root=output_root,
                logs_root=logs_root,
                judge=judge,
                tail_lines=20,
                tail_lines_fallback=50,
            )

            runner.run()

            mirrored = output_root / "modes" / "openai_gpt-5.2-xhigh" / "search_i_results_i_plani_k5" / "eval_results.csv"
            self.assertTrue(mirrored.exists())
            with mirrored.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["exact_match"], "1.0")
            self.assertEqual(rows[0]["semantic_match"], "1")
            self.assertEqual(rows[1]["semantic_bucket"], "answer_unknown_blank")
            self.assertEqual(rows[1]["log_error_bucket"], "error_tokens_reached")
            self.assertEqual(len(judge.file_calls), 1)
            self.assertEqual(len(judge.row_calls), 2)
            self.assertFalse((output_root / "modes" / "openai_gpt-5.2-xhigh" / "search_i_results_i_plani_k5" / "task_2.log").exists())
            self.assertEqual(judge.row_calls[0]["log_required"], False)
            self.assertEqual(judge.row_calls[1]["log_tail_lines_used"], 50)

    def test_successful_but_wrong_row_can_leave_log_error_bucket_blank(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results-ec2"
            logs_root = root / "logs-ec2"
            eval_path = source_root / "modes" / "openai_gpt-5.2-xhigh" / "search_i_results_i_plann_k5" / "eval_results.csv"
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            with eval_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "task_id",
                        "expected_answer",
                        "predicted_answer",
                        "exact_match",
                        "success",
                        "error",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "tasks_mini/k-1-d-1/task_3.json",
                        "expected_answer": "2023",
                        "predicted_answer": "[2022]",
                        "exact_match": "0.0",
                        "success": "True",
                        "error": "",
                    }
                )

            log_path = (
                logs_root
                / "modes"
                / "openai_gpt-5.2-xhigh"
                / "search_i_results_i_plann_k5"
                / "k-1-d-1"
                / "task_3.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Answer submitted: [2022]\n")

            class _WrongAnswerJudge(_StubJudge):
                def audit_row(self, *, source_eval_path, row, log_tail, log_tail_lines_used, log_required):
                    return auditor.RowAuditResult(
                        semantic_match=0,
                        semantic_reason="wrong year",
                        semantic_bucket="semantic_incorrect",
                        log_error_bucket="error_unknown",
                        log_error_evidence="No clear bucket",
                    )

            output_root = root / "results-ec2_semantic"
            runner = auditor.SemanticEvalAuditor(
                source_root=source_root,
                output_root=output_root,
                logs_root=logs_root,
                judge=_WrongAnswerJudge(),
                tail_lines=20,
                tail_lines_fallback=50,
            )
            runner.run()

            mirrored = output_root / "modes" / "openai_gpt-5.2-xhigh" / "search_i_results_i_plann_k5" / "eval_results.csv"
            with mirrored.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["semantic_bucket"], "semantic_incorrect")
            self.assertEqual(rows[0]["log_error_bucket"], "")
            self.assertEqual(rows[0]["log_error_evidence"], "")


if __name__ == "__main__":
    unittest.main()

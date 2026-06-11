import tempfile
import unittest
from pathlib import Path

from sana_analysis.run_sana_mode_analysis import prepare_sana_modes


class RunSanaModeAnalysisTests(unittest.TestCase):
    def test_prepare_sana_modes_exposes_raw_sana_runs_as_mode_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "sana-results"
            model = "openai_gpt-5.4-nano"
            log_model = "openai-gpt-5.4-nano"
            variant = "sana_delegation_search3_inspect8_si_ri_pd_k5"

            raw_result = root / "sana" / model / variant / model
            raw_result.mkdir(parents=True)
            (raw_result / "eval_results.csv").write_text("task_id,exact_match\nx,0\n")
            (raw_result / "agent_results.jsonl").write_text('{"task_id":"x"}\n')
            (raw_result / "tools_breakdown.csv").write_text("tool,count\n")

            raw_log = root / "sana" / model / variant / log_model / "tasks_core_quality" / "k-1-d-1"
            raw_log.mkdir(parents=True)
            (raw_log / "task_2.log").write_text("done\n")

            raw_trace = root / "traces" / "sana" / model / variant / "k-1-d-1"
            raw_trace.mkdir(parents=True)
            (raw_trace / "task_2.jsonl").write_text('{"event":"done"}\n')

            prepared = prepare_sana_modes(root)

            self.assertEqual(prepared, 1)
            self.assertEqual(
                (root / "modes" / model / variant / "eval_results.csv").read_text(),
                "task_id,exact_match\nx,0\n",
            )
            self.assertEqual(
                (root / "logs" / "modes" / model / variant / "tasks_core_quality" / "k-1-d-1" / "task_2.log").read_text(),
                "done\n",
            )
            self.assertEqual(
                (root / "traces" / "modes" / model / variant / "k-1-d-1" / "task_2.jsonl").read_text(),
                '{"event":"done"}\n',
            )


if __name__ == "__main__":
    unittest.main()

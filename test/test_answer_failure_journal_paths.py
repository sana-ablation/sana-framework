import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis.answer_failure.audit_runner import (
    CURRENT_AUDIT_SCHEMA_VERSION,
    AuditLayout,
    _load_journal_audits,
    _short_mode_variant,
)
from sana_analysis.variants import DISK_VARIANT_NAMES


class TestShortModeVariant(unittest.TestCase):
    def test_gen1_output_is_unchanged(self):
        """Journals written before the decoder landed must still resolve."""
        self.assertEqual(
            _short_mode_variant("search_i_results_i_plani_computei_k5_skills_off"),
            "search_i_plani_computei",
        )
        self.assertEqual(
            _short_mode_variant("search_d_results_i_pland_k5_skills_off"),
            "search_d_pland",
        )

    def test_gen4_has_no_empty_segments(self):
        self.assertEqual(
            _short_mode_variant("search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"),
            "search_ideal_plan_ideal_compute_ideal",
        )

    def test_no_name_on_disk_produces_an_empty_segment(self):
        for name in DISK_VARIANT_NAMES:
            short = _short_mode_variant(name)
            self.assertNotIn("__", short, f"{name} shortened to {short!r}")
            self.assertFalse(short.startswith("_") or short.endswith("_"), short)

    def test_results_k_and_skills_are_still_dropped(self):
        short = _short_mode_variant("search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off")
        for dropped in ("results", "rich", "k5", "skills", "off"):
            self.assertNotIn(dropped, short)

    def test_flags_survive(self):
        """`nos3` distinguishes the web arm from its no-S3 twin; dropping it would collide."""
        self.assertEqual(
            _short_mode_variant("search_web__plan_standard__compute_standard__results_minimal__nos3__skills_off"),
            "search_web_plan_standard_compute_standard_nos3",
        )

    def test_a_non_variant_string_is_passed_through_safely(self):
        self.assertEqual(_short_mode_variant("not_a_variant"), "not_a_variant")


class TestJournalCollisionIsHarmless(unittest.TestCase):
    """Two variants differing only in dropped segments share a journal file.

    `results` and `k` are dropped deliberately, so these two collide. Keeping
    either to break the tie would change gen-1 output. The collision is safe
    because journal records are filtered on read; this proves it rather than
    assuming it.
    """

    MINIMAL = "search_standard__plan_standard__compute_standard__results_minimal__skills_off"
    RICH = "search_standard__plan_standard__compute_standard__results_rich__k5__skills_off"

    def _layout(self, root: Path, mode: str) -> AuditLayout:
        eval_path = root / "results" / "modes" / "openai_gpt-5-mini" / mode / "eval_results.csv"
        return AuditLayout(
            repo_root=root,
            source_root=root / "results",
            output_root=root / "out",
            logs_root=root / "logs",
            eval_path=eval_path,
            mirrored_eval_path=root / "out" / "eval_results.csv",
            mirrored_events_path=root / "out" / "events.csv",
            mirrored_report_path=root / "out" / "report.md",
            model_variant="openai_gpt-5-mini",
            mode_variant=mode,
        )

    def test_the_two_variants_do_collide(self):
        self.assertEqual(_short_mode_variant(self.MINIMAL), _short_mode_variant(self.RICH))

    def test_a_shared_journal_does_not_leak_audits_between_them(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal = root / "shared.jsonl"
            with journal.open("w") as handle:
                for mode, task_id in ((self.MINIMAL, "task_a"), (self.RICH, "task_b")):
                    handle.write(json.dumps({
                        "status": "ok",
                        "model_variant": "openai_gpt-5-mini",
                        "mode_variant": mode,
                        "source_eval": str(self._layout(root, mode).eval_path),
                        "task_id": task_id,
                        "audit": {"answer_failure_audit_schema_version": CURRENT_AUDIT_SCHEMA_VERSION},
                    }) + "\n")

            self.assertEqual(sorted(_load_journal_audits(journal, self._layout(root, self.MINIMAL))), ["task_a"])
            self.assertEqual(sorted(_load_journal_audits(journal, self._layout(root, self.RICH))), ["task_b"])


if __name__ == "__main__":
    unittest.main()

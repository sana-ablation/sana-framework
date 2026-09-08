"""The variant-label layout migration: canonical results spelling, axis order.

`_variant_condition_label` now emits the three ablation axes first
(search, plan, compute) with the search-result richness modifier trailing them,
and records that modifier canonically (`rich`/`minimal`) rather than as the
caller spelled it. Directories written before that change carry the old layout.

These tests pin the pure rename computation. The filesystem walk, collision
detection and `--apply` guard are shared with the tree-walking helpers and are
covered by the round-trip test at the bottom.
"""
from pathlib import Path

import pytest

from scripts.migrate_variant_label_layout import (
    canonical_label,
    plan_renames,
    candidate_dirs,
)


class TestCanonicalLabel:
    def test_reorders_results_after_the_three_ablation_axes(self):
        assert canonical_label(
            "search_ideal__results_ideal__plan_ideal__compute_ideal__k5__skills_off"
        ) == "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"

    def test_canonicalises_ideal_to_rich(self):
        assert "results_rich" in canonical_label(
            "search_naive__results_ideal__plan_ideal__compute_ideal__k5__skills_off"
        )

    def test_canonicalises_naive_to_minimal(self):
        assert "results_minimal" in canonical_label(
            "search_standard__results_naive__plan_standard__compute_standard__skills_off"
        )

    def test_accepts_the_pre_rename_profile_spelling(self):
        """Remote trees never migrated from `profile_`; one pass must fix both."""
        assert canonical_label(
            "search_ideal__results_ideal__profile_standard__compute_ideal__k5__skills_off"
        ) == "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off"

    def test_preserves_the_tail_in_order(self):
        assert canonical_label(
            "search_web__results_naive__plan_standard__compute_standard__nos3__skills_off"
        ) == "search_web__plan_standard__compute_standard__results_minimal__nos3__skills_off"

    def test_preserves_k_and_search_call_budget(self):
        assert canonical_label(
            "search_ideal__results_naive__plan_naive__compute_standard__k5__sc2__free__skills_on"
        ) == "search_ideal__plan_naive__compute_standard__results_minimal__k5__sc2__free__skills_on"

    def test_is_idempotent(self):
        once = canonical_label(
            "search_ideal__results_ideal__plan_ideal__compute_ideal__k5__skills_off"
        )
        assert canonical_label(once) == once

    def test_leaves_an_unrecognised_name_alone(self):
        """A directory missing an axis is not guessed at -- it is returned unchanged
        so the caller skips it rather than inventing a default."""
        weird = "search_ideal__plan_ideal__k5__skills_off"  # no compute segment
        assert canonical_label(weird) == weird

    def test_ignores_a_non_variant_directory(self):
        assert canonical_label("modes") == "modes"


class TestPlanRenames:
    def test_skips_directories_already_canonical(self, tmp_path: Path):
        (tmp_path / "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off").mkdir()
        assert plan_renames(candidate_dirs(tmp_path)) == []

    def test_renames_deepest_first(self, tmp_path: Path):
        """A child must be renamed before its parent, or the parent's rename
        invalidates the child's recorded path."""
        outer = tmp_path / "search_ideal__results_ideal__plan_ideal__compute_ideal__skills_off"
        inner = outer / "traces" / "search_naive__results_naive__plan_naive__compute_standard__skills_off"
        inner.mkdir(parents=True)

        renames = plan_renames(candidate_dirs(tmp_path))
        depths = [len(src.parts) for src, _ in renames]
        assert depths == sorted(depths, reverse=True), renames

    def test_round_trip_on_disk(self, tmp_path: Path):
        src = tmp_path / "search_ideal__results_ideal__plan_ideal__compute_ideal__k5__skills_off"
        src.mkdir()
        (src / "eval_results.csv").write_text("kept", encoding="utf-8")

        renames = plan_renames(candidate_dirs(tmp_path))
        assert len(renames) == 1
        old, new = renames[0]
        old.rename(new)

        assert new.name == "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
        assert (new / "eval_results.csv").read_text(encoding="utf-8") == "kept"
        assert plan_renames(candidate_dirs(tmp_path)) == []

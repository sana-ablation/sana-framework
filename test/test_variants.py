import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.variants import (
    AXIS_DEFAULTS,
    RESULT_MODES,
    Variant,
    parse_variant,
    try_parse_variant,
)


class TestParseVariant(unittest.TestCase):
    def test_gen1_literal(self):
        v = parse_variant("search_i_results_i_plani_computei_k5_skills_off")
        self.assertEqual(v.search, "ideal")
        self.assertEqual(v.plan, "ideal")
        self.assertEqual(v.compute, "ideal")
        self.assertEqual(v.results, "rich")
        self.assertEqual(v.k, 5)
        self.assertFalse(v.skills)

    def test_gen1_compute_omitted_defaults_to_standard(self):
        v = parse_variant("search_i_results_i_plani_k5_skills_off")
        self.assertEqual(v.compute, "standard")
        self.assertEqual(v.search, "ideal")
        self.assertEqual(v.plan, "ideal")

    def test_gen2_directory_name_from_the_remote(self):
        v = parse_variant(
            "search_ideal__results_ideal__profile_ideal__compute_ideal__k5__skills_off"
        )
        self.assertEqual(v.plan, "ideal")
        self.assertEqual(v.results, "rich")
        self.assertEqual(v.compute, "ideal")

    def test_gen3_encoder_output(self):
        v = parse_variant(
            "search_ideal__results_ideal__plan_ideal__compute_ideal__k5__skills_off"
        )
        self.assertEqual(v.plan, "ideal")
        self.assertEqual(v.results, "rich")

    def test_gen4_canonical(self):
        v = parse_variant(
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
        )
        self.assertEqual(v.search, "ideal")
        self.assertEqual(v.plan, "ideal")
        self.assertEqual(v.compute, "ideal")
        self.assertEqual(v.results, "rich")
        self.assertEqual(v.k, 5)

    def test_every_spelling_of_the_planning_axis_agrees(self):
        names = [
            "search_i_results_i_plani_computei_k5_skills_off",
            "search_ideal__results_ideal__profile_ideal__compute_ideal__k5__skills_off",
            "search_ideal__results_ideal__plan_ideal__compute_ideal__k5__skills_off",
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
        ]
        self.assertEqual({parse_variant(n).plan for n in names}, {"ideal"})

    def test_search_web_with_nos3(self):
        v = parse_variant(
            "search_web__plan_standard__compute_standard__results_minimal__nos3__skills_off"
        )
        self.assertEqual(v.search, "web")
        self.assertEqual(v.results, "minimal")
        self.assertIn("nos3", v.flags)
        self.assertIsNone(v.k)

    def test_unknown_axis_value_does_not_raise(self):
        v = parse_variant("search_quantum__plan_ideal__compute_ideal__results_rich__skills_off")
        self.assertEqual(v.search, "quantum")

    def test_malformed_input_raises(self):
        with self.assertRaises(ValueError):
            parse_variant("not-a-variant-name")
        with self.assertRaises(ValueError):
            parse_variant("")

    def test_try_parse_returns_none_instead_of_raising(self):
        self.assertIsNone(try_parse_variant("not-a-variant-name"))
        self.assertIsNotNone(try_parse_variant("search_i_results_i_plani_computei_k5_skills_off"))

    def test_search_calls_and_flags_and_skills_on(self):
        v = parse_variant(
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__sc3__free__skills_on"
        )
        self.assertEqual(v.search_calls, 3)
        self.assertIn("free", v.flags)
        self.assertTrue(v.skills)

    def test_debug_suffix_is_tolerated(self):
        v = parse_variant(
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off__debug_tools"
        )
        self.assertEqual(v.search, "ideal")
        self.assertEqual(v.plan, "ideal")

    def test_defaults_fill_absent_segments(self):
        v = parse_variant("search_ideal__skills_off")
        self.assertEqual(v.plan, AXIS_DEFAULTS["plan"])
        self.assertEqual(v.compute, AXIS_DEFAULTS["compute"])
        self.assertEqual(v.results, AXIS_DEFAULTS["results"])

    def test_results_is_only_ever_minimal_or_rich(self):
        for name in (
            "search_i_results_i_plani_computei_k5_skills_off",
            "search_i_results_n_plani_computei_k5_skills_off",
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
            "search_ideal__plan_ideal__compute_ideal__results_minimal__skills_off",
        ):
            self.assertIn(parse_variant(name).results, RESULT_MODES, name)

    def test_raw_is_preserved(self):
        name = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
        self.assertEqual(parse_variant(name).raw, name)

    def test_variant_is_frozen_and_hashable(self):
        v = parse_variant("search_i_results_i_plani_computei_k5_skills_off")
        self.assertIsInstance(hash(v), int)
        with self.assertRaises(Exception):
            v.search = "naive"


class TestVariantMatches(unittest.TestCase):
    def setUp(self):
        self.v = parse_variant(
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
        )

    def test_matching_subset_of_axes(self):
        self.assertTrue(self.v.matches(search="ideal"))
        self.assertTrue(self.v.matches(search="ideal", plan="ideal", compute="ideal"))

    def test_non_matching_axis(self):
        self.assertFalse(self.v.matches(search="naive"))

    def test_unnamed_axes_are_ignored(self):
        self.assertTrue(self.v.matches(search="ideal"))  # k, skills, flags not consulted

    def test_no_axes_matches_everything(self):
        self.assertTrue(self.v.matches())

    def test_unknown_axis_name_raises(self):
        with self.assertRaises(TypeError):
            self.v.matches(plna="ideal")

    def test_matches_cannot_be_fooled_by_raw(self):
        with self.assertRaises(TypeError):
            self.v.matches(raw="anything")


if __name__ == "__main__":
    unittest.main()

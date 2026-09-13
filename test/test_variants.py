import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.variants import (
    AXIS_DEFAULTS,
    CONDITION_ORDER,
    DISK_VARIANT_NAMES,
    RESULT_MODES,
    Variant,
    find_variant,
    parse_variant,
    select_conditions,
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

    def test_an_unknown_results_token_passes_through_rather_than_being_coerced(self):
        # Deliberate. `naive`/`ideal` are absorbed to `minimal`/`rich` because they
        # are the retired spellings of known values. A genuinely unknown token is
        # not: coercing it to `rich` would average a distinct experimental arm into
        # the rich cell, and raising would crash a whole analysis run on one
        # unrecognised directory. Passing it through leaves the arm absent from
        # figures, which is what the spec asks for.
        v = parse_variant("search_ideal__plan_ideal__compute_ideal__results_quantum__skills_off")
        self.assertEqual(v.results, "quantum")

    def test_retired_results_spellings_are_always_absorbed(self):
        for spelling, expected in (("naive", "minimal"), ("ideal", "rich")):
            name = f"search_ideal__plan_ideal__compute_ideal__results_{spelling}__skills_off"
            self.assertEqual(parse_variant(name).results, expected, name)


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
        # Verify that k, skills, and flags do not affect matching by constructing
        # two variants that differ only in those attributes, then asserting both
        # match the same three-axis predicate.
        v1 = parse_variant("search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off")
        v2 = parse_variant("search_ideal__plan_ideal__compute_ideal__results_rich__k10__sc3__free__skills_on")
        # Both variants have identical axes (search, plan, compute, results)
        # but differ in k, search_calls, flags, and skills
        self.assertTrue(v1.matches(search="ideal", plan="ideal", compute="ideal", results="rich"))
        self.assertTrue(v2.matches(search="ideal", plan="ideal", compute="ideal", results="rich"))

    def test_no_axes_matches_everything(self):
        self.assertTrue(self.v.matches())

    def test_unknown_axis_name_raises(self):
        with self.assertRaises(TypeError):
            self.v.matches(plna="ideal")

    def test_matches_cannot_be_fooled_by_raw(self):
        with self.assertRaises(TypeError):
            self.v.matches(raw="anything")


# The seven gen-1 literals the seven modules used to hardcode, paired with the
# condition each one named. This is the frozen before-picture: the decoder must
# assign every one of them the same label the literal did.
GEN1_LITERALS = [
    ("No Plan", "search_i_results_i_plann_computei_k5_skills_off"),
    ("Standard Plan", "search_i_results_i_pland_computei_k5_skills_off"),
    ("BM25", "search_n_results_i_plani_computei_k5_skills_off"),
    ("Pneuma Hybrid", "search_d_results_i_plani_computei_k5_skills_off"),
    ("Standard Computation", "search_i_results_i_plani_k5_skills_off"),
    ("Ideal", "search_i_results_i_plani_computei_k5_skills_off"),
    ("Preloaded", "search_p_results_i_plani_computei_k5_skills_off"),
]

# The gen-4 name each of those seven conditions is written as today.
GEN4_EQUIVALENTS = {
    "No Plan": "search_ideal__plan_naive__compute_ideal__results_rich__k5__skills_off",
    "Standard Plan": "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off",
    "BM25": "search_naive__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "Pneuma Hybrid": "search_standard__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "Standard Computation": "search_ideal__plan_ideal__compute_standard__results_rich__k5__skills_off",
    "Ideal": "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    "Preloaded": "search_preloaded__plan_ideal__compute_ideal__results_rich__k5__skills_off",
}


class TestConditionOrder(unittest.TestCase):
    def test_seven_conditions_in_the_reported_order(self):
        self.assertEqual(
            [label for label, _axes in CONDITION_ORDER],
            [
                "No Plan",
                "Standard Plan",
                "BM25",
                "Pneuma Hybrid",
                "Standard Computation",
                "Ideal",
                "Preloaded",
            ],
        )

    def test_a_gen1_literal_and_its_gen4_equivalent_decode_alike(self):
        for label, literal in GEN1_LITERALS:
            a = parse_variant(literal)
            b = parse_variant(GEN4_EQUIVALENTS[label])
            self.assertEqual(
                (a.search, a.plan, a.compute, a.results),
                (b.search, b.plan, b.compute, b.results),
                label,
            )


class TestDiskCorpus(unittest.TestCase):
    def test_every_name_on_disk_parses(self):
        for name in DISK_VARIANT_NAMES:
            self.assertIsNotNone(try_parse_variant(name), name)

    def test_no_condition_is_ambiguous_over_the_disk_corpus(self):
        # CONDITION_ORDER constrains search/plan/compute but not results, so two
        # directories differing only in results could in principle collide.
        # None do. If one ever does, find_variant raises rather than guessing.
        for label, axes in CONDITION_ORDER:
            matched = [
                name for name in DISK_VARIANT_NAMES if parse_variant(name).matches(**axes)
            ]
            self.assertLessEqual(len(matched), 1, f"{label} matched {matched}")


class TestFindVariant(unittest.TestCase):
    def test_returns_the_observed_name_not_a_synthesised_one(self):
        found = find_variant(DISK_VARIANT_NAMES, search="ideal", plan="ideal", compute="ideal")
        self.assertEqual(
            found, "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
        )

    def test_returns_none_when_absent(self):
        self.assertIsNone(find_variant(DISK_VARIANT_NAMES, search="quantum"))

    def test_ignores_entries_that_are_not_variant_names(self):
        names = ["README.md", "search_i_results_i_plani_computei_k5_skills_off"]
        self.assertEqual(
            find_variant(names, search="ideal", plan="ideal", compute="ideal"),
            "search_i_results_i_plani_computei_k5_skills_off",
        )

    def test_ambiguity_raises_rather_than_guessing(self):
        names = [
            "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
            "search_ideal__plan_ideal__compute_ideal__results_minimal__skills_off",
        ]
        with self.assertRaises(ValueError):
            find_variant(names, search="ideal", plan="ideal", compute="ideal")


class TestSelectConditions(unittest.TestCase):
    def test_returns_label_and_observed_name_in_condition_order(self):
        names = [
            GEN4_EQUIVALENTS["Ideal"],
            GEN4_EQUIVALENTS["BM25"],
            GEN4_EQUIVALENTS["No Plan"],
        ]
        self.assertEqual(
            select_conditions(names),
            [
                ("No Plan", GEN4_EQUIVALENTS["No Plan"]),
                ("BM25", GEN4_EQUIVALENTS["BM25"]),
                ("Ideal", GEN4_EQUIVALENTS["Ideal"]),
            ],
        )

    def test_absent_conditions_are_omitted_not_blanked(self):
        self.assertEqual(
            select_conditions([GEN4_EQUIVALENTS["Ideal"]]),
            [("Ideal", GEN4_EQUIVALENTS["Ideal"])],
        )

    def test_label_subset_selects_and_preserves_condition_order(self):
        names = list(GEN4_EQUIVALENTS.values())
        self.assertEqual(
            select_conditions(names, labels=["Ideal", "No Plan"]),
            [
                ("No Plan", GEN4_EQUIVALENTS["No Plan"]),
                ("Ideal", GEN4_EQUIVALENTS["Ideal"]),
            ],
        )

    def test_unknown_label_raises(self):
        with self.assertRaises(KeyError):
            select_conditions(list(GEN4_EQUIVALENTS.values()), labels=["Nonexistent"])


if __name__ == "__main__":
    unittest.main()

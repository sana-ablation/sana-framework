import unittest

from sana_evaluation.helper.prompt_sections import split_prompt_sections, upsert_prompt_section


class PromptSectionTests(unittest.TestCase):
    def test_upsert_preserves_unknown_trailing_sections(self):
        prompt = "BASE\n\n## CURRENT PLAN\nold plan\n\n## STATIC TAIL\nkeep me"

        updated = upsert_prompt_section(prompt, "CURRENT PLAN", "new plan")

        self.assertIn("## CURRENT PLAN\nnew plan", updated)
        self.assertIn("## STATIC TAIL\nkeep me", updated)
        self.assertLess(updated.index("## CURRENT PLAN"), updated.index("## STATIC TAIL"))

    def test_split_sections_does_not_fold_unknown_tail_into_known_section(self):
        prompt = "BASE\n\n## CURRENT PLAN\nold plan\n\n## STATIC TAIL\nkeep me"

        base, sections = split_prompt_sections(prompt)

        self.assertEqual(base, "BASE")
        self.assertEqual(sections["CURRENT PLAN"], "old plan")


if __name__ == "__main__":
    unittest.main()

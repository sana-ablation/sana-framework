import unittest

from sana_evaluation.helper.text_utils import _clean_answer


class CleanAnswerTests(unittest.TestCase):
    def test_clean_answer_preserves_single_number_inside_non_numeric_answer(self):
        self.assertEqual(_clean_answer("District 9"), "District 9")
        self.assertEqual(_clean_answer("[District 9]"), "District 9")
        self.assertEqual(_clean_answer("Final answer: May 2024"), "May 2024")
        self.assertEqual(_clean_answer("The answer is 3%"), "3%")

    def test_clean_answer_still_normalizes_standalone_numeric_answers(self):
        self.assertEqual(_clean_answer("Final answer: 1,234."), "1234")
        self.assertEqual(_clean_answer("[42]"), "42")


if __name__ == "__main__":
    unittest.main()

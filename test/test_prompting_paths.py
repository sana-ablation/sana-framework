import os
import tempfile
import unittest

from sana_evaluation.helper.prompting import compose_managed_prompt


class PromptPathTests(unittest.TestCase):
    def test_prompt_loading_does_not_depend_on_process_cwd(self):
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                prompt = compose_managed_prompt("naive")
            finally:
                os.chdir(previous_cwd)

        self.assertIn("search_value", prompt)
        self.assertIn("search_schema", prompt)

    def test_managed_prompt_does_not_advertise_missing_context_tool(self):
        prompt = compose_managed_prompt("naive")

        self.assertNotIn("summarize_context", prompt)


if __name__ == "__main__":
    unittest.main()

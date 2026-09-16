"""The decoder owns its vocabulary; this asserts it still matches the runtime's.

`sana_analysis.variants` deliberately does not import `sana_evaluation`:
`runner.modes` pulls strands, boto3, lancedb and torch (measured 1.26s) because
it carries tool wiring alongside axis resolution. The coupling lives here
instead, so both packages stay importable alone and adding an axis value to the
runtime without teaching the decoder fails loudly.
"""

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.variants import (
    AXIS_DEFAULTS,
    COMPUTE_MODES,
    PLAN_MODES,
    RESULT_MODE_ALIASES,
    RESULT_MODES,
    SEARCH_MODES,
)


def _cli_choices(flag: str) -> frozenset:
    """The choices sana_evaluation's parser accepts for one axis flag."""
    from sana_evaluation.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if flag in action.option_strings:
            return frozenset(action.choices or ())
    raise AssertionError(f"{flag} is not an argument of sana_evaluation's parser")


class TestVocabularyContract(unittest.TestCase):
    def test_result_modes_agree(self):
        from sana_evaluation.runner.modes import _RESULT_MODES

        self.assertEqual(set(RESULT_MODES), set(_RESULT_MODES))

    def test_result_mode_aliases_agree(self):
        from sana_evaluation.runner.modes import _RESULT_MODE_ALIASES

        self.assertEqual(dict(RESULT_MODE_ALIASES), dict(_RESULT_MODE_ALIASES))

    def test_compute_modes_agree(self):
        from sana_evaluation.runner.modes import _COMPUTATION_MODES

        self.assertEqual(set(COMPUTE_MODES), set(_COMPUTATION_MODES))

    def test_axis_defaults_agree(self):
        from sana_evaluation.config import AXIS_DEFAULTS as RUNTIME_DEFAULTS

        self.assertEqual(
            AXIS_DEFAULTS,
            {
                "search": RUNTIME_DEFAULTS["search_tool_mode"],
                "plan": RUNTIME_DEFAULTS["plan_mode"],
                "compute": RUNTIME_DEFAULTS["computation_tool_mode"],
                "results": RUNTIME_DEFAULTS["search_results_mode"],
            },
        )

    def test_search_choices_agree(self):
        self.assertEqual(_cli_choices("--search"), SEARCH_MODES)

    def test_plan_choices_agree(self):
        self.assertEqual(_cli_choices("--plan"), PLAN_MODES)

    def test_compute_choices_agree(self):
        self.assertEqual(_cli_choices("--compute"), COMPUTE_MODES)

    def test_results_choices_are_the_modes_plus_their_aliases(self):
        self.assertEqual(
            _cli_choices("--results"), RESULT_MODES | frozenset(RESULT_MODE_ALIASES)
        )

    def test_decoder_does_not_import_the_runtime(self):
        source = (
            Path(__file__).resolve().parent.parent / "sana_analysis" / "variants.py"
        ).read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.assertNotIn("sana_evaluation", stripped, stripped)


class TestEncoderRoundTrip(unittest.TestCase):
    def test_the_encoder_output_decodes_back_to_its_inputs(self):
        from sana_evaluation.cli import _variant_condition_label

        from sana_analysis.variants import parse_variant

        for search in sorted(SEARCH_MODES):
            for plan in sorted(PLAN_MODES):
                for compute in sorted(COMPUTE_MODES):
                    for results in sorted(RESULT_MODES):
                        name = _variant_condition_label(
                            search_tool=search,
                            search_results=results,
                            plan=plan,
                            computation_tool=compute,
                            k=5,
                        )
                        decoded = parse_variant(name)
                        self.assertEqual(
                            (decoded.search, decoded.plan, decoded.compute, decoded.results),
                            (search, plan, compute, results),
                            name,
                        )


if __name__ == "__main__":
    unittest.main()

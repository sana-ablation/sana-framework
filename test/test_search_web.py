import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sana_evaluation.tools.external.search_web_tools as search_web_tools
from sana_evaluation.agent_with_mode import (
    _validate_search_mode_combination,
    build_mode_bundle,
    build_search,
)
from sana_evaluation.config import RunConfig
from sana_evaluation.helper.prompting import skill_paths_for_modes
from sana_evaluation.tools.external.ideal.search_wrapper import search_tool_names_in


def _response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


_PAYLOAD = {
    "search_id": "search_abc",
    "results": [
        {
            "url": "https://en.wikipedia.org/wiki/Sheldon_Independent_School_District",
            "title": "Sheldon Independent School District",
            "publish_date": "2024-01-02",
            "excerpts": ["Sheldon ISD became an independent district in 1952."],
        }
    ],
    "session_id": "sess_1",
    "warnings": None,
    "usage": [{"name": "search", "count": 1}],
}


class TestParallelSearchContract(unittest.TestCase):
    """Pin the documented Parallel Search API request/response contract."""

    def setUp(self) -> None:
        search_web_tools.set_max_results(None)
        self._env = patch.dict("os.environ", {"PARALLEL_API_KEY": "test-key"}, clear=False)
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        search_web_tools.set_max_results(None)

    def test_posts_documented_endpoint_headers_and_body(self) -> None:
        with patch.object(search_web_tools.requests, "post", return_value=_response(_PAYLOAD)) as post:
            search_web_tools.search_parallel(["sheldon isd independent year"], objective="founding year")

        url = post.call_args.args[0]
        kwargs = post.call_args.kwargs
        self.assertEqual(url, "https://api.parallel.ai/v1/search")
        self.assertEqual(kwargs["headers"]["x-api-key"], "test-key")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertNotIn("parallel-beta", kwargs["headers"])
        self.assertEqual(kwargs["json"]["search_queries"], ["sheldon isd independent year"])
        self.assertEqual(kwargs["json"]["objective"], "founding year")
        self.assertEqual(kwargs["json"]["mode"], "advanced")

    def test_optional_beta_header_and_endpoint_are_env_overridable(self) -> None:
        overrides = {
            "PARALLEL_SEARCH_URL": "https://api.parallel.ai/v1beta/search",
            "PARALLEL_BETA_HEADER": "search-extract-2025-10-10",
            "PARALLEL_SEARCH_MODE": "turbo",
        }
        with patch.dict("os.environ", overrides, clear=False):
            with patch.object(search_web_tools.requests, "post", return_value=_response(_PAYLOAD)) as post:
                search_web_tools.search_parallel(["a b c"])
        self.assertEqual(post.call_args.args[0], "https://api.parallel.ai/v1beta/search")
        self.assertEqual(post.call_args.kwargs["headers"]["parallel-beta"], "search-extract-2025-10-10")
        self.assertEqual(post.call_args.kwargs["json"]["mode"], "turbo")

    def test_fixed_k_limits_returned_results(self) -> None:
        many = dict(_PAYLOAD)
        many["results"] = [
            {"url": f"https://e/{i}", "title": str(i), "publish_date": None, "excerpts": ["x"]}
            for i in range(25)
        ]
        search_web_tools.set_max_results(5)
        with patch.object(search_web_tools.requests, "post", return_value=_response(many)):
            out = search_web_tools.search_parallel(["a b c"])
        self.assertEqual(out["count"], 5)

    def test_excerpts_are_capped_to_protect_the_context_window(self) -> None:
        huge = dict(_PAYLOAD)
        huge["results"] = [
            {"url": "https://e/1", "title": "t", "publish_date": None, "excerpts": ["x" * 50_000]}
        ]
        with patch.object(search_web_tools.requests, "post", return_value=_response(huge)):
            out = search_web_tools.search_parallel(["a b c"])
        total = sum(len(e) for r in out["results"] for e in r["excerpts"])
        self.assertLessEqual(total, search_web_tools._RESULT_CHAR_CAP)

    def test_results_carry_no_lake_addressing_fields(self) -> None:
        with patch.object(search_web_tools.requests, "post", return_value=_response(_PAYLOAD)):
            out = search_web_tools.search_parallel(["a b c"])
        for result in out["results"]:
            self.assertNotIn("s3_uri", result)
            self.assertNotIn("dataset_id", result)

    def test_errors_are_returned_not_raised(self) -> None:
        with patch.object(search_web_tools.requests, "post", side_effect=RuntimeError("boom")):
            out = search_web_tools.search_web(["a b c"])
        self.assertIn("boom", out["error"])
        self.assertEqual(out["results"], [])
        self.assertEqual(out["count"], 0)

    def test_missing_api_key_is_reported_through_the_tool_result(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            out = search_web_tools.search_web(["a b c"])
        self.assertIn("PARALLEL_API_KEY", out["error"])

    def test_empty_queries_rejected(self) -> None:
        self.assertIn("non-empty", search_web_tools.search_parallel([])["error"])


class TestWebSearchMode(unittest.TestCase):
    def _run_config(self, **overrides) -> RunConfig:
        base = dict(
            search_tool_mode="web",
            search_results_mode="naive",
            profile_mode="naive",
            computation_tool_mode="standard",
            search_k=5,
            benchmark="lakeqa",
        )
        base.update(overrides)
        return RunConfig(**base)

    def test_web_mode_exposes_only_search_web(self) -> None:
        tools = build_search("web", fixed_k=5)
        self.assertEqual([t.tool_spec["name"] for t in tools], ["search_web"])

    def test_fixed_k_is_applied_since_results_axis_does_not_wrap_web(self) -> None:
        build_search("web", fixed_k=7)
        self.assertEqual(search_web_tools.max_results(), 7)
        search_web_tools.set_max_results(None)

    def test_search_web_counts_as_a_search_tool_for_budgeting(self) -> None:
        tools = build_search("web", fixed_k=5)
        self.assertEqual(search_tool_names_in(tools), ("search_web",))

    def test_bundle_uses_the_web_overlay_and_hides_lake_search_tools(self) -> None:
        bundle = build_mode_bundle(self._run_config(), data_tools=[], task_context={})
        names = [t.tool_spec["name"] for t in bundle.tools]
        self.assertEqual(names, ["search_web"])
        self.assertIn("`search_web`", bundle.system_prompt)
        self.assertIn("There is no data lake in this run", bundle.system_prompt)
        self.assertNotIn("search_value", bundle.system_prompt)
        self.assertNotIn("search_ideal", bundle.system_prompt)

    def test_web_mode_requests_no_discover_data_skill(self) -> None:
        paths = skill_paths_for_modes("web", "standard")
        self.assertFalse(any("discover-data" in p for p in paths))


class TestWebModeAxisGuard(unittest.TestCase):
    """Ideal axes are lake-profile-backed and would mask what web search retrieved."""

    def _validate(self, **overrides) -> None:
        args = dict(
            search_tool_mode="web",
            search_results_mode="naive",
            profile_mode="naive",
            computation_tool_mode="standard",
        )
        args.update(overrides)
        _validate_search_mode_combination(**args)

    def test_rejects_ideal_computation(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._validate(computation_tool_mode="ideal")
        self.assertIn("--computation_tool ideal", str(ctx.exception))

    def test_rejects_ideal_profile(self) -> None:
        with self.assertRaises(ValueError):
            self._validate(profile_mode="ideal")

    def test_rejects_ideal_results(self) -> None:
        with self.assertRaises(ValueError):
            self._validate(search_results_mode="ideal")

    def test_reports_every_conflicting_axis_at_once(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._validate(
                computation_tool_mode="ideal", profile_mode="ideal", search_results_mode="ideal"
            )
        message = str(ctx.exception)
        for label in ("--computation_tool ideal", "--profile ideal", "--search_results ideal"):
            self.assertIn(label, message)

    def test_allows_the_supported_web_combination(self) -> None:
        self._validate()
        self._validate(profile_mode="standard")

    def test_does_not_constrain_non_web_search_modes(self) -> None:
        _validate_search_mode_combination(
            search_tool_mode="ideal",
            search_results_mode="ideal",
            profile_mode="ideal",
            computation_tool_mode="ideal",
        )


if __name__ == "__main__":
    unittest.main()

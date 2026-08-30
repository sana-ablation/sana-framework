"""Tests for --no-s3 mode: web download gated by the search_web URL allowlist.

Under --no-s3 the agent loses every S3-backed file tool and keeps only
`download` (fetching http(s) URLs) plus `execute_code`. `download` may fetch a
URL ONLY if a prior `search_web` call in the same task returned it, so the arm
measures retrieval rather than URLs memorised during pretraining.
"""

import json
import multiprocessing as mp
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sana_evaluation.tools.agent_tools as agent_tools
import sana_evaluation.tools.external.search_web_tools as search_web_tools
import sana_evaluation.tools.external.web_fetch_tools as web_fetch_tools
from sana_evaluation.agent_with_mode import (
    _validate_search_mode_combination,
    build_data_tools,
)
from sana_evaluation.helper.prompting import compose_managed_prompt
from sana_evaluation.run_mode_eval import _variant_condition_label


def _search_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _get_response(chunks, *, status=200, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {"Content-Type": "text/csv"}
    resp.raise_for_status.return_value = None
    resp.iter_content.return_value = iter(chunks)
    resp.__enter__ = lambda s: s
    resp.__exit__ = lambda s, *a: False
    return resp


_URL = "https://data.example.gov/crime/rows.csv"
_OTHER_URL = "https://memorised.example.org/known/ucr.csv"

_SEARCH_PAYLOAD = {
    "search_id": "search_abc",
    "results": [
        {
            "url": _URL,
            "title": "Crime rows",
            "publish_date": None,
            "excerpts": ["county,crimes\nAlpha,12\n"],
        }
    ],
}


def _tool_names(tools):
    return {t.tool_spec["name"] for t in tools}


class _SandboxTestCase(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self._tmp.name)
        agent_tools.set_sandbox_dir(self.sandbox)
        search_web_tools.set_max_results(None)
        self._env = patch.dict("os.environ", {"PARALLEL_API_KEY": "test-key"}, clear=False)
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        search_web_tools.set_max_results(None)
        self._tmp.cleanup()


class TestUrlAllowlist(_SandboxTestCase):
    """The allowlist must survive the spawn boundary between search and download."""

    def test_search_web_records_returned_urls(self) -> None:
        with patch.object(search_web_tools.requests, "post", return_value=_search_response(_SEARCH_PAYLOAD)):
            search_web_tools.search_web(search_queries=["crime rows county"])

        self.assertIn(_URL, web_fetch_tools.allowed_urls())

    def test_allowlist_is_file_backed_in_the_sandbox(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        path = self.sandbox / ".web_urls.json"
        self.assertTrue(path.is_file(), "allowlist must be on disk, not module state")
        self.assertEqual(json.loads(path.read_text()), [_URL])

    def test_allowlist_does_not_duplicate_repeated_urls(self) -> None:
        web_fetch_tools.record_search_urls([_URL, _URL])
        web_fetch_tools.record_search_urls([_URL])

        self.assertEqual(web_fetch_tools.allowed_urls(), [_URL])

    def test_allowlist_readable_from_a_spawned_subprocess(self) -> None:
        # download() runs via _run_tool_with_timeout, which uses a *spawned*
        # process — in-memory state does not cross that boundary.
        web_fetch_tools.record_search_urls([_URL])

        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(target=_read_allowlist_in_child, args=(queue, str(self.sandbox)))
        proc.start()
        proc.join(60)

        self.assertEqual(queue.get_nowait(), [_URL])

    def test_allowlist_is_empty_before_any_search(self) -> None:
        self.assertEqual(web_fetch_tools.allowed_urls(), [])


def _read_allowlist_in_child(queue, sandbox_dir: str) -> None:
    import sana_evaluation.tools.agent_tools as at
    import sana_evaluation.tools.external.web_fetch_tools as wft

    at.set_sandbox_dir(Path(sandbox_dir))
    queue.put(wft.allowed_urls())


class TestDownloadGating(_SandboxTestCase):
    def test_rejects_a_url_search_web_never_returned(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        result = web_fetch_tools._download_web_impl([{"url": _OTHER_URL}])

        self.assertEqual(result["download_count"], 0)
        self.assertIn("errors", result)
        self.assertIn("search_web", result["errors"][0]["error"])

    def test_rejection_names_the_urls_that_are_allowed(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        result = web_fetch_tools._download_web_impl([{"url": _OTHER_URL}])

        self.assertIn(_URL, json.dumps(result["errors"][0]))

    def test_rejects_s3_uris_under_no_s3(self) -> None:
        result = web_fetch_tools._download_web_impl(
            [{"s3_uri": "s3://sana-lake/datagov/x/files/rows.txt"}]
        )

        self.assertEqual(result["download_count"], 0)
        self.assertIn("errors", result)

    def test_fetches_an_allowlisted_url_into_the_sandbox(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        with patch.object(
            web_fetch_tools.requests, "get", return_value=_get_response([b"county,crimes\n", b"Alpha,12\n"])
        ):
            result = web_fetch_tools._download_web_impl([{"url": _URL}])

        self.assertEqual(result["download_count"], 1)
        local = Path(result["downloaded"][0]["local_path"])
        self.assertTrue(local.is_file())
        self.assertEqual(local.read_text(), "county,crimes\nAlpha,12\n")

    def test_accepts_a_bare_url_string(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        with patch.object(web_fetch_tools.requests, "get", return_value=_get_response([b"a,b\n"])):
            result = web_fetch_tools._download_web_impl([_URL])

        self.assertEqual(result["download_count"], 1)

    def test_rejects_more_than_five_files(self) -> None:
        result = web_fetch_tools._download_web_impl([{"url": _URL}] * 6)

        self.assertIn("error", result)
        self.assertIn("5", result["error"])

    def test_rejects_an_empty_file_list(self) -> None:
        self.assertIn("error", web_fetch_tools._download_web_impl([]))


class TestDownloadFailureModes(_SandboxTestCase):
    def test_http_error_becomes_an_error_entry_not_a_crash(self) -> None:
        web_fetch_tools.record_search_urls([_URL])
        boom = MagicMock(side_effect=web_fetch_tools.requests.RequestException("404 Not Found"))

        with patch.object(web_fetch_tools.requests, "get", boom):
            result = web_fetch_tools._download_web_impl([{"url": _URL}])

        self.assertEqual(result["download_count"], 0)
        self.assertIn("404", json.dumps(result["errors"]))

    def test_oversized_response_is_rejected(self) -> None:
        web_fetch_tools.record_search_urls([_URL])
        chunk = b"x" * (1024 * 1024)
        oversized = [chunk] * (web_fetch_tools._MAX_DOWNLOAD_BYTES // len(chunk) + 2)

        with patch.object(web_fetch_tools.requests, "get", return_value=_get_response(oversized)):
            result = web_fetch_tools._download_web_impl([{"url": _URL}])

        self.assertEqual(result["download_count"], 0)
        self.assertIn("too large", json.dumps(result["errors"]).lower())


class TestManifestIntegration(_SandboxTestCase):
    """A fetched file must be visible to execute_code exactly like an S3 download."""

    def test_download_registers_the_file_in_the_download_manifest(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        with patch.object(web_fetch_tools.requests, "get", return_value=_get_response([b"a,b\n1,2\n"])):
            result = web_fetch_tools._download_web_impl([{"url": _URL}])

        self.assertIn(_URL, result["path_map"])
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        self.assertEqual(manifest["path_map"][_URL], result["downloaded"][0]["local_path"])

    def test_execute_code_sees_the_fetched_file(self) -> None:
        web_fetch_tools.record_search_urls([_URL])
        with patch.object(web_fetch_tools.requests, "get", return_value=_get_response([b"a,b\n1,2\n"])):
            web_fetch_tools._download_web_impl([{"url": _URL}])

        out = agent_tools._execute_code_impl(
            "print(len(FILES)); print(sorted(DOWNLOAD_PATHS)[0])"
        )

        self.assertTrue(out["success"], out)
        self.assertIn(_URL, out["output"])

    def test_execute_code_still_blocks_network(self) -> None:
        # Load-bearing: download is the only network path, which is what makes
        # the allowlist enforceable rather than advisory.
        out = agent_tools._execute_code_impl("import socket; socket.socket()")

        self.assertFalse(out["success"])
        self.assertIn("Network access is disabled", out["error"])


class TestToolSurface(unittest.TestCase):
    def test_no_s3_exposes_only_download_execute_code_and_submit(self) -> None:
        names = _tool_names(build_data_tools(no_s3=True))

        self.assertEqual(names, {"download", "execute_code", "submit_answer"})

    def test_default_surface_keeps_the_s3_tools(self) -> None:
        names = _tool_names(build_data_tools(no_s3=False))

        self.assertIn("read_file", names)
        self.assertIn("query_file", names)
        self.assertIn("peek_file", names)

    def test_no_s3_download_is_the_web_fetcher(self) -> None:
        tools = {t.tool_spec["name"]: t for t in build_data_tools(no_s3=True)}

        self.assertIn("search_web", tools["download"].tool_spec["description"])


class TestNoS3Validation(unittest.TestCase):
    def test_no_s3_requires_web_search(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_search_mode_combination(
                search_tool_mode="ideal",
                search_results_mode="naive",
                profile_mode="standard",
                computation_tool_mode="standard",
                no_s3=True,
            )

        self.assertIn("--no-s3", str(ctx.exception))

    def test_web_with_no_s3_is_allowed(self) -> None:
        _validate_search_mode_combination(
            search_tool_mode="web",
            search_results_mode="naive",
            profile_mode="standard",
            computation_tool_mode="standard",
            no_s3=True,
        )

    def test_web_without_no_s3_is_still_allowed(self) -> None:
        _validate_search_mode_combination(
            search_tool_mode="web",
            search_results_mode="naive",
            profile_mode="standard",
            computation_tool_mode="standard",
            no_s3=False,
        )


class TestConditionLabel(unittest.TestCase):
    def test_no_s3_is_marked_in_the_condition_label(self) -> None:
        label = _variant_condition_label(
            search_tool="web",
            search_results="naive",
            profile="standard",
            computation_tool="standard",
            no_s3=True,
        )

        self.assertIn("nos3", label)

    def test_label_without_no_s3_is_unchanged(self) -> None:
        label = _variant_condition_label(
            search_tool="web",
            search_results="naive",
            profile="standard",
            computation_tool="standard",
            no_s3=False,
        )

        self.assertNotIn("nos3", label)


class TestPrompt(unittest.TestCase):
    def test_no_s3_prompt_tells_the_agent_download_works(self) -> None:
        prompt = compose_managed_prompt("web", no_s3=True)

        self.assertIn("download", prompt)
        self.assertIn("execute_code", prompt)

    def test_default_web_prompt_still_forbids_download(self) -> None:
        prompt = compose_managed_prompt("web", no_s3=False)

        self.assertIn("cannot open, download", prompt)


if __name__ == "__main__":
    unittest.main()


class TestRunConfigWiring(unittest.TestCase):
    """no_s3 must reach the composed prompt and tool surface through RunConfig."""

    def _bundle(self, *, no_s3: bool):
        from sana_evaluation.agent_with_mode import build_mode_bundle
        from sana_evaluation.config import RunConfig

        cfg = RunConfig(
            search_tool_mode="web",
            search_results_mode="naive",
            profile_mode="standard",
            computation_tool_mode="standard",
            benchmark="lakeqa",
            no_s3=no_s3,
        )
        return build_mode_bundle(cfg, data_tools=build_data_tools(no_s3=no_s3))

    def test_no_s3_bundle_uses_the_fetch_and_compute_prompt(self) -> None:
        bundle = self._bundle(no_s3=True)

        self.assertIn("FETCH AND COMPUTE", bundle.system_prompt)

    def test_no_s3_bundle_drops_the_s3_tools(self) -> None:
        names = _tool_names(self._bundle(no_s3=True).tools)

        self.assertNotIn("read_file", names)
        self.assertIn("download", names)
        self.assertIn("search_web", names)

    def test_web_without_no_s3_keeps_the_excerpts_only_prompt(self) -> None:
        bundle = self._bundle(no_s3=False)

        self.assertNotIn("FETCH AND COMPUTE", bundle.system_prompt)


class TestLocalFileNaming(_SandboxTestCase):
    """Filenames must carry a usable extension and index cleanly in path_map."""

    def test_dotted_host_still_gets_a_content_type_extension(self) -> None:
        # A host like `myarmybenefits.us.army.mil` puts dots in the name, which
        # must not be mistaken for the file already having an extension.
        url = "https://myarmybenefits.us.army.mil/Benefit-Library/Wyoming"
        web_fetch_tools.record_search_urls([url])

        with patch.object(
            web_fetch_tools.requests,
            "get",
            return_value=_get_response([b"<html></html>"], headers={"Content-Type": "text/html; charset=utf-8"}),
        ):
            result = web_fetch_tools._download_web_impl([{"url": url}])

        self.assertEqual(result["download_count"], 1, result)
        self.assertTrue(
            result["downloaded"][0]["local_path"].endswith(".html"),
            result["downloaded"][0]["local_path"],
        )

    def test_path_map_does_not_double_the_web_prefix(self) -> None:
        web_fetch_tools.record_search_urls([_URL])

        with patch.object(web_fetch_tools.requests, "get", return_value=_get_response([b"a,b\n"])):
            result = web_fetch_tools._download_web_impl([{"url": _URL}])

        doubled = [key for key in result["path_map"] if "web/web/" in key or "web:web/" in key]
        self.assertEqual(doubled, [], f"path_map has a doubled prefix: {doubled}")

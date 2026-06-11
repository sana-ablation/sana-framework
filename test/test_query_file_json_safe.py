"""
Tests for the JSON-safe row conversion used by query_file.

Regression: query_file was passing DuckDB row tuples directly to json.dumps,
which crashes with `Object of type date is not JSON serializable` whenever a
result set contains DATE / TIMESTAMP / TIME / UUID / DECIMAL / BLOB columns.
This is the largest single platform-side bug in tool_error_findings.md
(~93 query_file errors fired on plain `SELECT * FROM t LIMIT N`).
"""

import datetime
import decimal
import json
import unittest
import uuid
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

import sana_evaluation.tools.agent_tools as agent_tools
from sana_evaluation.tools.agent_tools import (
    _rewrite_execute_code_error,
    _parse_s3_reference,
    _resolve_file_reference,
    download,
    execute_code,
)
from sana_evaluation.tools.agent_tools_v2 import (
    _QUERY_MAX_FILE_BYTES,
    _query_file_impl,
    _normalize_sql_backticks,
    _rewrite_query_error,
    _rewrite_unqueryable_family_error,
    _strip_folder_prefix,
    _to_json_safe,
    query_file,
)


class TestToJsonSafe(unittest.TestCase):
    def test_passthrough_primitives(self):
        for v in [None, True, False, 0, 1, -3, 1.5, "abc", ""]:
            self.assertEqual(_to_json_safe(v), v)

    def test_date_converts_to_iso_string(self):
        self.assertEqual(_to_json_safe(datetime.date(2024, 7, 4)), "2024-07-04")

    def test_datetime_converts_to_iso_string(self):
        self.assertEqual(
            _to_json_safe(datetime.datetime(2024, 7, 4, 13, 5, 9)),
            "2024-07-04T13:05:09",
        )

    def test_time_converts_to_iso_string(self):
        self.assertEqual(_to_json_safe(datetime.time(13, 5, 9)), "13:05:09")

    def test_timedelta_converts_to_total_seconds(self):
        self.assertEqual(
            _to_json_safe(datetime.timedelta(hours=1, minutes=30)), 5400.0
        )

    def test_decimal_converts_to_string(self):
        # str preserves precision; float would lose it
        self.assertEqual(_to_json_safe(decimal.Decimal("12345.6789")), "12345.6789")

    def test_uuid_converts_to_string(self):
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        self.assertEqual(_to_json_safe(u), "12345678-1234-5678-1234-567812345678")

    def test_bytes_converts_to_base64_string(self):
        result = _to_json_safe(b"hello")
        self.assertIsInstance(result, str)
        # round-trippable as base64
        import base64
        self.assertEqual(base64.b64decode(result), b"hello")

    def test_bytearray_converts_to_base64_string(self):
        result = _to_json_safe(bytearray(b"world"))
        self.assertIsInstance(result, str)

    def test_list_recurses(self):
        self.assertEqual(
            _to_json_safe([datetime.date(2024, 1, 1), 1, "x"]),
            ["2024-01-01", 1, "x"],
        )

    def test_tuple_recurses_and_becomes_list(self):
        self.assertEqual(
            _to_json_safe((datetime.date(2024, 1, 1), 1)),
            ["2024-01-01", 1],
        )

    def test_dict_recurses(self):
        self.assertEqual(
            _to_json_safe({"d": datetime.date(2024, 1, 1), "n": 1}),
            {"d": "2024-01-01", "n": 1},
        )

    def test_nested_structures(self):
        # Mimics a DuckDB STRUCT column with a date inside
        value = {"created": datetime.datetime(2024, 1, 1, 0, 0, 0), "tags": [b"a", b"b"]}
        result = _to_json_safe(value)
        # Must be JSON-serializable end-to-end
        json.dumps(result)

    def test_full_row_set_serializes(self):
        # Simulates the exact shape that crashes query_file at line 515
        rows = [
            (1, "Alice", datetime.date(2024, 1, 15), decimal.Decimal("100.50")),
            (2, "Bob",   datetime.date(2024, 2, 28), decimal.Decimal("200.75")),
        ]
        safe_rows = [[_to_json_safe(v) for v in r] for r in rows]
        result = {
            "columns": ["id", "name", "created", "amount"],
            "rows": safe_rows,
            "row_count": len(safe_rows),
            "truncated": False,
        }
        # Pre-fix this raised TypeError; post-fix it must succeed.
        json.dumps(result)


class TestRewriteQueryError(unittest.TestCase):
    """
    Regression: when DuckDB's read_json_auto rejects a >100MB JSON object, the
    raw error tells the agent "Try increasing maximum_object_size" — which it
    cannot do. The agent then thrashes (retries the same query, malforms a
    download, etc.). _rewrite_query_error must convert it into the same
    actionable hint that the pre-check at line 483 uses: "use download +
    execute_code instead".
    """

    def test_maximum_object_size_error_is_rewritten_with_actionable_hint(self):
        raw = (
            'Invalid Input Error: "maximum_object_size" of 104857600 bytes '
            'exceeded while reading file '
            '"s3://lakeqa-yc4103-datalake/datagov/parking-violations-issued-in-january-2019/files/data.txt" '
            '(>115079012 bytes).\n Try increasing "maximum_object_size".'
        )
        rewritten = _rewrite_query_error(raw)
        self.assertIn("download", rewritten.lower())
        self.assertIn("execute_code", rewritten)
        # The misleading "Try increasing" hint must NOT be passed through.
        self.assertNotIn("increasing", rewritten.lower())

    def test_maximum_object_size_error_includes_observed_size(self):
        raw = (
            'Invalid Input Error: "maximum_object_size" of 104857600 bytes '
            'exceeded while reading file "s3://b/k.json" (>115079012 bytes).'
        )
        rewritten = _rewrite_query_error(raw)
        # Tells the agent how big the object actually was so it can plan
        self.assertIn("115079012", rewritten)

    def test_unrecognized_error_is_passed_through(self):
        raw = "Binder Error: Referenced column \"foo\" not found"
        self.assertEqual(_rewrite_query_error(raw), raw)

    def test_io_timeout_error_is_rewritten(self):
        raw = (
            "IO Error: Timeout was reached error for HTTP GET to "
            "'https://lakeqa-yc4103-datalake.s3.amazonaws.com/foo/bar.csv'"
        )
        rewritten = _rewrite_query_error(raw)
        # Should hint at the download path for big files that time out too
        self.assertIn("download", rewritten.lower())

    def test_parser_error_backtick_is_rewritten_with_quote_hint(self):
        # DuckDB rejects backtick identifiers with a Parser Error. The agent
        # has been observed using `column name` style ~17 times. Tell it the
        # actual fix: double quotes.
        raw = (
            'Parser Error: syntax error at or near "`OVERALL GRADE`"\n'
            'LINE 1: SELECT `OVERALL GRADE`, COUNT(*) FROM t GROUP BY 1'
        )
        rewritten = _rewrite_query_error(raw)
        # Must explicitly tell it to use double quotes
        self.assertIn('double quotes', rewritten.lower())
        self.assertIn('"', rewritten)
        # Original error context still surfaced for debugging
        self.assertIn('Parser Error', rewritten)

    def test_parser_error_without_backticks_is_passed_through(self):
        # A parser error that ISN'T about backticks should pass through
        # unchanged so we don't mislead the agent.
        raw = "Parser Error: syntax error at or near \"FRUM\""
        self.assertEqual(_rewrite_query_error(raw), raw)


class TestQueryFileJsonReaderLimit(unittest.TestCase):
    def test_json_reader_uses_direct_query_file_size_limit(self):
        executed_sql = []

        class FakeRelation:
            description = [("answer",)]

            def fetchmany(self, _limit):
                return [(1,)]

        class FakeConnection:
            def execute(self, sql):
                executed_sql.append(sql)
                return FakeRelation()

        with (
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._resolve_file_reference",
                return_value={
                    "dataset_id": "example",
                    "file_path": "files/data.json",
                    "s3_uri": "s3://bucket/example/files/data.json",
                    "key": "example/files/data.json",
                },
            ),
            mock.patch("sana_evaluation.tools.agent_tools_v2._get_s3_client"),
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._s3_head",
                return_value=_QUERY_MAX_FILE_BYTES,
            ),
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._s3_range_get",
                return_value=b'{"features":[]}',
            ),
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._duckdb_connection",
                return_value=FakeConnection(),
            ),
        ):
            result = _query_file_impl(
                s3_uri="s3://bucket/example/files/data.json",
                sql="SELECT 1 AS answer",
            )

        self.assertNotIn("error", result)
        create_view_sql = executed_sql[0]
        self.assertIn(
            f"maximum_object_size={_QUERY_MAX_FILE_BYTES}",
            create_view_sql,
        )


class TestQueryFileCsvReader(unittest.TestCase):
    def test_csv_reader_sets_standard_quote_character(self):
        executed_sql = []

        class FakeRelation:
            description = [("answer",)]

            def fetchmany(self, _limit):
                return [(1,)]

        class FakeConnection:
            def execute(self, sql):
                executed_sql.append(sql)
                return FakeRelation()

        with (
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._resolve_file_reference",
                return_value={
                    "dataset_id": "example",
                    "file_path": "files/data.csv",
                    "s3_uri": "s3://bucket/example/files/data.csv",
                    "key": "example/files/data.csv",
                },
            ),
            mock.patch("sana_evaluation.tools.agent_tools_v2._get_s3_client"),
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._s3_head",
                return_value=100,
            ),
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._s3_range_get",
                return_value=b'name,value\n"contains, comma",1\n',
            ),
            mock.patch(
                "sana_evaluation.tools.agent_tools_v2._duckdb_connection",
                return_value=FakeConnection(),
            ),
        ):
            result = _query_file_impl(
                s3_uri="s3://bucket/example/files/data.csv",
                sql="SELECT 1 AS answer",
            )

        self.assertNotIn("error", result)
        create_view_sql = executed_sql[0]
        self.assertIn(
            """read_csv_auto('s3://bucket/example/files/data.csv', quote='"')""",
            create_view_sql,
        )


class TestNormalizeSqlBackticks(unittest.TestCase):
    """
    Auto-fix MySQL-style backtick identifiers in agent SQL by converting them
    to DuckDB's double-quoted identifier syntax. Eliminates the ~17 backtick
    Parser Errors per eval at the source instead of after the fact. Must NOT
    touch backticks that appear inside string literals.
    """

    def test_simple_backtick_identifier_is_converted(self):
        self.assertEqual(
            _normalize_sql_backticks("SELECT `col` FROM t"),
            'SELECT "col" FROM t',
        )

    def test_identifier_with_spaces_is_converted(self):
        self.assertEqual(
            _normalize_sql_backticks("SELECT `OVERALL GRADE`, COUNT(*) FROM t GROUP BY 1"),
            'SELECT "OVERALL GRADE", COUNT(*) FROM t GROUP BY 1',
        )

    def test_multiple_identifiers_are_converted(self):
        self.assertEqual(
            _normalize_sql_backticks("SELECT `a`, `b`, `c` FROM `tbl`"),
            'SELECT "a", "b", "c" FROM "tbl"',
        )

    def test_no_backticks_unchanged(self):
        sql = 'SELECT col FROM t WHERE x = 1'
        self.assertEqual(_normalize_sql_backticks(sql), sql)

    def test_backtick_inside_single_quoted_literal_is_preserved(self):
        # The literal contains an apostrophe-like backtick — must not be touched
        sql = "SELECT * FROM t WHERE name = 'O`Brien'"
        self.assertEqual(_normalize_sql_backticks(sql), sql)

    def test_mixed_identifier_and_literal_with_backtick(self):
        self.assertEqual(
            _normalize_sql_backticks(
                "SELECT `name` FROM t WHERE label = 'has`backtick' AND `id` = 1"
            ),
            'SELECT "name" FROM t WHERE label = \'has`backtick\' AND "id" = 1',
        )

    def test_escaped_single_quote_inside_literal(self):
        # SQL escapes single quotes by doubling them: 'it''s' is the literal "it's"
        # Backticks inside that literal must still be preserved.
        sql = "SELECT `col` FROM t WHERE x = 'it''s `code`'"
        self.assertEqual(
            _normalize_sql_backticks(sql),
            'SELECT "col" FROM t WHERE x = \'it\'\'s `code`\'',
        )

    def test_backtick_inside_double_quoted_identifier_is_preserved(self):
        # An identifier already double-quoted that happens to contain a literal
        # backtick should be left alone.
        sql = 'SELECT "weird`col" FROM t'
        self.assertEqual(_normalize_sql_backticks(sql), sql)

    def test_empty_sql(self):
        self.assertEqual(_normalize_sql_backticks(""), "")

    def test_only_backticks(self):
        self.assertEqual(_normalize_sql_backticks("``"), '""')


class TestDownloadValidation(unittest.TestCase):
    """
    Regression: download was the single biggest bucket of agent-side errors
    (~132 calls, ~42% of all download attempts) — almost all "missing files
    wrapper" malforms where the agent passed `dataset_id`/`file_path` flat
    instead of `files=[{...}]`. The validation error messages and the
    docstring must show the correct envelope shape so the agent can self-
    correct AND must loudly mention the 5-file cap.
    """

    def _call(self, **kwargs):
        from sana_evaluation.tools.agent_tools import download
        fn = getattr(download, "original_function", download)
        return fn(**kwargs)

    def test_empty_list_error_shows_correct_envelope(self):
        result = self._call(files=[])
        self.assertIn("error", result)
        # Must show the literal envelope shape so the agent can copy it
        self.assertIn("files=[", result["error"])
        self.assertIn("dataset_id", result["error"])
        self.assertIn("file_path", result["error"])

    def test_non_list_error_shows_correct_envelope(self):
        # Agent often passes a single dict instead of a list of dicts
        result = self._call(files={"dataset_id": "x", "file_path": "y"})
        self.assertIn("error", result)
        self.assertIn("files=[", result["error"])
        self.assertIn("list", result["error"].lower())

    def test_too_many_files_error_mentions_cap_and_split_advice(self):
        files = [
            {"dataset_id": f"ds-{i}", "file_path": "files/data.txt"}
            for i in range(6)
        ]
        result = self._call(files=files)
        self.assertIn("error", result)
        # The cap itself
        self.assertIn("5", result["error"])
        # Tell the agent the remediation: split into multiple calls
        self.assertIn("split", result["error"].lower())

    def test_docstring_shows_correct_envelope(self):
        from sana_evaluation.tools.agent_tools import download
        fn = getattr(download, "original_function", download)
        doc = fn.__doc__ or ""
        # The docstring is what the agent reads when picking the tool — it
        # must contain the right shape, the 5-file cap, AND a CORRECT/WRONG
        # block to mirror peek_multiple's loud disambiguation.
        self.assertIn("files=[", doc)
        self.assertIn("Maximum 5", doc)
        self.assertIn("CORRECT", doc)
        self.assertIn("WRONG", doc)


class TestPeekMultipleRename(unittest.TestCase):
    """
    Regression: peek_files (plural) was malformed in 92% of agent calls because
    its name and signature collided with peek_file (singular). It was renamed
    to peek_multiple. This test pins the rename so it can't silently revert.
    """

    def test_peek_multiple_is_importable(self):
        from sana_evaluation.tools.agent_tools_v2 import peek_multiple  # noqa: F401

    def test_peek_files_is_no_longer_exported(self):
        import sana_evaluation.tools.agent_tools_v2 as mod
        self.assertNotIn("peek_files", mod.__all__)
        self.assertIn("peek_multiple", mod.__all__)
        self.assertFalse(hasattr(mod, "peek_files"))

    def test_peek_multiple_validates_files_argument(self):
        from sana_evaluation.tools.agent_tools_v2 import peek_multiple
        # The actual @tool decorator wraps the function — call the underlying
        # function directly via .original_function if present, else via the
        # wrapper.
        fn = getattr(peek_multiple, "original_function", peek_multiple)
        # Empty/missing files should yield an actionable error message that
        # explains the right shape (the malformation we're trying to prevent).
        result = fn()
        self.assertIn("error", result)
        self.assertIn("peek_multiple", result["error"])
        self.assertIn("peek_file", result["error"])  # mentions the alternative
        self.assertIn("files=[", result["error"])    # shows the right shape

        result = fn(files=[])
        self.assertIn("error", result)
        self.assertIn("peek_multiple", result["error"])
        self.assertIn("peek_file", result["error"])  # mentions the alternative
        self.assertIn("files=[", result["error"])    # shows the right shape


class TestRewriteUnqueryableFamilyError(unittest.TestCase):
    """
    Regression: query_file rejects non-CSV/JSON files with the bare message
    "File family 'text' is not queryable with SQL". The agent then doesn't
    know what to do — observed 22 times. Replace with an actionable hint
    naming the right tool for each family.
    """

    def test_text_family_message_suggests_text_tools(self):
        msg = _rewrite_unqueryable_family_error("text")
        self.assertIn("text", msg.lower())
        # Names the right alternative tools
        self.assertIn("peek_file", msg)
        self.assertIn("grep_file", msg)
        self.assertIn("read_file", msg)
        # Explicitly says query_file doesn't apply
        self.assertIn("query_file", msg)

    def test_xml_family_message_does_not_suggest_execute_code(self):
        msg = _rewrite_unqueryable_family_error("xml")
        self.assertIn("XML/KML", msg)
        self.assertIn("parse_xml_records", msg)
        self.assertIn("peek_file", msg)
        self.assertIn("grep_file", msg)
        self.assertNotIn("download + execute_code", msg)
        self.assertNotIn("xml.etree.ElementTree", msg)

    def test_unknown_family_message_still_useful(self):
        msg = _rewrite_unqueryable_family_error("binary")
        self.assertIn("binary", msg)
        self.assertIn("query_file", msg)
        self.assertNotIn("download + execute_code", msg)


class TestStripFolderPrefix(unittest.TestCase):
    """
    Regression: read_file (~38), peek_file (~2), and grep_file (~1) errors
    per eval all originate from the agent constructing dataset_ids of the
    form `wikipedia/<page>` after seeing a Wikipedia mention. The actual id
    is the bare name. _strip_folder_prefix silently normalizes these so the
    tools resolve correctly instead of returning "Dataset not found".
    """

    def test_strips_wikipedia_prefix(self):
        self.assertEqual(_strip_folder_prefix("wikipedia/Logan_Fontenelle"), "Logan_Fontenelle")

    def test_strips_datagov_prefix(self):
        self.assertEqual(_strip_folder_prefix("datagov/index-crimes-by-county"), "index-crimes-by-county")

    def test_case_insensitive_prefix(self):
        self.assertEqual(_strip_folder_prefix("WIKIPEDIA/Barack_Obama"), "Barack_Obama")
        self.assertEqual(_strip_folder_prefix("Wikipedia/Barack_Obama"), "Barack_Obama")
        self.assertEqual(_strip_folder_prefix("DataGov/foo-bar"), "foo-bar")

    def test_strips_leading_slash_then_prefix(self):
        self.assertEqual(_strip_folder_prefix("/wikipedia/Houston_Rockets"), "Houston_Rockets")

    def test_bare_dataset_id_unchanged(self):
        self.assertEqual(_strip_folder_prefix("Logan_Fontenelle"), "Logan_Fontenelle")
        self.assertEqual(_strip_folder_prefix("index-crimes-by-county"), "index-crimes-by-county")

    def test_preserves_underscores_and_hyphens_in_name(self):
        # The prefix is only stripped at the start; characters elsewhere are untouched
        self.assertEqual(
            _strip_folder_prefix("wikipedia/Some_Page_With_Many_Underscores"),
            "Some_Page_With_Many_Underscores",
        )

    def test_idempotent(self):
        once = _strip_folder_prefix("wikipedia/Logan_Fontenelle")
        twice = _strip_folder_prefix(once)
        self.assertEqual(once, twice)

    def test_empty_string_unchanged(self):
        self.assertEqual(_strip_folder_prefix(""), "")

    def test_none_unchanged(self):
        self.assertIsNone(_strip_folder_prefix(None))

    def test_non_string_unchanged(self):
        # Defensive: don't crash if a caller passes a non-string
        self.assertEqual(_strip_folder_prefix(123), 123)

    def test_prefix_only_returns_empty_string(self):
        # `wikipedia/` with nothing after — caller's empty-check will catch it
        self.assertEqual(_strip_folder_prefix("wikipedia/"), "")
        self.assertEqual(_strip_folder_prefix("datagov/"), "")

    def test_unrelated_prefix_unchanged(self):
        # A dataset_id that happens to look like it has a prefix but doesn't match
        # one of the two known folders is left alone.
        self.assertEqual(_strip_folder_prefix("other/foo"), "other/foo")
        self.assertEqual(_strip_folder_prefix("wiki/foo"), "wiki/foo")

    def test_substring_match_does_not_strip(self):
        # `wikipedia` without trailing slash is not a prefix to strip
        self.assertEqual(_strip_folder_prefix("wikipediathing"), "wikipediathing")


class TestS3ReferenceHelpers(unittest.TestCase):
    def test_parse_full_s3_uri(self):
        parsed = _parse_s3_reference(
            "s3://lakeqa-yc4103-datalake/datagov/index-crimes-by-county/files/rows.txt"
        )
        self.assertEqual(parsed["folder"], "datagov")
        self.assertEqual(parsed["dataset_id"], "index-crimes-by-county")
        self.assertEqual(parsed["file_path"], "files/rows.txt")

    def test_parse_bucket_relative_key(self):
        parsed = _parse_s3_reference(
            "datagov/index-crimes-by-county/files/rows.txt"
        )
        self.assertEqual(parsed["dataset_id"], "index-crimes-by-county")
        self.assertEqual(parsed["file_path"], "files/rows.txt")

    def test_parse_rejects_wrong_bucket(self):
        parsed = _parse_s3_reference("s3://other-bucket/datagov/foo/files/rows.txt")
        self.assertIn("error", parsed)
        self.assertIn("lakeqa-yc4103-datalake", parsed["error"])

    def test_resolve_prefers_s3_uri(self):
        parsed = _resolve_file_reference(
            s3_uri="s3://lakeqa-yc4103-datalake/wikipedia/Barack_Obama/content.txt"
        )
        self.assertEqual(parsed["folder"], "wikipedia")
        self.assertEqual(parsed["dataset_id"], "Barack_Obama")
        self.assertEqual(parsed["file_path"], "content.txt")


class TestRewriteExecuteCodeError(unittest.TestCase):
    """
    Regression: 82 execute_code errors per eval. The two trivial fixes
    (SANDBOX_DIR env var + ijson pre-install) clear ~22. The rewrite helper
    appends a `hint` field to the remaining ~20 known patterns so the agent
    has actionable next-step guidance instead of staring at a raw traceback.
    Each test case below uses a real traceback fragment from the eval logs.
    """

    def test_keyerror_features_suggests_peek_for_geojson(self):
        err = "KeyError: 'features'"
        tb = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 8, in <module>\n'
            "KeyError: 'features'\n"
        )
        hint = _rewrite_execute_code_error(err, tb)
        self.assertIsNotNone(hint)
        self.assertIn("peek_file", hint)
        self.assertIn("GeoJSON", hint)

    def test_jsondecodeerror_first_byte_suggests_peek(self):
        err = "JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
        tb = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 4, in <module>\n'
            "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
        )
        hint = _rewrite_execute_code_error(err, tb)
        self.assertIsNotNone(hint)
        self.assertIn("peek_file", hint)
        # Should mention CSV since many .txt files in this lake are CSV
        self.assertIn("CSV", hint)

    def test_nonetype_plus_str_suggests_field_check(self):
        err = "TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'"
        tb = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 12, in <module>\n'
            "TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'\n"
        )
        hint = _rewrite_execute_code_error(err, tb)
        self.assertIsNotNone(hint)
        self.assertIn("None", hint)
        self.assertIn("peek_file", hint)

    def test_infer_datetime_format_suggests_drop_arg(self):
        err = "TypeError: to_datetime() got an unexpected keyword argument 'infer_datetime_format'"
        tb = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 7, in <module>\n'
            "TypeError: to_datetime() got an unexpected keyword argument 'infer_datetime_format'\n"
        )
        hint = _rewrite_execute_code_error(err, tb)
        self.assertIsNotNone(hint)
        self.assertIn("infer_datetime_format", hint)
        # Should tell agent to drop the arg
        self.assertIn("Drop", hint)

    def test_max_empty_iterable_suggests_length_check(self):
        for variant in (
            "ValueError: max() iterable argument is empty",
            "ValueError: min() iterable argument is empty",
            "ValueError: max() arg is an empty sequence",
            "ValueError: attempt to get argmax of an empty sequence",
        ):
            with self.subTest(variant=variant):
                hint = _rewrite_execute_code_error(variant, "")
                self.assertIsNotNone(hint)
                self.assertIn("empty", hint)

    def test_usecols_mismatch_suggests_peek(self):
        err = (
            "ValueError: Usecols do not match columns, columns expected but not found: "
            "['Incident Date']"
        )
        hint = _rewrite_execute_code_error(err, "")
        self.assertIsNotNone(hint)
        self.assertIn("peek_file", hint)
        self.assertIn("usecols", hint)

    def test_module_not_found_lists_available_modules(self):
        err = "ModuleNotFoundError: No module named 'pyarrow'"
        tb = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "ModuleNotFoundError: No module named 'pyarrow'\n"
        )
        hint = _rewrite_execute_code_error(err, tb)
        self.assertIsNotNone(hint)
        # Names the missing module
        self.assertIn("pyarrow", hint)
        # Lists at least the common pre-imports including ijson
        self.assertIn("pandas", hint)
        self.assertIn("ijson", hint)
        # Tells agent network is blocked so they can't pip install
        self.assertIn("network", hint.lower())

    def test_xml_parser_on_non_xml_suggests_peek(self):
        err = "ParseError: not well-formed (invalid token): line 1, column 0"
        tb = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 16, in <module>\n'
            "xml.etree.ElementTree.ParseError: not well-formed (invalid token): line 1, column 0\n"
        )
        hint = _rewrite_execute_code_error(err, tb)
        self.assertIsNotNone(hint)
        self.assertIn("peek_file", hint)
        self.assertIn("parse_xml_records", hint)
        self.assertIn("Do not use execute_code", hint)

    def test_execute_code_docstring_points_xml_to_parse_xml_records(self):
        fn = getattr(
            execute_code,
            "_tool_func",
            getattr(execute_code, "original_function", execute_code),
        )
        doc = fn.__doc__ or ""
        self.assertIn("XML/KML", doc)
        self.assertIn("parse_xml_records", doc)

    def test_unknown_error_returns_none(self):
        # Errors we don't have a pattern for should pass through unchanged
        err = "RuntimeError: something exotic that we don't recognize"
        self.assertIsNone(_rewrite_execute_code_error(err, ""))

    def test_empty_error_returns_none(self):
        self.assertIsNone(_rewrite_execute_code_error("", ""))
        self.assertIsNone(_rewrite_execute_code_error(None, None))


class TestExecuteCodeSandboxEnvAndIjson(unittest.TestCase):
    """
    Integration tests for the two direct fixes (scope A):
      - os.environ['SANDBOX_DIR'] is set during exec and restored after
      - ijson is pre-imported and available

    These run real execute_code calls so they exercise the full setup/teardown
    path including the env var save/restore in the finally block.
    """

    def test_sandbox_dir_is_in_os_environ_during_exec(self):
        result = execute_code("import os; print(os.environ['SANDBOX_DIR'])")
        self.assertTrue(result.get("success"), result)
        self.assertIn("output", result)
        # The printed sandbox dir should match the one in the result
        self.assertEqual(result["output"].strip(), result["sandbox_dir"])

    def test_sandbox_dir_env_var_is_restored_after_exec(self):
        # Set a sentinel before, confirm it's restored after
        prior_value = "SENTINEL_VALUE_FOR_TEST"
        import os as _os
        _os.environ['SANDBOX_DIR'] = prior_value
        try:
            execute_code("print('hello')")
            self.assertEqual(_os.environ.get('SANDBOX_DIR'), prior_value)
        finally:
            _os.environ.pop('SANDBOX_DIR', None)

    def test_sandbox_dir_env_var_is_removed_after_exec_when_unset(self):
        # When SANDBOX_DIR was not set before the call, it should not exist after
        import os as _os
        _os.environ.pop('SANDBOX_DIR', None)
        execute_code("print('hello')")
        self.assertNotIn('SANDBOX_DIR', _os.environ)

    def test_ijson_is_pre_imported(self):
        # The agent should be able to use ijson without an explicit import
        result = execute_code("print(ijson.__name__)")
        self.assertTrue(result.get("success"), result)
        self.assertEqual(result["output"].strip(), "ijson")

    def test_download_manifest_path_is_available_during_exec(self):
        old_sandbox = agent_tools._SANDBOX_DIR
        old_override = agent_tools._SANDBOX_OVERRIDE
        with TemporaryDirectory() as tmpdir:
            sandbox = Path(tmpdir)
            manifest_path = sandbox / ".download_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "downloaded": [],
                        "local_paths": ["/sandbox/example/files/data.csv"],
                        "path_map": {
                            "example/files/data.csv": "/sandbox/example/files/data.csv",
                        },
                    }
                )
            )
            try:
                agent_tools.set_sandbox_dir(sandbox)
                result = agent_tools._execute_code_impl(
                    "import os\n"
                    "print(DOWNLOAD_MANIFEST_PATH == os.environ['DOWNLOAD_MANIFEST_PATH'])\n"
                    "print(DOWNLOAD_PATHS['example/files/data.csv'])"
                )
            finally:
                agent_tools._SANDBOX_DIR = old_sandbox
                agent_tools._SANDBOX_OVERRIDE = old_override

        self.assertTrue(result.get("success"), result)
        self.assertIn("True", result["output"])
        self.assertIn("/sandbox/example/files/data.csv", result["output"])

    def test_module_not_found_attaches_hint(self):
        # Errors should get the hint field appended via _rewrite_execute_code_error
        result = execute_code("import nonexistent_module_xyz")
        self.assertFalse(result.get("success"))
        self.assertIn("hint", result)
        self.assertIn("nonexistent_module_xyz", result["hint"])
        self.assertIn("ijson", result["hint"])

    def test_unknown_error_has_no_hint(self):
        # Errors without a known pattern should not get a hint field
        result = execute_code("raise RuntimeError('something exotic that we don\\'t recognize')")
        self.assertFalse(result.get("success"))
        self.assertNotIn("hint", result)

    @mock.patch("sana_evaluation.tools.agent_tools._run_tool_with_timeout", return_value=(False, None))
    def test_execute_code_timeout_returns_failure(self, _patched_timeout):
        result = execute_code("print('hello')")
        self.assertFalse(result.get("success"))
        self.assertIn("timed out after 150s", result.get("error", ""))
        self.assertIn("sandbox_dir", result)


class TestToolTimeoutWrappers(unittest.TestCase):
    @mock.patch("sana_evaluation.tools.agent_tools._run_tool_with_timeout", return_value=(False, None))
    def test_download_timeout_returns_error(self, _patched_timeout):
        result = download([{"dataset_id": "example", "file_path": "files/data.txt"}])
        self.assertIn("timed out after 150s", result.get("error", ""))
        self.assertEqual(result.get("download_count"), 0)

    @mock.patch("sana_evaluation.tools.agent_tools_v2._run_tool_with_timeout", return_value=(False, None))
    def test_query_file_timeout_returns_error(self, _patched_timeout):
        result = query_file(dataset_id="example", file_path="files/data.txt", sql="SELECT 1")
        self.assertIn("timed out after 150s", result.get("error", ""))
        self.assertIn("download + execute_code", result.get("error", ""))


class TestDownloadManifestAndErrors(unittest.TestCase):
    def setUp(self):
        self.old_sandbox = agent_tools._SANDBOX_DIR
        self.old_override = agent_tools._SANDBOX_OVERRIDE
        self.old_bucket = agent_tools.BUCKET
        agent_tools.configure_benchmark("lakeqa")

    def tearDown(self):
        agent_tools._SANDBOX_DIR = self.old_sandbox
        agent_tools._SANDBOX_OVERRIDE = self.old_override
        agent_tools.BUCKET = self.old_bucket

    def test_download_writes_manifest_and_returns_path_map(self):
        class FakeS3:
            def download_file(self, Bucket, Key, Filename):
                Path(Filename).write_text("a\n1\n", encoding="utf-8")

        with TemporaryDirectory() as tmpdir:
            sandbox = Path(tmpdir)
            agent_tools.set_sandbox_dir(sandbox)
            with mock.patch("sana_evaluation.tools.agent_tools._get_s3_client", return_value=FakeS3()):
                result = agent_tools._download_impl(
                    [
                        {
                            "s3_uri": "s3://lakeqa-yc4103-datalake/datagov/example/files/data.csv",
                        }
                    ]
                )
            manifest_exists = Path(result["manifest_path"]).is_file()

        self.assertEqual(result["download_count"], 1, result)
        local_path = result["downloaded"][0]["local_path"]
        self.assertEqual(
            result["path_map"]["s3://lakeqa-yc4103-datalake/datagov/example/files/data.csv"],
            local_path,
        )
        self.assertEqual(result["path_map"]["example/files/data.csv"], local_path)
        self.assertNotIn("local_paths", result)
        self.assertNotIn("python_snippet", result)
        self.assertTrue(manifest_exists)

    def test_download_missing_key_returns_bucket_key_and_close_candidates(self):
        class FakeS3:
            def download_file(self, Bucket, Key, Filename):
                raise agent_tools.ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )

            def list_objects_v2(self, Bucket, Prefix, MaxKeys):
                return {
                    "Contents": [
                        {"Key": "datagov/example/files/right-file.csv"},
                        {"Key": "datagov/example/files/other.csv"},
                    ]
                }

        with TemporaryDirectory() as tmpdir:
            agent_tools.set_sandbox_dir(Path(tmpdir))
            with mock.patch("sana_evaluation.tools.agent_tools._get_s3_client", return_value=FakeS3()):
                result = agent_tools._download_impl(
                    [
                        {
                            "s3_uri": "s3://lakeqa-yc4103-datalake/datagov/example/files/right_fiel.csv",
                        }
                    ]
                )

        error = result["errors"][0]
        self.assertEqual(error["bucket"], "lakeqa-yc4103-datalake")
        self.assertEqual(error["key"], "datagov/example/files/right_fiel.csv")
        self.assertIn("object not found", error["error"])
        self.assertIn("files/right-file.csv", error["available_files_preview"])
        self.assertTrue(error["did_you_mean"], error)


if __name__ == "__main__":
    unittest.main()

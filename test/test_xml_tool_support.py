import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sana_evaluation.tools import agent_tools_v2
from sana_evaluation.tools.helper.detect import detect_family


class TestDetectFamilyXml(unittest.TestCase):
    def test_xml_declaration_returns_xml(self):
        content = '<?xml version="1.0" encoding="UTF-8"?><root><item>1</item></root>'
        self.assertEqual(detect_family(content), "xml")

    def test_kml_root_returns_xml(self):
        content = '<kml xmlns="http://www.opengis.net/kml/2.2"><Document /></kml>'
        self.assertEqual(detect_family(content), "xml")

    def test_json_detection_unchanged(self):
        self.assertEqual(detect_family('{"name": "alice"}'), "json")

    def test_csv_detection_unchanged(self):
        self.assertEqual(detect_family("name,city\nAlice,Boston\nBob,Denver\n"), "csv")

    def test_plain_text_with_angle_bracket_later_is_not_xml(self):
        content = "Plain text first.\nA later snippet uses <tag>inline markup</tag>."
        self.assertEqual(detect_family(content), "text")


class TestPeekFileXmlSupport(unittest.TestCase):
    def _call_peek(self, text: str, size_bytes: int | None = None):
        fn = getattr(
            agent_tools_v2.peek_file,
            "_tool_func",
            getattr(agent_tools_v2.peek_file, "original_function", agent_tools_v2.peek_file),
        )
        payload = text.encode("utf-8")
        reported_size = len(payload) if size_bytes is None else size_bytes
        with (
            patch.object(agent_tools_v2, "_get_s3_client", return_value=Mock()),
            patch.object(agent_tools_v2, "_s3_head", return_value=reported_size),
            patch.object(agent_tools_v2, "_s3_range_get", return_value=payload),
        ):
            return fn(
                s3_uri="s3://lakeqa-yc4103-datalake/datagov/demo-dataset/files/data.xml",
                max_rows=20,
            )

    def test_complete_small_xml_uses_parsed_preview_mode(self):
        text = (
            '<?xml version="1.0"?>\n'
            '<root xmlns="urn:test">\n'
            '  <record><name>Alice</name></record>\n'
            '  <record><name>Bob</name></record>\n'
            '</root>\n'
        )
        result = self._call_peek(text)
        self.assertEqual(result["family"], "xml")
        self.assertEqual(result["xml_preview_mode"], "parsed")
        self.assertEqual(result["xml_root_tag"], "root")
        self.assertEqual(result["xml_namespaces"], {"default": "urn:test"})
        self.assertIn("record", result["xml_record_tag_candidates"])

    def test_truncated_xml_snippet_uses_heuristic_mode_without_crashing(self):
        text = (
            '<?xml version="1.0"?>\n'
            '<root xmlns="urn:test">\n'
            '  <record><name>Alice</name></record>\n'
            '  <record>\n'
        )
        result = self._call_peek(text, size_bytes=agent_tools_v2._PEEK_BYTES + 128)
        self.assertEqual(result["family"], "xml")
        self.assertEqual(result["xml_preview_mode"], "heuristic")
        self.assertEqual(result["xml_root_tag"], "root")
        self.assertEqual(result["xml_namespaces"], {"default": "urn:test"})
        self.assertIn("record", result["xml_record_tag_candidates"])

    def test_kml_preview_extracts_schema_fields_and_placemark(self):
        text = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            '  <Document>\n'
            '    <Schema name="schools" id="schools">\n'
            '      <SimpleField name="NCESSCH" type="string"/>\n'
            '      <SimpleField name="CITY" type="string"/>\n'
            '    </Schema>\n'
            '    <Placemark>\n'
            '      <name>School A</name>\n'
            '      <ExtendedData>\n'
            '        <SchemaData>\n'
            '          <SimpleData name="NCESSCH">0001</SimpleData>\n'
            '          <SimpleData name="CITY">Austin</SimpleData>\n'
            '        </SchemaData>\n'
            '      </ExtendedData>\n'
            '    </Placemark>\n'
            '  </Document>\n'
            '</kml>\n'
        )
        result = self._call_peek(text)
        self.assertEqual(result["family"], "xml")
        self.assertEqual(result["xml_preview_mode"], "parsed")
        self.assertEqual(result["xml_root_tag"], "kml")
        self.assertEqual(result["xml_schema_fields"], ["NCESSCH", "CITY"])
        self.assertIn("Placemark", result["xml_record_tag_candidates"])


class TestQueryFileXmlSupport(unittest.TestCase):
    def test_xml_family_error_is_actionable(self):
        msg = agent_tools_v2._rewrite_unqueryable_family_error("xml")
        self.assertIn("XML/KML", msg)
        self.assertIn("parse_xml_records", msg)
        self.assertIn("peek_file", msg)
        self.assertIn("grep_file", msg)
        self.assertIn("read_file", msg)
        self.assertNotIn("download + execute_code", msg)
        self.assertNotIn("xml.etree.ElementTree", msg)

    def test_query_file_detects_xml_and_returns_xml_specific_hint(self):
        text = '<?xml version="1.0"?><root><record>1</record></root>'
        with (
            patch.object(agent_tools_v2, "_resolve_dataset_folder", return_value="datagov"),
            patch.object(agent_tools_v2, "_get_s3_client", return_value=Mock()),
            patch.object(agent_tools_v2, "_s3_head", return_value=len(text.encode("utf-8"))),
            patch.object(agent_tools_v2, "_s3_range_get", return_value=text.encode("utf-8")),
        ):
            result = agent_tools_v2._query_file_impl(
                s3_uri="s3://lakeqa-yc4103-datalake/datagov/demo-dataset/files/data.xml",
                sql="SELECT * FROM t LIMIT 1",
            )

        self.assertIn("error", result)
        self.assertIn("XML/KML", result["error"])
        self.assertIn("parse_xml_records", result["error"])
        self.assertNotIn("plain text", result["error"].lower())

    def test_query_file_rejects_large_xml_txt_before_large_file_hint(self):
        text = '<kml xmlns="http://www.opengis.net/kml/2.2"><Document /></kml>'
        with (
            patch.object(agent_tools_v2, "_get_s3_client", return_value=Mock()),
            patch.object(agent_tools_v2, "_s3_head", return_value=agent_tools_v2._QUERY_MAX_FILE_BYTES + 1),
            patch.object(agent_tools_v2, "_s3_range_get", return_value=text.encode("utf-8")),
        ):
            result = agent_tools_v2._query_file_impl(
                s3_uri="s3://lakeqa-yc4103-datalake/datagov/demo-dataset/files/data.txt",
                sql="SELECT * FROM t LIMIT 1",
            )

        self.assertIn("error", result)
        self.assertIn("XML/KML", result["error"])
        self.assertIn("parse_xml_records", result["error"])
        self.assertNotIn("File too large", result["error"])
        self.assertNotIn("download + execute_code", result["error"])

    def test_query_file_does_not_open_duckdb_for_xml_txt_file(self):
        text = '<?xml version="1.0"?><root><record>1</record></root>'
        with (
            patch.object(agent_tools_v2, "_get_s3_client", return_value=Mock()),
            patch.object(agent_tools_v2, "_s3_head", return_value=len(text.encode("utf-8"))),
            patch.object(agent_tools_v2, "_s3_range_get", return_value=text.encode("utf-8")),
            patch.object(agent_tools_v2, "_duckdb_connection") as duckdb_connection,
        ):
            result = agent_tools_v2._query_file_impl(
                s3_uri="s3://lakeqa-yc4103-datalake/datagov/demo-dataset/files/data.txt",
                sql="SELECT * FROM t LIMIT 1",
            )

        self.assertIn("error", result)
        self.assertIn("parse_xml_records", result["error"])
        duckdb_connection.assert_not_called()

    def test_query_file_docstring_mentions_xml_detection_behavior(self):
        fn = getattr(
            agent_tools_v2.query_file,
            "_tool_func",
            getattr(agent_tools_v2.query_file, "original_function", agent_tools_v2.query_file),
        )
        doc = fn.__doc__ or ""
        self.assertIn("Supported file types: CSV", doc)
        self.assertIn("XML/KML is detected but not queryable", doc)


class TestParseXmlRecords(unittest.TestCase):
    def _call_parse_xml_records(self, text: str, **kwargs):
        payload = text.encode("utf-8")
        s3 = Mock()
        s3.get_object.return_value = {"Body": io.BytesIO(payload)}
        with (
            patch.object(agent_tools_v2, "_get_s3_client", return_value=s3),
            patch.object(agent_tools_v2, "_s3_head", return_value=len(payload)),
            patch.object(agent_tools_v2, "_s3_range_get", return_value=payload),
        ):
            result = agent_tools_v2._parse_xml_records_impl(
                s3_uri="s3://lakeqa-yc4103-datalake/datagov/demo-dataset/files/data.txt",
                **kwargs,
            )
        return result, s3

    def test_kml_simpledata_records_can_be_filtered_and_grouped(self):
        text = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            '  <Document>\n'
            '    <Placemark><name>School A</name><ExtendedData><SchemaData>\n'
            '      <SimpleData name="STFIP">06</SimpleData>\n'
            '      <SimpleData name="NMCNTY">Los Angeles County</SimpleData>\n'
            '    </SchemaData></ExtendedData></Placemark>\n'
            '    <Placemark><name>School B</name><ExtendedData><SchemaData>\n'
            '      <SimpleData name="STFIP">06</SimpleData>\n'
            '      <SimpleData name="NMCNTY">Los Angeles County</SimpleData>\n'
            '    </SchemaData></ExtendedData></Placemark>\n'
            '    <Placemark><name>School C</name><ExtendedData><SchemaData>\n'
            '      <SimpleData name="STFIP">06</SimpleData>\n'
            '      <SimpleData name="NMCNTY">San Diego County</SimpleData>\n'
            '    </SchemaData></ExtendedData></Placemark>\n'
            '    <Placemark><name>School D</name><ExtendedData><SchemaData>\n'
            '      <SimpleData name="STFIP">48</SimpleData>\n'
            '      <SimpleData name="NMCNTY">Travis County</SimpleData>\n'
            '    </SchemaData></ExtendedData></Placemark>\n'
            '  </Document>\n'
            '</kml>\n'
        )
        payload = text.encode("utf-8")
        s3 = Mock()
        s3.get_object.return_value = {"Body": io.BytesIO(payload)}

        with (
            patch.object(agent_tools_v2, "_get_s3_client", return_value=s3),
            patch.object(agent_tools_v2, "_s3_head", return_value=len(payload)),
            patch.object(agent_tools_v2, "_s3_range_get", return_value=payload),
        ):
            result = agent_tools_v2._parse_xml_records_impl(
                s3_uri=(
                    "s3://lakeqa-yc4103-datalake/datagov/"
                    "public-school-locations-current-23297/files/schools.kml"
                ),
                record_tag="Placemark",
                filters={"STFIP": "06"},
                group_by=["NMCNTY"],
                limit=10,
            )

        self.assertNotIn("error", result)
        self.assertEqual(result["family"], "xml")
        self.assertEqual(result["record_tag"], "Placemark")
        self.assertEqual(result["scanned_records"], 4)
        self.assertEqual(result["matched_records"], 3)
        self.assertEqual(
            result["rows"],
            [
                {"NMCNTY": "Los Angeles County", "count": 2},
                {"NMCNTY": "San Diego County", "count": 1},
            ],
        )

    def test_plain_xml_records_can_be_auto_detected_filtered_selected_and_limited(self):
        text = (
            "<root>"
            "<row><name>Alice</name><state>CA</state><score>10</score></row>"
            "<row><name>Bob</name><state>WA</state><score>8</score></row>"
            "<row><name>Cyd</name><state>CA</state><score>7</score></row>"
            "</root>"
        )

        result, _s3 = self._call_parse_xml_records(
            text,
            fields=["name", "score"],
            filters={"state": ["CA", "WA"]},
            limit=2,
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["record_tag"], "row")
        self.assertEqual(result["scanned_records"], 3)
        self.assertEqual(result["matched_records"], 3)
        self.assertTrue(result["truncated"])
        self.assertEqual(
            result["rows"],
            [
                {"name": "Alice", "score": "10"},
                {"name": "Bob", "score": "8"},
            ],
        )

    def test_namespaced_record_tags_and_record_attributes_are_extracted(self):
        text = (
            '<feed xmlns:a="urn:demo">'
            '<a:item id="A1" kind="school"><a:name>Alpha</a:name></a:item>'
            '<a:item id="B2" kind="library"><a:name>Beta</a:name></a:item>'
            "</feed>"
        )

        result, _s3 = self._call_parse_xml_records(
            text,
            record_tag="a:item",
            fields=["id", "kind", "name"],
            filters={"kind": "school"},
        )

        self.assertNotIn("error", result)
        self.assertEqual(result["record_tag"], "item")
        self.assertEqual(result["matched_records"], 1)
        self.assertEqual(result["rows"], [{"id": "A1", "kind": "school", "name": "Alpha"}])

    def test_non_xml_input_is_rejected_without_streaming_body(self):
        result, s3 = self._call_parse_xml_records(
            "name,state\nAlice,CA\nBob,WA\n",
            record_tag="row",
        )

        self.assertIn("error", result)
        self.assertIn("only supports XML/KML", result["error"])
        s3.get_object.assert_not_called()

    def test_malformed_xml_returns_parse_error(self):
        text = (
            '<?xml version="1.0"?>'
            "<root><row><name>Alice</name></row><row><name>Bob</name></root>"
        )

        result, _s3 = self._call_parse_xml_records(text, record_tag="row")

        self.assertIn("error", result)
        self.assertIn("XML/KML parse failed", result["error"])


class TestPromptContract(unittest.TestCase):
    def test_system_prompt_mentions_xml_preview_and_query_limit(self):
        prompt = Path("sana_evaluation/prompts/baseline.txt").read_text()
        self.assertIn("CSV/JSON/XML/text", prompt)
        self.assertIn("parse_xml_records", prompt)
        self.assertIn("do not use `execute_code` for XML/KML extraction", prompt)


if __name__ == "__main__":
    unittest.main()

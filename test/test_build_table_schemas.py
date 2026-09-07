"""Deriving table schemas from stored bytes rather than a catalogue.

Every case here is taken from the bucket. The two rejection cases are the ones
that made the obvious tests unusable: detect_family calls any line with a comma
"csv", and detect.is_table_content -- which checks delimiter consistency across
lines -- both accepts a list of "Surname, Given" names and rejects a real table
whose quoted WKT geometry spans lines.
"""
import json

from dataindexing.cli.build_table_schemas import derive_table


def _key(name="rows.txt"):
    return f"datagov/some-dataset/files/{name}"


class TestAccepts:
    def test_real_csv_header(self):
        head = ("the_geom,ExpireDT,IssueDT,Route,UpdateDT,linktxt\n"
                '"MULTILINESTRING ((-93.83 42.03, -93.83 42.03))",1,2,3,4,5\n')
        rec = derive_table(_key(), head)
        assert rec is not None
        assert rec["table_kind"] == "delimited_text"
        assert rec["delimiter"] == ","
        assert rec["columns"][:3] == ["the_geom", "ExpireDT", "IssueDT"]

    def test_header_survives_a_multiline_quoted_field(self):
        # The geometry's commas break a line-by-line delimiter-consistency check,
        # which is why that test is not the gate.
        head = ('a,b,c,d\n"MULTILINESTRING ((1 2,\n3 4,\n5 6))",1,2,3\n')
        assert derive_table(_key(), head) is not None

    def test_json_object_columns_from_keys(self):
        head = json.dumps({"beta": 1, "alpha": 2, "gamma": 3})
        rec = derive_table(_key("data.txt"), head)
        assert rec["table_kind"] == "json"
        assert rec["columns"] == ["alpha", "beta", "gamma"]

    def test_geojson_columns_come_from_feature_properties(self):
        head = json.dumps({
            "type": "FeatureCollection",
            "features": [{"type": "Feature",
                          "properties": {"pop": 1, "name": "x", "area": 2}}],
        })
        rec = derive_table(_key("data.txt"), head)
        assert rec["table_kind"] == "geojson"
        assert rec["columns"] == ["area", "name", "pop"]

    def test_relative_path_is_taken_from_the_files_segment(self):
        rec = derive_table(_key("rows.txt"), "a,b,c\n1,2,3\n")
        assert rec["relative_path"] == "rows.txt"


class TestRejects:
    def test_author_list_is_not_a_table(self):
        # Every line has exactly one comma, so delimiter consistency is perfect
        # and the "header" would be the first data row.
        head = "Petersen, Mark D.\nMueller, Charles S.\nMoschetti, Morgan P.\n"
        assert derive_table(_key("USGS.58ab2b35.txt"), head) is None

    def test_title_split_on_commas_is_not_a_table(self):
        head = ("Watershed characteristics and streamwater constituent load data,"
                "models,and estimates for 15 watersheds in Gwinnett County,"
                "Georgia,2000-2021 - ScienceBase-Catalog\n")
        assert derive_table(_key("58795a8c.txt"), head) is None

    def test_two_column_split_is_not_enough(self):
        assert derive_table(_key(), "Seismic Hazard,Risk\nfoo,bar\n") is None

    def test_plain_prose_yields_nothing(self):
        assert derive_table(_key("readme.txt"), "This dataset describes things.\n") is None

    def test_empty_content_yields_nothing(self):
        assert derive_table(_key(), "") is None


class TestBinaryAndMetadata:
    """Guards added after a 3000-dataset sample, each for something it found."""

    def test_zip_stored_as_txt_is_rejected(self):
        # should_skip only sees filenames; these arrive as .txt and their
        # compressed bytes contain commas, which parsed as a 2180-column header.
        head = "PK\x03\x04\x14\x00\x00\x00ag_roosevelt_1936.json,Project,Agency\n"
        assert derive_table(_key("archive.txt"), head) is None

    def test_content_with_nul_bytes_is_rejected(self):
        assert derive_table(_key(), "a,b,c\n\x00\x00\x00binary\n") is None

    def test_absurd_column_count_is_rejected(self):
        assert derive_table(_key(), ",".join(f"c{i}" for i in range(900)) + "\n") is None

    def test_json_ld_envelope_is_not_a_table(self):
        head = json.dumps({"@context": {}, "@id": "x", "@type": "dcat:Catalog",
                           "dataset": [], "describedBy": "y"})
        assert derive_table(_key("catalog-record.txt"), head) is None

    def test_service_descriptor_is_not_a_table(self):
        head = json.dumps({"capabilities": "Query", "supportedQueryFormats": "JSON",
                           "cacheMaxAge": 0, "name": "layer"})
        assert derive_table(_key("arcgis-layer.txt"), head) is None

    def test_single_key_envelope_is_not_a_table(self):
        assert derive_table(_key("tags.txt"), json.dumps({"tags": ["a", "b"]})) is None


class TestLargeJsonFallback:
    def test_geojson_larger_than_the_peek_still_yields_columns(self):
        # A FeatureCollection is typically megabytes, so it never parses whole
        # from a 32 KiB peek; ijson reads far enough to reach the first feature.
        truncated = (
            '{\n  "type": "FeatureCollection",\n  "features": [\n'
            '    {"type": "Feature", "properties": '
            '{"ADDDATE": "2019", "CITY": "DC", "DETAILS": "x", "WARD": 1},'
            ' "geometry": {"type": "Point", "coordinates": [1, 2'
        )
        rec = derive_table(_key("data.txt"), truncated)
        assert rec is not None, "truncated GeoJSON should still yield columns"
        assert rec["columns"] == ["ADDDATE", "CITY", "DETAILS", "WARD"]

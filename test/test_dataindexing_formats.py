"""dataindexing had its own detect_family with no XML branch."""
from dataindexing import formats
from dataindexing.sources import s3


def test_xml_declaration_is_detected():
    assert formats.detect_family('<?xml version="1.0"?><root><a/></root>') == "xml"


def test_kml_is_detected():
    assert formats.detect_family('<kml xmlns="http://www.opengis.net/kml/2.2"><Document/></kml>') == "xml"


def test_bare_root_tag_is_detected():
    assert formats.detect_family("<records><record><id>1</id></record></records>") == "xml"


def test_delimiter_in_an_xml_first_line_is_not_csv():
    # The drifted copy classified this "csv" because line 1 contains commas.
    assert formats.detect_family('<?xml version="1.0"?><r a="1,2,3" b="4,5"/>') == "xml"


def test_sources_s3_uses_the_one_implementation():
    assert s3.detect_family is formats.detect_family
    assert s3.should_skip is formats.should_skip


def test_content_family_includes_xml():
    assert "xml" in getattr(formats.ContentFamily, "__args__", ())

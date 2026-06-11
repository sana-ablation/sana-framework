from dataindexing.hybrid_search.text_builders import (
    build_description_chunk,
    build_schema_text,
    tokenize_column_name,
)
from dataindexing.hybrid_search.uri_matching import uri_to_schema_stem


def test_uri_to_schema_stem_strips_bucket_v1_and_extension():
    assert (
        uri_to_schema_stem("s3://bucket/datagov/foo/v1/files/rows.csv")
        == "datagov/foo/files/rows"
    )


def test_basic_and_infused_text_builders_preserve_mode_boundaries():
    desc = {
        "generated_metadata": "Generated metadata",
        "description": "Generated description",
        "original_metadata": "original uri words",
    }

    infused = build_schema_text(
        build_mode="infused",
        uri="s3://bucket/datagov/example/files/rows.txt",
        doc="fallback raw words",
        desc_row=desc,
        schema_cols=["PermitNumber", "agency_name"],
    )
    basic = build_schema_text(
        build_mode="basic",
        uri="s3://bucket/datagov/example/files/rows.txt",
        doc="fallback raw words",
        desc_row=desc,
        schema_cols=["PermitNumber", "agency_name"],
    )

    assert "Generated metadata" in infused
    assert "Generated description" not in infused
    assert "original uri words" in infused
    assert "Permit Number" in infused
    assert "agency name" in infused
    assert "Generated metadata" not in basic
    assert "Generated description" not in basic
    assert "original uri words" not in basic
    assert "datagov example files rows" in basic
    assert tokenize_column_name("PermitNumber") == "Permit Number"
    assert build_description_chunk("infused", desc) == "Generated metadata Generated description"
    assert build_description_chunk("basic", desc) is None

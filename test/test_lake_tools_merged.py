import sana_evaluation.tools.lake as lake


def test_the_whole_live_tool_surface_is_in_one_module():
    for name in (
        "submit_answer", "search", "search_prefix", "search_keyword", "list_files",
        "download", "execute_code", "cleanup_sandbox", "get_sandbox_info",
        "peek_file", "peek_multiple", "read_file", "grep_file",
        "parse_xml_records", "query_file", "configure_benchmark", "set_sandbox_dir",
    ):
        assert hasattr(lake, name), f"tools.lake is missing {name}"


def test_inspect_file_is_gone():
    assert not hasattr(lake, "inspect_file"), "peek_file replaced inspect_file"


def test_download_smart_was_never_real():
    assert not hasattr(lake, "download_smart")
    assert "download_smart" not in (lake.__doc__ or ""), \
        "the v2 docstring advertised a function that never existed"

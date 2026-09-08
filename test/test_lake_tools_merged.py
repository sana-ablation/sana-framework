import sana_evaluation.tools.lake as lake
import sana_evaluation.tools.computation.standard as computation_standard


def test_the_whole_live_tool_surface_is_in_one_module():
    """Eleven of the thirteen live lake tools are owned by tools.lake.

    ``query_file`` and ``execute_code`` are the exception, and deliberately so:
    they are the computation axis's standard arm, swapped per run against
    ``query_ideal``/``execute_ideal``, so their @tool surfaces live beside that
    arm in ``tools.computation.standard``. See the companion assertion below --
    the pair is still checked, just against the module that now owns it.
    """
    for name in (
        "submit_answer", "search", "search_prefix", "search_keyword", "list_files",
        "download", "cleanup_sandbox", "get_sandbox_info",
        "peek_file", "peek_multiple", "read_file", "grep_file",
        "parse_xml_records", "configure_benchmark", "set_sandbox_dir",
    ):
        assert hasattr(lake, name), f"tools.lake is missing {name}"


def test_the_computation_axis_owns_query_file_and_execute_code():
    for name in ("query_file", "execute_code"):
        assert hasattr(computation_standard, name), \
            f"tools.computation.standard is missing {name}"
        assert not hasattr(lake, name), \
            f"{name} is the computation axis's tool surface; lake keeps only its _impl"

    # The bodies stay in lake -- this is a layering, not a copy.
    for name in ("_query_file_impl", "_execute_code_impl"):
        assert hasattr(lake, name), f"tools.lake is missing {name}"


def test_the_swapped_pair_still_reports_the_same_tool_names():
    """runner.modes._apply_computation_tool_mode matches on tool name."""
    from sana_evaluation.runner.modes import _tool_name

    assert _tool_name(computation_standard.query_file) == "query_file"
    assert _tool_name(computation_standard.execute_code) == "execute_code"


def test_inspect_file_is_gone():
    assert not hasattr(lake, "inspect_file"), "peek_file replaced inspect_file"


def test_download_smart_was_never_real():
    assert not hasattr(lake, "download_smart")
    assert "download_smart" not in (lake.__doc__ or ""), \
        "the v2 docstring advertised a function that never existed"

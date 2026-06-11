from sana_analysis.running_analysis.discovery_metrics import compute_discovery_metrics
from sana_evaluation.helper import result as result_module
from sana_evaluation.instrumentation import read_trace_plugin


def test_read_trace_treats_query_ideal_as_dataset_read() -> None:
    assert "query_ideal" in read_trace_plugin._READ_TOOLS
    read_ids = read_trace_plugin._extract_read_dataset_ids(
        "query_ideal",
        {
            "s3_uri": (
                "s3://lakeqa-yc4103-datalake/datagov/"
                "school-directory/files/rows.txt"
            )
        },
    )
    assert read_ids == ["school-directory"]


def test_read_trace_treats_execute_ideal_as_dataset_read() -> None:
    assert "execute_ideal" in read_trace_plugin._READ_TOOLS
    read_ids = read_trace_plugin._extract_read_dataset_ids(
        "execute_ideal",
        {
            "s3_uri": (
                "s3://lakeqa-yc4103-datalake/datagov/"
                "school-directory/files/rows.txt"
            )
        },
    )
    assert read_ids == ["school-directory"]


def test_read_trace_treats_download_as_dataset_read() -> None:
    assert "download" in read_trace_plugin._READ_TOOLS
    read_ids = read_trace_plugin._extract_read_dataset_ids(
        "download",
        {
            "files": [
                {
                    "s3_uri": (
                        "s3://lakeqa-yc4103-datalake/datagov/"
                        "school-directory/files/rows.txt"
                    )
                },
                {"dataset_id": "datagov/teacher-directory", "file_path": "files/rows.txt"},
            ]
        },
    )
    assert read_ids == ["school-directory", "teacher-directory"]


def test_read_trace_extracts_peek_multiple_string_and_dict_entries() -> None:
    read_ids = read_trace_plugin._extract_read_dataset_ids(
        "peek_multiple",
        {
            "files": [
                (
                    "s3://lakeqa-yc4103-datalake/datagov/"
                    "school-directory/files/rows.txt"
                ),
                {"dataset_id": "datagov/teacher-directory"},
                {
                    "s3_uri": (
                        "s3://lakeqa-yc4103-datalake/wikipedia/"
                        "National_Park_Service/content.txt"
                    )
                },
                {"not_a_file": "ignored"},
            ]
        },
    )
    assert read_ids == [
        "school-directory",
        "teacher-directory",
        "National_Park_Service",
    ]


def test_tool_metrics_classify_query_ideal_as_api_read_tool() -> None:
    assert "query_ideal" in result_module._API_TOOL_NAMES


def test_discovery_metrics_ignore_ideal_subagent_cost_traces() -> None:
    metrics = compute_discovery_metrics(
        {
            "k-1-d-1/task_1": [
                {"tool": "search_ideal", "result_dataset_ids": ["ds_a"]},
                {"tool": "query_ideal", "read_dataset_ids": ["ds_a"]},
                {"tool": "execute_ideal", "read_dataset_ids": ["ds_a"]},
                {"event": "ideal_subagent_cost", "tool": "search_ideal"},
                {"event": "ideal_subagent_cost", "tool": "query_ideal"},
                {"event": "ideal_subagent_cost", "tool": "execute_ideal"},
            ]
        },
        {"k-1-d-1/task_1": ["ds_a"]},
    )

    row = metrics["task_metrics"][0]
    assert row["num_search_calls"] == 1
    assert row["num_read_calls"] == 2


def test_discovery_metrics_ignore_ideal_repair_bookkeeping_traces() -> None:
    metrics = compute_discovery_metrics(
        {
            "k-1-d-1/task_1": [
                {"tool": "search_ideal", "result_dataset_ids": ["ds_a"]},
                {"tool": "query_ideal", "read_dataset_ids": ["ds_a"]},
                {"tool": "execute_ideal", "read_dataset_ids": ["ds_a"]},
                {"event": "repair_agent_invoked", "tool": "query_ideal"},
                {"event": "repair_agent_completed", "tool": "query_ideal"},
                {"event": "repair_agent_failed", "tool": "execute_ideal"},
            ]
        },
        {"k-1-d-1/task_1": ["ds_a"]},
    )

    row = metrics["task_metrics"][0]
    assert row["num_search_calls"] == 1
    assert row["num_read_calls"] == 2

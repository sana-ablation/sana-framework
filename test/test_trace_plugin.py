import json
from types import SimpleNamespace
from unittest.mock import patch

from sana_evaluation.instrumentation.read_trace_plugin import ReadTracePlugin
from sana_evaluation.instrumentation.trace_plugin import TracePlugin, set_trace_context


def _event(tool_name: str, tool_input: dict, tool_use_id: str = "tool-1", result: dict | None = None):
    return SimpleNamespace(
        tool_use={
            "name": tool_name,
            "input": tool_input,
            "toolUseId": tool_use_id,
        },
        result=result or {"content": [{"text": json.dumps({"results": []})}]},
    )


def _trace_rows(trace_root, task_dir: str = "k-1-d-1", task_name: str = "task_1") -> list[dict]:
    trace_path = trace_root / task_dir / f"{task_name}.jsonl"
    return [json.loads(line) for line in trace_path.read_text().splitlines()]


def test_trace_plugin_tracks_search_state_per_tool_use_id(tmp_path):
    set_trace_context("tasks_mini/k-1-d-1/task_1.json", ["ds1"], str(tmp_path))
    plugin = TracePlugin()

    with patch(
        "sana_evaluation.instrumentation.trace_plugin.time.time",
        side_effect=[10, 20, 30, 31, 35, 36],
    ):
        plugin.on_before_tool(_event("search_value", {"query": "first query"}, "search-a"))
        plugin.on_before_tool(_event("search_value", {"query": "second query"}, "search-b"))
        plugin.on_after_tool(
            _event(
                "search_value",
                {"query": "first query"},
                "search-a",
                {"content": [{"text": json.dumps({"results": [{"dataset_id": "ds1"}]})}]},
            )
        )
        plugin.on_after_tool(
            _event(
                "search_value",
                {"query": "second query"},
                "search-b",
                {"content": [{"text": json.dumps({"results": [{"dataset_id": "ds2"}]})}]},
            )
        )

    rows = _trace_rows(tmp_path)
    assert rows[0]["query"] == "first query"
    assert rows[0]["latency_ms"] == 20_000
    assert rows[1]["query"] == "second query"
    assert rows[1]["latency_ms"] == 15_000


def test_read_trace_plugin_tracks_latency_per_tool_use_id(tmp_path):
    set_trace_context("tasks_mini/k-1-d-1/task_1.json", ["school-directory"], str(tmp_path))
    plugin = ReadTracePlugin()

    with patch(
        "sana_evaluation.instrumentation.read_trace_plugin.time.time",
        side_effect=[5, 7, 11, 12, 13, 14],
    ):
        plugin.on_before_tool(
            _event(
                "read_file",
                {"s3_uri": "s3://lakeqa-yc4103-datalake/datagov/school-directory/files/rows.txt"},
                "read-a",
            )
        )
        plugin.on_before_tool(
            _event(
                "read_file",
                {"s3_uri": "s3://lakeqa-yc4103-datalake/datagov/other-dataset/files/rows.txt"},
                "read-b",
            )
        )
        plugin.on_after_tool(
            _event(
                "read_file",
                {"s3_uri": "s3://lakeqa-yc4103-datalake/datagov/school-directory/files/rows.txt"},
                "read-a",
            )
        )
        plugin.on_after_tool(
            _event(
                "read_file",
                {"s3_uri": "s3://lakeqa-yc4103-datalake/datagov/other-dataset/files/rows.txt"},
                "read-b",
            )
        )

    rows = _trace_rows(tmp_path)
    assert rows[0]["read_dataset_ids"] == ["school-directory"]
    assert rows[0]["latency_ms"] == 6_000
    assert rows[1]["read_dataset_ids"] == ["other-dataset"]
    assert rows[1]["latency_ms"] == 6_000


def test_trace_plugin_writes_submit_answer_record(tmp_path):
    set_trace_context("tasks_mini/k-1-d-1/task_1.json", [], str(tmp_path))
    plugin = TracePlugin()

    plugin.on_after_tool(
        _event(
            "submit_answer",
            {"answer": "[42]", "reasoning": "computed from the table"},
            "submit-1",
            {"content": [{"text": "Answer submitted: [42]"}], "status": "success"},
        )
    )

    rows = _trace_rows(tmp_path)
    assert rows == [
        {
            "task_id": "tasks_mini/k-1-d-1/task_1.json",
            "tool": "submit_answer",
            "answer_text": "[42]",
            "reasoning": "computed from the table",
            "status": "success",
            "timestamp_ms": rows[0]["timestamp_ms"],
        }
    ]

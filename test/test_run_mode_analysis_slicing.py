"""Slicing axes inlined from the deleted search_depth / reasoning_density /
tool_error_analysis modules.

These had no direct test before -- the modules' own compute_* entry points were
dead, and the pipeline imported only the constants. Inlining them surfaced a
NameError that the whole suite passed straight over, so they are pinned here.
"""
import json
from pathlib import Path

from sana_analysis.run_mode_analysis import (
    _DATA_TOOLS,
    _assign_reasoning_density_bin,
    _assign_search_depth_bin,
    load_task_gold_counts,
)


def test_search_depth_bins_cover_the_range():
    assert [_assign_search_depth_bin(n) for n in (1, 2, 3, 4, 6, 7, 10, 11, 30)] == [
        "1", "2-3", "2-3", "4-6", "4-6", "7-10", "7-10", "11-30", "11-30"
    ]


def test_search_depth_bin_saturates_above_the_top_bin():
    assert _assign_search_depth_bin(999) == "11-30"


def test_reasoning_density_bins_cover_the_range():
    assert [_assign_reasoning_density_bin(n) for n in (0, 2, 3, 4, 5, 7, 8, 10, 11)] == [
        "<=2", "<=2", "3-4", "3-4", "5-7", "5-7", "8-10", "8-10", ">10"
    ]


def test_load_task_gold_counts_keys_by_both_path_and_stem(tmp_path):
    task_dir = tmp_path / "k-1-d-2"
    task_dir.mkdir()
    (task_dir / "task_1.json").write_text(json.dumps({"datasets_used": ["a", "b"]}))

    counts = load_task_gold_counts(str(tmp_path))

    assert set(counts.values()) == {2}
    # One entry keyed by full path, one by stem -- callers hold either.
    assert len(counts) == 2
    assert any(k.endswith("task_1.json") for k in counts)


def test_load_task_gold_counts_on_a_real_task_set():
    counts = load_task_gold_counts("benchmarks/lakeqa/tasks_20_subset/tasks")
    assert counts, "no gold counts loaded from the shipped task set"
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())


def test_data_tools_are_the_data_touching_ones():
    assert "query_file" in _DATA_TOOLS and "read_file" in _DATA_TOOLS
    # Planning and answer submission are not data tools.
    assert "plan" not in _DATA_TOOLS and "submit_answer" not in _DATA_TOOLS

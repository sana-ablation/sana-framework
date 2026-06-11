from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from sana_evaluation.instrumentation.trace_plugin import _normalize_dataset_id

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "sana_evaluation" / "prompts"
_MODES = {"naive", "standard", "ideal", "preloaded"}
_DEBUG_MODES = {"decision_notes"}

_PLAN_AGENT_SKILL = "sana_evaluation/tools/skills/plan-agent"
_PLAN_IDEAL_SKILL = "sana_evaluation/tools/skills/plan-ideal"
_DISCOVER_SKILL_PATHS = {
    "naive": "sana_evaluation/tools/skills/discover-data-naive",
    "standard": "sana_evaluation/tools/skills/discover-data-standard",
    "ideal": "sana_evaluation/tools/skills/discover-data-ideal",
}
_QUERY_DATA_SKILL = "sana_evaluation/tools/skills/query-data"


def _normalize_mode(value: Optional[str], default: str, label: str) -> str:
    mode = (value or default).strip().lower()
    if mode not in _MODES:
        raise ValueError(
            f"Unsupported {label} mode '{value}'. Expected one of: {', '.join(sorted(_MODES))}"
        )
    return mode


def normalize_debug_mode(value: Optional[str]) -> Optional[str]:
    mode = (value or "").strip().lower()
    if not mode or mode == "none":
        return None
    if mode not in _DEBUG_MODES:
        raise ValueError(
            f"Unsupported debug mode '{value}'. Expected one of: none, {', '.join(sorted(_DEBUG_MODES))}"
        )
    return mode


def load_prompt_text(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Required prompt file missing: {prompt_path}. "
            "Run preflight to confirm your sana_evaluation/prompts/ directory is complete."
        )
    return prompt_path.read_text()


def _compose_search_overlay_prompt(
    base_prompt_path: str | Path,
    search_tool_mode: Optional[str],
    *,
    benchmark: str = "lakeqa",
) -> str:
    mode = _normalize_mode(search_tool_mode, "naive", "search_tool")
    base_prompt = load_prompt_text(base_prompt_path).rstrip()
    benchmark_name = (benchmark or "lakeqa").strip().lower()
    benchmark_overlay = _PROMPTS_DIR / f"search_{mode}_{benchmark_name}.txt"
    overlay_path = benchmark_overlay if benchmark_overlay.is_file() else _PROMPTS_DIR / f"search_{mode}.txt"
    overlay = load_prompt_text(overlay_path).strip()
    return f"{base_prompt}\n\n{overlay}"


def _remove_prompt_section(prompt: str, heading: str) -> str:
    lines = prompt.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == heading:
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            while out and out[-1] == "":
                out.pop()
            out.append("")
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip()


def _remove_skill_references(prompt: str) -> str:
    prompt = _remove_prompt_section(prompt, "## SKILLS")
    lines = [
        line
        for line in prompt.splitlines()
        if "Skill loading" not in line
    ]
    return "\n".join(lines).strip()


def compose_managed_prompt(search_tool_mode: Optional[str], *, include_skills: bool = True) -> str:
    prompt = _compose_search_overlay_prompt(
        _PROMPTS_DIR / "managed.txt",
        search_tool_mode,
    )
    if not include_skills:
        prompt = _remove_skill_references(prompt)
    return prompt


def compose_kramabench_prompt(search_tool_mode: Optional[str], *, include_skills: bool = True) -> str:
    prompt = _compose_search_overlay_prompt(
        _PROMPTS_DIR / "managed_kramabench.txt",
        search_tool_mode,
        benchmark="kramabench",
    )
    if not include_skills:
        prompt = _remove_skill_references(prompt)
    return prompt


def compose_baseline_prompt(search_tool_mode: Optional[str]) -> str:
    return _compose_search_overlay_prompt(
        _PROMPTS_DIR / "baseline.txt",
        search_tool_mode,
    )


def compose_preloaded_block(source_sequence: List[str]) -> str:
    if not source_sequence:
        raise ValueError("Preloaded mode requires a non-empty source_sequence.")

    lines = [
        "## PRELOADED DATASETS",
        "",
        "You have been given the complete set of datasets required to answer this",
        "task. Do not search. Proceed directly to inspect and query the files",
        "listed below using the available data tools.",
        "Pass `uri` values as `s3_uri` directly when calling file tools; do",
        "not reconstruct dataset_id + file_path unless you need to.",
        "",
    ]
    for source in source_sequence:
        dataset_id = _normalize_dataset_id(source)
        lines.append(f"- dataset_id: {dataset_id} | uri: {source}")
    return "\n".join(lines)


def planning_skill_path(profile_mode: Optional[str]) -> str:
    management_mode = _normalize_mode(profile_mode, "standard", "profile")
    return _PLAN_IDEAL_SKILL if management_mode == "ideal" else _PLAN_AGENT_SKILL


def discover_skill_path(search_tool_mode: Optional[str]) -> str:
    mode = _normalize_mode(search_tool_mode, "naive", "search_tool")
    if mode == "preloaded":
        raise ValueError("Preloaded mode does not use a discover-data skill.")
    return _DISCOVER_SKILL_PATHS[mode]


def skill_paths_for_modes(
    search_tool_mode: Optional[str],
    profile_mode: Optional[str],
) -> List[str]:
    mode = _normalize_mode(search_tool_mode, "naive", "search_tool")
    if mode == "preloaded":
        return [
            planning_skill_path(profile_mode),
            _QUERY_DATA_SKILL,
        ]
    return [
        planning_skill_path(profile_mode),
        discover_skill_path(mode),
        _QUERY_DATA_SKILL,
    ]


def inject_debug_prompt(prompt: str, debug_mode: Optional[str]) -> str:
    mode = normalize_debug_mode(debug_mode)
    if mode is None:
        return prompt

    if mode == "decision_notes":
        section = (
            "\n\n## DEBUG DECISION NOTES\n"
            "- Debug mode is active.\n"
            "- Before every tool call, emit a short structured note immediately before the tool-use block.\n"
            "- Keep it concise and action-oriented. Do not dump long reasoning.\n"
            "- Use exactly these four fields:\n"
            "goal: <one short sentence>\n"
            "why_this_tool: <one short sentence>\n"
            "what_success_looks_like: <one short sentence>\n"
            "confidence: <low|medium|high>\n"
            "- Then call the tool in the same response.\n"
        )
        return prompt.rstrip() + section

    return prompt


__all__ = [
    "compose_baseline_prompt",
    "compose_kramabench_prompt",
    "compose_managed_prompt",
    "compose_preloaded_block",
    "discover_skill_path",
    "inject_debug_prompt",
    "load_prompt_text",
    "normalize_debug_mode",
    "planning_skill_path",
    "skill_paths_for_modes",
]

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from sana_evaluation.instrumentation.trace_plugin import _normalize_dataset_id

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = _REPO_ROOT / "sana_evaluation" / "prompting" / "fragments"
_MODES = {"naive", "standard", "ideal", "preloaded", "web"}
_DEBUG_MODES = {"decision_notes"}

_PLAN_AGENT_SKILL = "sana_evaluation/prompting/skills/plan-agent"
_PLAN_IDEAL_SKILL = "sana_evaluation/prompting/skills/plan-ideal"
_DISCOVER_SKILL_PATHS = {
    "naive": "sana_evaluation/prompting/skills/discover-data-naive",
    "standard": "sana_evaluation/prompting/skills/discover-data-standard",
    "ideal": "sana_evaluation/prompting/skills/discover-data-ideal",
}
_QUERY_DATA_SKILL = "sana_evaluation/prompting/skills/query-data"


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
            "Run preflight to confirm your sana_evaluation/prompting/fragments/ directory is complete."
        )
    return prompt_path.read_text()


def fragment_paths(
    *,
    plan: str,
    search: str,
    benchmark: str = "lakeqa",
    skills: bool = True,
) -> List[Path]:
    """The ordered fragment list for one axis combination.

    Order reproduces the historical section order: framing and the operating
    envelope are two fragments precisely so ``limits`` can stay at the end,
    where VERIFY DATA SOURCES / GENERAL TIPS / TURN AND TIME LIMITS sit today.
    See the spec's "Section order is preserved" section.

    ``skills`` selects no file -- it is a text filter applied by :func:`build`.
    It is in the signature so preflight and the runtime call this with one
    shape and cannot drift apart.
    """
    del skills
    plan_mode = _normalize_mode(plan, "standard", "plan")
    search_mode = _normalize_mode(search, "naive", "search_tool")
    benchmark_name = (benchmark or "lakeqa").strip().lower()

    paths = [_PROMPTS_DIR / "base" / "framing.txt"]
    # naive contributes the plain tool-list heading; every other plan mode adds
    # skills, planning style and the planning tool itself.
    paths.append(_PROMPTS_DIR / "plan" / ("naive.txt" if plan_mode == "naive" else "managed.txt"))
    if search_mode == "web":
        # The web arm has no data lake, so it gets neither the lake tools nor a
        # corpus description. Nothing to fall back to, nothing to contradict.
        paths.append(_PROMPTS_DIR / "data-access" / "web.txt")
    else:
        paths.append(_PROMPTS_DIR / "data-access" / "lake.txt")
        if benchmark_name != "kramabench":
            # query_file is disabled for kramabench, so the bullet, the cost
            # ladder and the query discipline it governs are simply not composed.
            paths.append(_PROMPTS_DIR / "data-access" / "lake-query.txt")
        paths.append(_PROMPTS_DIR / "benchmark" / f"{benchmark_name}.txt")
    paths.append(_PROMPTS_DIR / "base" / "limits.txt")
    paths.append(_PROMPTS_DIR / "search" / f"{search_mode}.txt")
    return paths


def _unloadable_skill_markers(search_mode: str) -> tuple[str, ...]:
    """Skill bullets the shared plan fragment lists that this search mode never loads.

    The SKILLS list is plan-axis content, but one of its bullets is selected by
    the search axis: ``skill_paths_for_modes`` gives web only planning and
    query-data, so ``discover_skill_path`` raises for it. Advertising
    discover-data there is the tool-advertisement bug one axis over, and the
    section sits above the data-access fragment, so no fragment of the search
    axis can drop it -- the skills axis is already filtered as text below, and
    this uses the same mechanism.

    ``preloaded`` has the identical gap and has had it since before the split.
    It is deliberately not fixed here: it changes four further golden files that
    this commit's review did not cover.
    """
    return ('skills("discover-data")',) if search_mode == "web" else ()


def build(
    *,
    plan: str,
    search: str,
    benchmark: str = "lakeqa",
    skills: bool = True,
) -> str:
    """Compose one prompt by concatenating one fragment per axis, in order."""
    parts = [
        load_prompt_text(path).strip()
        for path in fragment_paths(plan=plan, search=search, benchmark=benchmark, skills=skills)
    ]
    prompt = "\n\n".join(part for part in parts if part)
    if not skills:
        prompt = _remove_skill_references(prompt)
    markers = _unloadable_skill_markers(_normalize_mode(search, "naive", "search_tool"))
    if markers:
        prompt = "\n".join(
            line for line in prompt.splitlines()
            if not any(marker in line for marker in markers)
        )
    return prompt


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


def compose_managed_prompt(
    search_tool_mode: Optional[str],
    *,
    include_skills: bool = True,
    no_s3: bool = False,
) -> str:
    """Managed lakeqa prompt. ``no_s3`` is accepted and ignored: web mode has a
    single overlay now, and it used to be what selected between two."""
    del no_s3
    return build(plan="standard", search=search_tool_mode, skills=include_skills)


def compose_kramabench_prompt(search_tool_mode: Optional[str], *, include_skills: bool = True) -> str:
    """Kramabench prompt. Composes the managed plan fragment for every plan
    mode, including naive -- which is what the kramabench base did before the
    split, since there was never a baseline_kramabench.txt to select."""
    return build(
        plan="standard",
        search=search_tool_mode,
        benchmark="kramabench",
        skills=include_skills,
    )


def compose_baseline_prompt(search_tool_mode: Optional[str], *, no_s3: bool = False) -> str:
    """Naive-plan lakeqa prompt: no skills, no planning tool. ``no_s3`` is
    accepted and ignored for the same reason as in the managed wrapper."""
    del no_s3
    return build(plan="naive", search=search_tool_mode, skills=False)


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


def planning_skill_path(plan_mode: Optional[str]) -> str:
    mode = _normalize_mode(plan_mode, "standard", "plan")
    return _PLAN_IDEAL_SKILL if mode == "ideal" else _PLAN_AGENT_SKILL


def discover_skill_path(search_tool_mode: Optional[str]) -> str:
    mode = _normalize_mode(search_tool_mode, "naive", "search_tool")
    if mode == "preloaded":
        raise ValueError("Preloaded mode does not use a discover-data skill.")
    if mode == "web":
        raise ValueError("Web mode does not use a discover-data skill.")
    return _DISCOVER_SKILL_PATHS[mode]


def skill_paths_for_modes(
    search_tool_mode: Optional[str],
    plan_mode: Optional[str],
) -> List[str]:
    mode = _normalize_mode(search_tool_mode, "naive", "search_tool")
    # Neither mode does lake discovery, so neither gets a discover-data skill.
    if mode in {"preloaded", "web"}:
        return [
            planning_skill_path(plan_mode),
            _QUERY_DATA_SKILL,
        ]
    return [
        planning_skill_path(plan_mode),
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
    "build",
    "compose_baseline_prompt",
    "compose_kramabench_prompt",
    "compose_managed_prompt",
    "compose_preloaded_block",
    "discover_skill_path",
    "fragment_paths",
    "inject_debug_prompt",
    "load_prompt_text",
    "normalize_debug_mode",
    "planning_skill_path",
    "skill_paths_for_modes",
]

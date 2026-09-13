"""Helpers for persistent agent-authored system-prompt sections."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


_KNOWN_HEADINGS = ("CURRENT PLAN", "CURRENT SPRINT")
_KNOWN_HEADING_SET = set(_KNOWN_HEADINGS)
_HEADING_RE = re.compile(r"(?P<prefix>\A|\n\n)## (?P<heading>[^\n]+)\n")


def _normalize_heading(heading: str) -> str:
    return heading.strip().upper()


def _split_prompt_parts(prompt: str) -> Tuple[str, Dict[str, str], List[str]]:
    text = prompt or ""
    matches = list(_HEADING_RE.finditer(text))
    first_known_idx = next(
        (
            idx
            for idx, match in enumerate(matches)
            if _normalize_heading(match.group("heading")) in _KNOWN_HEADING_SET
        ),
        None,
    )
    if first_known_idx is None:
        return text, {}, []

    base = text[: matches[first_known_idx].start()]
    sections: Dict[str, str] = {}
    unknown_sections: List[str] = []
    for idx in range(first_known_idx, len(matches)):
        match = matches[idx]
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = _normalize_heading(match.group("heading"))
        if heading in _KNOWN_HEADING_SET:
            sections[heading] = text[start:end].strip()
        else:
            chunk = text[match.start():end].strip()
            if chunk:
                unknown_sections.append(chunk)
    return base, sections, unknown_sections


def split_prompt_sections(prompt: str) -> Tuple[str, Dict[str, str]]:
    """Split a prompt into base text and known persistent sections."""
    base, sections, _ = _split_prompt_parts(prompt)
    return base, sections


def upsert_prompt_section(
    prompt: str,
    heading: str,
    body: str,
    *,
    ordered_headings: Iterable[str] = _KNOWN_HEADINGS,
) -> str:
    """Insert or replace a known prompt section, preserving section order."""
    normalized_heading = heading.strip().upper()
    if normalized_heading not in _KNOWN_HEADINGS:
        raise ValueError(f"Unknown prompt section heading: {heading!r}")

    base, sections, unknown_sections = _split_prompt_parts(prompt)
    sections[normalized_heading] = (body or "").strip()

    out = base.rstrip()
    for section_heading in ordered_headings:
        if section_heading in sections:
            out += f"\n\n## {section_heading}\n{sections[section_heading]}"
    for section in unknown_sections:
        out += f"\n\n{section}"
    return out


__all__ = ["split_prompt_sections", "upsert_prompt_section"]

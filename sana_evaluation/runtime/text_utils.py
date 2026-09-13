"""
Text utilities extracted from agent_runner.py.
"""

import re
from typing import Any

_STANDALONE_NUMBER_RE = re.compile(r"^[-+]?\d[\d,]*(?:\.\d+)?\.?$")
_ANSWER_IS_NUMBER_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer\s+is\s+([-+]?\d[\d,]*(?:\.\d+)?\.?)$",
    re.IGNORECASE,
)


def _clean_answer(answer: Any) -> str:
    """Extract a concise answer from bracketed or verbose responses."""
    if not answer:
        return ""

    if isinstance(answer, dict):
        answer = (
            answer.get("answer")
            or answer.get("value")
            or answer.get("result")
            or str(answer)
        )
    if not isinstance(answer, str):
        answer = str(answer)

    text = answer.strip()

    # Prefer bracketed content if present: [Answer]
    bracket_match = re.search(r"\[([^\[\]]+)\]", text)
    if bracket_match:
        text = bracket_match.group(1).strip()

    # Remove common answer prefixes
    text = re.sub(
        r"^(final answer|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s+is\s+", "", text, flags=re.IGNORECASE
    )

    # Strip surrounding quotes
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()

    # Normalize only clearly numeric answers. Do not strip numbers embedded in
    # entity names or dates such as "District 9" or "May 2024".
    if _STANDALONE_NUMBER_RE.fullmatch(text):
        return text.replace(",", "").rstrip(".")
    numeric_phrase = _ANSWER_IS_NUMBER_RE.fullmatch(text)
    if numeric_phrase:
        return numeric_phrase.group(1).replace(",", "").rstrip(".")

    return text


def _normalize_dataset_path(dataset_path: str) -> str:
    if not dataset_path:
        return ""
    path = dataset_path.rstrip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    if parts[0] in ("datagov", "wikipedia"):
        parts = parts[1:]
    return parts[0] if parts else ""


def _truncate_output(text: str, max_chars: int = 15000) -> str:
    """Truncate output to avoid triggering content filters on large data dumps."""
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        f"{text[:half]}\n\n"
        f"... [TRUNCATED {len(text) - max_chars} chars - use code to analyze full data] ...\n\n"
        f"{text[-half:]}"
    )

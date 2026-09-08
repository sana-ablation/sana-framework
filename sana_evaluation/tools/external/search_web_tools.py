"""Web search tools for the `web` search-tool mode, backed by the Parallel Search API.

Contract as documented at https://docs.parallel.ai/api-reference/search-api/search:

    POST https://api.parallel.ai/v1/search
    headers: x-api-key: <key>, Content-Type: application/json
    body:    {"search_queries": [str, ...],        # required, 3-6 words each
              "objective": str | null,             # optional
              "mode": "turbo"|"fast"|"basic"|"advanced",   # optional, default advanced
              "max_chars_total": int | null,       # optional
              "session_id": str | null}            # optional
    200:     {"search_id": str,
              "results": [{"url": str,
                           "title": str | null,
                           "publish_date": str | null,     # YYYY-MM-DD
                           "excerpts": [str, ...]}],       # markdown
              "session_id": str,
              "warnings": [...] | null,
              "usage": [...] | null}

Endpoint, mode and beta header are env-overridable so an account still provisioned
on the older `/v1beta/search` + `parallel-beta` alpha can be pointed at it without a
code change.

Unlike the lake search tools this returns open-web pages, so it deliberately does not
emit `dataset_id` / `s3_uri` fields — nothing it returns is addressable by the
lake data tools. See `search_web.txt` for how that is surfaced to the agent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from strands import tool

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://api.parallel.ai/v1/search"
_DEFAULT_MODE = "advanced"
_ALLOWED_MODES = ("turbo", "fast", "basic", "advanced")

# Keep a single tool result from dominating the context window. Mirrors the
# _TOOL_RESULT_CHAR_CAP convention in tools/lake.py.
_RESULT_CHAR_CAP = 6_000
_DEFAULT_MAX_RESULTS = 10
_TIMEOUT_SECONDS = 60

_MAX_RESULTS: Optional[int] = None


def set_max_results(k: Optional[int]) -> None:
    """Pin the result count exposed to the agent (wired from --k)."""
    global _MAX_RESULTS
    _MAX_RESULTS = int(k) if k else None


def max_results() -> int:
    return _MAX_RESULTS or _DEFAULT_MAX_RESULTS


def _endpoint() -> str:
    return os.getenv("PARALLEL_SEARCH_URL", _DEFAULT_URL)


def _mode() -> str:
    mode = (os.getenv("PARALLEL_SEARCH_MODE") or _DEFAULT_MODE).strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"Unsupported PARALLEL_SEARCH_MODE '{mode}'. Expected one of: {', '.join(_ALLOWED_MODES)}"
        )
    return mode


def _headers() -> Dict[str, str]:
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PARALLEL_API_KEY is not set. The web search mode requires a Parallel API key."
        )
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    beta = os.getenv("PARALLEL_BETA_HEADER")
    if beta:
        headers["parallel-beta"] = beta
    return headers


def _truncate(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    return text[: budget - 3].rstrip() + "..."


def _shape_results(payload: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Trim the raw Parallel payload to a bounded, agent-facing shape."""
    raw = payload.get("results") or []
    shaped: List[Dict[str, Any]] = []
    remaining = _RESULT_CHAR_CAP
    for item in raw[:limit]:
        if remaining <= 0:
            break
        excerpts: List[str] = []
        for excerpt in item.get("excerpts") or []:
            if remaining <= 0:
                break
            piece = _truncate(str(excerpt), remaining)
            if piece:
                excerpts.append(piece)
                remaining -= len(piece)
        shaped.append(
            {
                "url": item.get("url"),
                "title": item.get("title"),
                "publish_date": item.get("publish_date"),
                "excerpts": excerpts,
            }
        )
    return shaped


def search_parallel(search_queries: List[str], objective: str = "") -> Dict[str, Any]:
    """Raw Parallel Search call. Separated from the tool wrapper so it is unit-testable."""
    if not isinstance(search_queries, list) or not search_queries:
        return {"error": "search_queries must be a non-empty list of strings.", "results": [], "count": 0}

    queries = [str(q).strip() for q in search_queries if str(q).strip()]
    if not queries:
        return {"error": "search_queries must contain at least one non-empty query.", "results": [], "count": 0}

    limit = max_results()
    body: Dict[str, Any] = {"search_queries": queries, "mode": _mode()}
    if objective and str(objective).strip():
        body["objective"] = str(objective).strip()
    body["max_chars_total"] = _RESULT_CHAR_CAP

    response = requests.post(
        _endpoint(), json=body, headers=_headers(), timeout=_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    payload = response.json()

    results = _shape_results(payload, limit)
    return {
        "results": results,
        "count": len(results),
        "queries": queries,
        "search_id": payload.get("search_id"),
    }


@tool
def search_web(search_queries: List[str], objective: str = "") -> Dict[str, Any]:
    """Search the open web and return ranked pages with extended excerpts.

    Returns web pages, NOT data-lake datasets. The URLs it returns are not
    addressable by `read_file`, `download`, or `query_file` — those only accept
    data-lake sources. Answer from the excerpts.

    Args:
        search_queries: 2-3 diverse keyword queries, 3-6 words each. Vary the
                        entity and angle between queries. Not full sentences.
        objective: Optional natural-language statement of what you are trying
                   to establish, used to rank and select excerpts.
    """
    try:
        result = search_parallel(search_queries, objective)
        # Record what search offered so `download` can gate on it under --no-s3.
        # Best-effort: a failure here must never break search itself.
        try:
            from sana_evaluation.tools.external.web_fetch_tools import record_search_urls

            record_search_urls(
                item.get("url") for item in (result.get("results") or []) if item.get("url")
            )
        except Exception as record_exc:  # noqa: BLE001
            logger.warning("could not record search_web URLs: %s", record_exc)
        return result
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent, mirrors sibling tools
        logger.warning("search_web failed: %s: %s", type(exc).__name__, exc)
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "queries": search_queries,
            "results": [],
            "count": 0,
        }


__all__ = ["search_web", "search_parallel", "set_max_results", "max_results"]

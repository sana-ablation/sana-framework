"""Search tool wrappers for search mode experiments.

Supports:
1. Hard-coding result limit (`k`) by removing user-controlled limit params.
2. Returning enriched table descriptions from benchmark descriptions.jsonl.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from strands import tool
from strands.tools.decorator import DecoratedFunctionTool
from sana_evaluation.tools.external.description_rows import reject_forbidden_description_row

logger = logging.getLogger(__name__)

_QUERY_TOOLS = {"search_value", "search_schema", "search_reranked"}
_PREFIX_TOOLS = {"search_prefix"}
_SEARCH_TOOL_NAMES = _QUERY_TOOLS | _PREFIX_TOOLS

_TABLE_DESCRIPTIONS_PATH = Path("benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl")

_DESC_CACHE_LOADED = False
_DESC_BY_URI: Dict[str, str] = {}
_DESC_BY_DATASET_ID: Dict[str, str] = {}


def _dataset_id_from_uri(uri: str) -> Optional[str]:
    raw = str(uri or "")
    if "://" not in raw:
        return None
    try:
        # s3://bucket/<folder>/<dataset_id>/...
        tail = raw.split("://", 1)[1]
        parts = tail.split("/")
        if len(parts) >= 3:
            return parts[2]
    except Exception:
        return None
    return None


def _load_description_cache() -> None:
    global _DESC_CACHE_LOADED, _DESC_BY_URI, _DESC_BY_DATASET_ID

    if _DESC_CACHE_LOADED:
        return

    if not _TABLE_DESCRIPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Required dependency '{_TABLE_DESCRIPTIONS_PATH}' not found. "
            "search description mode requires the benchmark artifact descriptions JSONL."
        )

    _DESC_CACHE_LOADED = True
    uri_map: Dict[str, str] = {}
    dataset_map: Dict[str, str] = {}

    with _TABLE_DESCRIPTIONS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            reject_forbidden_description_row(obj, path=_TABLE_DESCRIPTIONS_PATH)

            uri = str(obj.get("dataset_uri") or obj.get("uri") or "").strip()
            description = str(obj.get("description") or "").strip()
            if not uri or not description:
                continue

            if uri not in uri_map:
                uri_map[uri] = description
            dsid = _dataset_id_from_uri(uri)
            if dsid and dsid not in dataset_map:
                dataset_map[dsid] = description

    _DESC_BY_URI = uri_map
    _DESC_BY_DATASET_ID = dataset_map
    logger.info(
        "Loaded %d URI descriptions and %d dataset-level descriptions from %s",
        len(_DESC_BY_URI),
        len(_DESC_BY_DATASET_ID),
        _TABLE_DESCRIPTIONS_PATH.name,
    )


def _lookup_description(item: Dict[str, Any]) -> Optional[str]:
    uri = str(item.get("dataset_uri") or item.get("uri") or "").strip()
    if uri and uri in _DESC_BY_URI:
        return _DESC_BY_URI[uri]

    dsid = str(item.get("dataset_id") or "").strip()
    if dsid and dsid in _DESC_BY_DATASET_ID:
        return _DESC_BY_DATASET_ID[dsid]

    if uri:
        dsid_from_uri = _dataset_id_from_uri(uri)
        if dsid_from_uri and dsid_from_uri in _DESC_BY_DATASET_ID:
            return _DESC_BY_DATASET_ID[dsid_from_uri]

    return None


def _apply_description_mode(payload: Any, mode: str) -> Any:
    if mode != "description":
        return payload
    if not isinstance(payload, dict):
        return payload
    if "results" not in payload or not isinstance(payload["results"], list):
        return payload

    _load_description_cache()
    if not _DESC_BY_URI and not _DESC_BY_DATASET_ID:
        return payload

    updated_results: List[Any] = []
    for row in payload["results"]:
        if not isinstance(row, dict):
            updated_results.append(row)
            continue
        new_row = dict(row)
        desc = _lookup_description(new_row)
        if desc:
            new_row["description"] = desc
        updated_results.append(new_row)

    out = dict(payload)
    out["results"] = updated_results
    return out


def _compose_description(base_description: str, *, fixed_k: Optional[int], mode: str) -> str:
    description = base_description.strip()
    notes: List[str] = []
    if fixed_k is not None:
        notes.append(f"Result limit is fixed at {fixed_k}; callers cannot change it.")
    if mode == "description":
        notes.append(
            "Each result may include an added `description` field from benchmark descriptions.jsonl."
        )
    if not notes:
        return description
    return f"{description} {' '.join(notes)}"


def _query_default(spec: Dict[str, Any], field_name: str, fallback: int) -> int:
    try:
        props = spec.get("inputSchema", {}).get("json", {}).get("properties", {})
        if field_name in props and "default" in props[field_name]:
            return int(props[field_name]["default"])
    except Exception:
        pass
    return fallback


def _wrap_query_tool(
    base_tool: DecoratedFunctionTool,
    *,
    fixed_k: Optional[int],
    mode: str,
) -> DecoratedFunctionTool:
    spec = base_tool.tool_spec
    tool_name = spec["name"]
    base_description = spec.get("description", "")
    wrapped_description = _compose_description(base_description, fixed_k=fixed_k, mode=mode)
    top_k_default = _query_default(spec, "top_k", 10)

    if fixed_k is not None:
        def _wrapped(query: str) -> dict:
            result = base_tool(query=query, top_k=fixed_k)
            return _apply_description_mode(result, mode)

        return tool(_wrapped, name=tool_name, description=wrapped_description)

    def _wrapped(query: str, top_k: int = top_k_default) -> dict:
        result = base_tool(query=query, top_k=top_k)
        return _apply_description_mode(result, mode)

    return tool(_wrapped, name=tool_name, description=wrapped_description)


def _wrap_prefix_tool(
    base_tool: DecoratedFunctionTool,
    *,
    fixed_k: Optional[int],
    mode: str,
) -> DecoratedFunctionTool:
    spec = base_tool.tool_spec
    tool_name = spec["name"]
    base_description = spec.get("description", "")
    wrapped_description = _compose_description(base_description, fixed_k=fixed_k, mode=mode)
    limit_default = _query_default(spec, "limit", 50)

    if fixed_k is not None:
        def _wrapped(prefixes: List[str]) -> dict:
            result = base_tool(prefixes=prefixes, limit=fixed_k)
            return _apply_description_mode(result, mode)

        return tool(_wrapped, name=tool_name, description=wrapped_description)

    def _wrapped(prefixes: List[str], limit: int = limit_default) -> dict:
        result = base_tool(prefixes=prefixes, limit=limit)
        return _apply_description_mode(result, mode)

    return tool(_wrapped, name=tool_name, description=wrapped_description)


def build_search_tools(
    base_tools: Sequence[DecoratedFunctionTool],
    *,
    fixed_k: Optional[int],
    search_descriptions: str,
) -> List[DecoratedFunctionTool]:
    """Return wrapped search tools for a run configuration."""
    mode = str(search_descriptions or "naive").strip().lower()
    if mode not in {"naive", "description"}:
        raise ValueError(
            f"Unsupported search_descriptions mode '{search_descriptions}'. Expected: naive|description"
        )

    if fixed_k is None and mode == "naive":
        return list(base_tools)

    wrapped: List[DecoratedFunctionTool] = []
    for tool_obj in base_tools:
        tool_name = tool_obj.tool_spec["name"]
        if tool_name in _QUERY_TOOLS:
            wrapped.append(_wrap_query_tool(tool_obj, fixed_k=fixed_k, mode=mode))
        elif tool_name in _PREFIX_TOOLS:
            wrapped.append(_wrap_prefix_tool(tool_obj, fixed_k=fixed_k, mode=mode))
        else:
            wrapped.append(tool_obj)
    return wrapped


def search_tool_names_in(tools: Sequence[DecoratedFunctionTool]) -> Tuple[str, ...]:
    names = [
        t.tool_spec["name"]
        for t in tools
        if t.tool_spec.get("name") in _SEARCH_TOOL_NAMES
    ]
    return tuple(dict.fromkeys(names))

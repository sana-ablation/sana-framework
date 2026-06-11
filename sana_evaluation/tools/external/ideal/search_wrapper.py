"""Search tool wrappers for multi-axis ablation experiments.

This module wraps search tools to control:
1) Fixed result limits (k)
2) Search result payload richness (naive | ideal)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from strands import tool
from strands.tools.decorator import DecoratedFunctionTool
from sana_evaluation.tools.external.description_rows import reject_forbidden_description_row

_QUERY_TOOLS = {"search_value", "search_schema", "search_reranked", "search_ideal"}
_PREFIX_TOOLS = {"search_prefix"}
_SEARCH_TOOL_NAMES = _QUERY_TOOLS | _PREFIX_TOOLS

_IDEAL_SNIPPET_WORDS = 100

_TABLE_DESCRIPTIONS_PATH = Path("benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl")
_SNIPPETS_PATH = Path("benchmarks/lakeqa/tasks-mini/artifacts/snippets.jsonl")
_SCHEMAS_PATH = Path("benchmarks/lakeqa/tasks-mini/artifacts/table_schemas_full.jsonl")

_DESC_CACHE_LOADED = False
_DESC_BY_URI: Dict[str, str] = {}
_DESC_ROW_BY_URI: Dict[str, Dict[str, Any]] = {}

_SNIPPET_CACHE_LOADED = False
_SNIPPET_BY_URI: Dict[str, str] = {}

_SCHEMAS_CACHE_LOADED = False
_SCHEMA_BY_SLUG_FILENAME: Dict[Tuple[str, str], Dict[str, Any]] = {}

_TABULAR_KIND_PRIORITY = ("delimited_text", "tsv", "csv", "parquet", "json", "geojson")


def configure_dependency_paths(
    *,
    descriptions: str | Path | None = None,
    snippets: str | Path | None = None,
    schemas: str | Path | None = None,
) -> None:
    """Configure ideal-search enrichment artifact paths and clear caches."""
    global _TABLE_DESCRIPTIONS_PATH, _SNIPPETS_PATH, _SCHEMAS_PATH
    global _DESC_CACHE_LOADED, _DESC_BY_URI, _DESC_ROW_BY_URI
    global _SNIPPET_CACHE_LOADED, _SNIPPET_BY_URI
    global _SCHEMAS_CACHE_LOADED, _SCHEMA_BY_SLUG_FILENAME

    if descriptions is not None:
        _TABLE_DESCRIPTIONS_PATH = Path(descriptions)
    if snippets is not None:
        _SNIPPETS_PATH = Path(snippets)
    if schemas is not None:
        _SCHEMAS_PATH = Path(schemas)

    _DESC_CACHE_LOADED = False
    _DESC_BY_URI = {}
    _DESC_ROW_BY_URI = {}
    _SNIPPET_CACHE_LOADED = False
    _SNIPPET_BY_URI = {}
    _SCHEMAS_CACHE_LOADED = False
    _SCHEMA_BY_SLUG_FILENAME = {}


def _dataset_id_from_uri(uri: str) -> Optional[str]:
    raw = str(uri or "")
    if "://" not in raw:
        return None
    try:
        tail = raw.split("://", 1)[1]
        parts = tail.split("/")
        if len(parts) >= 3:
            return parts[2]
    except Exception:
        return None
    return None


def _first_non_empty(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value:
            return str(value)
    return None


def _extract_uri(item: Dict[str, Any]) -> Optional[str]:
    return _first_non_empty([item.get("s3_uri"), item.get("dataset_uri"), item.get("uri")])


def _extract_dataset_id(item: Dict[str, Any]) -> Optional[str]:
    dsid = _first_non_empty([item.get("dataset_id")])
    if dsid:
        return dsid
    uri = _extract_uri(item)
    if uri:
        return _dataset_id_from_uri(uri)
    return None


def _extract_search_text(item: Dict[str, Any]) -> str:
    text = _first_non_empty(
        [
            item.get("dataset_snippet"),
            item.get("snippet"),
            item.get("document"),
            item.get("text"),
            item.get("content"),
        ]
    )
    return text or ""


def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _load_desc_cache() -> None:
    global _DESC_CACHE_LOADED, _DESC_BY_URI, _DESC_ROW_BY_URI
    if _DESC_CACHE_LOADED:
        return

    if not _TABLE_DESCRIPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Required dependency '{_TABLE_DESCRIPTIONS_PATH}' not found. "
            "search_results=ideal requires the benchmark artifact descriptions JSONL."
        )

    uri_map: Dict[str, str] = {}
    row_map: Dict[str, Dict[str, Any]] = {}
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
            desc = str(obj.get("description") or "").strip()
            if not uri or not desc:
                continue
            if uri not in uri_map:
                uri_map[uri] = desc
                row_map[uri] = obj

    _DESC_BY_URI = uri_map
    _DESC_ROW_BY_URI = row_map
    _DESC_CACHE_LOADED = True


def _load_snippet_cache() -> None:
    global _SNIPPET_CACHE_LOADED, _SNIPPET_BY_URI
    if _SNIPPET_CACHE_LOADED:
        return

    if not _SNIPPETS_PATH.exists():
        raise FileNotFoundError(
            f"Required dependency '{_SNIPPETS_PATH}' not found. "
            "Run the snippet builder first to create the benchmark snippets artifact."
        )

    _SNIPPET_CACHE_LOADED = True
    uri_map: Dict[str, str] = {}
    with _SNIPPETS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uri = str(obj.get("dataset_uri") or obj.get("uri") or "").strip()
            snippet = _truncate_words(
                str(obj.get("dataset_snippet") or obj.get("snippet") or obj.get("content") or "").strip(),
                max_words=_IDEAL_SNIPPET_WORDS,
            )
            if uri and snippet and uri not in uri_map:
                uri_map[uri] = snippet

    _SNIPPET_BY_URI = uri_map


def _slug_stem_from_uri(uri: str) -> Optional[Tuple[str, str]]:
    raw = (uri or "").strip()
    if not raw:
        return None
    if "://" in raw:
        raw = raw.split("://", 1)[1]
        if "/" not in raw:
            return None
        raw = raw.split("/", 1)[1]
    parts = raw.split("/")
    if len(parts) < 4 or parts[0] != "datagov":
        return None
    slug = parts[1]
    if "files" not in parts:
        return None
    idx = parts.index("files")
    if idx + 1 >= len(parts):
        return None
    filename = parts[idx + 1]
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return (slug, stem)


def _kind_rank(kind: str) -> int:
    lower = (kind or "").strip().lower()
    for rank, candidate in enumerate(_TABULAR_KIND_PRIORITY):
        if lower == candidate:
            return rank
    return len(_TABULAR_KIND_PRIORITY)


def _format_schema_entry(table: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(table.get("table_kind") or "").strip() or "unknown"
    raw_columns = table.get("columns")
    columns = [str(c) for c in raw_columns] if isinstance(raw_columns, list) else []
    return {"kind": kind, "columns": columns}


def _load_schemas_cache() -> None:
    global _SCHEMAS_CACHE_LOADED, _SCHEMA_BY_SLUG_FILENAME
    if _SCHEMAS_CACHE_LOADED:
        return

    if not _SCHEMAS_PATH.exists():
        raise FileNotFoundError(
            f"Required dependency '{_SCHEMAS_PATH}' not found. "
            "search_results=ideal requires table_schemas_full.jsonl at the repo root."
        )

    chosen: Dict[Tuple[str, str], Tuple[int, Dict[str, Any]]] = {}
    with _SCHEMAS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = str(obj.get("dataset_slug") or "").strip()
            if not slug:
                continue
            tables = obj.get("tables") or []
            if not isinstance(tables, list):
                continue
            for table in tables:
                if not isinstance(table, dict):
                    continue
                rel = str(table.get("relative_path") or "").strip()
                if not rel:
                    continue
                filename = rel.rsplit("/", 1)[-1]
                stem = filename.rsplit(".", 1)[0] if "." in filename else filename
                key = (slug, stem)
                rank = _kind_rank(str(table.get("table_kind") or ""))
                existing = chosen.get(key)
                if existing is None or rank < existing[0]:
                    chosen[key] = (rank, _format_schema_entry(table))

    _SCHEMA_BY_SLUG_FILENAME = {key: entry for key, (_, entry) in chosen.items()}
    _SCHEMAS_CACHE_LOADED = True


def _lookup_schema(uri: str) -> Optional[Dict[str, Any]]:
    key = _slug_stem_from_uri(uri)
    if key is None:
        return None
    _load_schemas_cache()
    return _SCHEMA_BY_SLUG_FILENAME.get(key)


def _fallback_type_from_uri(uri: str) -> str:
    name = (uri or "").rsplit("/", 1)[-1]
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    return "unknown"


def _schema_fields(uri: str) -> Dict[str, Any]:
    schema = _lookup_schema(uri)
    if not schema:
        return {"type": _fallback_type_from_uri(uri)}

    columns = schema.get("columns") or []
    if columns:
        return {"columns": list(columns)}

    return {"type": str(schema.get("kind") or _fallback_type_from_uri(uri))}


def _reshape_row(row: Dict[str, Any], mode: str) -> Dict[str, Any]:
    dataset_id = _extract_dataset_id(row)
    uri = _extract_uri(row)

    out: Dict[str, Any] = {}
    if dataset_id:
        out["dataset_id"] = dataset_id
    if uri:
        out["s3_uri"] = uri

    if mode == "naive":
        return out

    desc = _DESC_BY_URI.get(uri or "", "")
    if desc:
        out["llm_desc"] = desc

    if uri:
        out.update(_schema_fields(uri))

    snippet = _SNIPPET_BY_URI.get(uri or "", "")
    if not snippet:
        snippet = _truncate_words(_extract_search_text(row), max_words=_IDEAL_SNIPPET_WORDS)
    if snippet:
        out["dataset_snippet"] = snippet

    return out


def reshape_search_payload(payload: Any, mode: str) -> Any:
    """Transform a search payload into naive/ideal result richness."""
    normalized = str(mode or "naive").strip().lower()
    if normalized not in {"naive", "ideal"}:
        raise ValueError(f"Unsupported search_results mode '{mode}'. Expected: naive|ideal")

    if not isinstance(payload, dict):
        return payload
    if "results" not in payload or not isinstance(payload["results"], list):
        return payload

    if normalized == "ideal":
        _load_desc_cache()
        _load_snippet_cache()
        _load_schemas_cache()

    shaped: List[Any] = []
    for row in payload["results"]:
        if not isinstance(row, dict):
            shaped.append(row)
            continue
        shaped.append(_reshape_row(row, normalized))

    transformed = dict(payload)
    transformed["results"] = shaped
    transformed["count"] = len(shaped)
    return transformed


def _query_default(spec: Dict[str, Any], field_name: str, fallback: int) -> int:
    try:
        props = spec.get("inputSchema", {}).get("json", {}).get("properties", {})
        if field_name in props and "default" in props[field_name]:
            return int(props[field_name]["default"])
    except Exception:
        pass
    return fallback


def _compose_description(
    base_description: str,
    *,
    tool_name: str,
    fixed_k: Optional[int],
    mode: str,
) -> str:
    desc = base_description.strip()
    notes: List[str] = []
    if fixed_k is not None:
        if tool_name == "search_ideal":
            notes.append(
                "Returns the planned sources most relevant to your query. "
                "Count is chosen automatically based on query intent. "
                "Returned datasets are most likely needed for the task."
            )
        else:
            notes.append(f"Result limit is fixed at {fixed_k}; callers cannot change it.")
    if mode == "naive":
        notes.append("Each result returns dataset_id and s3_uri.")
    else:
        notes.append(
            "Each result returns dataset_id, s3_uri, llm_desc, columns or type, and dataset_snippet."
        )
    return f"{desc} {' '.join(notes)}".strip()


def _wrap_query_tool(
    base_tool: DecoratedFunctionTool,
    *,
    fixed_k: Optional[int],
    mode: str,
) -> DecoratedFunctionTool:
    spec = base_tool.tool_spec
    tool_name = spec["name"]
    effective_fixed_k = 100 if tool_name == "search_ideal" else fixed_k
    wrapped_description = _compose_description(
        spec.get("description", ""),
        tool_name=tool_name,
        fixed_k=effective_fixed_k,
        mode=mode,
    )
    top_k_default = _query_default(spec, "top_k", 10)

    if effective_fixed_k is not None:

        def _wrapped(query: str) -> dict:
            return reshape_search_payload(base_tool(query=query, top_k=effective_fixed_k), mode)

        return tool(_wrapped, name=tool_name, description=wrapped_description)

    def _wrapped(query: str, top_k: int = top_k_default) -> dict:
        return reshape_search_payload(base_tool(query=query, top_k=top_k), mode)

    return tool(_wrapped, name=tool_name, description=wrapped_description)


def _wrap_prefix_tool(
    base_tool: DecoratedFunctionTool,
    *,
    fixed_k: Optional[int],
    mode: str,
) -> DecoratedFunctionTool:
    spec = base_tool.tool_spec
    tool_name = spec["name"]
    wrapped_description = _compose_description(
        spec.get("description", ""),
        tool_name=tool_name,
        fixed_k=fixed_k,
        mode=mode,
    )
    limit_default = _query_default(spec, "limit", 50)

    if fixed_k is not None:

        def _wrapped(prefixes: List[str]) -> dict:
            return reshape_search_payload(base_tool(prefixes=prefixes, limit=fixed_k), mode)

        return tool(_wrapped, name=tool_name, description=wrapped_description)

    def _wrapped(prefixes: List[str], limit: int = limit_default) -> dict:
        return reshape_search_payload(base_tool(prefixes=prefixes, limit=limit), mode)

    return tool(_wrapped, name=tool_name, description=wrapped_description)


def build_search_tools(
    base_tools: Sequence[DecoratedFunctionTool],
    *,
    fixed_k: Optional[int],
    results_mode: str,
) -> List[DecoratedFunctionTool]:
    """Return search tools wrapped with k-control and payload shaping."""
    mode = str(results_mode or "naive").strip().lower()
    if mode not in {"naive", "ideal"}:
        raise ValueError(
            f"Unsupported results_mode '{results_mode}'. Expected: naive|ideal"
        )

    wrapped: List[DecoratedFunctionTool] = []
    for tool_obj in base_tools:
        tool_name = tool_obj.tool_spec.get("name", "")
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

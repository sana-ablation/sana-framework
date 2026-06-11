"""Shared raw dataset-profile lookup for profile-aware tools.

The canonical profile artifact lives under the LakeQA benchmark artifacts.
Rows are returned as stored; callers that need schema/snippet fallbacks should
layer those fallbacks outside this loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_PROFILES_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/lakeqa/tasks-mini/artifacts/table_profiles.jsonl"
)

_PROFILE_BY_URI: Dict[str, Dict[str, Any]] = {}
_PROFILE_BY_SLUG_FILENAME: Dict[Tuple[str, str], Dict[str, Any]] = {}
_PROFILES_LOADED: bool = False

_PROFILE_SCALAR_FIELDS = (
    "row_count",
    "column_count",
    "llm_description",
    "snippet",
    "record_tag",
)
_PROFILE_LIST_FIELDS = (
    "top_2_rows",
)


def _stem(name: str) -> str:
    value = str(name or "").strip().rsplit("/", 1)[-1]
    return value.rsplit(".", 1)[0] if "." in value else value


def _slug_stem_from_uri(uri: str) -> Optional[Tuple[str, str]]:
    raw = str(uri or "").strip()
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
    return (slug, _stem(filename))


def _profile_key_from_row(obj: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    slug = str(obj.get("slug") or obj.get("dataset_slug") or obj.get("dataset_id") or "").strip()
    filename = str(obj.get("filename") or obj.get("file_path") or obj.get("relative_path") or "").strip()
    if not slug or not filename:
        return None
    return (slug, _stem(filename))


def _load_profiles_cache() -> None:
    """Load raw precomputed dataset profiles into in-process caches."""
    global _PROFILES_LOADED, _PROFILE_BY_URI, _PROFILE_BY_SLUG_FILENAME
    if _PROFILES_LOADED:
        return

    profiles_by_uri: Dict[str, Dict[str, Any]] = {}
    profiles_by_slug_filename: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if _PROFILES_PATH.exists():
        with _PROFILES_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                s3_uri = str(obj.get("s3_uri") or "").strip()
                if s3_uri and s3_uri not in profiles_by_uri:
                    profiles_by_uri[s3_uri] = obj
                key = _profile_key_from_row(obj)
                if key is not None and key not in profiles_by_slug_filename:
                    profiles_by_slug_filename[key] = obj

    _PROFILE_BY_URI = profiles_by_uri
    _PROFILE_BY_SLUG_FILENAME = profiles_by_slug_filename
    _PROFILES_LOADED = True


def load_dataset_profile(s3_uri: str) -> Optional[Dict[str, Any]]:
    """Return a raw cached profile for a dataset file URI, or None."""
    if not s3_uri:
        return None
    _load_profiles_cache()
    profile = _PROFILE_BY_URI.get(str(s3_uri))
    if profile is not None:
        return dict(profile)
    key = _slug_stem_from_uri(str(s3_uri))
    if key is None:
        return None
    profile = _PROFILE_BY_SLUG_FILENAME.get(key)
    if profile is None:
        return None
    return dict(profile)


def _selected_columns(raw_columns: Any) -> Optional[list[Any]]:
    if not isinstance(raw_columns, list):
        return None
    columns = []
    for column in raw_columns:
        if isinstance(column, dict):
            selected: Dict[str, Any] = {}
            if "name" in column:
                selected["name"] = column["name"]
            if "type" in column:
                selected["type"] = column["type"]
            if selected:
                columns.append(selected)
        elif column is not None:
            columns.append(str(column))
    return columns or None


def select_dataset_profile_fields(profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the compact profile shape exposed to agent tools."""
    if not isinstance(profile, dict):
        return None

    schema_status = str(profile.get("schema_status") or "").strip()
    has_schema_error = bool(profile.get("schema_error"))
    if has_schema_error or schema_status in {"unavailable", "metadata", "archive"}:
        return None

    out: Dict[str, Any] = {}
    for key in _PROFILE_SCALAR_FIELDS:
        if key in profile:
            out[key] = profile[key]

    if schema_status == "single_column":
        return out or None

    columns = _selected_columns(profile.get("columns"))
    if columns is not None:
        out["columns"] = columns

    for key in _PROFILE_LIST_FIELDS:
        if key in profile:
            out[key] = profile[key]

    return out or None


__all__ = ["load_dataset_profile", "select_dataset_profile_fields"]

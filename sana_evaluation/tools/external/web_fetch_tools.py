"""Web download tool for `--no-s3` runs, gated by the search_web URL allowlist.

Under `--no-s3` the agent loses every S3-backed file tool and keeps only this
`download` plus `execute_code`. A URL is fetchable ONLY if a prior `search_web`
call in the same task returned it, so the arm measures what web search actually
retrieved rather than URLs the model memorised during pretraining.

The allowlist is written to `<sandbox>/.web_urls.json` rather than held in
module state on purpose: `download` runs through
``agent_tools._run_tool_with_timeout``, which uses a *spawned* subprocess, so
nothing in the parent's memory reaches it. The file doubles as a per-task audit
record of what search offered versus what the agent chose to fetch.

The sandbox is created and deleted per task (``agent_with_mode`` builds one via
``_create_isolated_sandbox`` and removes it in its ``finally``), so the
allowlist needs no explicit reset between tasks.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

import requests
from strands import tool

from sana_evaluation.tools.agent_tools import (
    _download_manifest_path,
    _get_sandbox_dir,
    _load_download_manifest,
    _run_tool_with_timeout,
    _tool_timeout_seconds,
    _write_download_manifest,
)

logger = logging.getLogger(__name__)

_ALLOWLIST_NAME = ".web_urls.json"
_WEB_DIR_NAME = "web"
_MAX_FILES_PER_CALL = 5
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024
_TIMEOUT_SECONDS = 60
_MAX_LISTED_URLS = 20
_MAX_NAME_CHARS = 120

# Only used when the URL path carries no usable extension of its own.
_CONTENT_TYPE_EXTENSIONS = {
    "text/csv": ".csv",
    "text/plain": ".txt",
    "text/html": ".html",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "application/pdf": ".pdf",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_KNOWN_EXTENSIONS = frozenset(_CONTENT_TYPE_EXTENSIONS.values()) | {".txt", ".tsv", ".zip", ".htm"}


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


def _allowlist_path() -> Path:
    return _get_sandbox_dir() / _ALLOWLIST_NAME


def allowed_urls() -> List[str]:
    """Return the URLs search_web has returned for the current task."""
    path = _allowlist_path()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [str(url) for url in raw if url]


def record_search_urls(urls: Iterable[str]) -> None:
    """Add search_web result URLs to the per-task allowlist, preserving order."""
    existing = allowed_urls()
    seen = set(existing)
    for url in urls:
        candidate = str(url or "").strip()
        if candidate and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)
    _allowlist_path().write_text(json.dumps(existing, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _is_http_url(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("http://", "https://"))


def _local_name_for(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    host = _SLUG_RE.sub("_", parsed.netloc) or "web"
    stem = _SLUG_RE.sub("_", parsed.path.strip("/")) or "index"

    # The extension must come from the URL *path*, never the whole name: a host
    # like `myarmybenefits.us.army.mil` is full of dots that are not a suffix.
    suffix = Path(parsed.path).suffix.lower()
    if suffix in _KNOWN_EXTENSIONS:
        stem = stem[: -len(suffix)]
    else:
        base = (content_type or "").split(";")[0].strip().lower()
        suffix = _CONTENT_TYPE_EXTENSIONS.get(base, ".html" if base.startswith("text/") else ".bin")

    return f"{host}__{stem}"[: _MAX_NAME_CHARS - len(suffix)] + suffix


def _fetch_to_sandbox(url: str, sandbox: Path) -> Dict[str, Any]:
    """Stream one URL into the sandbox. Returns a downloaded entry or an error."""
    with requests.get(
        url,
        stream=True,
        timeout=_TIMEOUT_SECONDS,
        allow_redirects=True,
        headers={"User-Agent": "sana-eval/1.0"},
    ) as resp:
        resp.raise_for_status()
        content_type = str((resp.headers or {}).get("Content-Type", ""))

        payload = bytearray()
        for chunk in resp.iter_content(chunk_size=_CHUNK_BYTES):
            if not chunk:
                continue
            payload.extend(chunk)
            if len(payload) > _MAX_DOWNLOAD_BYTES:
                return {
                    "error": (
                        f"Response is too large (over {_MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB). "
                        "Pick a more specific URL."
                    ),
                    "url": url,
                }

    web_dir = sandbox / _WEB_DIR_NAME
    web_dir.mkdir(parents=True, exist_ok=True)
    file_name = _local_name_for(url, content_type)
    local_path = web_dir / file_name
    local_path.write_bytes(bytes(payload))

    return {
        "local_path": str(local_path),
        "file_path": file_name,
        "dataset_id": _WEB_DIR_NAME,
        "s3_uri": url,
        "url": url,
        "content_type": content_type,
        "size": len(payload),
        "status": "downloaded",
    }


def _download_web_impl(files: List[Any]) -> Dict[str, Any]:
    if not isinstance(files, list):
        return {"error": (
            "download requires `files` to be a list. Example: "
            'download(files=[{"url": "https://example.gov/data.csv"}])'
        )}
    if len(files) == 0:
        return {"error": (
            "download requires a non-empty `files` list. Example: "
            'download(files=[{"url": "https://example.gov/data.csv"}])'
        )}
    if len(files) > _MAX_FILES_PER_CALL:
        return {"error": (
            f"Maximum {_MAX_FILES_PER_CALL} files per download call (got {len(files)}). "
            "Split your request into multiple download calls."
        )}

    sandbox = _get_sandbox_dir()
    allowed = allowed_urls()
    allowed_set = set(allowed)

    downloaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for spec in files:
        if isinstance(spec, str):
            spec = {"url": spec}
        if not isinstance(spec, dict):
            errors.append({"error": "Each file must be a dict with a `url`, or a URL string."})
            continue

        url = str(
            spec.get("url") or spec.get("uri") or spec.get("s3_uri") or spec.get("file_path") or ""
        ).strip()

        if not url:
            errors.append({"error": "Each file needs a `url`.", "file_spec": spec})
            continue

        if not _is_http_url(url):
            errors.append({
                "error": (
                    "This run has no data-lake access: `download` accepts only http(s) URLs "
                    "returned by `search_web`, not dataset_id/file_path or s3_uri."
                ),
                "file_spec": spec,
            })
            continue

        if url not in allowed_set:
            errors.append({
                "error": (
                    "That URL was not returned by `search_web` in this task, so it cannot be "
                    "downloaded. Search first, then download a URL from the results."
                ),
                "url": url,
                "allowed_urls": allowed[:_MAX_LISTED_URLS],
            })
            continue

        try:
            outcome = _fetch_to_sandbox(url, sandbox)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent, mirrors the S3 tool
            logger.warning("web download failed: %s: %s", type(exc).__name__, exc)
            errors.append({"error": f"{type(exc).__name__}: {exc}", "url": url})
            continue

        if "error" in outcome:
            errors.append(outcome)
        else:
            downloaded.append(outcome)

    manifest = (
        _write_download_manifest(sandbox, downloaded) if downloaded else _load_download_manifest(sandbox)
    )

    result: Dict[str, Any] = {
        "downloaded": downloaded,
        "download_count": len(downloaded),
        "sandbox_dir": str(sandbox),
        "path_map": manifest.get("path_map", {}),
        "manifest_path": str(_download_manifest_path(sandbox)),
        "usage_note": (
            "Use path_map or DOWNLOAD_MANIFEST_PATH in execute_code; do not guess sandbox paths."
        ),
    }
    if errors:
        result["errors"] = errors
    return result


_DOWNLOAD_DESCRIPTION = """Download web pages or files to the local sandbox so you can compute over them.

Only URLs that `search_web` returned earlier in THIS task can be downloaded.
Search first, then pass URLs from the results. There is no data lake in this
run: dataset_id / file_path / s3_uri are not accepted.

    CORRECT:
        download(files=[{"url": "https://data.example.gov/rows.csv"}])

    WRONG:
        download(files=[{"dataset_id": "census", "file_path": "rows.csv"}])

Maximum 5 files per call. After downloading, read or analyse the file with
`execute_code` — the file appears in `FILES` and `DOWNLOAD_PATHS`.

Args:
    files: NON-EMPTY list (max 5) of dicts, each with a `url` key. A bare URL
           string is also accepted.

Returns:
    Dict with 'downloaded', 'download_count', 'sandbox_dir', 'path_map' and
    'manifest_path'. Failures appear under 'errors'.
"""


def _download_entry(files: List[Any]) -> Dict[str, Any]:
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(_download_web_impl, files, timeout_seconds=timeout_seconds)
    if completed:
        return result or {"error": "download failed without returning a result"}
    return {
        "error": f"download timed out after {timeout_seconds}s. Try one URL at a time.",
        "downloaded": [],
        "download_count": 0,
        "sandbox_dir": str(_get_sandbox_dir()),
    }


download_web = tool(_download_entry, name="download", description=_DOWNLOAD_DESCRIPTION)


__all__ = [
    "allowed_urls",
    "download_web",
    "record_search_urls",
]

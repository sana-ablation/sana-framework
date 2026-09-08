"""Data-lake access tools for the evaluation agent.

Search, file inspection, SQL-over-S3, sandboxed execution and answer
submission over the benchmark bucket. One module: the tools share S3 client,
sandbox and timeout plumbing too densely to split usefully.

Bucket: lakeqa-yc4103-datalake
Folders: wikipedia/, datagov/
"""

from __future__ import annotations

import base64
import datetime
import decimal
import difflib
import io
import json
import multiprocessing as _mp
import os
import queue as _queue
import re
import shutil
import sys
import tempfile
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, deque
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

import duckdb
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from strands import tool

from sana_evaluation.runtime.peek_profile import load_dataset_profile, select_dataset_profile_fields

from dataindexing.formats import (
    build_xml_preview,
    detect_family,
    local_xml_name,
    normalize_xml_record_tag,
    should_skip,
    xml_record_to_row,
)
from dataindexing.sources.s3 import (
    BENCHMARK_BUCKETS,
    DEFAULT_BUCKET,
    FOLDERS,
    REGION,
    build_s3_client,
)

# Load AWS credentials from .env
load_dotenv()

# The active bucket. Unlike the imported constants it is mutable runtime state
# the agent owns: configure_benchmark() rebinds it (and drops the cached S3
# clients) when a run switches benchmark.
BUCKET = os.getenv("LAKEQA_BUCKET", DEFAULT_BUCKET)

# Sandbox directory on main disk (500G) instead of /tmp (63G tmpfs)
SANDBOX_BASE_DIR = Path(__file__).resolve().parent.parent.parent / ".sandbox"

_TOOL_RESULT_CHAR_CAP = 6_000  # ~1.5k tokens — keeps single tool results from dominating context
_DEFAULT_TOOL_TIMEOUT_SECONDS = 150
_DOWNLOAD_MANIFEST_NAME = ".download_manifest.json"

# Global sandbox directory (created per session)
_SANDBOX_DIR = None
# Optional override to force a specific sandbox directory (set by callers for isolation)
_SANDBOX_OVERRIDE = None
# Cached S3 clients and resolved access mode
_S3_SIGNED_CLIENT = None
_S3_UNSIGNED_CLIENT = None
_S3_CLIENT_MODE: Optional[str] = None  # signed | unsigned

# Slot written by submit_answer; read back by SubmitAnswerPlugin in agent.py
_submitted_answer: Optional[Dict[str, Any]] = None


def _empty_download_manifest() -> Dict[str, Any]:
    return {
        "downloaded": [],
        "local_paths": [],
        "path_map": {},
    }


def _download_manifest_path(sandbox: Path) -> Path:
    return sandbox / _DOWNLOAD_MANIFEST_NAME


def _load_download_manifest(sandbox: Path) -> Dict[str, Any]:
    manifest_path = _download_manifest_path(sandbox)
    if not manifest_path.is_file():
        return _empty_download_manifest()
    try:
        with manifest_path.open(encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return _empty_download_manifest()
    manifest = _empty_download_manifest()
    if isinstance(raw, dict):
        if isinstance(raw.get("downloaded"), list):
            manifest["downloaded"] = raw["downloaded"]
        if isinstance(raw.get("local_paths"), list):
            manifest["local_paths"] = raw["local_paths"]
        if isinstance(raw.get("path_map"), dict):
            manifest["path_map"] = raw["path_map"]
    return manifest


def _manifest_key_values(dataset_id: str, file_path: str, s3_uri: str) -> List[str]:
    keys = [
        s3_uri,
        f"{dataset_id}/{file_path}",
        f"{dataset_id}:{file_path}",
    ]
    return [key for key in keys if key]


def _write_download_manifest(sandbox: Path, downloaded: List[Dict[str, Any]]) -> Dict[str, Any]:
    manifest = _load_download_manifest(sandbox)
    by_s3_uri: Dict[str, Dict[str, Any]] = {
        str(item.get("s3_uri")): item
        for item in manifest["downloaded"]
        if isinstance(item, dict) and item.get("s3_uri")
    }
    for item in downloaded:
        s3_uri = str(item.get("s3_uri") or "")
        if s3_uri:
            by_s3_uri[s3_uri] = item
        local_path = str(item.get("local_path") or "")
        if local_path and local_path not in manifest["local_paths"]:
            manifest["local_paths"].append(local_path)
        dataset_id = str(item.get("dataset_id") or "")
        file_path = str(item.get("file_path") or "")
        for key in _manifest_key_values(dataset_id, file_path, s3_uri):
            manifest["path_map"][key] = local_path

    manifest["downloaded"] = list(by_s3_uri.values())
    manifest_path = _download_manifest_path(sandbox)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def configure_benchmark(benchmark: Optional[str] = None) -> str:
    """Configure the active data-lake bucket for a benchmark label."""
    global BUCKET, _S3_SIGNED_CLIENT, _S3_UNSIGNED_CLIENT, _S3_CLIENT_MODE

    explicit_benchmark = benchmark is not None
    normalized = (benchmark or os.getenv("LAKEQA_BENCHMARK") or "lakeqa").strip().lower()
    if normalized not in BENCHMARK_BUCKETS:
        expected = ", ".join(sorted(BENCHMARK_BUCKETS))
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected one of: {expected}")

    bucket = (
        BENCHMARK_BUCKETS[normalized]
        if explicit_benchmark
        else os.getenv("LAKEQA_BUCKET", BENCHMARK_BUCKETS[normalized])
    )
    os.environ["LAKEQA_BENCHMARK"] = normalized
    os.environ["LAKEQA_BUCKET"] = bucket
    if bucket != BUCKET:
        BUCKET = bucket
        _S3_SIGNED_CLIENT = None
        _S3_UNSIGNED_CLIENT = None
        _S3_CLIENT_MODE = None
    return BUCKET


def get_submitted_answer() -> Optional[Dict[str, Any]]:
    """Return the last submitted answer dict, or None if not yet submitted."""
    return _submitted_answer


def clear_submitted_answer() -> None:
    """Reset the submission slot (call before each task run)."""
    global _submitted_answer
    _submitted_answer = None


@tool
def submit_answer(answer: str, reasoning: str = "") -> str:
    """Submit the final answer to the question.

    Call this tool when you have found the definitive answer. The agent loop
    will stop immediately after this tool returns.

    Args:
        answer: The final answer, wrapped in square brackets e.g. [42]
        reasoning: Brief explanation of how you arrived at the answer
    """
    global _submitted_answer
    _submitted_answer = {
        "answer": answer,
        "reasoning": reasoning,
    }
    return f"Answer submitted: {answer}"


def set_sandbox_dir(path: Path) -> None:
    """Force the sandbox directory to a specific path (per-process isolation)."""
    global _SANDBOX_DIR, _SANDBOX_OVERRIDE
    _SANDBOX_OVERRIDE = Path(path)
    _SANDBOX_OVERRIDE.mkdir(parents=True, exist_ok=True)
    _SANDBOX_DIR = _SANDBOX_OVERRIDE


def _get_signed_s3_client():
    global _S3_SIGNED_CLIENT
    if _S3_SIGNED_CLIENT is None:
        _S3_SIGNED_CLIENT = build_s3_client(unsigned=False)
    return _S3_SIGNED_CLIENT


def _get_unsigned_s3_client():
    global _S3_UNSIGNED_CLIENT
    if _S3_UNSIGNED_CLIENT is None:
        _S3_UNSIGNED_CLIENT = build_s3_client(unsigned=True)
    return _S3_UNSIGNED_CLIENT


def _requested_s3_mode() -> str:
    """
    Read requested S3 mode from env.

    Values:
      - auto (default): signed first, fallback to unsigned on AccessDenied
      - signed: force signed requests
      - unsigned/public/anonymous/no-sign-request: force unsigned requests
    """
    raw = (os.getenv("S3_ACCESS_MODE", "auto") or "auto").strip().lower()
    if raw in {"signed", "private"}:
        return "signed"
    if raw in {"unsigned", "public", "anonymous", "anon", "no-sign-request"}:
        return "unsigned"
    return "auto"


def _should_fallback_to_unsigned(exc: Exception) -> bool:
    """Return True when signed auth failed and public fallback should be attempted."""
    if isinstance(exc, ClientError):
        code = (exc.response or {}).get("Error", {}).get("Code", "")
        if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}:
            return True
    msg = str(exc).lower()
    return "accessdenied" in msg or "explicit deny" in msg


def _is_s3_access_denied(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = (exc.response or {}).get("Error", {}).get("Code", "")
        return code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
    msg = str(exc).lower()
    return "accessdenied" in msg or "access denied" in msg or "explicit deny" in msg


def _get_s3_client():
    """
    Get S3 client.

    In auto mode, probe signed once and fallback to unsigned if signed auth is
    denied. This supports public buckets when IAM credentials are explicitly
    denied by policy.
    """
    global _S3_CLIENT_MODE

    requested = _requested_s3_mode()
    if requested == "signed":
        _S3_CLIENT_MODE = "signed"
        return _get_signed_s3_client()
    if requested == "unsigned":
        _S3_CLIENT_MODE = "unsigned"
        return _get_unsigned_s3_client()

    if _S3_CLIENT_MODE == "signed":
        return _get_signed_s3_client()
    if _S3_CLIENT_MODE == "unsigned":
        return _get_unsigned_s3_client()

    signed = _get_signed_s3_client()
    try:
        signed.list_objects_v2(Bucket=BUCKET, Prefix=f"{FOLDERS[0]}/", MaxKeys=1)
        _S3_CLIENT_MODE = "signed"
        return signed
    except Exception as exc:
        if _should_fallback_to_unsigned(exc):
            _S3_CLIENT_MODE = "unsigned"
            return _get_unsigned_s3_client()
        raise


def s3_access_mode() -> str:
    """Return resolved S3 access mode after client initialization."""
    _get_s3_client()
    return _S3_CLIENT_MODE or "signed"


def _get_sandbox_dir() -> Path:
    """Get or create the sandbox directory for downloaded files."""
    global _SANDBOX_DIR, _SANDBOX_OVERRIDE

    # If a caller pinned the sandbox (per-process isolation), use it
    if _SANDBOX_OVERRIDE is not None:
        _SANDBOX_DIR = _SANDBOX_OVERRIDE
        return _SANDBOX_DIR

    if _SANDBOX_DIR is None or not _SANDBOX_DIR.exists():
        SANDBOX_BASE_DIR.mkdir(parents=True, exist_ok=True)
        _SANDBOX_DIR = Path(tempfile.mkdtemp(prefix="task_", dir=SANDBOX_BASE_DIR))
    return _SANDBOX_DIR


def _tool_timeout_seconds() -> int:
    raw = str(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", _DEFAULT_TOOL_TIMEOUT_SECONDS)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TOOL_TIMEOUT_SECONDS
    return value if value > 0 else _DEFAULT_TOOL_TIMEOUT_SECONDS


def _collect_sandbox_snapshot(sandbox: Path) -> Tuple[List[str], List[str]]:
    sandbox_files: List[str] = []
    datasets_in_sandbox = set()
    if sandbox.exists():
        for file_path in sandbox.rglob('*'):
            if not file_path.is_file():
                continue
            try:
                rel_path = file_path.relative_to(sandbox)
            except ValueError:
                continue
            sandbox_files.append(str(rel_path))
            if rel_path.parts:
                datasets_in_sandbox.add(rel_path.parts[0])
    return sandbox_files, sorted(datasets_in_sandbox)


def _tool_worker_entry(
    result_queue,
    func: Callable[..., Dict[str, Any]],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    sandbox_dir: str,
) -> None:
    try:
        set_sandbox_dir(Path(sandbox_dir))
        result_queue.put(("ok", func(*args, **kwargs)))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
        )


def _run_tool_with_timeout(
    func: Callable[..., Dict[str, Any]],
    *args: Any,
    timeout_seconds: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    timeout = timeout_seconds if timeout_seconds is not None else _tool_timeout_seconds()
    if timeout <= 0:
        return True, func(*args, **kwargs)

    sandbox = _get_sandbox_dir()
    ctx = _mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_tool_worker_entry,
        args=(result_queue, func, args, kwargs, str(sandbox)),
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        return False, None

    try:
        status, payload = result_queue.get_nowait()
    except _queue.Empty:
        payload = {"error": f"Tool subprocess exited with code {proc.exitcode} without returning a result."}
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

    return True, payload


def _dataset_exists(s3, folder: str, dataset_id: str) -> bool:
    """Check whether a dataset exists under a given folder."""
    try:
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f"{folder}/{dataset_id}/",
            MaxKeys=1
        )
    except Exception as exc:
        if _is_s3_access_denied(exc):
            return False
        raise
    return "Contents" in response or "CommonPrefixes" in response


def _resolve_dataset_folder(dataset_id: str) -> Optional[str]:
    """Resolve dataset folder (datagov or wikipedia) for a dataset_id."""
    if not dataset_id:
        return None
    s3 = _get_s3_client()
    matches = []
    for folder in FOLDERS:
        if _dataset_exists(s3, folder, dataset_id):
            matches.append(folder)
    if len(matches) == 1:
        return matches[0]
    return None


def _strip_known_folder_prefix(dataset_id: str) -> str:
    """Strip a leading wikipedia/ or datagov/ prefix from a dataset id."""
    if not dataset_id or not isinstance(dataset_id, str):
        return dataset_id
    candidate = dataset_id.lstrip("/")
    lowered = candidate.lower()
    for folder in FOLDERS:
        prefix = f"{folder}/"
        if lowered.startswith(prefix):
            return candidate[len(prefix):]
    return dataset_id


def _canonicalize_file_path(folder: str, file_path: str) -> str:
    """Normalize common file-path omissions before building an S3 key."""
    normalized = (file_path or "").lstrip("/")
    if folder == "datagov" and normalized and "/" not in normalized:
        return f"files/{normalized}"
    return normalized


def _looks_like_s3_reference(value: str) -> bool:
    """Return True when the string looks like an S3 URI or bucket-relative key."""
    if not value or not isinstance(value, str):
        return False
    candidate = value.strip()
    if candidate.startswith("s3://"):
        return True
    candidate = candidate.lstrip("/")
    if candidate.startswith(f"{BUCKET}/"):
        return True
    parts = candidate.split("/", 2)
    return len(parts) >= 3 and parts[0] in FOLDERS


def _parse_s3_reference(value: str) -> Dict[str, Any]:
    """
    Parse an S3 URI or bucket-relative key into folder / dataset / file pieces.

    Accepted forms:
    - s3://<bucket>/<folder>/<dataset_id>/<file_path>
    - <bucket>/<folder>/<dataset_id>/<file_path>
    - <folder>/<dataset_id>/<file_path>
    """
    if not value or not isinstance(value, str):
        return {"error": "s3_uri must be a non-empty string"}

    raw = value.strip()
    candidate = raw

    if raw.startswith("s3://"):
        remainder = raw[len("s3://") :]
        bucket, sep, key = remainder.partition("/")
        if not bucket or not sep or not key:
            return {"error": "s3_uri must include bucket, dataset, and file path"}
        if bucket != BUCKET:
            return {"error": f"s3_uri bucket must be {BUCKET}, got {bucket}"}
        candidate = key
    else:
        candidate = raw.lstrip("/")
        bucket_prefix = f"{BUCKET}/"
        if candidate.startswith(bucket_prefix):
            candidate = candidate[len(bucket_prefix) :]

    parts = candidate.split("/", 2)
    if len(parts) < 3 or parts[0] not in FOLDERS:
        return {
            "error": (
                "s3_uri must point to a lake object like "
                f"s3://{BUCKET}/datagov/<dataset_id>/<file_path>"
            )
        }

    folder, dataset_id, file_path = (
        parts[0],
        parts[1],
        _canonicalize_file_path(parts[0], parts[2]),
    )
    if not dataset_id or not file_path:
        return {"error": "s3_uri must include both dataset_id and file_path"}

    key = f"{folder}/{dataset_id}/{file_path}"
    return {
        "bucket": BUCKET,
        "folder": folder,
        "dataset_id": dataset_id,
        "file_path": file_path,
        "key": key,
        "s3_uri": f"s3://{BUCKET}/{key}",
    }


def _resolve_file_reference(
    dataset_id: Optional[str] = None,
    file_path: Optional[str] = None,
    s3_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve either dataset_id+file_path or a single S3 URI into a canonical ref.
    """
    if s3_uri:
        return _parse_s3_reference(s3_uri)

    if dataset_id and not file_path and _looks_like_s3_reference(dataset_id):
        return _parse_s3_reference(dataset_id)

    normalized_dataset_id = _strip_known_folder_prefix(dataset_id or "")
    normalized_file_path = (file_path or "").lstrip("/")

    if not normalized_dataset_id:
        return {"error": "dataset_id or s3_uri is required"}
    if not normalized_file_path:
        return {"error": "file_path is required unless you pass s3_uri"}

    folder = _resolve_dataset_folder(normalized_dataset_id)
    if folder is None:
        return {"error": f"Dataset not found or ambiguous: {normalized_dataset_id}"}

    normalized_file_path = _canonicalize_file_path(folder, normalized_file_path)
    key = f"{folder}/{normalized_dataset_id}/{normalized_file_path}"
    return {
        "bucket": BUCKET,
        "folder": folder,
        "dataset_id": normalized_dataset_id,
        "file_path": normalized_file_path,
        "key": key,
        "s3_uri": f"s3://{BUCKET}/{key}",
    }


def _s3_error_code(exc: BaseException) -> str:
    if isinstance(exc, ClientError):
        return str(exc.response.get("Error", {}).get("Code") or "")
    return ""


def _is_missing_s3_key_error(exc: BaseException) -> bool:
    code = _s3_error_code(exc)
    if code in {"NoSuchKey", "404", "NotFound"}:
        return True
    text = str(exc)
    return "NoSuchKey" in text or "Not Found" in text or "404" in text


def _candidate_prefix_for_key(key: str) -> str:
    parts = (key or "").split("/", 2)
    if len(parts) >= 2 and parts[0] in FOLDERS:
        return f"{parts[0]}/{parts[1]}/"
    return str(Path(key).parent).strip(".") + "/" if "/" in key else ""


def _nearby_s3_candidates(s3: Any, *, bucket: str, key: str, file_path: str) -> Dict[str, Any]:
    prefix = _candidate_prefix_for_key(key)
    if not prefix:
        return {}
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
    except Exception as exc:
        return {"candidate_error": f"Could not list nearby files under {prefix}: {exc}"}

    keys = [
        str(item.get("Key") or "")
        for item in response.get("Contents", [])
        if item.get("Key") and str(item.get("Key")) != prefix
    ]
    relative_candidates = [
        candidate[len(prefix):]
        for candidate in keys
        if candidate.startswith(prefix) and candidate != prefix
    ]
    relative_candidates = [candidate for candidate in relative_candidates if candidate]
    close = difflib.get_close_matches(file_path, relative_candidates, n=5, cutoff=0.55)
    if not close:
        close = difflib.get_close_matches(key, keys, n=5, cutoff=0.55)
    return {
        "candidate_prefix": prefix,
        "did_you_mean": close,
        "available_files_preview": relative_candidates[:20],
    }


def _download_error_entry(
    exc: BaseException,
    *,
    s3: Any,
    dataset_id: str,
    file_path: str,
    s3_key: str,
    s3_uri: str,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "error": f"Failed to download: {str(exc)}",
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "bucket": BUCKET,
        "key": s3_key,
    }
    if _is_missing_s3_key_error(exc):
        entry["error"] = f"Failed to download: object not found at s3://{BUCKET}/{s3_key}"
        entry["s3_error_code"] = _s3_error_code(exc) or "NoSuchKey"
        entry.update(_nearby_s3_candidates(s3, bucket=BUCKET, key=s3_key, file_path=file_path))
    return entry


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text) if text else ""
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_by_query(query_tokens: List[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    common = query_set.intersection(text_tokens)
    if not common:
        return 0.0
    coverage = len(common) / len(query_set)
    density = len(common) / len(text_tokens)
    return (coverage * 0.8) + (density * 0.2)


def _search_wikipedia_titles(query: str) -> List[Dict[str, Any]]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }
    headers = {"User-Agent": "DataLakeAgentTools/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    results = data.get("query", {}).get("search", [])
    return [{"title": item.get("title"), "api_score": item.get("score")} for item in results]


def _search_datagov_packages(query: str) -> List[Dict[str, Any]]:
    url = "https://catalog.data.gov/api/3/action/package_search"
    params = {"q": query}
    headers = {"User-Agent": "DataLakeAgentTools/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError("data.gov search failed")
    return data.get("result", {}).get("results", [])


# =============================================================================
# Tool 1: Search
# =============================================================================
@tool
def search(prefixes: List[str], limit: int = 50) -> Dict[str, Any]:
    """
    Search for datasets matching one or more prefixes.

    Searches across BOTH wikipedia/ and datagov/ folders automatically.
    Uses S3's native prefix search which is efficient even with billions of objects.

    Args:
        prefixes: List of search prefixes. Each prefix must be at least 2 characters.
                  Examples: ["Barack"], ["climate", "census", "weather"]
        limit: Maximum results per folder per prefix (default 50)

    Returns:
        Dict with 'results' containing dataset identifiers

    Examples:
        >>> search(["Barack"])
        >>> search(["Barack", "climate", "census"])
    """
    # Validate input
    if not isinstance(prefixes, list):
        return {'error': "prefixes must be a list of strings."}
    prefix_list = prefixes

    # Validate all prefixes
    for p in prefix_list:
        if not p or len(p) < 2:
            return {'error': f"Prefix '{p}' must be at least 2 characters."}

    results_by_prefix = {}
    all_results = []
    seen_ids = set()

    for prefix in prefix_list:
        prefix_results = []
        for folder in FOLDERS:
            s3 = _get_s3_client()
            full_prefix = f"{folder}/{prefix}"

            # First try to find datasets (directories)
            try:
                response = s3.list_objects_v2(
                    Bucket=BUCKET,
                    Prefix=full_prefix,
                    Delimiter='/',
                    MaxKeys=limit
                )
            except Exception as exc:
                if _is_s3_access_denied(exc):
                    continue
                raise

            # Get dataset-level results (CommonPrefixes are "directories")
            if 'CommonPrefixes' in response:
                for p in response['CommonPrefixes']:
                    dataset_id = p['Prefix'].split('/')[1]
                    result_entry = {'dataset_id': dataset_id, 'type': 'dataset'}
                    prefix_results.append(result_entry)
                    if dataset_id not in seen_ids:
                        seen_ids.add(dataset_id)
                        all_results.append(result_entry)

        results_by_prefix[prefix] = prefix_results

    return {
        'results': all_results,
        'results_by_prefix': results_by_prefix,
        'count': len(all_results),
        'prefixes': prefix_list
    }


@tool
def search_prefix(prefixes: List[str], limit: int = 50) -> Dict[str, Any]:
    """
    Search for datasets matching one or more prefixes (S3 native prefix search).

    Searches across BOTH wikipedia/ and datagov/ folders automatically.
    Use for known dataset name fragments or entity names.

    Args:
        prefixes: List of search prefixes (min 2 chars each).
                  Examples: ["Erie_County"], ["index-crimes", "violent-crime"]
        limit: Maximum results per folder per prefix (default 50)

    Returns:
        Dict with 'results' containing dataset identifiers
    """
    return search(prefixes=prefixes, limit=limit)

@tool
def search_keyword(
    keywords: List[str],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Tag-style keyword search filtered by S3 existence.

    Args:
        keywords: List of short tag-style keywords. Long sentences will impair performance.
                  Examples: ["police"], ["police", "crime", "traffic"]
        limit: Optional cap on results after ranking; may omit relevant datasets

    Returns:
        Dict with 'results' list of dataset identifiers and metadata.
    """
    # Validate input
    if not isinstance(keywords, list):
        return {'error': "keywords must be a list of strings."}
    keyword_list = keywords

    # Validate all keywords
    for kw in keyword_list:
        if not kw or not kw.strip():
            return {'error': "All keywords must be non-empty."}

    s3 = _get_s3_client()
    results = []
    seen_ids = set()
    results_by_keyword = {}

    for keyword in keyword_list:
        query_tokens = _tokenize(keyword)
        keyword_results = []

        try:
            wiki_hits = _search_wikipedia_titles(keyword)
        except Exception:
            wiki_hits = []

        for item in wiki_hits:
            title = item.get("title") or ""
            dataset_id = title.replace(' ', '_')
            if dataset_id and _dataset_exists(s3, "wikipedia", dataset_id):
                result_entry = {
                    "title": title,
                    "dataset_id": dataset_id,
                    "score": _score_by_query(query_tokens, title),
                }
                keyword_results.append(result_entry)
                if dataset_id not in seen_ids:
                    seen_ids.add(dataset_id)
                    results.append(result_entry)

        try:
            datagov_hits = _search_datagov_packages(keyword)
        except Exception:
            datagov_hits = []

        for item in datagov_hits:
            name = item.get("name") or ""
            title = item.get("title") or name
            if name and _dataset_exists(s3, "datagov", name):
                score_text = f"{title} {name}".strip()
                result_entry = {
                    "title": title,
                    "dataset_id": name,
                    "score": _score_by_query(query_tokens, score_text),
                }
                keyword_results.append(result_entry)
                if name not in seen_ids:
                    seen_ids.add(name)
                    results.append(result_entry)

        # Sort and clean keyword-specific results
        keyword_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        results_by_keyword[keyword] = [{"title": r.get("title", ""), "dataset_id": r.get("dataset_id", "")} for r in keyword_results]

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    if limit is not None and limit < len(results):
        results = results[:limit]

    cleaned = [{"title": r.get("title", ""), "dataset_id": r.get("dataset_id", "")} for r in results]

    return {
        "results": cleaned,
        "results_by_keyword": results_by_keyword,
        "count": len(results),
        "keywords": keyword_list
    }

@tool
def list_files(dataset_ids: List[str], limit: int = 100) -> Dict[str, Any]:
    """
    List files within one or more datasets/directories.

    WARNING: Only use this for datasets with a SMALL number of files (< 100).
    The data lake contains billions of objects. If you try to list files in
    a large dataset or use a broad path, this operation may be very slow or
    return truncated results. Always provide a specific dataset path.

    Args:
        dataset_ids: List of dataset identifiers.
                     Examples: ["Barack_Obama"], ["Barack_Obama", "climate-data"]
        limit: Maximum files to return per dataset (default 100)

    Returns:
        Dict with 'files' list grouped by dataset_id. Each file entry includes
        `path`, `dataset_id`, `size`, and `s3_uri`.

    Example:
        >>> list_files(["Barack_Obama"])
        >>> list_files(["Barack_Obama", "climate-data"])
    """
    # Validate input
    if not isinstance(dataset_ids, list):
        return {'error': "dataset_ids must be a list of strings."}
    id_list = dataset_ids

    # Validate all dataset_ids
    for ds_id in id_list:
        if not ds_id:
            return {'error': "All dataset_ids must be non-empty."}

    s3 = _get_s3_client()
    all_files = []
    results_by_dataset = {}
    any_truncated = False

    for dataset_id in id_list:
        folder = _resolve_dataset_folder(dataset_id)
        if folder is None:
            results_by_dataset[dataset_id] = {'error': f"Dataset not found or ambiguous: {dataset_id}"}
            continue

        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=f"{folder}/{dataset_id}/",
            MaxKeys=limit
        )

        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if not obj['Key'].endswith('/'):
                    relative_path = obj['Key'].split(f"{folder}/{dataset_id}/", 1)[-1]
                    file_entry = {
                        'path': relative_path,
                        'size': obj['Size'],
                        'dataset_id': dataset_id,
                        's3_uri': f"s3://{BUCKET}/{folder}/{dataset_id}/{relative_path}"
                    }
                    files.append(file_entry)
                    all_files.append(file_entry)

        results_by_dataset[dataset_id] = {
            'files': files,
            'count': len(files),
            'truncated': response.get('IsTruncated', False)
        }
        if response.get('IsTruncated', False):
            any_truncated = True

    return {
        'files': all_files,
        'count': len(all_files),
        'dataset_ids': id_list,
        'by_dataset': results_by_dataset,
        'truncated': any_truncated
    }


# =============================================================================
# Tool 2: Download
# =============================================================================
@tool
def download(files: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Download one or more files from S3 to the local sandbox directory.

    REQUIRED ARGUMENT SHAPE — read carefully. You MUST pass a `files` list of
    dicts. Do NOT pass `dataset_id` / `file_path` directly at the top level.
    Prefer per-entry `s3_uri` when list_files/search/preloaded results gave you
    one. For datagov files, both "rows.txt" and "files/rows.txt" are accepted,
    but "files/rows.txt" is the canonical path.

        CORRECT (multiple files, up to 5):
            download(files=[
                {"dataset_id": "Barack_Obama", "file_path": "content.txt"},
                {"dataset_id": "climate-data", "file_path": "files/data.txt"},
            ])

        WRONG (these will all error — observed in eval logs):
            download(dataset_id="Barack_Obama", file_path="content.txt")
            download({})                                            # empty payload


    Maximum 5 files per call. If you need more, split into multiple `download`
    calls — there is no batch override.

    Args:
        files: NON-EMPTY list of dicts (max length 5). Each dict needs either
               ('dataset_id', 'file_path') or a single 's3_uri' pointing at
               the object.

    Returns:
        Dict with 'downloaded' list of successful downloads, 'download_count',
        'sandbox_dir', 'path_map', and 'manifest_path'. If any downloads fail,
        includes 'errors' with the attempted bucket/key and close candidates
        for missing S3 keys when available.
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(_download_impl, files, timeout_seconds=timeout_seconds)
    if completed:
        return result or {'error': "download failed without returning a result"}

    sandbox = _get_sandbox_dir()
    return {
        'error': (
            f"download timed out after {timeout_seconds}s. "
            "Try fewer files per call, download one file at a time, or narrow the task before downloading."
        ),
        'downloaded': [],
        'download_count': 0,
        'sandbox_dir': str(sandbox),
    }


def _download_impl(files: List[Dict[str, str]]) -> Dict[str, Any]:
    if not isinstance(files, list):
        return {'error': (
            "download requires `files` to be a list of dicts, not a single "
            "dict or other value. Example: "
            'download(files=[{"dataset_id": "Barack_Obama", "file_path": "content.txt"}])'
        )}

    if len(files) > 5:
        return {'error': (
            f"Maximum 5 files per download call (got {len(files)}). "
            "Split your request into multiple download calls — there is no "
            "batch override."
        )}

    if len(files) == 0:
        return {'error': (
            "download requires a non-empty `files` list. Example: "
            'download(files=[{"dataset_id": "Barack_Obama", "file_path": "content.txt"}])'
        )}

    s3 = _get_s3_client()
    sandbox = _get_sandbox_dir()

    downloaded: List[Dict[str, Any]] = []
    errors = []

    for file_spec in files:
        if isinstance(file_spec, str):
            file_spec = {'s3_uri': file_spec}
        if not isinstance(file_spec, dict):
            errors.append({'error': "Each file must be a dict with dataset_id/file_path or s3_uri"})
            continue

        ref = _resolve_file_reference(
            dataset_id=file_spec.get('dataset_id', ''),
            file_path=file_spec.get('file_path') or file_spec.get('path') or '',
            s3_uri=file_spec.get('s3_uri') or file_spec.get('uri') or '',
        )
        if 'error' in ref:
            error_entry = {'error': ref['error'], 'file_spec': file_spec}
            errors.append(error_entry)
            continue

        dataset_id = ref['dataset_id']
        file_path = ref['file_path']
        s3_key = ref['key']
        s3_uri = ref['s3_uri']

        # Create local path structure (no folder prefix)
        local_path = sandbox / dataset_id / file_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3.download_file(BUCKET, s3_key, str(local_path))
            file_size = local_path.stat().st_size

            downloaded.append({
                'local_path': str(local_path),
                'file_path': file_path,
                'dataset_id': dataset_id,
                's3_uri': s3_uri,
                'size': file_size,
                'status': 'downloaded'
            })
        except Exception as e:
            errors.append(
                _download_error_entry(
                    e,
                    s3=s3,
                    dataset_id=dataset_id,
                    file_path=file_path,
                    s3_key=s3_key,
                    s3_uri=s3_uri,
                )
            )

    manifest = _write_download_manifest(sandbox, downloaded) if downloaded else _load_download_manifest(sandbox)
    manifest_path = _download_manifest_path(sandbox)

    result = {
        'downloaded': downloaded,
        'download_count': len(downloaded),
        'sandbox_dir': str(sandbox),
        'path_map': manifest.get('path_map', {}),
        'manifest_path': str(manifest_path),
        'usage_note': (
            "Use path_map or DOWNLOAD_MANIFEST_PATH in execute_code; do not guess "
            "sandbox paths such as /mnt/data or files/..."
        ),
    }

    if errors:
        result['errors'] = errors

    return result

@tool
def get_sandbox_info() -> Dict[str, Any]:
    """
    Get information about the current sandbox directory and downloaded files.

    Returns:
        Dict with sandbox_dir path and list of downloaded files
    """
    sandbox = _get_sandbox_dir()

    files = []
    total_size = 0
    for path in sandbox.rglob('*'):
        if path.is_file():
            size = path.stat().st_size
            files.append({
                'path': str(path),
                'relative_path': str(path.relative_to(sandbox)),
                'size': size
            })
            total_size += size

    return {
        'sandbox_dir': str(sandbox),
        'files': files,
        'file_count': len(files),
        'total_size': total_size
    }


# =============================================================================
# Tool 3: Execute Code (Python Sandbox)
# =============================================================================

def _rewrite_execute_code_error(error_str: str, traceback_str: str) -> Optional[str]:
    """
    Pattern-match common execute_code failure modes and return a one-line
    actionable hint for the agent. Returns None when the error doesn't match
    any known pattern.

    Each pattern below maps to a real failure observed in eval logs (see
    tool_error_findings.md). The hint is appended to the error result as a
    separate `hint` field — the original `error` and `traceback` are unchanged.
    """
    if not error_str:
        return None

    combined = f"{error_str}\n{traceback_str or ''}"

    # KeyError: 'features' — agent assumes any JSON file is a GeoJSON
    # FeatureCollection. ~4 errors per eval.
    if "KeyError: 'features'" in combined:
        return (
            "Not all JSON files are GeoJSON FeatureCollections. Use peek_file "
            "to confirm the top-level shape (json_keys) before assuming "
            "data['features']."
        )

    # JSONDecodeError on the first byte — agent ran json.load on a non-JSON
    # file (often a CSV with .txt extension). ~7 errors per eval.
    if "JSONDecodeError" in combined and "line 1 column 1" in combined:
        return (
            "File is not valid JSON. Use peek_file to check the family — many "
            "`.txt` files in this lake are CSV, not JSON."
        )

    # TypeError: NoneType + str — `.get(key)` returned None and was concatenated.
    # ~6 errors per eval.
    if "TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'" in combined:
        return (
            "A `.get(key)` returned None and was used in string concatenation. "
            "Confirm the field exists with peek_file (json_keys / header_columns) "
            "before assuming it's present."
        )

    # pandas 3.0 removed infer_datetime_format — argument is auto-detected now.
    if "infer_datetime_format" in combined and "unexpected keyword argument" in combined:
        return (
            "pandas 3.0 removed `infer_datetime_format` — datetime format is "
            "auto-detected now. Drop the argument and call to_datetime(s, errors='coerce')."
        )

    # Empty-iterable reductions — empty dataframe / filter result.
    if (
        "max() iterable argument is empty" in combined
        or "min() iterable argument is empty" in combined
        or "max() arg is an empty sequence" in combined
        or "min() arg is an empty sequence" in combined
        or "argmax of an empty sequence" in combined
    ):
        return (
            "The iterable is empty (likely an empty filter result or dataframe). "
            "Check `len(...)` or `if not df.empty` before reducing."
        )

    # pandas Usecols mismatch — agent guessed column names without peeking.
    if "Usecols do not match columns" in combined:
        return (
            "Column names in `usecols` don't exist in the file. Use peek_file "
            "to see the real header_columns before naming columns in read_csv."
        )

    # ModuleNotFoundError — list what IS available so the agent doesn't
    # repeatedly try other modules.
    mnfe_match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", combined)
    if mnfe_match:
        module = mnfe_match.group(1)
        return (
            f"`{module}` is not available in the sandbox. Pre-installed modules: "
            "pandas, json, csv, os, glob, re, pathlib, ijson. The sandbox blocks "
            "network access, so pip install is not possible — use what's available."
        )

    # XML parser on non-XML file — usually a JSON or CSV.
    if "ParseError" in combined and "not well-formed" in combined:
        return (
            "XML parser hit a non-XML file. Use peek_file to check the family "
            "before parsing. For XML/KML structured records, use parse_xml_records. "
            "Do not use execute_code for XML/KML extraction."
        )

    return None


@tool
def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a sandbox environment with access to downloaded files.
    Use only for tabular or JSON-like sources. Do not use execute_code to
    parse Wikipedia/content.txt, prose/plain text, XML/KML, HTML, PDFs, binary
    files, or other non-tabular/non-JSON sources. Use parse_xml_records for
    XML/KML structured records.

    The code runs with:
    - Working directory set to the sandbox directory
    - Pre-imported: pandas, json, csv, os, glob, re, pathlib, ijson
    - Variable `SANDBOX_DIR` pointing to the sandbox directory (also available
      as os.environ['SANDBOX_DIR'])
    - Variable `FILES` containing list of downloaded file paths
    - Variable `DOWNLOAD_MANIFEST_PATH` and env var of the same name pointing
      to `.download_manifest.json`
    - Variable `DOWNLOAD_PATHS` containing the manifest `path_map`

    Write your analysis code and print() results. The printed output will be returned.
    Do not use this tool to view or extract facts from non-tabular text files;
    use read_file or grep_file for those sources.
    Note: execution has a timeout; avoid inefficient code.

    For large JSON files (100+ MB), use the pre-imported `ijson` module to
    stream-parse without loading everything into memory:
        for feat in ijson.items(open(path, 'rb'), 'features.item'): ...

    Args:
        code: Python code to execute

    Returns:
        Dict with 'output' (stdout), 'error' (if any), 'success' (bool)

    Example:
        >>> execute_code('''
        ... import pandas as pd
        ... df = pd.read_csv(SANDBOX_DIR + "/wikipedia/Barack_Obama/table_0.csv")
        ... print(df.head())
        ... print(f"Total rows: {len(df)}")
        ... ''')
        {'output': '   col1  col2\\n...\\nTotal rows: 50', 'success': True}
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(_execute_code_impl, code, timeout_seconds=timeout_seconds)
    if completed:
        return result or {'error': "execute_code failed without returning a result", 'success': False}

    sandbox = _get_sandbox_dir()
    sandbox_files, datasets_in_sandbox = _collect_sandbox_snapshot(sandbox)
    return {
        'output': '',
        'error': (
            f"execute_code timed out after {timeout_seconds}s. "
            "Use a narrower script, stream large files, or print less intermediate data."
        ),
        'success': False,
        'sandbox_dir': str(sandbox),
        'sandbox_files': sandbox_files,
        'datasets_in_sandbox': datasets_in_sandbox,
    }


def _execute_code_impl(code: str) -> Dict[str, Any]:
    if not code or not code.strip():
        return {'error': "No code provided", 'success': False}

    sandbox = _get_sandbox_dir()
    manifest_path = _download_manifest_path(sandbox)
    download_manifest = _load_download_manifest(sandbox)

    # Collect downloaded files
    downloaded_files = []
    if sandbox.exists():
        for path in sandbox.rglob('*'):
            if path.is_file():
                downloaded_files.append(str(path))
    sandbox_files, datasets_in_sandbox = _collect_sandbox_snapshot(sandbox)

    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'SANDBOX_DIR': str(sandbox),
        'FILES': downloaded_files,
        'DOWNLOAD_MANIFEST_PATH': str(manifest_path),
        'DOWNLOAD_PATHS': download_manifest.get('path_map', {}),
    }

    # Block all outgoing network traffic by disabling socket
    import socket as _socket
    _original_socket = _socket.socket

    def _blocked_socket(*args, **kwargs):
        raise OSError("Network access is disabled in sandbox. Use the download() tool to fetch data from the datalake.")

    _socket.socket = _blocked_socket

    # Pre-import common libraries. ijson is included so the agent can
    # stream-parse 100+ MB JSON files without ModuleNotFoundError (~8 errors
    # per eval before this).
    pre_imports = """
import pandas as pd
import json
import csv
import os
import glob
import re
import ijson
from pathlib import Path
"""

    # Capture stdout
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_cwd = os.getcwd()
    # Save and set SANDBOX_DIR in os.environ — the agent frequently writes
    # `os.environ['SANDBOX_DIR']` instead of using the injected local
    # (~14 KeyError per eval). Both forms now work.
    _prev_sandbox_env = os.environ.get('SANDBOX_DIR')
    _prev_download_manifest_env = os.environ.get('DOWNLOAD_MANIFEST_PATH')
    os.environ['SANDBOX_DIR'] = str(sandbox)
    os.environ['DOWNLOAD_MANIFEST_PATH'] = str(manifest_path)

    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        # Change to sandbox directory
        os.chdir(sandbox)

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Execute pre-imports
        exec(pre_imports, exec_globals)

        # Execute user code
        exec(code, exec_globals)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        stdout_overflow_path = None
        if len(output) > _TOOL_RESULT_CHAR_CAP:
            dump_path = sandbox / "_stdout_overflow.txt"
            dump_path.write_text(output)
            stdout_overflow_path = str(dump_path)
            output = (
                output[:_TOOL_RESULT_CHAR_CAP]
                + f"\n... [stdout truncated at {_TOOL_RESULT_CHAR_CAP} chars. "
                f"Full output written to: {dump_path} — read it with: "
                f"open('{dump_path}').read(). Print less data or query more specifically.]"
            )

        result = {
            'output': output,
            'success': True,
            'sandbox_dir': str(sandbox),
            'sandbox_files': sandbox_files,
            'datasets_in_sandbox': datasets_in_sandbox,
        }
        if stdout_overflow_path:
            result['local_result_path'] = stdout_overflow_path
            result['truncation_note'] = (
                f"stdout exceeded {_TOOL_RESULT_CHAR_CAP} chars and was truncated. "
                f"Full output at: {stdout_overflow_path}"
            )

        if errors:
            result['stderr'] = errors

        return result

    except BaseException as e:
        # Catch BaseException to handle SystemExit, KeyboardInterrupt, etc.
        # that the agent's code might raise (these bypass "except Exception")
        error_str = f"{type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()
        result = {
            'output': stdout_capture.getvalue(),
            'error': error_str,
            'traceback': traceback_str,
            'success': False,
            'sandbox_dir': str(sandbox),
            'sandbox_files': sandbox_files,
            'datasets_in_sandbox': datasets_in_sandbox
        }
        hint = _rewrite_execute_code_error(error_str, traceback_str)
        if hint:
            result['hint'] = hint
        return result
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.chdir(old_cwd)
        _socket.socket = _original_socket  # Restore network access
        # Restore the prior SANDBOX_DIR env var (or remove if it wasn't set)
        if _prev_sandbox_env is None:
            os.environ.pop('SANDBOX_DIR', None)
        else:
            os.environ['SANDBOX_DIR'] = _prev_sandbox_env
        if _prev_download_manifest_env is None:
            os.environ.pop('DOWNLOAD_MANIFEST_PATH', None)
        else:
            os.environ['DOWNLOAD_MANIFEST_PATH'] = _prev_download_manifest_env


# =============================================================================
# Utility Functions
# =============================================================================
def cleanup_sandbox() -> Dict[str, Any]:
    """
    Clean up the sandbox directory and delete all downloaded files.

    Returns:
        Dict with cleanup status
    """
    global _SANDBOX_DIR, _SANDBOX_OVERRIDE

    if _SANDBOX_DIR is None or not _SANDBOX_DIR.exists():
        return {'status': 'no_sandbox', 'deleted_files': 0}

    try:
        file_count = sum(1 for _ in _SANDBOX_DIR.rglob('*') if _.is_file())
        shutil.rmtree(_SANDBOX_DIR)
        _SANDBOX_DIR = None
        _SANDBOX_OVERRIDE = None
        return {'status': 'cleaned', 'deleted_files': file_count}
    except Exception as e:
        return {'error': f"Failed to cleanup: {str(e)}"}


# ---------------------------------------------------------------------------
# Budget constants (mirrored from streams.py S3Config defaults)
# ---------------------------------------------------------------------------
_PEEK_BYTES = 65_536          # 64 KB for initial range-GET / peek
_QUERY_ROW_CAP = 200
_SEARCH_MAX_MATCHES = 20
_QUERY_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB — above this, download first
_MAX_SPREADSHEET_PEEK_BYTES = 128 * 1024 * 1024
_MAX_SPREADSHEET_PREVIEW_COLUMNS = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _s3_range_get(s3, key: str, start: int, end: int) -> bytes:
    """Sync range-GET from the configured bucket."""
    resp = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def _s3_head(s3, key: str) -> int:
    """Return file size in bytes via HeadObject."""
    meta = s3.head_object(Bucket=BUCKET, Key=key)
    return meta.get("ContentLength", 0)


def _s3_peek_text(s3, key: str, size_bytes: int) -> Tuple[str, int]:
    """Return UTF-8 preview text plus sampled byte count for a possibly empty object."""
    if size_bytes <= 0:
        return "", 0
    end = min(_PEEK_BYTES - 1, size_bytes - 1)
    raw = _s3_range_get(s3, key, 0, end)
    return raw.decode("utf-8", errors="replace"), len(raw)


def _s3_read_bounded(s3, key: str, size_bytes: int, max_bytes: int) -> bytes:
    """Read a whole small object, failing clearly when it exceeds the cap."""
    if size_bytes > max_bytes:
        raise ValueError(f"file is {size_bytes} bytes, above {max_bytes} byte spreadsheet peek cap")
    if size_bytes <= 0:
        return b""
    return _s3_range_get(s3, key, 0, size_bytes - 1)


def _duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create a fresh in-memory DuckDB connection with httpfs and AWS credentials."""
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    region = os.getenv("AWS_DEFAULT_REGION", REGION)
    conn.execute(f"SET s3_region='{region}'")
    if s3_access_mode() == "unsigned":
        # Keep DuckDB S3 access anonymous for public buckets.
        conn.execute("SET s3_access_key_id=''")
        conn.execute("SET s3_secret_access_key=''")
        conn.execute("SET s3_session_token=''")
    else:
        aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        aws_session = os.getenv("AWS_SESSION_TOKEN", "")
        if aws_key:
            conn.execute(f"SET s3_access_key_id='{aws_key}'")
        if aws_secret:
            conn.execute(f"SET s3_secret_access_key='{aws_secret}'")
        if aws_session:
            conn.execute(f"SET s3_session_token='{aws_session}'")
    return conn


_MAX_OBJECT_SIZE_RE = re.compile(
    r'"maximum_object_size".*?bytes\s*exceeded.*?\(>(\d+)\s*bytes\)',
    re.DOTALL,
)


def _normalize_sql_backticks(sql: str) -> str:
    """
    Convert MySQL-style backtick identifiers to DuckDB's double-quoted form.

    Agents trained on MySQL/SQLite repeatedly write `` `column name` `` even
    though DuckDB only accepts `"column name"`. Observed ~17 times in eval
    logs as Parser Errors. Auto-fix at the source eliminates the round-trip.

    Carefully preserves backticks that fall INSIDE single-quoted string
    literals (e.g. `WHERE name = 'O\\`Brien'`) and inside already-double-
    quoted identifiers (e.g. `"weird\\`col"`). Handles escaped single quotes
    (`''`) inside literals.
    """
    out: List[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if in_single:
            if ch == "'":
                # `''` is an escaped single quote inside a literal
                if i + 1 < n and sql[i + 1] == "'":
                    out.append("''")
                    i += 2
                    continue
                in_single = False
            out.append(ch)
        elif in_double:
            if ch == '"':
                # `""` is an escaped double quote inside an identifier
                if i + 1 < n and sql[i + 1] == '"':
                    out.append('""')
                    i += 2
                    continue
                in_double = False
            out.append(ch)
        else:
            if ch == "'":
                in_single = True
                out.append(ch)
            elif ch == '"':
                in_double = True
                out.append(ch)
            elif ch == "`":
                out.append('"')
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def _rewrite_query_error(raw: str) -> str:
    """
    Rewrite low-level DuckDB error strings into actionable remediation hints
    so the agent stops thrashing on platform-side limits.

    The headline case: when DuckDB's read_json_auto rejects a JSON object
    larger than `maximum_object_size`, the stock hint says "Try increasing
    maximum_object_size" — which is misleading because the cap is hard-coded
    in this module and the agent cannot change it. Logs show the agent then
    retries the same query, malforms a download call, or pivots to peek_file
    which doesn't help. The right remediation is the same one used by the
    pre-check at line 483: download the file and process it with execute_code.
    """
    m = _MAX_OBJECT_SIZE_RE.search(raw)
    if m:
        observed = m.group(1)
        limit_mb = _QUERY_MAX_FILE_BYTES // (1024 * 1024)
        return (
            f"File contains a JSON object larger than the {limit_mb} MB query limit "
            f"({observed} bytes observed). query_file cannot stream it. "
            "Use download to fetch the file, then execute_code with a "
            "streaming JSON parser (e.g. ijson) or pandas.read_json with "
            "chunksize."
        )
    if "Timeout was reached" in raw and "HTTP GET" in raw:
        return (
            "S3 read timed out — file is likely too large to query directly "
            "over httpfs. Use download to fetch it locally, then execute_code "
            "to process it. Original: " + raw
        )
    if "Parser Error" in raw and "`" in raw:
        # Agent is using MySQL-style backtick identifiers; DuckDB requires
        # double quotes. Observed ~17 times in production logs.
        return (
            'DuckDB uses double quotes for identifiers, not backticks. '
            'Replace `column name` with "column name" in your SQL. '
            "Original: " + raw
        )
    return raw


def _rewrite_unqueryable_family_error(family: str) -> str:
    """
    Replace the bare "File family '<X>' is not queryable with SQL" message
    with a hint naming the right tool to use instead. The agent has no way
    to know that text files should go through grep/read/peek; spell it out.
    """
    if family == "xml":
        return (
            "XML/KML was detected. query_file does not support XML because "
            "SQL has no stable row model for arbitrary XML documents. "
            "Use peek_file to inspect tags and schema fields, parse_xml_records "
            "to extract/filter/group structured XML/KML records, grep_file to "
            "search for specific values, or read_file to inspect nearby text. "
            "Do not use query_file or execute_code for XML/KML sources."
        )
    if family == "text":
        return (
            "File contents are plain text — query_file only handles CSV and "
            "JSON. Use peek_file to inspect structure, grep_file to search "
            "for specific values, or read_file to load lines directly."
        )
    return (
        f"File family '{family}' is not queryable with SQL via query_file "
        "(only CSV and JSON are supported). Use peek_file to inspect what "
        "the file actually contains, then pick a tool that matches its "
        "format. Do not use query_file or execute_code unless the source is "
        "tabular or JSON."
    )


def _to_json_safe(value: Any) -> Any:
    """
    Convert DuckDB row values into JSON-serializable primitives.

    DuckDB returns native Python types for SQL DATE / TIMESTAMP / TIME / INTERVAL /
    DECIMAL / UUID / BLOB columns, which json.dumps cannot serialize. Without
    this conversion, even `SELECT * FROM t LIMIT 1` against any table with a
    date column raises `Object of type date is not JSON serializable`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, datetime.time):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_safe(v) for v in value]
    # Fallback for unexpected types (e.g., numpy scalars). Stringify rather
    # than crash — surfaces the value while keeping the tool call alive.
    return str(value)


def _strip_folder_prefix(dataset_id: str) -> str:
    """
    Silently strip a hallucinated leading `wikipedia/` or `datagov/` prefix
    from a dataset_id.

    The agent frequently constructs dataset ids of the form `wikipedia/<page>`
    after seeing a Wikipedia mention (~38 read_file errors per eval, plus a
    handful in peek_file/grep_file). The actual dataset_id is the bare name
    (e.g. `Logan_Fontenelle`), which the agent already knows one turn later
    via search_prefix. Per tool_error_findings.md the recommended fix is to
    auto-strip the prefix and resolve.

    Strips a single optional leading slash, then a case-insensitive
    `wikipedia/` or `datagov/` segment. Idempotent. Returns the input
    unchanged for empty/None or strings that don't start with one of the
    known folder prefixes.
    """
    if not dataset_id or not isinstance(dataset_id, str):
        return dataset_id
    candidate = dataset_id.lstrip("/")
    lowered = candidate.lower()
    for folder in ("wikipedia/", "datagov/"):
        if lowered.startswith(folder):
            return candidate[len(folder):]
    return dataset_id


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _xml_row_matches_filters(row: Dict[str, str], filters: Dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = row.get(str(field))
        if isinstance(expected, (list, tuple, set)):
            allowed = {str(value) for value in expected}
            if actual not in allowed:
                return False
        elif actual != str(expected):
            return False
    return True


def _select_xml_row_fields(row: Dict[str, str], fields: List[str]) -> Dict[str, str]:
    if not fields:
        return row
    return {field: row.get(field, "") for field in fields}


def _is_excel_path(file_path: str) -> bool:
    return file_path.lower().rsplit("?", 1)[0].endswith((".xlsx", ".xlsm", ".xltx", ".xltm"))


def _format_spreadsheet_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return str(value)


def _first_nonempty_rows(worksheet, max_rows: int, max_columns: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    max_seen_columns = 0
    for row in worksheet.iter_rows(values_only=True):
        values = [_format_spreadsheet_cell(value) for value in row]
        while values and values[-1] == "":
            values.pop()
        if not any(value != "" for value in values):
            continue
        max_seen_columns = max(max_seen_columns, len(values))
        if len(values) > max_columns:
            values = values[:max_columns] + [f"... {len(values) - max_columns} more columns"]
        rows.append(values)
        if len(rows) >= max_rows:
            break
    return rows, max_seen_columns


def _build_excel_preview(raw: bytes, max_rows: int) -> Dict[str, Any]:
    try:
        from openpyxl import load_workbook
    except Exception as e:
        return {
            "family": "xlsx",
            "preview_text": (
                "Excel workbook detected, but openpyxl is not installed. "
                "Install openpyxl or use download plus an environment with Excel support."
            ),
            "excel_error": f"openpyxl unavailable: {e}",
        }

    try:
        workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except Exception as e:
        return {
            "family": "xlsx",
            "preview_text": f"Excel workbook detected, but preview failed: {e}",
            "excel_error": str(e),
        }

    sheet_names = list(workbook.sheetnames)
    sheet_infos: list[Dict[str, Any]] = []
    preview_blocks = [f"Excel workbook with sheets: {', '.join(sheet_names)}"]
    for sheet_name in sheet_names[:8]:
        worksheet = workbook[sheet_name]
        rows, max_seen_columns = _first_nonempty_rows(
            worksheet,
            max(max_rows, 1) + 1,
            _MAX_SPREADSHEET_PREVIEW_COLUMNS,
        )
        info: Dict[str, Any] = {
            "name": sheet_name,
            "max_row": worksheet.max_row,
            "max_column": worksheet.max_column,
        }
        if max_seen_columns > _MAX_SPREADSHEET_PREVIEW_COLUMNS:
            info["preview_column_limit"] = _MAX_SPREADSHEET_PREVIEW_COLUMNS
            info["preview_columns_truncated"] = max_seen_columns - _MAX_SPREADSHEET_PREVIEW_COLUMNS
        if rows:
            info["header_columns"] = rows[0]
            info["preview_rows"] = rows[1:max_rows + 1]
        sheet_infos.append(info)

        preview_blocks.append(f"\nSheet: {sheet_name} ({worksheet.max_row} rows x {worksheet.max_column} columns)")
        if rows:
            preview_blocks.extend(",".join(row) for row in rows[:max_rows + 1])
        else:
            preview_blocks.append("(no non-empty rows found)")

    return {
        "family": "xlsx",
        "preview_text": "\n".join(preview_blocks),
        "sheet_names": sheet_names,
        "sheets": sheet_infos,
    }


# ---------------------------------------------------------------------------
# Tool 1: peek_file
# ---------------------------------------------------------------------------

@tool
def peek_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    max_rows: int = 20,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Inspect a SINGLE file via a budget range-GET. Returns the content family
    (csv/json/xml/text), column headers or XML tags/schema hints, and a
    preview — no full download. When available, also returns a compact
    `profile` field from the benchmark table_profiles.jsonl artifact.

    USE THIS for one file at a time. For multiple files in one call, use
    `peek_multiple` instead (different signature: takes a `files` list).

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path.

    Args:
        dataset_id: ONE dataset identifier as a bare string, e.g. "Barack_Obama"
        file_path:  ONE relative path within the dataset, e.g. "files/data.txt"
        s3_uri:     Optional full object URI instead of dataset_id/file_path
        max_rows:   Maximum preview rows to include (default 20)

    Example call:
        peek_file(dataset_id="index-crimes-by-county", file_path="files/rows.txt")

    Returns:
        Dict with keys: family, preview_text, header_columns, row_count_estimate,
        size_bytes, dataset_id, file_path. XML previews may also include
        xml_root_tag, xml_namespaces, xml_schema_fields,
        xml_record_tag_candidates, xml_preview_mode.
        May also include `profile` with selected cached metadata. Strict
        queryable profiles include columns as name/type pairs, row_count,
        top_2_rows, and llm_description. Strict XML/KML profiles may also
        include record_tag. Single-column CSV/TXT profiles include
        column_count, snippet, and llm_description. Profiles retried from a
        safe prefix expose the same columns/top_2_rows shape. Profiles with
        parser errors, metadata, archives, or unavailable schemas are omitted.
        On error: {error: ...}
    """
    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    s3 = _get_s3_client()
    key = ref["key"]

    try:
        size_bytes = _s3_head(s3, key)
    except Exception as e:
        return {"error": f"HeadObject failed: {e}"}

    if _is_excel_path(file_path):
        try:
            raw = _s3_read_bounded(s3, key, size_bytes, _MAX_SPREADSHEET_PEEK_BYTES)
        except Exception as e:
            return {"error": f"Excel preview failed: {e}"}

        result: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "size_bytes": size_bytes,
        }
        result.update(_build_excel_preview(raw, max_rows=max_rows))
        try:
            profile = load_dataset_profile(s3_uri)
        except Exception:
            profile = None
        profile = select_dataset_profile_fields(profile)
        if profile is not None:
            result["profile"] = profile
        return result

    try:
        text, sampled_bytes = _s3_peek_text(s3, key, size_bytes)
    except Exception as e:
        return {"error": f"Range-GET failed: {e}"}

    family = detect_family(text)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    preview_lines = lines[: max_rows + 1]  # +1 to include header for csv
    preview_text = "\n".join(preview_lines[:max_rows])

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "size_bytes": size_bytes,
        "family": family,
        "preview_text": preview_text,
    }

    # Estimate row count from sampled bytes
    if size_bytes > 0 and sampled_bytes > 0:
        lines_in_sample = len(lines)
        bytes_per_line = sampled_bytes / max(lines_in_sample, 1)
        result["row_count_estimate"] = int(size_bytes / bytes_per_line) if bytes_per_line > 0 else None
    else:
        result["row_count_estimate"] = len(lines)

    # CSV: extract header columns
    if family == "csv" and lines:
        header = lines[0]
        for delim in (",", "\t", "|", ";"):
            if delim in header:
                result["header_columns"] = [c.strip() for c in header.split(delim)]
                break

    # JSON: extract keys from first object
    if family == "json":
        try:
            first_obj = json.loads(lines[0]) if lines else None
            if isinstance(first_obj, dict):
                result["json_keys"] = sorted(first_obj.keys())
            elif isinstance(first_obj, list) and first_obj and isinstance(first_obj[0], dict):
                result["json_keys"] = sorted(first_obj[0].keys())
        except Exception:
            pass

    if family == "xml":
        result.update(build_xml_preview(text, size_bytes, peek_bytes=_PEEK_BYTES))

    try:
        profile = load_dataset_profile(s3_uri)
    except Exception:
        profile = None
    profile = select_dataset_profile_fields(profile)
    if profile is not None:
        result["profile"] = profile

    return result


# ---------------------------------------------------------------------------
# Tool 1b: peek_multiple (batch wrapper around peek_file)
# ---------------------------------------------------------------------------

@tool
def peek_multiple(
    files: Optional[List[Dict[str, str]]] = None,
    entries: Optional[List[Dict[str, str]]] = None,
    max_rows: int = 20,
) -> Dict[str, Any]:
    """
    Inspect SEVERAL files in ONE call — a batch wrapper around peek_file.
    Results may include compact `profile` fields when cached profiles are available.

    USE THIS when you already know which 2+ files you need
    (e.g. immediately after `list_files` returned several relevant paths).
    For a single file, use `peek_file` instead — its signature is simpler.

    Prefer per-entry `s3_uri` when list_files/search/preloaded results gave
    you one.

    REQUIRED ARGUMENT SHAPE: You MUST pass a `files` list of dicts, NOT
    `dataset_id`/`file_path` directly:

        CORRECT:
            peek_multiple(files=[
                {"dataset_id": "census-2021", "file_path": "files/rows.txt"},
            ])
        WRONG (these will all error):
            peek_multiple(max_rows=5)                             # missing files

    Args:
        files:    NON-EMPTY list of dicts. Each dict needs 'dataset_id' and
                  'file_path'. The key 'path' is also accepted as an alias for
                  'file_path' so raw list_files output can be passed directly.
                  A per-entry `s3_uri` is also accepted.
        entries:  Alias for `files`. Accepted to be forgiving when the agent
                  uses the older/wrong wrapper key.
        max_rows: Maximum preview rows per file (default 20).

    Returns:
        Dict with 'results' list (one entry per file, same shape as peek_file)
        and 'count'. Each result may include `profile` with selected cached
        metadata for queryable files: columns as name/type pairs, row_count,
        top_2_rows, llm_description, and, for XML/KML, record_tag. Sampled
        retry profiles expose the same columns/top_2_rows shape. Profiles with
        parser errors, metadata, archives, or unavailable schemas are omitted.
    """
    if files is None and entries is not None:
        files = entries
    if isinstance(files, dict):
        files = [files]
    if not isinstance(files, list) or not files:
        return {
            "error": (
                "peek_multiple requires a non-empty `files` list of "
                "{dataset_id, file_path} dicts. Use peek_multiple for 2+ files "
                "after list_files, or peek_file(dataset_id, file_path) for one "
                "file. Example: "
                'peek_multiple(files=[{"dataset_id": "census", "file_path": "files/rows.txt"}], max_rows=5)'
            )
        }

    results = []
    for spec in files:
        if isinstance(spec, str):
            results.append(peek_file(s3_uri=spec, max_rows=max_rows))
            continue
        if not isinstance(spec, dict):
            results.append({"error": "each entry must be a dict with dataset_id/file_path or s3_uri"})
            continue
        ds = spec.get("dataset_id", "")
        fp = spec.get("file_path") or spec.get("path") or ""
        uri = spec.get("s3_uri") or spec.get("uri") or ""
        results.append(peek_file(dataset_id=ds, file_path=fp, max_rows=max_rows, s3_uri=uri))

    return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Tool 2: read_file
# ---------------------------------------------------------------------------


@tool
def read_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    start_line: int = 0,
    max_lines: int = 10000,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Read lines from a file in the data lake without downloading it.
    Use this to read text, CSV, or JSON files directly. Supports pagination
    via start_line for large files.

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path. For datagov
    files, both "rows.txt" and "files/rows.txt" are accepted, but
    "files/rows.txt" is the canonical path.

    Args:
        dataset_id: Dataset identifier (e.g. "census")
        file_path:  Relative path within the dataset (e.g. "files/data.txt")
        start_line: Line index to start reading from, 0-based (default 0)
        max_lines:  Number of lines to return, capped at 10000 (default 10000)
        s3_uri:     Optional full object URI instead of dataset_id/file_path
    """
    if start_line < 0:
        return {"error": "start_line must be >= 0"}

    max_lines = min(max_lines, 10_000)
    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    s3 = _get_s3_client()
    key = ref["key"]

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"]
        lines = []
        current = 0
        for raw_line in body.iter_lines():
            if current < start_line:
                current += 1
                continue
            lines.append(raw_line.decode("utf-8", errors="replace"))
            current += 1
            if len(lines) >= max_lines:
                break
        body.close()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    result = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "start_line": start_line,
        "returned_lines": len(lines),
        "lines": lines,
    }
    result_json = json.dumps(result)
    if len(result_json) > _TOOL_RESULT_CHAR_CAP:
        # Write full content to sandbox, return truncated lines + pointer
        sandbox = _get_sandbox_dir()
        safe_name = file_path.lstrip("/").replace("/", "_")
        dump_path = sandbox / dataset_id / f"{safe_name}.read_result.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(result_json)
        budget = _TOOL_RESULT_CHAR_CAP - 600
        char_count, truncated_lines = 0, []
        for line in lines:
            if char_count + len(line) + 4 > budget:
                break
            truncated_lines.append(line)
            char_count += len(line) + 4
        return {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "start_line": start_line,
            "returned_lines": len(truncated_lines),
            "truncated": True,
            "local_result_path": str(dump_path),
            "truncation_note": (
                f"Output truncated to {len(truncated_lines)} of {len(lines)} fetched lines. "
                f"Full content written to: {dump_path} (use execute_code to read it). "
                f"Or use start_line={start_line + len(truncated_lines)} to page forward, "
                "or grep_file for specific values."
            ),
            "lines": truncated_lines,
        }
    return result


# ---------------------------------------------------------------------------
# Tool 3: grep_file
# ---------------------------------------------------------------------------

@tool
def grep_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    regex_pattern: str = "",
    context_lines: int = 2,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Search for a regex pattern inside a file without downloading it.
    Streams the file from S3 and returns matching lines with surrounding context.
    Use this to locate specific values, IDs, or keywords in large files.

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path. For datagov
    files, both "rows.txt" and "files/rows.txt" are accepted, but
    "files/rows.txt" is the canonical path.

    Args:
        dataset_id:    Dataset identifier (e.g. "public-school-locations-current-23297")
        file_path:     Relative path within the dataset (e.g. "files/data.txt")
        regex_pattern: Case-insensitive regex to search for (e.g. "King County")
        context_lines: Lines of context before/after each match (default 2, max 5)
        s3_uri:        Optional full object URI instead of dataset_id/file_path

    Returns:
        Dict with keys: match_count, matches (list of {line_number, line,
        context_before, context_after}), truncated. On error: {error: ...}
    """
    if not regex_pattern:
        return {"error": "regex_pattern is required"}

    context_lines = min(context_lines, 5)
    # filename = file_path.rsplit("/", 1)[-1]
    # if should_skip(filename):
    #     return {"error": f"File skipped (metadata/binary): {file_path}"}

    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Invalid regex pattern: {e}"}

    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    s3 = _get_s3_client()
    key = ref["key"]

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        line_iter = resp["Body"].iter_lines()
        
        matches = []
        pending_matches = []
        # deque automatically pushes old lines out when it hits maxlen!
        history = deque(maxlen=context_lines) 

        for i, raw_line in enumerate(line_iter):
            line = raw_line.decode("utf-8", errors="replace")

            # 1. Fill the "after" context for any matches we found previously
            for m in pending_matches:
                if len(m["context_after"]) < context_lines:
                    m["context_after"].append(line)

            # 2. Move fully-populated matches into our final list
            completed = [m for m in pending_matches if len(m["context_after"]) == context_lines]
            for m in completed:
                matches.append(m)
                pending_matches.remove(m)
                
            if len(matches) >= _SEARCH_MAX_MATCHES:
                break

            # 3. Evaluate the CURRENT line for a new match
            if pattern.search(line):
                new_match = {
                    "line_number": i,
                    "line": line,
                    "context_before": list(history),  # Snapshot the current history
                    "context_after": []
                }
                if context_lines == 0:
                    matches.append(new_match)
                    if len(matches) >= _SEARCH_MAX_MATCHES:
                        break
                else:
                    pending_matches.append(new_match)

            # 4. Add current line to history for future matches
            history.append(line)

        # Catch any pending matches if we hit the end of the file early
        for m in pending_matches:
            if m not in matches:
                matches.append(m)

        resp["Body"].close()

    except Exception as e:
        return {"error": f"Stream search failed: {e}", "traceback": traceback.format_exc()}

    result = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "match_count": len(matches),
        "truncated_matches": len(matches) >= _SEARCH_MAX_MATCHES,
        "matches": matches,
    }
    result_json = json.dumps(result)
    if len(result_json) > _TOOL_RESULT_CHAR_CAP:
        sandbox = _get_sandbox_dir()
        safe_name = file_path.lstrip("/").replace("/", "_")
        dump_path = sandbox / dataset_id / f"{safe_name}.grep_result.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(result_json)
        budget = _TOOL_RESULT_CHAR_CAP - 600
        char_count, capped_matches = 0, []
        for match in matches:
            match_json = json.dumps(match)
            if char_count + len(match_json) + 2 > budget:
                break
            capped_matches.append(match)
            char_count += len(match_json) + 2
        return {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "match_count": len(matches),
            "returned_matches": len(capped_matches),
            "truncated_matches": True,
            "local_result_path": str(dump_path),
            "truncation_note": (
                f"Match output exceeded the {_TOOL_RESULT_CHAR_CAP} char limit. "
                f"Showing {len(capped_matches)} of {len(matches)} matches. "
                f"Full result written to: {dump_path}. "
                "Tighten the regex or reduce context_lines if you need a smaller in-context result."
            ),
            "matches": capped_matches,
        }
    return result

# ---------------------------------------------------------------------------
# Tool 4: parse_xml_records
# ---------------------------------------------------------------------------

@tool
def parse_xml_records(
    dataset_id: str | None = None,
    file_path: str | None = None,
    record_tag: str | None = None,
    fields: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[List[str]] = None,
    limit: int = 50,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Parse XML/KML records directly from S3 without downloading or executing
    arbitrary Python.

    Use this for XML/KML sources after peek_file has shown record tags/schema
    fields. It supports KML SimpleData fields, plain XML leaf-tag fields,
    exact-match filters, and COUNT-style grouping.

    Do not use this for CSV/JSON; use query_file for those.

    Args:
        dataset_id: Dataset identifier (e.g. "public-school-locations-current-23297")
        file_path: Relative path within the dataset (e.g. "files/schools.kml")
        record_tag: XML tag for one logical record (e.g. "Placemark"). If omitted,
                    parse_xml_records uses the first record candidate from peek_file.
        fields: Optional field names to return for each matched record.
        filters: Optional exact-match filters, e.g. {"STFIP": "06"}.
        group_by: Optional field names to group by. When provided, rows contain
                  the group fields plus "count".
        limit: Maximum rows/groups to return, capped at 200.
        s3_uri: Optional full object URI instead of dataset_id/file_path.

    Returns:
        Dict with keys: rows, row_count, scanned_records, matched_records,
        truncated. On error: {error: ...}
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(
        _parse_xml_records_impl,
        dataset_id,
        file_path,
        record_tag,
        fields,
        filters,
        group_by,
        limit,
        s3_uri,
        timeout_seconds=timeout_seconds,
    )
    if completed:
        return result or {"error": "parse_xml_records failed without returning a result"}

    return {
        "error": (
            f"XML/KML parsing timed out after {timeout_seconds}s. "
            "Use peek_file to confirm the record_tag and filters, then retry "
            "with a narrower filter or lower limit."
        )
    }


def _parse_xml_records_impl(
    dataset_id: str | None = None,
    file_path: str | None = None,
    record_tag: str | None = None,
    fields: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[List[str]] = None,
    limit: int = 50,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    fields_list = _coerce_string_list(fields)
    group_fields = _coerce_string_list(group_by)
    filters = filters if isinstance(filters, dict) else {}
    try:
        limit = max(1, min(int(limit), _QUERY_ROW_CAP))
    except (TypeError, ValueError):
        limit = 50

    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    key = ref["key"]
    s3 = _get_s3_client()

    try:
        size = _s3_head(s3, key)
        text, _sampled_bytes = _s3_peek_text(s3, key, size)
        family = detect_family(text)
    except Exception as e:
        return {"error": f"Could not inspect XML/KML file: {e}"}

    if family != "xml":
        return {
            "error": (
                f"parse_xml_records only supports XML/KML files. Detected {family!r}. "
                "Use query_file for CSV/JSON or read_file/grep_file for text."
            )
        }

    preview = build_xml_preview(text, size, peek_bytes=_PEEK_BYTES)
    chosen_record_tag = normalize_xml_record_tag(record_tag)
    if not chosen_record_tag:
        candidates = preview.get("xml_record_tag_candidates") or []
        chosen_record_tag = normalize_xml_record_tag(candidates[0]) if candidates else None
    if not chosen_record_tag:
        return {
            "error": (
                "Could not infer an XML record_tag. Call peek_file first and pass one "
                "of xml_record_tag_candidates to parse_xml_records."
            ),
            **preview,
        }

    rows: List[Dict[str, Any]] = []
    groups: Counter = Counter()
    scanned_records = 0
    matched_records = 0
    body = None

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"]
        for _event, elem in ET.iterparse(body, events=("end",)):
            if local_xml_name(elem.tag) != chosen_record_tag:
                continue

            scanned_records += 1
            row = xml_record_to_row(elem)
            if filters and not _xml_row_matches_filters(row, filters):
                elem.clear()
                continue

            matched_records += 1
            if group_fields:
                group_key = tuple(row.get(field, "") for field in group_fields)
                groups[group_key] += 1
            elif len(rows) < limit:
                rows.append(_select_xml_row_fields(row, fields_list))

            elem.clear()
    except ET.ParseError as e:
        return {"error": f"XML/KML parse failed: {e}", "traceback": traceback.format_exc()}
    except Exception as e:
        return {"error": f"XML/KML stream parse failed: {e}", "traceback": traceback.format_exc()}
    finally:
        if body is not None and hasattr(body, "close"):
            body.close()

    if group_fields:
        rows = []
        for group_key, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))[:limit]:
            group_row = {field: group_key[i] for i, field in enumerate(group_fields)}
            group_row["count"] = count
            rows.append(group_row)
        truncated = len(groups) > limit
    else:
        truncated = matched_records > len(rows)

    return {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "family": family,
        "record_tag": chosen_record_tag,
        "fields": fields_list,
        "filters": filters,
        "group_by": group_fields,
        "scanned_records": scanned_records,
        "matched_records": matched_records,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Tool 5: query_file
# ---------------------------------------------------------------------------

@tool
def query_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    sql: str = "",
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Run a SQL query directly against an S3 file using DuckDB httpfs.
    No download required. The file is referenced as table alias 't'.

    Supported file types: CSV (.csv), JSON (.json, .jsonl, .ndjson).
    Do not use query_file for non-tabular/non-JSON sources such as
    Wikipedia/content.txt, prose/plain text, XML/KML, HTML, PDFs, or binary files.
    XML/KML is detected but not queryable here; query_file returns a hint
    to use parse_xml_records, peek_file, grep_file, or read_file instead.
    Results are capped at 200 rows.

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path. For datagov
    files, both "rows.txt" and "files/rows.txt" are accepted, but
    "files/rows.txt" is the canonical path.

    Args:
        dataset_id: Dataset identifier (e.g. "Barack_Obama")
        file_path:  Relative path within the dataset (e.g. "table_0.csv")
        sql:        SQL query. Use 't' as the table alias.
                    Example: "SELECT * FROM t LIMIT 10"
                    Example: "SELECT col1, COUNT(*) FROM t GROUP BY col1"
        s3_uri:     Optional full object URI instead of dataset_id/file_path

    Returns:
        Dict with keys: columns, rows, row_count, truncated. On error: {error: ...}
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(
        _query_file_impl,
        dataset_id,
        file_path,
        sql,
        s3_uri,
        timeout_seconds=timeout_seconds,
    )
    if completed:
        return result or {"error": "query_file failed without returning a result"}

    return {
        "error": (
            f"Query timed out after {timeout_seconds}s. "
            "Narrow the SQL (SELECT fewer columns, add WHERE/GROUP BY/LIMIT), "
            "or use download + execute_code for long-running tabular/JSON work."
        )
    }


def _query_file_impl(
    dataset_id: str | None = None,
    file_path: str | None = None,
    sql: str = "",
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    if not sql or not sql.strip():
        return {"error": "sql is required"}

    # Auto-fix MySQL-style backtick identifiers — DuckDB only accepts double
    # quotes. Preserves backticks inside string literals. Eliminates ~17
    # Parser Errors per eval at the source.
    sql = _normalize_sql_backticks(sql)

    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]

    # Determine reader function — peek first for all extensions
    try:
        s3 = _get_s3_client()
        key = ref["key"]
        size = _s3_head(s3, key)
        text, _sampled_bytes = _s3_peek_text(s3, key, size)
        family = detect_family(text)
    except Exception as e:
        return {"error": f"Could not detect file family: {e}"}

    if size > _QUERY_MAX_FILE_BYTES:
        if family not in {"csv", "json"}:
            return {"error": _rewrite_unqueryable_family_error(family)}
        return {
            "error": (
                f"File too large to query directly ({size // (1024*1024)} MB). "
                "Use download + execute_code only if the source is tabular or JSON."
            )
        }

    if family == "csv":
        reader = f"read_csv_auto('{s3_uri}', quote='\"')"
    elif family == "json":
        reader = f"read_json_auto('{s3_uri}', maximum_object_size={_QUERY_MAX_FILE_BYTES})"
    else:
        return {"error": _rewrite_unqueryable_family_error(family)}

    try:
        conn = _duckdb_connection()
        # Create view aliased as 't'
        conn.execute(f"CREATE VIEW t AS SELECT * FROM {reader}")
        # Execute user SQL, cap rows
        rel = conn.execute(sql)
        rows = rel.fetchmany(_QUERY_ROW_CAP + 1)
        columns = [desc[0] for desc in rel.description]
        truncated = len(rows) > _QUERY_ROW_CAP
        if truncated:
            rows = rows[:_QUERY_ROW_CAP]
        # Convert rows to JSON-safe primitives up front so DATE / TIMESTAMP /
        # DECIMAL / UUID / BLOB columns don't crash json.dumps below.
        safe_rows = [[_to_json_safe(v) for v in r] for r in rows]
        result = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "columns": columns,
            "rows": safe_rows,
            "row_count": len(safe_rows),
            "truncated": truncated,
        }
        result_json = json.dumps(result)
        if len(result_json) > _TOOL_RESULT_CHAR_CAP:
            # Write full result to sandbox so agent can read it via execute_code
            sandbox = _get_sandbox_dir()
            safe_name = file_path.lstrip("/").replace("/", "_")
            dump_path = sandbox / dataset_id / f"{safe_name}.query_result.json"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(result_json)
            # Pack as many rows as fit within the budget
            budget = _TOOL_RESULT_CHAR_CAP - 600  # headroom for metadata fields
            char_count, capped_rows = 0, []
            for row in safe_rows:
                row_json = json.dumps(row)
                if char_count + len(row_json) + 2 > budget:
                    break
                capped_rows.append(row)
                char_count += len(row_json) + 2
            avg_row_bytes = len(result_json) // max(len(safe_rows), 1)
            return {
                "truncated": True,
                "truncation_note": (
                    f"Result too large for context ({size // 1024}KB source file, "
                    f"~{int(size / max(1, avg_row_bytes))} rows estimated). "
                    f"Showing {len(capped_rows)} of {len(safe_rows)} rows within the {_TOOL_RESULT_CHAR_CAP} char limit. "
                    f"Full result written to: {dump_path} (use execute_code to read it). "
                    "Prefer: query_file with SELECT specific columns + WHERE/GROUP BY/LIMIT "
                    "for aggregates, or grep_file for searching specific values."
                ),
                "local_result_path": str(dump_path),
                "dataset_id": dataset_id,
                "file_path": file_path,
                "s3_uri": s3_uri,
                "size_bytes": size,
                "columns": columns,
                "rows": capped_rows,
                "row_count_shown": len(capped_rows),
                "row_count_estimate": int(size / max(1, avg_row_bytes)),
            }
        return result
    except Exception as e:
        return {
            "error": f"Query failed: {_rewrite_query_error(str(e))}",
            "traceback": traceback.format_exc(),
        }


# Export all public functions
__all__ = [
    'configure_benchmark',
    'search',
    'search_prefix',
    'search_keyword',
    'list_files',
    'download',
    'get_sandbox_info',
    'execute_code',
    'cleanup_sandbox',
    'set_sandbox_dir',
    'peek_file',
    'peek_multiple',
    'read_file',
    'grep_file',
    'parse_xml_records',
    'query_file',
]

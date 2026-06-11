"""
Data Lake Access Tools for LLM Agent

This module provides tools for LLM agents to access the data lake:

1. search(prefix) - Find datasets matching a prefix (searches both wikipedia and datagov)
2. download(dataset_id, file_path) - Download a file to local sandbox
3. execute_code(code) - Execute Python code against downloaded data
4. search_keyword(query) - Keyword search across wikipedia and data.gov with S3 validation

Bucket: lakeqa-yc4103-datalake
Folders: wikipedia/, datagov/
"""

import os
import json
import re
import sys
import tempfile
import shutil
import traceback
import multiprocessing as _mp
import queue as _queue
import difflib
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from strands import tool
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
from dotenv import load_dotenv

# Load AWS credentials from .env
load_dotenv()

# Configuration
DEFAULT_BUCKET = "lakeqa-yc4103-datalake"
BENCHMARK_BUCKETS = {
    "lakeqa": DEFAULT_BUCKET,
    "kramabench": "sana-kramabench",
}
BUCKET = os.getenv("LAKEQA_BUCKET", DEFAULT_BUCKET)
FOLDERS = ["wikipedia", "datagov"]
REGION = "us-east-1"

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


def _build_s3_client(unsigned: bool):
    """Build an S3 client in signed or unsigned mode."""
    s3_config = {"addressing_style": "path"}
    if unsigned:
        return boto3.client(
            "s3",
            region_name=REGION,
            config=Config(signature_version=UNSIGNED, s3=s3_config),
        )

    kwargs: Dict[str, Any] = {
        "region_name": REGION,
        "config": Config(s3=s3_config),
    }
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
    return boto3.client("s3", **kwargs)


def _get_signed_s3_client():
    global _S3_SIGNED_CLIENT
    if _S3_SIGNED_CLIENT is None:
        _S3_SIGNED_CLIENT = _build_s3_client(unsigned=False)
    return _S3_SIGNED_CLIENT


def _get_unsigned_s3_client():
    global _S3_UNSIGNED_CLIENT
    if _S3_UNSIGNED_CLIENT is None:
        _S3_UNSIGNED_CLIENT = _build_s3_client(unsigned=True)
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


def _guess_delimiter(line: str) -> str:
    candidates = [",", "\t", "|", ";"]
    best = ""
    best_count = 0
    for c in candidates:
        count = line.count(c)
        if count > best_count:
            best_count = count
            best = c
    return best if best_count > 0 else ""


def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return t.startswith("{") or t.startswith("[")

@tool
def inspect_file(dataset_id: str, file_path: str, max_lines: int = 5) -> Dict[str, Any]:
    """
    Inspect a file and return safe metadata (no raw content).
    """
    if not dataset_id:
        return {'error': "dataset_id is required"}
    if not file_path:
        return {'error': "file_path is required"}

    folder = _resolve_dataset_folder(dataset_id)
    if folder is None:
        return {'error': f"Dataset not found or ambiguous: {dataset_id}"}

    s3 = _get_s3_client()
    key = f"{folder}/{dataset_id}/{file_path.lstrip('/')}"

    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        size = head.get("ContentLength", 0)
    except Exception as e:
        return {'error': f"Failed to stat file: {str(e)}"}

    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key, Range="bytes=0-65535")
        raw = obj["Body"].read()
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return {'error': f"Failed to read file: {str(e)}"}

    lines = text.splitlines()
    preview = lines[: max_lines or 5]
    first_line = preview[0] if preview else ""

    meta = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "size_bytes": size,
        "line_count_sampled": len(preview),
        "looks_like_json": _looks_like_json(first_line),
    }

    if first_line:
        meta["delimiter_guess"] = _guess_delimiter(first_line)
        if meta["delimiter_guess"]:
            meta["header_columns"] = [c.strip() for c in first_line.split(meta["delimiter_guess"])]

    # If JSON, attempt to parse the first line for keys (safe)
    if meta["looks_like_json"]:
        try:
            first_obj = json.loads(first_line)
            if isinstance(first_obj, dict):
                meta["json_keys"] = sorted(first_obj.keys())
        except Exception:
            pass

    return meta


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


# Export all public functions
__all__ = [
    'configure_benchmark',
    'search',
    'search_prefix',
    'search_keyword',
    'list_files',
    'download',
    'inspect_file',
    'get_sandbox_info',
    'execute_code',
    'cleanup_sandbox',
    'set_sandbox_dir',
]

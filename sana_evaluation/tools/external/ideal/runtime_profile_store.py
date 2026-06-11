"""Shared loader/state for ideal-mode runtime profiles.

Default mappings:
- benchmarks/lakeqa/tasks-mini/tasks/... -> benchmarks/lakeqa/tasks-mini/runtime-profiles/...
- benchmarks/kramabench/tasks-mini/tasks/... -> benchmarks/kramabench/tasks-mini/runtime-profiles/...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_RUNTIME_PROFILES_ROOT = Path("benchmarks/lakeqa/tasks-mini/runtime-profiles")
_KRAMABENCH_RUNTIME_PROFILES_ROOT = Path("benchmarks/kramabench/tasks-mini/runtime-profiles")
_TASK_CONTEXT: Dict[str, Any] = {}


@dataclass(frozen=True)
class IdealComputationRecord:
    tool: str
    node_id: str
    dataset_id: str
    source: str
    intent: str
    payload: str
    answer: Any
    blocked: bool = False


@dataclass(frozen=True)
class IdealRuntimeProfile:
    task_id: str
    profile_path: Path
    dataset_sequence: List[str]
    source_sequence: List[str]
    reasoning_chain_text: str
    ideal_query: List[IdealComputationRecord]
    ideal_code: List[IdealComputationRecord]


def set_runtime_profiles_root(path: str | Path) -> None:
    global _RUNTIME_PROFILES_ROOT
    _RUNTIME_PROFILES_ROOT = Path(path)


def runtime_profiles_root() -> Path:
    return _RUNTIME_PROFILES_ROOT


def set_task_context(task_context: Optional[Dict[str, Any]]) -> None:
    global _TASK_CONTEXT
    _TASK_CONTEXT = dict(task_context or {})


def get_task_context() -> Dict[str, Any]:
    return dict(_TASK_CONTEXT)


def task_id_from_context(task_context: Optional[Dict[str, Any]]) -> str:
    return str((task_context or {}).get("task_id") or "").strip()


def _profile_location_from_task(task_id: str) -> tuple[Path, Path]:
    raw = str(task_id or "").strip()
    if not raw:
        raise ValueError("Missing task_id for ideal runtime profile lookup.")

    p = Path(raw)
    parts = p.parts

    def _benchmark_suffix(benchmark: str) -> Optional[tuple[Path, Path]]:
        marker = ("benchmarks", benchmark, "tasks-mini", "tasks")
        for idx in range(0, max(len(parts) - len(marker) + 1, 0)):
            if parts[idx : idx + len(marker)] == marker:
                suffix = parts[idx + len(marker) :]
                if not suffix:
                    raise ValueError(f"Could not derive relative profile path from task_id '{task_id}'.")
                root = (
                    _RUNTIME_PROFILES_ROOT
                    if benchmark == "lakeqa"
                    else _KRAMABENCH_RUNTIME_PROFILES_ROOT
                )
                return (root, Path(*suffix))
        return None

    lakeqa_location = _benchmark_suffix("lakeqa")
    if lakeqa_location is not None:
        return lakeqa_location

    kramabench_location = _benchmark_suffix("kramabench")
    if kramabench_location is not None:
        return kramabench_location

    if parts[:4] == ("benchmarks", "lakeqa", "tasks-mini", "tasks"):
        suffix = parts[4:]
        if not suffix:
            raise ValueError(f"Could not derive relative profile path from task_id '{task_id}'.")
        return (_RUNTIME_PROFILES_ROOT, Path(*suffix))
    if parts[:4] == ("benchmarks", "kramabench", "tasks-mini", "tasks"):
        suffix = parts[4:]
        if not suffix:
            raise ValueError(f"Could not derive relative profile path from task_id '{task_id}'.")
        return (_KRAMABENCH_RUNTIME_PROFILES_ROOT, Path(*suffix))
    if p.is_absolute():
        return (runtime_profiles_root(), Path(p.name))
    return (runtime_profiles_root(), p)


def runtime_profile_path_for_task(task_id: str) -> Path:
    root, relpath = _profile_location_from_task(task_id)
    return root / relpath


def _validate_dataset_sequence(raw: Any, *, profile_path: Path) -> List[str]:
    if not isinstance(raw, list):
        raise ValueError(f"Invalid runtime profile at '{profile_path}': dataset_sequence must be a list.")

    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': dataset_sequence[{i}] must be a string."
            )
        dataset_id = item.strip()
        if not dataset_id:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': dataset_sequence[{i}] cannot be empty."
            )
        if "/" in dataset_id or "://" in dataset_id:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': dataset_sequence[{i}] must be a bare dataset ID, got '{dataset_id}'."
            )
        out.append(dataset_id)
    return out


def _validate_source_sequence(raw: Any, *, profile_path: Path) -> List[str]:
    if not isinstance(raw, list):
        raise ValueError(f"Invalid runtime profile at '{profile_path}': source_sequence must be a list.")

    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': source_sequence[{i}] must be a string."
            )
        source = item.strip()
        if not source:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': source_sequence[{i}] cannot be empty."
            )
        if "/" not in source:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': source_sequence[{i}] must be a dataset-relative file path, got '{source}'."
            )
        if "://" in source:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': source_sequence[{i}] must be a dataset-relative file path, got '{source}'."
            )
        out.append(source)
    if not out:
        raise ValueError(f"Invalid runtime profile at '{profile_path}': source_sequence cannot be empty.")
    return out


def _validate_reasoning_text(raw: Any, *, profile_path: Path) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        if not text:
            raise ValueError(f"Invalid runtime profile at '{profile_path}': reasoning_chain_text cannot be empty.")
        return text

    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"Invalid runtime profile at '{profile_path}': reasoning_chain_text cannot be an empty list.")

        lines: List[str] = []
        for i, item in enumerate(raw):
            if not isinstance(item, str):
                raise ValueError(
                    f"Invalid runtime profile at '{profile_path}': reasoning_chain_text[{i}] must be a string."
                )
            text = item.strip()
            if not text:
                raise ValueError(
                    f"Invalid runtime profile at '{profile_path}': reasoning_chain_text[{i}] cannot be empty."
                )
            lines.append(text)
        return "\n".join(lines)

    raise ValueError(
        f"Invalid runtime profile at '{profile_path}': reasoning_chain_text must be a string or a list of strings."
    )


def _dataset_id_from_source(source: str) -> str:
    value = str(source or "").strip()
    if value.startswith("s3://"):
        remainder = value[len("s3://") :]
        _bucket, _sep, value = remainder.partition("/")
    value = value.lstrip("/")
    parts = value.split("/")
    if len(parts) >= 2 and parts[0] in {"datagov", "wikipedia"}:
        return parts[1]
    return parts[0] if parts else value


def _source_for_dataset(dataset_id: str, source_sequence: List[str]) -> str:
    for source in source_sequence:
        if _dataset_id_from_source(source) == dataset_id:
            return source
    return ""


def _is_blocked_ideal_query_answer(answer: Any) -> bool:
    answer_text = str(answer or "").strip().lower()
    if answer_text.startswith("cannot execute sql:"):
        return True
    compact = answer_text.replace("_", "").replace("'", "")
    return "queryfile" in compact and (
        "doesnt run on xml" in compact
        or "does not run on xml" in compact
        or "doesnt run on kml" in compact
        or "does not run on kml" in compact
    )


def _validate_computation_records(
    raw: Any,
    *,
    key: str,
    tool: str,
    payload_key: str,
    source_sequence: List[str],
    profile_path: Path,
) -> List[IdealComputationRecord]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"Invalid runtime profile at '{profile_path}': {key} must be a list when present.")

    out: List[IdealComputationRecord] = []
    for i, item in enumerate(raw):
        label = f"{key}[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"Invalid runtime profile at '{profile_path}': {label} must be an object.")

        node_id = str(item.get("node_id") or item.get("node") or "").strip()
        if not node_id:
            raise ValueError(f"Invalid runtime profile at '{profile_path}': {label}.node_id is required.")

        dataset_id = str(item.get("dataset_id") or "").strip()
        source = str(item.get("source") or item.get("s3_uri") or item.get("uri") or "").strip()
        if not dataset_id and source:
            dataset_id = _dataset_id_from_source(source)
        if dataset_id and not source:
            source = _source_for_dataset(dataset_id, source_sequence)
        if not dataset_id:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': {label} requires dataset_id or source."
            )
        if not source:
            raise ValueError(
                f"Invalid runtime profile at '{profile_path}': {label} could not map dataset_id '{dataset_id}' to a source."
            )

        intent = str(item.get("intent") or item.get("subquestion") or "").strip()
        if not intent:
            raise ValueError(f"Invalid runtime profile at '{profile_path}': {label}.intent is required.")

        if "answer" not in item:
            raise ValueError(f"Invalid runtime profile at '{profile_path}': {label}.answer is required.")

        payload = str(item.get(payload_key) or "").strip()
        blocked = False
        if not payload:
            blocked = key == "ideal_query" and _is_blocked_ideal_query_answer(item.get("answer"))
            if not blocked:
                raise ValueError(f"Invalid runtime profile at '{profile_path}': {label}.{payload_key} is required.")

        out.append(
            IdealComputationRecord(
                tool=tool,
                node_id=node_id,
                dataset_id=dataset_id,
                source=source,
                intent=intent,
                payload=payload,
                answer=item["answer"],
                blocked=blocked,
            )
        )

    return out


def load_runtime_profile_for_task(task_id: str) -> IdealRuntimeProfile:
    profile_path = runtime_profile_path_for_task(task_id)
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing ideal runtime profile file for task '{task_id}': expected '{profile_path}'."
        )

    try:
        payload = json.loads(profile_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in ideal runtime profile file '{profile_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runtime profile at '{profile_path}': root JSON value must be an object.")

    dataset_sequence = _validate_dataset_sequence(
        payload.get("dataset_sequence"),
        profile_path=profile_path,
    )
    source_sequence = _validate_source_sequence(
        payload.get("source_sequence"),
        profile_path=profile_path,
    )
    reasoning_chain_text = _validate_reasoning_text(
        payload.get("reasoning_chain_text"),
        profile_path=profile_path,
    )
    ideal_query = _validate_computation_records(
        payload.get("ideal_query"),
        key="ideal_query",
        tool="query",
        payload_key="sql",
        source_sequence=source_sequence,
        profile_path=profile_path,
    )
    ideal_code = _validate_computation_records(
        payload.get("ideal_code"),
        key="ideal_code",
        tool="code",
        payload_key="code",
        source_sequence=source_sequence,
        profile_path=profile_path,
    )

    return IdealRuntimeProfile(
        task_id=str(task_id),
        profile_path=profile_path,
        dataset_sequence=dataset_sequence,
        source_sequence=source_sequence,
        reasoning_chain_text=reasoning_chain_text,
        ideal_query=ideal_query,
        ideal_code=ideal_code,
    )


def load_runtime_profile_for_context(task_context: Optional[Dict[str, Any]] = None) -> IdealRuntimeProfile:
    ctx = task_context if task_context is not None else get_task_context()
    task_id = task_id_from_context(ctx)
    return load_runtime_profile_for_task(task_id)


__all__ = [
    "IdealComputationRecord",
    "IdealRuntimeProfile",
    "set_runtime_profiles_root",
    "runtime_profiles_root",
    "set_task_context",
    "get_task_context",
    "task_id_from_context",
    "runtime_profile_path_for_task",
    "load_runtime_profile_for_task",
    "load_runtime_profile_for_context",
]

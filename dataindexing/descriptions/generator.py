"""
Simple parquet/manifest -> prompt (and optional LLM description) pipeline.

Dataflow:
metadata (trimmed s3 uri) + content (first N words) -> prompt -> optional LLM output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from dataindexing.sources.parquet_cache import iter_parquet_records


DEFAULT_OUTPUT = "benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl"
DEFAULT_MODEL = "gpt-4o-mini"
PARQUET_ROW_GROUP_SIZE = 512

# Standard pricing per 1M tokens. Cached-input pricing is intentionally excluded.
MODEL_COST_PER_1M: dict[str, dict[str, float]] = {
    # GPT-5.4 family
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
    # GPT-5.2 / 5.1 / 5 families
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    # 4.1 + 4o
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

PROMPT_TEMPLATE = """Generate dataset metadata and a concise description for this dataset file.

S3 URI: {dataset_uri}
Metadata: {metadata}
Content sample (first {max_words} words): {dataset_sample}

Requirements:
1) Return JSON with exactly these keys: "generated_metadata", "description".
2) "generated_metadata" should be a cleaned, human-readable metadata phrase (around 5-20 words).
3) "description" must be <= 120 words, factual, and readable.
4) If sample quality is low/noisy, mention uncertainty briefly.
"""


class LLMInferenceGenerator:
    """Optional OpenAI inference wrapper."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.client = None

    def setup(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required for --infer. Install with: pip install openai"
            ) from exc
        self.client = AsyncOpenAI()

    def _extract_usage(self, resp: Any) -> tuple[int | None, int | None]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None, None

        # OpenAI SDK object shape
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)

        # Dict fallback
        if isinstance(usage, dict):
            in_tok = usage.get("input_tokens", in_tok)
            out_tok = usage.get("output_tokens", out_tok)

        return in_tok, out_tok

    def _cost_components_usd(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> tuple[float | None, float | None, float | None]:
        price = MODEL_COST_PER_1M.get(self.model)
        if price is None:
            return None, None, None

        input_cost = None
        output_cost = None
        if input_tokens is not None:
            input_cost = (input_tokens / 1_000_000.0) * price["input"]
        if output_tokens is not None:
            output_cost = (output_tokens / 1_000_000.0) * price["output"]

        if input_cost is None and output_cost is None:
            total_cost = None
        else:
            total_cost = float((input_cost or 0.0) + (output_cost or 0.0))
        return input_cost, output_cost, total_cost

    async def describe(self, prompt: str) -> dict[str, Any]:
        if self.client is None:
            self.setup()
        resp = await self.client.responses.create(model=self.model, input=prompt)
        input_tokens, output_tokens = self._extract_usage(resp)
        input_cost_usd, output_cost_usd, total_cost_usd = self._cost_components_usd(
            input_tokens, output_tokens
        )

        text = getattr(resp, "output_text", None)
        if text:
            return {
                "raw_output": text.strip(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": input_cost_usd,
                "output_cost_usd": output_cost_usd,
                "cost_usd": total_cost_usd,
            }

        output = getattr(resp, "output", None) or []
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for part in content:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        return {
            "raw_output": "\n".join(chunks).strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_usd": input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "cost_usd": total_cost_usd,
        }


def metadata_from_s3_uri(uri: str) -> str:
    """Trim bucket/extension and split path parts into words."""
    path = re.sub(r"^s3://[^/]+/", "", uri)
    parts = [p for p in path.strip("/").split("/") if p]
    words: list[str] = []
    for part in parts:
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", part)
        words.extend([w for w in re.split(r"[-_]", stem) if w])
    return " ".join(words)


def first_n_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def build_prompt(dataset_uri: str, metadata: str, dataset_sample: str, max_words: int) -> str:
    return PROMPT_TEMPLATE.format(
        dataset_uri=dataset_uri,
        metadata=metadata,
        dataset_sample=dataset_sample,
        max_words=max_words,
    )


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_generated_fields(raw_output: str, fallback_metadata: str) -> tuple[str, str]:
    """
    Parse model output into (generated_metadata, description).
    Falls back safely when output is not valid JSON.
    """
    raw = (raw_output or "").strip()
    if not raw:
        return fallback_metadata, ""

    # Remove common code fence wrappers
    fenced = raw
    fenced = re.sub(r"^```(?:json)?\s*", "", fenced, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)

    candidates = [raw, fenced]

    # Also try extracting the first JSON object span.
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(fenced[start : end + 1])

    parsed: dict[str, Any] | None = None
    for candidate in candidates:
        parsed = _try_parse_json(candidate)
        if parsed is not None:
            break

    if parsed is None:
        # If not JSON, keep metadata fallback and treat output as description text.
        return fallback_metadata, raw

    generated_metadata = (
        parsed.get("generated_metadata")
        or parsed.get("metadata")
        or parsed.get("topic")
        or fallback_metadata
    )
    description = parsed.get("description") or ""
    return str(generated_metadata).strip() or fallback_metadata, str(description).strip()


def compose_content(generated_metadata: str, description: str) -> str:
    merged = f"{generated_metadata or ''} {description or ''}".strip()
    return " ".join(merged.split())


def build_row(
    dataset_uri: str,
    original_metadata: str,
    generated_metadata: str,
    description: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    input_cost_usd: float | None = None,
    output_cost_usd: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    generated_clean = " ".join((generated_metadata or "").split())
    description_clean = " ".join((description or "").split())
    content = compose_content(generated_clean, description_clean)
    return {
        # Core compatibility columns (hybrid_search expects these)
        "dataset_uri": dataset_uri,
        "metadata": generated_clean,
        "content": content,
        # Audit / provenance
        "original_metadata": original_metadata,
        "generated_metadata": generated_clean,
        "description": description_clean,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "cost_usd": cost_usd,
        "error": error,
    }


PARQUET_FIELDS = [
    "dataset_uri",
    "metadata",
    "content",
    "original_metadata",
    "generated_metadata",
    "description",
    "input_tokens",
    "output_tokens",
    "input_cost_usd",
    "output_cost_usd",
    "cost_usd",
    "error",
]

PARQUET_SCHEMA = pa.schema([
    ("dataset_uri", pa.string()),
    ("metadata", pa.string()),
    ("content", pa.string()),
    ("original_metadata", pa.string()),
    ("generated_metadata", pa.string()),
    ("description", pa.string()),
    ("input_tokens", pa.int64()),
    ("output_tokens", pa.int64()),
    ("input_cost_usd", pa.float64()),
    ("output_cost_usd", pa.float64()),
    ("cost_usd", pa.float64()),
    ("error", pa.string()),
])


class OutputSink:
    def __init__(self, jsonl_path: str, parquet_path: str | None) -> None:
        self._json_path = Path(jsonl_path)
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_file = self._json_path.open("w", encoding="utf-8")

        self._parquet_writer: pq.ParquetWriter | None = None
        self._parquet_rows: list[dict[str, Any]] = []
        self._parquet_written = 0
        self._parquet_path = Path(parquet_path) if parquet_path else None
        if self._parquet_path is not None:
            self._parquet_path.parent.mkdir(parents=True, exist_ok=True)
            self._parquet_writer = pq.ParquetWriter(
                str(self._parquet_path),
                PARQUET_SCHEMA,
                compression="zstd",
            )

    @property
    def parquet_written(self) -> int:
        return self._parquet_written

    def _flush_parquet_rows(self) -> None:
        if self._parquet_writer is None or not self._parquet_rows:
            return
        table = pa.Table.from_pydict(
            {name: [row.get(name) for row in self._parquet_rows] for name in PARQUET_FIELDS},
            schema=PARQUET_SCHEMA,
        )
        self._parquet_writer.write_table(table)
        self._parquet_written += len(self._parquet_rows)
        self._parquet_rows = []

    def write(self, row: dict[str, Any]) -> None:
        self._json_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Parquet export is success-only and requires non-empty content.
        if self._parquet_writer is None:
            return
        if row.get("error") not in (None, ""):
            return
        if not str(row.get("content", "")).strip():
            return
        self._parquet_rows.append(row)
        if len(self._parquet_rows) >= PARQUET_ROW_GROUP_SIZE:
            self._flush_parquet_rows()

    def close(self) -> None:
        self._json_file.close()
        self._flush_parquet_rows()
        if self._parquet_writer is not None:
            self._parquet_writer.close()


def get_parquet_row_count(path: str) -> int | None:
    """Best-effort parquet row count for tqdm totals."""
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def resolve_input_format(path: str, input_format: str) -> str:
    if input_format != "auto":
        return input_format

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".jsonl":
        try:
            first_record = next(iter_jsonl_records(path))
        except StopIteration:
            return "manifest-jsonl"
        if "local_table_path" in first_record and "columns" in first_record:
            return "kramabench-manifest-jsonl"
        return "manifest-jsonl"
    raise ValueError(
        f"Could not infer input format from path: {path}. "
        "Use --input-format parquet or --input-format manifest-jsonl."
    )


def count_jsonl_records(path: str) -> int:
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def load_retry_error_uris(path: str | None) -> set[str] | None:
    if not path:
        return None

    retry_path = Path(path)
    if not retry_path.exists():
        raise FileNotFoundError(retry_path)

    uris: set[str] = set()
    with retry_path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            if raw.startswith("{"):
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Retry file line {line_no} is not valid JSON") from exc
                if not isinstance(record, dict):
                    continue
                uri = record.get("dataset_uri") or record.get("s3_uri")
                if isinstance(uri, str) and uri and record.get("error") not in (None, ""):
                    uris.add(uri)
                continue

            uri = raw.split("\t", 1)[0].strip()
            if uri.startswith("s3://"):
                uris.add(uri)

    return uris


def _extract_manifest_uri(
    record: dict[str, Any],
    *,
    uri_field: str,
    line_no: int,
) -> str:
    uri = record.get(uri_field)
    if not uri and uri_field == "s3_uri":
        uri = record.get("dataset_uri")
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError(
            f"Manifest line {line_no} is missing a non-empty URI field "
            f"('{uri_field}' or fallback 'dataset_uri')."
        )
    return uri.strip()


def iter_manifest_uris(path: str, *, uri_field: str) -> Any:
    for line_no, record in iter_jsonl_records(path, with_line_numbers=True):
        yield _extract_manifest_uri(record, uri_field=uri_field, line_no=line_no)


def iter_jsonl_records(path: str, *, with_line_numbers: bool = False) -> Any:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Manifest line {line_no} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_no} must be a JSON object")
            if with_line_numbers:
                yield line_no, record
            else:
                yield record


def _kramabench_manifest_metadata(record: dict[str, Any]) -> str:
    table_name = str(record.get("table_name") or "")
    domain = str(record.get("domain") or "")
    dataset_id = str(record.get("dataset_id") or "")
    source_uri = str(record.get("source_uri") or "")
    columns = record.get("columns") or []
    if not isinstance(columns, list):
        columns = []
    column_text = " ".join(str(c) for c in columns[:24])
    parts = [
        "Kramabench",
        domain,
        table_name,
        dataset_id,
        metadata_from_s3_uri(source_uri) if source_uri else "",
        column_text,
    ]
    return " ".join(" ".join(part.replace("_", " ").replace("-", " ").split()) for part in parts if part)


def _kramabench_manifest_description(record: dict[str, Any]) -> str:
    domain = str(record.get("domain") or "unknown")
    table_name = str(record.get("table_name") or "table")
    dataset_id = str(record.get("dataset_id") or "unknown candidate")
    source_uri = str(record.get("source_uri") or "")
    row_count = record.get("row_count")
    columns = record.get("columns") or []
    if not isinstance(columns, list):
        columns = []

    column_phrase = ", ".join(str(c) for c in columns[:18])
    if len(columns) > 18:
        column_phrase += f", and {len(columns) - 18} more"
    rows_phrase = f"{row_count} rows" if isinstance(row_count, int) else "an unknown number of rows"

    source_phrase = ""
    if source_uri:
        source_path = re.sub(r"^s3://[^/]+/", "", source_uri)
        source_phrase = f" extracted from {source_path}"

    return (
        f"Kramabench {domain} table {table_name} for {dataset_id}{source_phrase}. "
        f"It contains {rows_phrase}. Columns include: {column_phrase}."
    ).strip()


def _resolve_manifest_local_table_path(record: dict[str, Any], manifest_path: str) -> Path:
    raw = record.get("local_table_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Kramabench manifest record is missing local_table_path")

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / path,
        repo_root / path,
        Path(manifest_path).resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _sample_local_table(path: Path, max_words: int) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    chunks: list[str] = []
    word_count = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for line in fh:
            clean_line = line.rstrip("\n")
            chunks.append(clean_line)
            word_count += len(clean_line.split())
            if word_count >= max_words:
                break
    return first_n_words("\n".join(chunks), max_words=max_words)


def _candidate_kramabench_parquet_paths(manifest_path: str) -> list[Path]:
    manifest = Path(manifest_path).expanduser()
    manifest_parent = manifest.resolve().parent
    repo_root = Path(__file__).resolve().parent.parent

    candidates = [
        manifest_parent / "kramabench_tables.parquet",
        repo_root / "outputs" / "kramabench_tables.parquet",
    ]

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def _load_kramabench_parquet_samples(manifest_path: str, max_words: int) -> dict[str, str]:
    parquet_path = next(
        (candidate for candidate in _candidate_kramabench_parquet_paths(manifest_path) if candidate.exists()),
        None,
    )
    if parquet_path is None:
        return {}

    samples: dict[str, str] = {}
    pf = pq.ParquetFile(parquet_path)
    available = set(pf.schema_arrow.names)
    if not {"dataset_uri", "content"}.issubset(available):
        return samples

    for row_group_index in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(row_group_index, columns=["dataset_uri", "content"])
        for dataset_uri, content in zip(
            batch.column("dataset_uri").to_pylist(),
            batch.column("content").to_pylist(),
        ):
            if dataset_uri and content:
                samples[str(dataset_uri)] = first_n_words(str(content), max_words=max_words)
    return samples


def _sample_kramabench_manifest_record(
    record: dict[str, Any],
    manifest_path: str,
    max_words: int,
    parquet_samples: dict[str, str],
) -> str:
    try:
        local_table_path = _resolve_manifest_local_table_path(record, manifest_path)
        return _sample_local_table(local_table_path, max_words=max_words)
    except (FileNotFoundError, ValueError) as exc:
        dataset_uri = record.get("dataset_uri")
        if isinstance(dataset_uri, str):
            sample = parquet_samples.get(dataset_uri)
            if sample:
                return sample
        raise exc


def _load_manifest_s3_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import aioboto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Manifest input with --infer requires aioboto3. "
            "Install with: pip install aioboto3"
        ) from exc

    from dataindexing.sources.s3 import S3Config, parse_s3_uri, s3_fetch, should_skip

    return aioboto3, S3Config, parse_s3_uri, s3_fetch, should_skip


def run(
    path: str,
    output: str,
    parquet_output: str | None,
    max_words: int,
    n: int | None,
    infer: bool,
    model: str,
    concurrency: int,
    input_format: str,
    uri_field: str,
    skip_unimportant: bool,
    retry_errors_from: str | None = None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    resolved_input_format = resolve_input_format(path, input_format)
    retry_error_uris = load_retry_error_uris(retry_errors_from)

    if resolved_input_format == "manifest-jsonl":
        if infer:
            return asyncio.run(
                run_manifest_infer_parallel(
                    path=path,
                    output=output,
                    parquet_output=parquet_output,
                    max_words=max_words,
                    n=n,
                    model=model,
                    concurrency=concurrency,
                    uri_field=uri_field,
                    skip_unimportant=skip_unimportant,
                    retry_error_uris=retry_error_uris,
                )
            )
        return run_manifest_no_infer(
            path=path,
            output=output,
            parquet_output=parquet_output,
            n=n,
            uri_field=uri_field,
            retry_error_uris=retry_error_uris,
        )

    if resolved_input_format == "kramabench-manifest-jsonl":
        if infer:
            return asyncio.run(
                run_kramabench_manifest_infer_parallel(
                    path=path,
                    output=output,
                    parquet_output=parquet_output,
                    max_words=max_words,
                    n=n,
                    model=model,
                    concurrency=concurrency,
                    retry_error_uris=retry_error_uris,
                )
            )
        return run_kramabench_manifest_no_infer(
            path=path,
            output=output,
            parquet_output=parquet_output,
            n=n,
            retry_error_uris=retry_error_uris,
        )

    if infer:
        return asyncio.run(
            run_infer_parallel(
                path=path,
                output=output,
                parquet_output=parquet_output,
                max_words=max_words,
                n=n,
                model=model,
                concurrency=concurrency,
                retry_error_uris=retry_error_uris,
            )
        )

    return run_no_infer(
        path=path,
        output=output,
        parquet_output=parquet_output,
        max_words=max_words,
        n=n,
        retry_error_uris=retry_error_uris,
    )


def run_manifest_no_infer(
    path: str,
    output: str,
    parquet_output: str | None,
    n: int | None,
    uri_field: str,
    retry_error_uris: set[str] | None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    written = 0
    sink = OutputSink(jsonl_path=output, parquet_path=parquet_output)
    try:
        for dataset_uri in iter_manifest_uris(path, uri_field=uri_field):
            if retry_error_uris is not None and dataset_uri not in retry_error_uris:
                continue
            if n is not None and written >= n:
                break

            metadata = metadata_from_s3_uri(dataset_uri)
            row = build_row(
                dataset_uri=dataset_uri,
                original_metadata=metadata,
                generated_metadata=metadata,
                description="",
                error=None,
            )
            sink.write(row)
            written += 1
    finally:
        sink.close()

    success_rows = written
    error_rows = 0
    return written, 0, 0, 0.0, 0.0, 0.0, sink.parquet_written, success_rows, error_rows


def run_kramabench_manifest_no_infer(
    path: str,
    output: str,
    parquet_output: str | None,
    n: int | None,
    retry_error_uris: set[str] | None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    written = 0
    sink = OutputSink(jsonl_path=output, parquet_path=parquet_output)
    try:
        for line_no, record in iter_jsonl_records(path, with_line_numbers=True):
            dataset_uri = _extract_manifest_uri(
                record,
                uri_field="dataset_uri",
                line_no=line_no,
            )
            if retry_error_uris is not None and dataset_uri not in retry_error_uris:
                continue
            if n is not None and written >= n:
                break

            metadata = _kramabench_manifest_metadata(record)
            description = _kramabench_manifest_description(record)
            row = build_row(
                dataset_uri=dataset_uri,
                original_metadata=metadata,
                generated_metadata=metadata,
                description=description,
                error=None,
            )
            sink.write(row)
            written += 1
    finally:
        sink.close()

    success_rows = written
    error_rows = 0
    return written, 0, 0, 0.0, 0.0, 0.0, sink.parquet_written, success_rows, error_rows


def run_no_infer(
    path: str,
    output: str,
    parquet_output: str | None,
    max_words: int,
    n: int | None,
    retry_error_uris: set[str] | None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    written = 0
    sink = OutputSink(jsonl_path=output, parquet_path=parquet_output)
    try:
        for dataset_uri, _metadata_unused, content in iter_parquet_records(path):
            if retry_error_uris is not None and dataset_uri not in retry_error_uris:
                continue
            if n is not None and written >= n:
                break

            metadata = metadata_from_s3_uri(dataset_uri)
            row = build_row(
                dataset_uri=dataset_uri,
                original_metadata=metadata,
                generated_metadata=metadata,
                description="",
                error=None,
            )
            sink.write(row)
            written += 1
    finally:
        sink.close()

    success_rows = written
    error_rows = 0
    return written, 0, 0, 0.0, 0.0, 0.0, sink.parquet_written, success_rows, error_rows


async def run_kramabench_manifest_infer_parallel(
    path: str,
    output: str,
    parquet_output: str | None,
    max_words: int,
    n: int | None,
    model: str,
    concurrency: int,
    retry_error_uris: set[str] | None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    if concurrency <= 0:
        raise ValueError("concurrency must be >= 1")

    inferencer = LLMInferenceGenerator(model=model)
    inferencer.setup()

    out_path = Path(output)
    error_path = out_path.with_suffix(out_path.suffix + ".errors.txt")

    written = 0
    input_token_sum = 0
    output_token_sum = 0
    input_cost_sum = 0.0
    output_cost_sum = 0.0
    cost_sum = 0.0
    success_rows = 0
    error_rows = 0

    if retry_error_uris is None:
        row_count = count_jsonl_records(path)
    else:
        row_count = 0
        for line_no, record in iter_jsonl_records(path, with_line_numbers=True):
            dataset_uri = _extract_manifest_uri(
                record,
                uri_field="dataset_uri",
                line_no=line_no,
            )
            if dataset_uri in retry_error_uris:
                row_count += 1
    target_rows = row_count if n is None else min(n, row_count)
    parquet_samples = _load_kramabench_parquet_samples(path, max_words=max_words)

    async def describe_record(record: dict[str, Any], line_no: int) -> tuple[str, str, str, Any]:
        dataset_uri = _extract_manifest_uri(
            record,
            uri_field="dataset_uri",
            line_no=line_no,
        )
        metadata = _kramabench_manifest_metadata(record)
        fallback_description = _kramabench_manifest_description(record)
        try:
            dataset_sample = _sample_kramabench_manifest_record(
                record,
                manifest_path=path,
                max_words=max_words,
                parquet_samples=parquet_samples,
            )
            prompt = build_prompt(
                dataset_uri,
                metadata,
                dataset_sample,
                max_words=max_words,
            )
            result = await inferencer.describe(prompt)
            return dataset_uri, metadata, fallback_description, result
        except Exception as exc:
            return dataset_uri, metadata, fallback_description, exc

    records_iter = iter(iter_jsonl_records(path, with_line_numbers=True))
    consumed = 0

    sink = OutputSink(jsonl_path=output, parquet_path=parquet_output)
    try:
        with (
            error_path.open("w", encoding="utf-8") as ef,
            tqdm(total=target_rows, desc="Processing Kramabench Tables") as pbar,
        ):
            while True:
                current_batch: list[tuple[int, dict[str, Any]]] = []

                while len(current_batch) < concurrency and consumed < target_rows:
                    try:
                        line_no, record = next(records_iter)
                    except StopIteration:
                        break
                    dataset_uri = _extract_manifest_uri(
                        record,
                        uri_field="dataset_uri",
                        line_no=line_no,
                    )
                    if retry_error_uris is not None and dataset_uri not in retry_error_uris:
                        continue
                    current_batch.append((line_no, record))
                    consumed += 1

                if not current_batch:
                    break

                results = await asyncio.gather(
                    *[describe_record(record, line_no) for line_no, record in current_batch]
                )

                for uri, meta, fallback_description, result in results:
                    if isinstance(result, Exception):
                        row = build_row(
                            dataset_uri=uri,
                            original_metadata=meta,
                            generated_metadata=meta,
                            description=fallback_description,
                            error=str(result),
                        )
                        sink.write(row)
                        ef.write(f"{uri}\t{type(result).__name__}: {result}\n")
                        written += 1
                        error_rows += 1
                        continue

                    generated_metadata, description = parse_generated_fields(
                        raw_output=str(result.get("raw_output", "")),
                        fallback_metadata=meta,
                    )
                    input_tokens = result.get("input_tokens")
                    output_tokens = result.get("output_tokens")
                    input_cost_usd = result.get("input_cost_usd")
                    output_cost_usd = result.get("output_cost_usd")
                    cost_usd = result.get("cost_usd")
                    row = build_row(
                        dataset_uri=uri,
                        original_metadata=meta,
                        generated_metadata=generated_metadata,
                        description=description or fallback_description,
                        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                        input_cost_usd=float(input_cost_usd)
                        if isinstance(input_cost_usd, (int, float))
                        else None,
                        output_cost_usd=float(output_cost_usd)
                        if isinstance(output_cost_usd, (int, float))
                        else None,
                        cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
                        error=None,
                    )
                    sink.write(row)
                    written += 1
                    success_rows += 1

                    if isinstance(input_tokens, int):
                        input_token_sum += input_tokens
                    if isinstance(output_tokens, int):
                        output_token_sum += output_tokens
                    if isinstance(input_cost_usd, (int, float)):
                        input_cost_sum += float(input_cost_usd)
                    if isinstance(output_cost_usd, (int, float)):
                        output_cost_sum += float(output_cost_usd)
                    if isinstance(cost_usd, (int, float)):
                        cost_sum += float(cost_usd)
                pbar.update(len(current_batch))
    finally:
        sink.close()

    return (
        written,
        input_token_sum,
        output_token_sum,
        input_cost_sum,
        output_cost_sum,
        cost_sum,
        sink.parquet_written,
        success_rows,
        error_rows,
    )


async def run_manifest_infer_parallel(
    path: str,
    output: str,
    parquet_output: str | None,
    max_words: int,
    n: int | None,
    model: str,
    concurrency: int,
    uri_field: str,
    skip_unimportant: bool,
    retry_error_uris: set[str] | None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    if concurrency <= 0:
        raise ValueError("concurrency must be >= 1")

    aioboto3, S3Config, parse_s3_uri, s3_fetch, should_skip = _load_manifest_s3_dependencies()

    inferencer = LLMInferenceGenerator(model=model)
    inferencer.setup()

    s3_cfg = S3Config()
    s3_cfg.skip_unimportant_files = skip_unimportant

    out_path = Path(output)
    error_path = out_path.with_suffix(out_path.suffix + ".errors.txt")

    written = 0
    input_token_sum = 0
    output_token_sum = 0
    input_cost_sum = 0.0
    output_cost_sum = 0.0
    cost_sum = 0.0
    success_rows = 0
    error_rows = 0

    if retry_error_uris is None:
        row_count = count_jsonl_records(path)
    else:
        row_count = sum(1 for uri in iter_manifest_uris(path, uri_field=uri_field) if uri in retry_error_uris)
    target_rows = row_count if n is None else min(n, row_count)

    async def fetch_and_describe(s3, uri: str) -> tuple[str, str, Any]:
        metadata = metadata_from_s3_uri(uri)
        try:
            bucket, key = parse_s3_uri(uri)
            if s3_cfg.skip_unimportant_files and should_skip(key):
                raise ValueError("Skipped by --skip-unimportant")
            raw = await s3_fetch(s3, s3_cfg, bucket, key)
            dataset_sample = first_n_words(raw, max_words=max_words)
            prompt = build_prompt(uri, metadata, dataset_sample, max_words=max_words)
            result = await inferencer.describe(prompt)
            return uri, metadata, result
        except Exception as exc:
            return uri, metadata, exc

    uris_iter = iter(iter_manifest_uris(path, uri_field=uri_field))
    consumed = 0

    sink = OutputSink(jsonl_path=output, parquet_path=parquet_output)
    session = aioboto3.Session()
    try:
        async with session.client("s3", region_name=s3_cfg.region) as s3:
            with (
                error_path.open("w", encoding="utf-8") as ef,
                tqdm(total=target_rows, desc="Processing Rows") as pbar,
            ):
                while True:
                    current_batch: list[str] = []

                    while len(current_batch) < concurrency and consumed < target_rows:
                        try:
                            uri = next(uris_iter)
                        except StopIteration:
                            break
                        if retry_error_uris is not None and uri not in retry_error_uris:
                            continue
                        current_batch.append(uri)
                        consumed += 1

                    if not current_batch:
                        break

                    results = await asyncio.gather(
                        *[fetch_and_describe(s3, uri) for uri in current_batch]
                    )

                    for uri, meta, result in results:
                        if isinstance(result, Exception):
                            row = build_row(
                                dataset_uri=uri,
                                original_metadata=meta,
                                generated_metadata=meta,
                                description="",
                                error=str(result),
                            )
                            sink.write(row)
                            ef.write(f"{uri}\t{type(result).__name__}: {result}\n")
                            written += 1
                            error_rows += 1
                            continue

                        generated_metadata, description = parse_generated_fields(
                            raw_output=str(result.get("raw_output", "")),
                            fallback_metadata=meta,
                        )
                        input_tokens = result.get("input_tokens")
                        output_tokens = result.get("output_tokens")
                        input_cost_usd = result.get("input_cost_usd")
                        output_cost_usd = result.get("output_cost_usd")
                        cost_usd = result.get("cost_usd")
                        row = build_row(
                            dataset_uri=uri,
                            original_metadata=meta,
                            generated_metadata=generated_metadata,
                            description=description,
                            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                            input_cost_usd=float(input_cost_usd)
                            if isinstance(input_cost_usd, (int, float))
                            else None,
                            output_cost_usd=float(output_cost_usd)
                            if isinstance(output_cost_usd, (int, float))
                            else None,
                            cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
                            error=None,
                        )
                        sink.write(row)
                        written += 1
                        success_rows += 1

                        if isinstance(input_tokens, int):
                            input_token_sum += input_tokens
                        if isinstance(output_tokens, int):
                            output_token_sum += output_tokens
                        if isinstance(input_cost_usd, (int, float)):
                            input_cost_sum += float(input_cost_usd)
                        if isinstance(output_cost_usd, (int, float)):
                            output_cost_sum += float(output_cost_usd)
                        if isinstance(cost_usd, (int, float)):
                            cost_sum += float(cost_usd)
                    pbar.update(len(current_batch))
    finally:
        sink.close()

    return (
        written,
        input_token_sum,
        output_token_sum,
        input_cost_sum,
        output_cost_sum,
        cost_sum,
        sink.parquet_written,
        success_rows,
        error_rows,
    )


async def run_infer_parallel(
    path: str,
    output: str,
    parquet_output: str | None,
    max_words: int,
    n: int | None,
    model: str,
    concurrency: int,
    retry_error_uris: set[str] | None,
) -> tuple[int, int, int, float, float, float, int, int, int]:
    if concurrency <= 0:
        raise ValueError("concurrency must be >= 1")

    inferencer = LLMInferenceGenerator(model=model)
    inferencer.setup()

    out_path = Path(output)
    error_path = out_path.with_suffix(out_path.suffix + ".errors.txt")

    written = 0
    input_token_sum = 0
    output_token_sum = 0
    input_cost_sum = 0.0
    output_cost_sum = 0.0
    cost_sum = 0.0
    success_rows = 0
    error_rows = 0

    if retry_error_uris is None:
        row_count = get_parquet_row_count(path)
    else:
        row_count = sum(
            1 for dataset_uri, _metadata_unused, _content in iter_parquet_records(path)
            if dataset_uri in retry_error_uris
        )
    if n is None:
        target_rows = row_count
    elif row_count is None:
        target_rows = n
    else:
        target_rows = min(n, row_count)

    async def process_batch(batch: list[tuple[str, str, str]]) -> list[Any]:
        tasks = [inferencer.describe(prompt) for _, _, prompt in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    records_iter = iter(iter_parquet_records(path))
    consumed = 0

    sink = OutputSink(jsonl_path=output, parquet_path=parquet_output)
    try:
        with (
            error_path.open("w", encoding="utf-8") as ef,
            tqdm(total=target_rows, desc="Processing Batches") as pbar,
        ):
            while True:
                current_batch: list[tuple[str, str, str]] = []

                while len(current_batch) < concurrency and (
                    target_rows is None or consumed < target_rows
                ):
                    try:
                        dataset_uri, _metadata_unused, content = next(records_iter)
                    except StopIteration:
                        break
                    if retry_error_uris is not None and dataset_uri not in retry_error_uris:
                        continue

                    metadata = metadata_from_s3_uri(dataset_uri)
                    dataset_sample = first_n_words(content, max_words=max_words)
                    prompt = build_prompt(dataset_uri, metadata, dataset_sample, max_words=max_words)
                    current_batch.append((dataset_uri, metadata, prompt))
                    consumed += 1

                if not current_batch:
                    break

                results = await process_batch(current_batch)
                for (uri, meta, _prompt), result in zip(current_batch, results, strict=True):
                    if isinstance(result, Exception):
                        row = build_row(
                            dataset_uri=uri,
                            original_metadata=meta,
                            generated_metadata=meta,
                            description="",
                            error=str(result),
                        )
                        sink.write(row)
                        ef.write(f"{uri}\t{type(result).__name__}: {result}\n")
                        written += 1
                        error_rows += 1
                        continue

                    generated_metadata, description = parse_generated_fields(
                        raw_output=str(result.get("raw_output", "")),
                        fallback_metadata=meta,
                    )
                    input_tokens = result.get("input_tokens")
                    output_tokens = result.get("output_tokens")
                    input_cost_usd = result.get("input_cost_usd")
                    output_cost_usd = result.get("output_cost_usd")
                    cost_usd = result.get("cost_usd")
                    row = build_row(
                        dataset_uri=uri,
                        original_metadata=meta,
                        generated_metadata=generated_metadata,
                        description=description,
                        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
                        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
                        input_cost_usd=float(input_cost_usd)
                        if isinstance(input_cost_usd, (int, float))
                        else None,
                        output_cost_usd=float(output_cost_usd)
                        if isinstance(output_cost_usd, (int, float))
                        else None,
                        cost_usd=float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
                        error=None,
                    )
                    sink.write(row)
                    written += 1
                    success_rows += 1

                    if isinstance(input_tokens, int):
                        input_token_sum += input_tokens
                    if isinstance(output_tokens, int):
                        output_token_sum += output_tokens
                    if isinstance(input_cost_usd, (int, float)):
                        input_cost_sum += float(input_cost_usd)
                    if isinstance(output_cost_usd, (int, float)):
                        output_cost_sum += float(output_cost_usd)
                    if isinstance(cost_usd, (int, float)):
                        cost_sum += float(cost_usd)
                pbar.update(len(current_batch))
    finally:
        sink.close()
    return (
        written,
        input_token_sum,
        output_token_sum,
        input_cost_sum,
        output_cost_sum,
        cost_sum,
        sink.parquet_written,
        success_rows,
        error_rows,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build prompts/descriptions from parquet rows or JSONL S3 manifests."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="datalake_silver.parquet",
        help="Input parquet or JSONL manifest path",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "parquet", "manifest-jsonl", "kramabench-manifest-jsonl"],
        default="auto",
        help="Input type (default: auto, based on file extension)",
    )
    parser.add_argument(
        "--uri-field",
        type=str,
        default="s3_uri",
        help="JSONL field containing the S3 URI for manifest input (default: s3_uri)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--parquet-output",
        type=str,
        default=None,
        help="Optional parquet export path for hybrid_search-compatible rows",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="Number of content words to include in prompt sample",
    )
    parser.add_argument(
        "--n",
        "--limit",
        dest="n",
        type=int,
        default=None,
        help="How many parquet rows/datasets to process (for testing)",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Call OpenAI model and include generated description",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model for --infer (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Parallel LLM request count when --infer is enabled",
    )
    parser.add_argument(
        "--skip-unimportant",
        action="store_true",
        help="Skip metadata-like/binary files when reading directly from a JSONL manifest",
    )
    parser.add_argument(
        "--retry-errors-from",
        type=str,
        default=None,
        help=(
            "Only process URIs that failed in a previous output JSONL or .errors.txt file. "
            "Use this to avoid spending tokens on rows that already succeeded."
        ),
    )
    args = parser.parse_args()

    (
        written,
        input_tokens,
        output_tokens,
        input_cost_usd,
        output_cost_usd,
        cost_usd,
        parquet_rows,
        success_rows,
        error_rows,
    ) = run(
        path=args.path,
        output=args.output,
        parquet_output=args.parquet_output,
        max_words=args.max_words,
        n=args.n,
        infer=args.infer,
        model=args.model,
        concurrency=args.concurrency,
        input_format=args.input_format,
        uri_field=args.uri_field,
        skip_unimportant=args.skip_unimportant,
        retry_errors_from=args.retry_errors_from,
    )
    print(f"Rows written: {written}")
    print(f"Success rows: {success_rows}")
    print(f"Error rows  : {error_rows}")
    if args.infer:
        print(f"Input tokens total : {input_tokens}")
        print(f"Output tokens total: {output_tokens}")
        print(f"Input cost total   : {input_cost_usd:.6f}")
        print(f"Output cost total  : {output_cost_usd:.6f}")
        print(f"Cost total (USD)   : {cost_usd:.6f}")
    print(f"Output     : {args.output}")
    if args.output == DEFAULT_OUTPUT:
        print("(Using default JSONL output path)")
    if args.parquet_output:
        print(f"Parquet     : {args.parquet_output}")
        print(f"Parquet rows: {parquet_rows}")


if __name__ == "__main__":
    main()

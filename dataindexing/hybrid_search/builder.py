import argparse
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pyarrow as pa
import lancedb
import torch

from dataindexing.hybrid_search.config import EMBED_PRESETS, HybridConfig, build_hybrid_config
from dataindexing.hybrid_search.embeddings import build_embed_model, encode_documents
from dataindexing.hybrid_search.text_builders import (
    build_description_chunk,
    build_schema_text,
)
from dataindexing.hybrid_search.uri_matching import uri_to_schema_stem

import asyncio
import re
from pathlib import Path

import aioboto3
from tqdm.asyncio import tqdm

from dataindexing.schemas.table_schemas import load_table_schemas
from dataindexing.sources.parquet_cache import (
    iter_parquet_records,
    load_descriptions_full,
    load_parquet_records,
)
from dataindexing.sources.s3 import (
    S3Config,
    parse_s3_uri,
    s3_fetch,
)

cfg = build_hybrid_config()
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
LAKEQA_ARTIFACT_ROOT = PROJECT_ROOT / "benchmarks" / "lakeqa" / "tasks-mini" / "artifacts"
DEFAULT_MANIFEST_PATH = LAKEQA_ARTIFACT_ROOT / "task_file_manifest.jsonl"
DEFAULT_DESCRIPTIONS_PATH = LAKEQA_ARTIFACT_ROOT / "descriptions.jsonl"
DEFAULT_SCHEMAS_PATH = LAKEQA_ARTIFACT_ROOT / "table_schemas_full.jsonl"
BUILD_MODES = ("infused", "basic")

def _uri_to_stem(uri: str) -> str:
    """Strip bucket prefix and file extension to match keys in load_table_schemas."""
    return uri_to_schema_stem(uri)


def resolve_cli_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    return Path(path_str).expanduser().resolve()


def resolve_config_path(path_str: str | None, default: Path | None = None) -> Path | None:
    if path_str is None:
        return default
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hybrid LanceDB index from a parquet cache or S3 manifest."
    )
    parser.add_argument(
        "--embed-preset",
        choices=sorted(EMBED_PRESETS),
        default=None,
        help="Embedding preset to use for indexing",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Override model name from the selected preset",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Override embedding dimensionality from the selected preset",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        help="Override embedding batch size from the selected preset",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default=None,
        help="Override torch dtype from the selected preset",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Override max sequence length from the selected preset",
    )
    parser.add_argument(
        "--parquet",
        "--input-parquet",
        dest="parquet",
        type=str,
        default=None,
        help="Input parquet cache path to use instead of S3Config.parquet_cache_path",
    )
    parser.add_argument(
        "--build-mode",
        choices=BUILD_MODES,
        default="infused",
        help=(
            "Index construction mode. 'infused' includes LLM descriptions; "
            "'basic' indexes only raw chunks and URI/column schema text."
        ),
    )
    parser.add_argument(
        "--manifest",
        "--input-manifest",
        dest="manifest",
        type=str,
        default=None,
        help="Input manifest path to use when parquet cache is unavailable "
             f"(default: {DEFAULT_MANIFEST_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        "--lance-path",
        dest="output_path",
        type=str,
        default=None,
        help="Output directory for the LanceDB dataset (default: config path)",
    )
    parser.add_argument(
        "--descriptions",
        "--descriptions-parquet",
        "--descriptions-jsonl",
        dest="descriptions",
        action="append",
        type=str,
        default=None,
        help="JSONL or parquet file with LLM-generated table descriptions. "
             "Repeat to layer multiple files; later files override earlier rows. "
             f"Defaults to canonical {DEFAULT_DESCRIPTIONS_PATH.relative_to(PROJECT_ROOT)}.",
    )
    parser.add_argument(
        "--schemas",
        "--schemas-jsonl",
        dest="schemas",
        type=str,
        default=None,
        help="JSONL with per-table column metadata "
             f"(default: {DEFAULT_SCHEMAS_PATH.relative_to(PROJECT_ROOT)})",
    )
    return parser.parse_args()


async def fetch_all(s3_cfg: S3Config, manifest_path: Path):
    if s3_cfg.parquet_cache_path and Path(s3_cfg.parquet_cache_path).exists():
        return [
            (uri, f"{meta} {content}")
            for uri, meta, content in load_parquet_records(s3_cfg.parquet_cache_path)
        ]

    uris = list(_iter_manifest_uris(manifest_path))

    sem = asyncio.Semaphore(s3_cfg.max_async)
    session = aioboto3.Session()
    async with session.client("s3", region_name=s3_cfg.region) as s3:
        async def fetch_one(uri: str) -> tuple[str, str] | None:
            bucket, key = parse_s3_uri(uri)
            async with sem:
                try:
                    text = await s3_fetch(s3, s3_cfg, bucket, key)

                    metadata_text = (key.replace("/", " ").replace(".csv", "").replace(".json", "")
                                     .replace(".txt", ""))
                    clean_document = f"{metadata_text} {text}"
                    print(f"Fetched: {uri}")
                    return uri, clean_document
                except Exception as e:
                    print(f"Failed: {uri}: {e}")
                    return None

        return await tqdm.gather(*[fetch_one(u) for u in uris])


def make_table_schema(embed_dim: int, with_chunk_kind: bool = False) -> pa.Schema:
    fields = [
        pa.field("id",     pa.string()),
        pa.field("text",   pa.string()),
        pa.field("uri",    pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embed_dim)),
    ]
    if with_chunk_kind:
        fields.append(pa.field("chunk_kind", pa.string()))
    return pa.schema(fields)


def get_or_create_lakeqa_table(db, cfg):
    return db.create_table(
        "lakeqa",
        schema=make_table_schema(cfg.embed_dim, with_chunk_kind=True),
        mode="overwrite",
    )


def get_or_create_schema_table(db, cfg):
    return db.create_table(
        "lakeqa_schema",
        schema=make_table_schema(cfg.embed_dim),
        mode="overwrite",
    )


def embed_documents(texts, model, batch_size):
    del batch_size
    return encode_documents(texts, model, cfg, show_progress_bar=True)


def _doc_stream(s3_cfg: S3Config, manifest_path: Path):
    """Yields (uri, clean_text) one doc at a time from parquet or S3."""
    if s3_cfg.parquet_cache_path and Path(s3_cfg.parquet_cache_path).exists():
        for uri, meta, content in iter_parquet_records(s3_cfg.parquet_cache_path):
            raw = f"{meta} {content}"
            words = raw.split()
            yield uri, " ".join(words[:10_000])
    else:
        for uri, text in asyncio.run(fetch_all(s3_cfg, manifest_path)):
            if text is not None:
                words = text.split()
                yield uri, " ".join(words[:10_000])


def _iter_manifest_uris(manifest_path: Path):
    """Yield S3 URIs from either a plain-text URI manifest or JSONL file manifest."""
    for raw_line in manifest_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uri = str(row.get("s3_uri") or row.get("dataset_uri") or row.get("uri") or "").strip()
            if uri:
                yield uri
            continue
        yield line


def _load_jsonl_descriptions(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("description_source") or "").strip() == "tasks_mini_manifest_fallback":
                uri = str(row.get("dataset_uri") or row.get("uri") or "").strip()
                raise ValueError(
                    f"{path}: description_source='tasks_mini_manifest_fallback' is not allowed "
                    f"for {uri or '<unknown uri>'}; regenerate this row with an LLM."
                )
            if row.get("error") not in (None, ""):
                continue
            uri = str(row.get("dataset_uri") or row.get("uri") or "").strip()
            if not uri:
                continue
            out[uri] = {
                "generated_metadata": str(row.get("generated_metadata") or ""),
                "description": str(row.get("description") or ""),
                "original_metadata": str(row.get("original_metadata") or ""),
            }
    return out


def _load_description_records(paths: list[Path]) -> dict[str, dict[str, str]]:
    descriptions: dict[str, dict[str, str]] = {}
    for path in paths:
        if path.suffix.lower() == ".parquet":
            rows = load_descriptions_full(str(path))
        else:
            rows = _load_jsonl_descriptions(path)
            if rows:
                print(f"Loaded {len(rows)} description records from: {path}")
        descriptions.update(rows)
    return descriptions


if __name__ == "__main__":
    args = parse_args()
    cfg = build_hybrid_config(
        embed_preset=args.embed_preset,
        embed_model=args.embed_model,
        embed_dim=args.embed_dim,
        embed_batch_size=args.embed_batch_size,
        torch_dtype=args.torch_dtype,
        max_seq_length=args.max_seq_length,
    )
    if args.output_path:
        output_path = resolve_cli_path(args.output_path)
        if output_path is None:
            raise ValueError("Output path could not be resolved")
        cfg.path = str(output_path)

    s3_cfg = S3Config()
    manifest_path = resolve_cli_path(args.manifest) if args.manifest else DEFAULT_MANIFEST_PATH
    parquet_path = resolve_cli_path(args.parquet)
    if parquet_path is not None:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet cache not found: {parquet_path}")
        s3_cfg.parquet_cache_path = str(parquet_path)
    elif s3_cfg.parquet_cache_path is not None:
        resolved_default_parquet = resolve_config_path(s3_cfg.parquet_cache_path)
        s3_cfg.parquet_cache_path = (
            str(resolved_default_parquet) if resolved_default_parquet is not None else None
        )

    if manifest_path is None:
        raise ValueError("Manifest path could not be resolved")
    if not manifest_path.exists() and not (
        s3_cfg.parquet_cache_path and Path(s3_cfg.parquet_cache_path).exists()
    ):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    descriptions = {}
    descriptions_paths: list[Path] = []
    if args.build_mode == "infused":
        if args.descriptions:
            descriptions_paths = [
                p for p in (resolve_cli_path(path) for path in args.descriptions)
                if p is not None
            ]
        else:
            descriptions_paths = [DEFAULT_DESCRIPTIONS_PATH]
        descriptions = _load_description_records(descriptions_paths)
        if args.descriptions and not descriptions:
            raise ValueError(
                f"--descriptions was set to {descriptions_paths} but no descriptions were loaded. "
                f"Check the file exists and has the expected columns/keys."
            )
    elif args.descriptions:
        print("WARN: --descriptions is ignored in basic build mode.")

    schemas_path = resolve_cli_path(args.schemas) if args.schemas else DEFAULT_SCHEMAS_PATH
    table_schemas = load_table_schemas(str(schemas_path)) if schemas_path is not None else {}
    if args.schemas and not table_schemas:
        raise ValueError(
            f"--schemas was set to {schemas_path} but no table schemas were loaded. "
            f"Check the file exists and is a valid JSONL with a tables[] array."
        )

    embed_model = build_embed_model(cfg)

    db = lancedb.connect(cfg.path)
    table = get_or_create_lakeqa_table(db, cfg)
    schema_table = get_or_create_schema_table(db, cfg)

    print("Chunking, embedding, and inserting into LanceDB...")
    state = {
        "chunk_ids": [], "chunk_texts": [], "chunk_uris": [], "chunk_kinds": [],
        "schema_texts": [], "schema_uris": [],
        "chunk_idx": 0, "schema_idx": 0, "batch_count": 1,
    }

    def flush_chunks():
        if not state["chunk_ids"]:
            return
        print(f"Embedding chunk batch {state['batch_count']} ({len(state['chunk_ids'])} chunks)...")
        vecs = embed_documents(state["chunk_texts"], embed_model, cfg.embed_batch_size)
        table.add([
            {"id": bid, "text": txt, "uri": uri, "vector": vec, "chunk_kind": kind}
            for bid, txt, uri, vec, kind in zip(
                state["chunk_ids"], state["chunk_texts"], state["chunk_uris"],
                vecs, state["chunk_kinds"],
            )
        ])
        state["chunk_ids"] = []
        state["chunk_texts"] = []
        state["chunk_uris"] = []
        state["chunk_kinds"] = []
        state["batch_count"] += 1
        torch.cuda.empty_cache()

    def flush_schema():
        if not state["schema_texts"]:
            return
        vecs = embed_documents(state["schema_texts"], embed_model, cfg.embed_batch_size)
        schema_table.add([
            {"id": str(state["schema_idx"] + i), "text": txt, "uri": uri, "vector": vec}
            for i, (txt, uri, vec) in enumerate(
                zip(state["schema_texts"], state["schema_uris"], vecs)
            )
        ])
        state["schema_idx"] += len(state["schema_texts"])
        state["schema_texts"], state["schema_uris"] = [], []
        torch.cuda.empty_cache()

    print(f"Embedding preset: {cfg.embed_preset}")
    print(f"Embedding model : {cfg.embed_model}")
    print(f"Embedding dim   : {cfg.embed_dim}")
    print(f"Embedding batch : {cfg.embed_batch_size}")
    print(f"Torch dtype     : {cfg.torch_dtype}")
    print(f"Max seq length  : {cfg.max_seq_length}")
    print(f"Build mode      : {args.build_mode}")
    print(f"Output path    : {cfg.path}")
    print(f"Manifest path  : {manifest_path}")
    print(f"Parquet cache  : {s3_cfg.parquet_cache_path}")
    print(f"Descriptions   : {descriptions_paths} ({len(descriptions)} loaded)")
    print(f"Schemas        : {schemas_path} ({len(table_schemas)} loaded)")

    for uri, doc in _doc_stream(s3_cfg, manifest_path):
        desc_row = descriptions.get(uri)

        schema_cols = table_schemas.get(_uri_to_stem(uri))
        schema_text = build_schema_text(
            build_mode=args.build_mode,
            uri=uri,
            doc=doc,
            desc_row=desc_row,
            schema_cols=schema_cols,
        )
        state["schema_texts"].append(schema_text)
        state["schema_uris"].append(uri)

        desc_chunk = build_description_chunk(args.build_mode, desc_row)
        if desc_chunk is not None:
            state["chunk_texts"].append(desc_chunk)
            state["chunk_ids"].append(str(state["chunk_idx"]))
            state["chunk_uris"].append(uri)
            state["chunk_kinds"].append("description")
            state["chunk_idx"] += 1
            if len(state["chunk_ids"]) >= cfg.batch_insert:
                flush_chunks()

        words = doc.split()
        for j in range(0, len(words), cfg.word_chunks - cfg.overlap):
            state["chunk_texts"].append(" ".join(words[j : j + cfg.word_chunks]))
            state["chunk_ids"].append(str(state["chunk_idx"]))
            state["chunk_uris"].append(uri)
            state["chunk_kinds"].append("content")
            state["chunk_idx"] += 1
            if len(state["chunk_ids"]) >= cfg.batch_insert:
                flush_chunks()
            if j + cfg.word_chunks >= len(words):
                break

        if len(state["schema_texts"]) >= cfg.batch_insert:
            flush_schema()

    flush_chunks()
    flush_schema()

    print("Building FTS indexes...")
    fts_kwargs = dict(
        replace=True,
        with_position=True,
        base_tokenizer="simple",
        language="English",
        lower_case=True,
        stem=True,
        remove_stop_words=False,
        ascii_folding=True,
    )
    table.create_fts_index("text", **fts_kwargs)
    schema_table.create_fts_index("text", **fts_kwargs)

    print("Building ANN indexes (IVF_HNSW_SQ)...")
    for tbl, label in [(table, "lakeqa"), (schema_table, "lakeqa_schema")]:
        num_partitions = max(1, tbl.count_rows() // 1_048_576)
        tbl.create_index(
            metric="cosine",
            num_partitions=num_partitions,
            ef_construction=cfg.ivf_ef_construction,
            vector_column_name="vector",
            replace=True,
            index_type="IVF_HNSW_SQ",
        )
        print(f"  {label} index done.")

    print("Success! Hybrid Vector database built.")

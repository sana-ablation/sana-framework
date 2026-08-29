import logging
import re

import lancedb
from lancedb.rerankers import CrossEncoderReranker, RRFReranker

from dataindexing.hybrid_search.config import build_hybrid_config
from dataindexing.hybrid_search.embeddings import build_embed_model, encode_query

logger = logging.getLogger(__name__)

_db = None
_embed_model = None
_reranker = None
cfg = build_hybrid_config()

# LanceDB parses a double-quoted span as a phrase query, which needs token
# positions in the FTS index. Indexes built before builder.py set
# with_position=True reject those outright instead of degrading, so a single
# quoted term turns a search into a hard error and costs the agent a turn.
_PHRASE_UNSUPPORTED_MARKER = "position is not found"
_QUOTE_CHARS = re.compile(r"[\"“”‘’]")


def strip_phrase_quotes(query: str) -> str:
    """Drop quote characters LanceDB would read as phrase-query syntax."""
    stripped = _QUOTE_CHARS.sub(" ", str(query or ""))
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def _fts_with_quote_fallback(run, query):
    """Run an FTS search, retrying unquoted when the index lacks positions.

    Agents reach for Google-style quoting to mean "these terms are required"
    rather than to request adjacency - observed queries quote single tokens
    like "311", where phrase semantics are a no-op. Falling back to keyword
    search preserves that intent; BM25 still ranks documents containing all
    the terms above the rest.
    """
    try:
        return run(query)
    except Exception as exc:
        if _PHRASE_UNSUPPORTED_MARKER not in str(exc):
            raise
        unquoted = strip_phrase_quotes(query)
        if not unquoted or unquoted == query:
            raise
        logger.warning(
            "FTS index has no positions; retrying %r as keyword search %r",
            query,
            unquoted,
        )
        return run(unquoted)


def setup(load_reranker: bool = True):
    __get_db()
    __get_embed_model()
    if load_reranker:
        __setup_reranker()


def setup_hybrid():
    """Eagerly load only the resources needed by hybrid RRF search."""
    setup(load_reranker=False)


def setup_sparse():
    """Eagerly load only the LanceDB connection needed by FTS search."""
    __get_db()


def __get_db():
    global _db
    if _db is None:
        _db = lancedb.connect(cfg.path)
    return _db


def get_connection():
    return lancedb.connect(cfg.path)


def get_table():
    return __get_db().open_table("lakeqa")


def get_schema_table():
    return __get_db().open_table("lakeqa_schema")


def __get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = build_embed_model(cfg)
    return _embed_model


def _encode_query(text: str) -> list:
    return encode_query(text, __get_embed_model(), cfg)


def _fmt(rows: list[dict], score_key: str = "_score", k: int | None = None,
         invert: bool = False) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        uri = row["uri"]
        if uri in seen:
            continue
        seen.add(uri)
        raw = row.get(score_key, 0)
        score = 1.0 - raw if invert else raw
        out.append({
            "score":    f"{score:.3f}",
            "uri":      uri,
        })
        if k and len(out) >= k:
            break
    return out


def sparse_search_schema(query, k=10):
    def _run(q):
        return (
            get_schema_table()
            .search(q, query_type="fts", fts_columns="text")
            .limit(k * 3)
            .select(["uri"])
            .to_list()
        )

    rows = _fts_with_quote_fallback(_run, query)
    return _fmt(rows, score_key="_score", k=k)


def hybrid_search_schema(query, sparse_query=None, k=10):
    vector = _encode_query(query)

    def _run(text_query):
        return (
            get_schema_table()
            .search(query_type="hybrid", vector_column_name="vector", fts_columns="text")
            .vector(vector)
            .text(text_query)
            .rerank(RRFReranker(K=cfg.rrf_k))
            .limit(k * 3)
            .select(["uri"])
            .to_list()
        )

    rows = _fts_with_quote_fallback(_run, sparse_query or query)
    return _fmt(rows, score_key="_relevance_score", k=k)


def sparse_search(query, k=10):
    def _run(q):
        return (
            get_table()
            .search(q, query_type="fts", fts_columns="text")
            .limit(k * 3)
            .select(["uri"])
            .to_list()
        )

    rows = _fts_with_quote_fallback(_run, query)
    return _fmt(rows, score_key="_score", k=k)


def vector_search(query, k=10):
    rows = (
        get_table()
        .search(_encode_query(query), query_type="vector", vector_column_name="vector")
        .limit(k * 3)
        .select(["uri"])
        .to_list()
    )
    return _fmt(rows, score_key="_distance", k=k, invert=True)


def hybrid_search(query, sparse_query=None, k=10):
    """Fast hybrid search using RRF. Scores are always tiny (max ~0.016) — they
    represent rank positions, not semantic relevance. Use this for low-latency queries."""
    vector = _encode_query(query)

    def _run(text_query):
        return (
            get_table()
            .search(query_type="hybrid", vector_column_name="vector", fts_columns="text")
            .vector(vector)
            .text(text_query)
            .rerank(RRFReranker(K=cfg.rrf_k))
            .limit(k * 3)
            .select(["uri"])
            .to_list()
        )

    rows = _fts_with_quote_fallback(_run, sparse_query or query)
    return _fmt(rows, score_key="_relevance_score", k=k)


def __setup_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            column="text",
        )
        _ = _reranker.model  # force cached_property to load weights now
    return _reranker


def hybrid_search_with_reranker(query, sparse_query=None, k=10):
    """Slower hybrid search using a cross-encoder reranker. Scores are meaningful
    (0-1 relevance) but results can feel unexpected — the reranker rewards surface-level
    token overlap, so short queries may rank oddly. Better for longer, specific queries."""
    vector = _encode_query(query)
    reranker = __setup_reranker()

    def _run(text_query):
        return (
            get_table()
            .search(query_type="hybrid", vector_column_name="vector", fts_columns="text")
            .vector(vector)
            .text(text_query)
            .rerank(reranker)
            .limit(k * 3)
            .select(["uri"])
            .to_list()
        )

    rows = _fts_with_quote_fallback(_run, sparse_query or query)
    return _fmt(rows, score_key="_relevance_score", k=k)

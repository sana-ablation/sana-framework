"""Search tools for standard mode: hybrid RRF, schema, and reranked."""

from strands import tool

from dataindexing.hybrid_search import api as _api


def set_db_path(path: str) -> None:
    """Override hybrid search LanceDB path for this process."""
    _api.cfg.path = str(path)
    _api._db = None


def setup() -> None:
    _api.setup_hybrid()



@tool
def search_value(query: str, top_k: int = 10) -> dict:
    """Hybrid semantic + keyword search over dataset content (RRF reranking).

    Args:
        query: Natural-language query or topic.
        top_k: Maximum number of results to return (default 10).
    """
    try:
        results = _api.hybrid_search(query, k=top_k)
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": str(e), "query": query, "results": [], "count": 0}


@tool
def search_schema(query: str, top_k: int = 10) -> dict:
    """Hybrid semantic + keyword search over dataset schemas (column names, types).

    Args:
        query: Natural-language query about schema structure.
        top_k: Maximum number of results to return (default 10).
    """
    try:
        results = _api.hybrid_search_schema(query, k=top_k)
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": str(e), "query": query, "results": [], "count": 0}


@tool
def search_reranked(query: str, top_k: int = 10) -> dict:
    """Hybrid semantic + keyword search with cross-encoder reranking. Slower but more accurate.

    Args:
        query: Natural-language query or topic.
        top_k: Maximum number of results to return (default 10).
    """
    try:
        results = _api.hybrid_search_with_reranker(query, k=top_k)
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": str(e), "query": query, "results": [], "count": 0}

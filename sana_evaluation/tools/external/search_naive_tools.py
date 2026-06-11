"""Search tools for naive mode: sparse FTS only."""

from strands import tool

from dataindexing.hybrid_search import api as _api


def set_db_path(path: str) -> None:
    """Override sparse/hybrid search LanceDB path for this process."""
    _api.cfg.path = str(path)
    _api._db = None


def setup() -> None:
    _api.setup_sparse()


@tool
def search_value(query: str, top_k: int = 10) -> dict:
    """Sparse keyword search over dataset content (FTS).

    Args:
        query: Keywords or short phrase to search for.
        top_k: Maximum number of results to return (default 10).
    """
    try:
        results = _api.sparse_search(query, k=top_k)
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": str(e), "query": query, "results": [], "count": 0}


@tool
def search_schema(query: str, top_k: int = 10) -> dict:
    """Sparse keyword search over dataset schemas (column names, types).

    Args:
        query: Keywords describing the column or field structure you need.
        top_k: Maximum number of results to return (default 10).
    """
    try:
        results = _api.sparse_search_schema(query, k=top_k)
        return {"results": results, "count": len(results), "query": query}
    except Exception as e:
        return {"error": str(e), "query": query, "results": [], "count": 0}

import os
from dataclasses import dataclass


EMBED_PRESETS = {
    "qwen3_0_6b": {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "embed_dim": 1024,
        "embed_batch_size": 16,
        "torch_dtype": "float16",
        "max_seq_length": 2048,
        "query_encode_mode": "prompt",
        "document_encode_mode": "prompt",
        "tokenizer_padding_side": "left",
    },
    "embeddinggemma_300m": {
        "embed_model": "google/embeddinggemma-300m",
        "embed_dim": 768,
        "embed_batch_size": 32,
        "torch_dtype": "bfloat16",
        "max_seq_length": 2048,
        "query_encode_mode": "sentence_transformers",
        "document_encode_mode": "sentence_transformers",
        "tokenizer_padding_side": None,
    },
}


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value is not None else None


@dataclass
class HybridConfig:
    # Embedding
    embed_preset: str = "qwen3_0_6b"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    embed_batch_size: int = 16
    torch_dtype: str = "float16"
    max_seq_length: int = 2048
    query_encode_mode: str = "prompt"
    document_encode_mode: str = "prompt"
    tokenizer_padding_side: str | None = "left"

    # Storage
    path: str = "./lance_data"

    # Chunking
    word_chunks: int = 800
    overlap: int = 100
    batch_insert: int = 128
    TOKEN_PATTERN: str = r'(?u)\b[a-zA-Z_]\w+\b'

    # ANN index tuning (IVF_HNSW_SQ); num_partitions = max(1, num_rows // 1_048_576)
    # ef_construction: higher = better recall, slower build (default 150)
    ivf_ef_construction: int = 150

    # Search
    rerank: bool = False
    document_word_cutoff: int = 100
    vector_weight: float = 0.7
    rrf_k: int = 60


def build_hybrid_config(
    *,
    embed_preset: str | None = None,
    embed_model: str | None = None,
    embed_dim: int | None = None,
    embed_batch_size: int | None = None,
    torch_dtype: str | None = None,
    max_seq_length: int | None = None,
) -> HybridConfig:
    cfg = HybridConfig()
    cfg.path = os.getenv("HYBRID_DB_PATH") or cfg.path

    preset_name = embed_preset or os.getenv("HYBRID_EMBED_PRESET") or cfg.embed_preset
    if preset_name not in EMBED_PRESETS:
        raise ValueError(
            f"Unknown embedding preset: {preset_name}. "
            f"Choose from {', '.join(sorted(EMBED_PRESETS))}"
        )
    preset = EMBED_PRESETS[preset_name]

    cfg.embed_preset = preset_name
    cfg.embed_model = embed_model or os.getenv("HYBRID_EMBED_MODEL") or preset["embed_model"]
    cfg.embed_dim = embed_dim or _env_int("HYBRID_EMBED_DIM") or preset["embed_dim"]
    cfg.embed_batch_size = (
        embed_batch_size
        or _env_int("HYBRID_EMBED_BATCH_SIZE")
        or preset["embed_batch_size"]
    )
    cfg.torch_dtype = torch_dtype or os.getenv("HYBRID_TORCH_DTYPE") or preset["torch_dtype"]
    cfg.max_seq_length = (
        max_seq_length
        or _env_int("HYBRID_MAX_SEQ_LENGTH")
        or preset["max_seq_length"]
    )
    cfg.query_encode_mode = os.getenv("HYBRID_QUERY_ENCODE_MODE") or preset["query_encode_mode"]
    cfg.document_encode_mode = (
        os.getenv("HYBRID_DOCUMENT_ENCODE_MODE") or preset["document_encode_mode"]
    )
    cfg.tokenizer_padding_side = (
        os.getenv("HYBRID_TOKENIZER_PADDING_SIDE") or preset["tokenizer_padding_side"]
    )
    return cfg

from __future__ import annotations

import torch
from typing import Any

from dataindexing.hybrid_search.config import HybridConfig


def resolve_torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def build_embed_model(cfg: HybridConfig):
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": resolve_torch_dtype(cfg.torch_dtype)}
    tokenizer_kwargs = {}
    if cfg.tokenizer_padding_side is not None:
        tokenizer_kwargs["padding_side"] = cfg.tokenizer_padding_side

    model = SentenceTransformer(
        cfg.embed_model,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs or None,
    )
    model.max_seq_length = cfg.max_seq_length
    return model


def encode_documents(
    texts: list[str],
    model: Any,
    cfg: HybridConfig,
    *,
    show_progress_bar: bool,
):
    kwargs = {
        "batch_size": cfg.embed_batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": show_progress_bar,
    }
    if cfg.document_encode_mode == "sentence_transformers":
        if not hasattr(model, "encode_document"):
            raise AttributeError(f"{cfg.embed_model} does not support encode_document()")
        vecs = model.encode_document(texts, **kwargs)
    elif cfg.document_encode_mode == "prompt":
        vecs = model.encode(texts, prompt_name="document", **kwargs)
    else:
        vecs = model.encode(texts, **kwargs)
    return vecs.astype("float32").tolist()


def encode_query(text: str, model: Any, cfg: HybridConfig) -> list[float]:
    kwargs = {
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    if cfg.query_encode_mode == "sentence_transformers":
        if not hasattr(model, "encode_query"):
            raise AttributeError(f"{cfg.embed_model} does not support encode_query()")
        vec = model.encode_query([text], **kwargs)
    elif cfg.query_encode_mode == "prompt":
        vec = model.encode([text], prompt_name="query", **kwargs)
    else:
        vec = model.encode([text], **kwargs)
    return vec[0].astype("float32").tolist()

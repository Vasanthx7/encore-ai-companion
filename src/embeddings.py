"""Embedder — text -> vector, behind a swappable, optional seam.

Default: `nomic-embed-text` served by Ollama, reached through the OpenAI SDK's
embeddings endpoint. An alternative such as bge-base via fastembed, or a hosted
OpenAI-compatible provider, can be swapped in behind this same interface with no
call-site changes (COMPANION_EMBED_BASE_URL / _API_KEY / _MODEL).

Embeddings are optional. When disabled (COMPANION_EMBEDDINGS_ENABLED=false) or
unreachable, embed()/embed_one() return None vectors instead of raising, and
callers (src/retrieval.py, src/reconcile.py) fall back to non-semantic matching.
"""

from __future__ import annotations

import sys

import numpy as np
from openai import OpenAI

import config

_client: OpenAI | None = None
_warned = False


def enabled() -> bool:
    return config.EMBEDDINGS_ENABLED


def _client_() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY)
    return _client


def _warn_once(exc: Exception) -> None:
    global _warned
    if not _warned:
        print(
            f"[embeddings] disabled for this session ({exc}); "
            "falling back to entity/recency-based retrieval. "
            "Set COMPANION_EMBEDDINGS_ENABLED=false to silence this.",
            file=sys.stderr,
        )
        _warned = True


def embed(texts: list[str]) -> list[np.ndarray | None]:
    """Embed a batch of texts -> list of float32 vectors (L2-normalized), or
    None per text if embeddings are disabled or the endpoint is unreachable.

    Normalizing here means cosine similarity later is a plain dot product.
    """
    if not texts:
        return []
    if not config.EMBEDDINGS_ENABLED:
        return [None] * len(texts)
    try:
        resp = _client_().embeddings.create(model=config.EMBED_MODEL, input=texts)
    except Exception as exc:  # noqa: BLE001 - never break the chat loop on a down embedder
        _warn_once(exc)
        return [None] * len(texts)
    out: list[np.ndarray | None] = []
    for item in resp.data:
        v = np.asarray(item.embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        out.append(v / n if n > 0 else v)
    return out


def embed_one(text: str) -> np.ndarray | None:
    return embed([text])[0]

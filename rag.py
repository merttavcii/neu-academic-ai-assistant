"""Sentence splitting, overlapping chunks, and vector retrieval."""

from __future__ import annotations

import os
import re



try:
    _raw_max = float(os.environ.get("RAG_MAX_COSINE_DISTANCE", "0.68"))
    MAX_RELEVANT_COSINE_DISTANCE = min(0.99, max(0.1, _raw_max))
except ValueError:
    MAX_RELEVANT_COSINE_DISTANCE = 0.68


def split_into_sentences(text: str) -> list[str]:
    """Split on sentence boundaries (. ! ?) followed by whitespace."""
    if not text or not text.strip():
        return []
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_sentences(
    sentences: list[str],
    *,
    max_sentences: int = 3,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Chunks of up to `max_sentences` (2–3 in practice) with overlap between chunks.
    Each step advances by (chunk_size - overlap_sentences).
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return [" ".join(sentences)]

    overlap = max(1, min(overlap_sentences, max_sentences - 1))
    chunks: list[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        remaining = n - i
        
        if remaining >= 3:
            size = 3
        elif remaining >= 2:
            size = 2
        else:
            size = 1

        piece = sentences[i : i + size]
        text = " ".join(piece).strip()
        if text:
            if size == 1 and chunks:
                chunks[-1] = f"{chunks[-1]} {text}".strip()
            else:
                chunks.append(text)

        if remaining <= size:
            break
        step = max(1, size - overlap)
        i += step

    return chunks


def chunk_text(text: str) -> list[str]:
    sents = split_into_sentences(text)
    return chunk_sentences(sents, max_sentences=3, overlap_sentences=1)


def retrieve_top_k(collection, query: str, k: int = 3) -> tuple[list[str], float | None]:
    """
    Return (chunk_texts, best_distance).
    best_distance is None if collection empty / no results.
    """
    if collection.count() == 0:
        return [], None
    n = min(k, collection.count())
    result = collection.query(query_texts=[query], n_results=n)
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    best: float | None = float(dists[0]) if dists else None
    return list(docs), best


def is_relevant(best_distance: float | None) -> bool:
    if best_distance is None:
        return False
    return best_distance <= MAX_RELEVANT_COSINE_DISTANCE


def format_context(chunks: list[str]) -> str:
    parts = []
    for idx, ch in enumerate(chunks, start=1):
        parts.append(f"[Parça {idx}]\n{ch}")
    return "\n\n".join(parts)

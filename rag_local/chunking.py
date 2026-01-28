# rag_local/chunking.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    meta: Dict[str, Any]


def make_chunks(
    doc_id: str,
    source: str,
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Deterministic fixed-window chunking.
    - Produces stable chunk boundaries for identical input
    - Produces unique IDs: f"{doc_id}#chunk{i}"
    """
    doc_id = str(doc_id)
    source = str(source)

    text = (text or "").strip()
    if not text:
        return []

    chunk_size = int(chunk_size)
    overlap = int(overlap)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        # enforce progress
        overlap = max(0, chunk_size // 4)

    chunks: List[Chunk] = []

    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            cid = f"{doc_id}#chunk{idx}"
            meta = {"start": start, "end": end}
            chunks.append(Chunk(id=cid, text=chunk_text, source=source, meta=meta))
            idx += 1

        if end >= n:
            break
        start = end - overlap

    return chunks

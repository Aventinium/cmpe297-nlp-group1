"""
rag_local.rag

Goal
----
Provide the baseline RAG pipeline functions:
- build_index(): load -> chunk -> index
- (later) answer_query(): retrieve -> generate

Ticket scope here: index build pipeline only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rag_local.loaders import load_documents, Document
from rag_local.chunking import chunk_text, Chunk
from rag_local.local_index import SimpleLocalIndex, Embedder


@dataclass(frozen=True)
class BuildStats:
    doc_count: int
    chunk_count: int


def build_index(
    *,
    data_dir: Union[str, Path],
    embedder: Embedder,
    chunk_size: int = 800,
    overlap: int = 200,
) -> Tuple[SimpleLocalIndex, BuildStats]:
    """
    Build a local vector index from files in data_dir.

    Pipeline:
      1) load_documents(data_dir) -> list[Document]
      2) chunk each document text -> list[Chunk]
      3) index.add(chunks)

    Determinism:
      - loaders should return deterministic doc ordering
      - chunking is deterministic
      - index.add preserves order of input chunks
    """
    docs: List[Document] = load_documents(data_dir)

    all_chunks: List[Chunk] = []
    for d in docs:
        doc_text = d.get("text", "")
        source = d.get("source", d.get("source_path", "unknown"))  # supports either naming
        doc_id = d.get("doc_id", None)

        # Each chunk should have stable IDs if doc_id provided
        chunks = chunk_text(
            doc_text,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc_id,
            include_spans=True,
        )

        # Optional: attach more metadata later (source/doc_id) if you extend Chunk schema
        # For now: keep minimal chunk contract stable.
        all_chunks.extend(chunks)

    index = SimpleLocalIndex(embedder=embedder)
    index.add(all_chunks)

    stats = BuildStats(doc_count=len(docs), chunk_count=len(all_chunks))
    return index, stats


def save_index(index: SimpleLocalIndex, path: Union[str, Path]) -> None:
    """
    Convenience wrapper so callers don't import local_index directly.
    """
    index.save(path)


def load_index(path: Union[str, Path], *, embedder: Embedder) -> SimpleLocalIndex:
    """
    Convenience wrapper for index reloading.
    """
    return SimpleLocalIndex.load(path, embedder=embedder)

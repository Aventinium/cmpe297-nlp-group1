"""
rag_local.chunking

Chunk Schema (Output Contract)
------------------------------
Each chunk MUST include:
- chunk_id: str
    Stable identifier for the chunk (deterministic across runs if inputs unchanged).
- text: str
    Chunk text.
Optional:
- start: int
    Start character index within the original text (inclusive).
- end: int
    End character index within the original text (exclusive).

Example chunk object
--------------------
chunk = {
    "chunk_id": "a1b2c3d4e5f6g7h8",
    "text": "This is the chunk text ...",
    "start": 0,
    "end": 500
}

Public API
----------
- chunk_text(text: str, *, chunk_size: int = 800, overlap: int = 200,
            doc_id: str | None = None, include_spans: bool = True) -> list[Chunk]

Notes
-----
- No NLP libraries required (pure Python).
- This implementation chunks by character count (baseline).
- You can later add token-based chunking, sentence boundaries, etc., while keeping the schema stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TypedDict
import hashlib


# -----------------------------
# Chunk schema (contract)
# -----------------------------

class Chunk(TypedDict, total=False):
    """
    Standard chunk schema for chunking output.

    Required keys:
      - chunk_id: stable identifier string
      - text: chunk text

    Optional keys:
      - start: start char index in original text (inclusive)
      - end: end char index in original text (exclusive)
    """
    chunk_id: str
    text: str
    start: int
    end: int


# -----------------------------
# Public entrypoint
# -----------------------------

def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 200,
    doc_id: Optional[str] = None,
    include_spans: bool = True,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks.

    Requirements satisfied (per ticket):
    - Fixed chunk size (chunk_size)
    - Configurable overlap (overlap)
    - Returns a list of chunk objects (Chunk schema)
    - Each chunk includes: chunk_id, text, optional start/end indices
    - No NLP libraries

    Parameters
    ----------
    text:
        Input document text.
    chunk_size:
        Maximum characters per chunk (baseline, char-based).
    overlap:
        Number of characters to overlap between consecutive chunks.
        Must be < chunk_size.
    doc_id:
        Optional document identifier used to make chunk_id stable across runs
        and unique across documents.
    include_spans:
        If True, include start/end indices in each chunk.

    Returns
    -------
    list[Chunk]
        List of chunk objects in deterministic order.

    Raises
    ------
    ValueError:
        If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    _validate_params(chunk_size=chunk_size, overlap=overlap)

    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []
    step = chunk_size - overlap

    # Deterministic traversal
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_str = text[start:end]

        # Optional: you can later add trimming rules here.
        # Keep baseline simple (do not alter text beyond slicing).
        chunk = _make_chunk(
            chunk_text=chunk_str,
            start=start,
            end=end,
            doc_id=doc_id,
            include_spans=include_spans,
        )
        chunks.append(chunk)

        if end == n:
            break
        start += step

    return chunks


# -----------------------------
# Helpers (modular building blocks)
# -----------------------------

def _validate_params(*, chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap must be < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")


def _make_chunk(
    *,
    chunk_text: str,
    start: int,
    end: int,
    doc_id: Optional[str],
    include_spans: bool,
) -> Chunk:
    """
    Normalize chunk output into the Chunk schema and generate a stable chunk_id.
    """
    chunk_id = make_chunk_id(doc_id=doc_id, start=start, end=end, chunk_text=chunk_text)
    chunk: Chunk = {
        "chunk_id": chunk_id,
        "text": chunk_text,
    }
    if include_spans:
        chunk["start"] = start
        chunk["end"] = end
    return chunk


def make_chunk_id(*, doc_id: Optional[str], start: int, end: int, chunk_text: str) -> str:
    """
    Create a deterministic chunk identifier.

    Strategy:
    - Hash(doc_id + start/end + chunk_text)
    - If doc_id is None, the id is still deterministic for the same text content,
      but doc_id is strongly recommended to avoid collisions across documents.
    """
    h = hashlib.sha256()
    h.update((doc_id or "NO_DOC_ID").encode("utf-8"))
    h.update(b"\n")
    h.update(f"{start}:{end}".encode("utf-8"))
    h.update(b"\n")
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


# -----------------------------
# Developer guide: how to extend
# -----------------------------

# ============================================================
# HOW TO EXTEND CHUNKING (Developer Guide)
# ============================================================
#
# Baseline today: character-based fixed-size chunks with overlap.
#
# Future upgrades you can add WITHOUT breaking downstream code:
# ------------------------------------------------------------
# 1) Sentence-aware chunking:
#    - Keep output schema the same (Chunk: chunk_id, text, start/end)
#    - Change chunk boundaries to respect sentence breaks.
#
# 2) Token-based chunking:
#    - Compute token spans, but still return chunk text + start/end (char indices)
#    - Add token metadata into an optional field later ONLY if needed
#
# 3) Metadata propagation:
#    - If you later chunk Documents instead of raw text, you can carry doc_id + source
#    - But keep `chunk_text(...)` stable as the low-level utility.
#
# Testing checklist before merging:
# ---------------------------------
# □ Deterministic output ordering
# □ overlap < chunk_size
# □ empty/whitespace text returns []
# □ chunk ids stable across runs (when doc_id + text unchanged)
# ============================================================

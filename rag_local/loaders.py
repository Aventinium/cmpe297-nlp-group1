"""
rag_local.loaders

Standard Document Schema (Loader Output Contract)
-------------------------------------------------
Every loader MUST return a list of Document objects with at least:

- doc_id: str
    Stable string identifier for this document. Should remain the same across runs
    if the underlying document has not changed.

- text: str
    Full extracted text for the document (plaintext).

- source: str
    Filename or relative path indicating where the document came from.
    For this project, `source` should usually be relative to rag_local/Data/.

- meta: dict (optional)
    Optional dictionary for any extra metadata (page numbers, title, author, etc.)

Example Document object
-----------------------
doc = {
    "doc_id": "lecture01_nlp_intro_v1",
    "text": "Natural Language Processing (NLP) is ...",
    "source": "notes/lecture01_intro.md",
    "meta": {"course": "CMPE297", "week": 1}
}

Public API
----------
- load_documents(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Document]

Team rule:
----------
If you change the Document schema, you must also update any downstream modules that
consume it (chunking, indexing, retrieval).
"""

# ============================================================
# HOW TO ADD A NEW LOADER (Developer Guide)
# ============================================================
#
# Goal:
# -----
# Add support for a new file type (pdf, md, txt, json, etc.)
# without breaking downstream modules (chunking, indexing, RAG).
#
# Rules:
# ------
# 1. Loader functions MUST return List[Document]
# 2. Always use normalize_document() to build outputs
# 3. Never return raw strings or custom dict shapes
# 4. Keep behavior deterministic (same input → same output order)
#
#
# Step-by-step example: Adding a Markdown loader
# ------------------------------------------------
#
# def load_markdown_file(path: Path) -> List[Document]:
#     """
#     Load one markdown file and return Document objects.
#     """
#
#     raw_text = path.read_text(encoding="utf-8")
#
#     # Optional: extract metadata here (title, headers, etc.)
#     meta = {
#         "type": "markdown",
#         "filename": path.name,
#     }
#
#     doc = normalize_document(
#         source=str(path.relative_to(DEFAULT_DATA_DIR)),
#         text=raw_text,
#         meta=meta,
#     )
#
#     return [doc]
#
#
# Then register it inside load_documents():
# -----------------------------------------
#
# def load_documents(...):
#     docs = []
#
#     for path in root.rglob("*"):
#         if path.suffix == ".md":
#             docs.extend(load_markdown_file(path))
#
#     docs.sort(key=lambda d: d["source"])  # keep CI deterministic
#     return docs
#
#
# Multi-document loaders (example: PDF pages)
# -------------------------------------------
# If one file produces multiple logical documents (like per-page PDFs),
# return multiple Document objects:
#
# return [
#     normalize_document(source="paper.pdf#page1", text=page1_text),
#     normalize_document(source="paper.pdf#page2", text=page2_text),
# ]
#
#
# Testing checklist before merging:
# ---------------------------------
# □ Returns List[Document]
# □ Each Document has doc_id, text, source
# □ No empty text fields
# □ Deterministic ordering
# □ No downstream module changes required
#
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
import hashlib


# -----------------------------
# Document schema (contract)
# -----------------------------

class Document(TypedDict, total=False):
    """
    Standard document schema for loader output.

    Required keys:
      - doc_id: stable identifier string
      - text: full document text (plaintext)
      - source: filename or relative path

    Optional keys:
      - meta: freeform metadata dictionary
    """
    doc_id: str
    text: str
    source: str
    meta: Dict[str, Any]


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_DATA_DIR = Path(__file__).parent / "Data"


# -----------------------------
# Public entrypoint (stub)
# -----------------------------

def load_documents(data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> List[Document]:
    """
    Load documents from `data_dir` and return a list of standardized Document objects.

    Current status:
    --------------
    This is an architecture stub. It defines the output contract and leaves file-type
    parsing to future incremental implementations.

    Expected behavior (eventually):
    -------------------------------
    - Walk rag_local/Data recursively
    - For each supported file type, call a dedicated loader function
    - Normalize output into the Document schema
    - Return deterministically ordered results (important for CI)

    Parameters
    ----------
    data_dir:
        Root folder containing raw files. Default: rag_local/Data/

    Returns
    -------
    list[Document]
        A list of documents adhering to the Document schema contract.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    # Placeholder: discover files (future) and call specific loaders (future)
    docs: List[Document] = []

    # CI-friendly: deterministic ordering (once you add discovery)
    # docs.sort(key=lambda d: d["source"])

    return docs


# -----------------------------
# Helper(s) you will build on
# -----------------------------

def make_doc_id(source: str, text: str) -> str:
    """
    Create a stable-ish identifier from source + content.

    This is a helper you can reuse across loaders. It provides a consistent strategy:
      - if source and text are unchanged, doc_id stays the same
      - if content changes, doc_id changes (useful for re-indexing)

    Note:
    You may later swap this with something else (front-matter IDs, filename-only IDs, etc.)
    but keep doc_id stable across runs.
    """
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def normalize_document(*, source: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Document:
    """
    Convert raw loader outputs into the standardized Document schema.
    Always use this function when adding new loaders so output stays consistent.
    """
    doc: Document = {
        "doc_id": make_doc_id(source=source, text=text),
        "text": text,
        "source": source,
    }
    if meta:
        doc["meta"] = meta
    return doc

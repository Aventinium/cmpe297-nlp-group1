# rag_local/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import os

# PDF parsing: try PyPDF2 first; if missing, PDFs will be skipped with a clear message.
try:
    import PyPDF2
except Exception:  # pragma: no cover
    PyPDF2 = None


@dataclass
class LoadedDoc:
    doc_id: str
    source: str
    text: str


def _read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    if PyPDF2 is None:
        raise ImportError("PyPDF2 not installed. Install it or use markdown docs only.")

    text_parts: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text_parts.append(t)

    return "\n".join(text_parts).strip()


def load_docs(docs_dir: str = "docs") -> List[LoadedDoc]:
    """
    Load all .md and .pdf from docs_dir.
    Returns LoadedDoc objects with:
      - doc_id: "file://<abs_path>"
      - source: filename
      - text: extracted content
    """
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Docs dir not found: {docs_dir}")

    out: List[LoadedDoc] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        if ext not in [".md", ".pdf"]:
            continue

        abs_path = p.resolve()
        doc_id = f"file://{abs_path}"
        source = p.name

        try:
            if ext == ".md":
                text = _read_md(p)
            else:
                text = _read_pdf(p)
        except Exception as e:
            # Skip unreadable docs but keep the pipeline running
            print(f"[load_docs] Skipping {p.name}: {e}")
            continue

        text = (text or "").strip()
        if len(text) < 50:
            # avoid indexing garbage/empty extractions
            print(f"[load_docs] Skipping {p.name}: extracted text too short ({len(text)} chars)")
            continue

        out.append(LoadedDoc(doc_id=doc_id, source=source, text=text))

    return out

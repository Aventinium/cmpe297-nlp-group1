"""
rag_local.openalex_fetch

OpenAlex search results -> PDF-only materialization for ingestion into local corpus.

Policy (PDF-only)
-----------------
- Use OpenAlex-provided PDF URL if available.
- If no PDF URL -> SKIP (no abstract-only ingestion).

Dataset contract
----------------
rag_local/Data/openalex/<safe_id>/
  - paper.pdf
  - paper.txt
  - meta.json
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

MIN_TEXT_LENGTH = 3000  # chars; raise if you want stricter filtering


def _safe_folder_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s or "unknown"


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        return ""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = " ".join(text.split())
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def materialize_openalex_selected(
    selected: List[Dict[str, Any]],
    data_dir: str | Path,
    query: str,
    timeout_s: int = 30,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    selected: list of dicts with at least:
      - openalex_id (or id)
      - title
      - year
      - url (landing page)
      - pdf_url (direct PDF)
      - score (optional; from reranker)
      - abstract (optional; not used for ingestion under PDF-only policy)

    Writes:
      data_dir/openalex/<safe_id>/paper.pdf
      data_dir/openalex/<safe_id>/paper.txt
      data_dir/openalex/<safe_id>/meta.json
    """
    data_dir = Path(data_dir)
    out_dir = data_dir / "openalex"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    skipped: List[Tuple[str, str]] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "cmpe297-local-rag/1.0"})

    for item in selected:
        openalex_id = str(item.get("openalex_id") or item.get("id") or "")
        title = str(item.get("title") or "")
        year = item.get("year")
        url = item.get("url")
        pdf_url = item.get("pdf_url")
        score = item.get("score")

        if not pdf_url:
            skipped.append((_safe_folder_name(openalex_id or title), "No PDF URL (PDF-only policy)"))
            continue

        # Download PDF
        try:
            r = session.get(str(pdf_url), timeout=timeout_s)
            r.raise_for_status()
            pdf_bytes = r.content
        except Exception as e:
            skipped.append((_safe_folder_name(openalex_id or title), f"PDF download failed: {e}"))
            continue

        # Extract text
        try:
            text = _extract_pdf_text(pdf_bytes)
        except Exception:
            skipped.append((_safe_folder_name(openalex_id or title), "PDF extraction failed"))
            continue

        if len(text) < MIN_TEXT_LENGTH:
            skipped.append((_safe_folder_name(openalex_id or title), f"Extracted text too short (<{MIN_TEXT_LENGTH})"))
            continue

        safe_id = _safe_folder_name(openalex_id.replace("https://openalex.org/", "OA_") or title)
        paper_dir = out_dir / safe_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        (paper_dir / "paper.pdf").write_bytes(pdf_bytes)
        (paper_dir / "paper.txt").write_text(text, encoding="utf-8")

        meta = {
            "source": "openalex",
            "openalex_id": openalex_id,
            "title": title,
            "year": year,
            "landing_url": url,
            "pdf_url": pdf_url,
            "query": query,
            "score": score,
            "raw": item.get("raw") or item,
        }
        (paper_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        saved.append(safe_id)

    return saved, skipped
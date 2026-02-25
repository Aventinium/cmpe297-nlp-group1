from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import requests


@dataclass
class RankedCandidate:
    id: str
    title: str
    abstract: str
    year: Optional[int]
    url: Optional[str]
    pdf_url: Optional[str]
    source: str
    score: float


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec)) or 1.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot / (_l2_norm(a) * _l2_norm(b)))


class OllamaEmbedder:
    """
    Robust embedding client for Ollama.

    - Truncates text to avoid oversized requests
    - Splits into small batches to avoid 500s
    - Tries /api/embed (batch) first, falls back to /api/embeddings (single)
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text", timeout_s: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def _truncate(self, s: str, max_chars: int) -> str:
        s = (s or "").strip()
        return s[:max_chars]

    def embed_batch(
        self,
        texts: List[str],
        *,
        batch_size: int = 8,
        max_chars_per_text: int = 4000,
        sleep_s: float = 0.0,
    ) -> List[List[float]]:
        if not texts:
            return []

        # truncate to avoid server-side 500 from too-large payloads
        texts = [self._truncate(t, max_chars_per_text) for t in texts]

        out: List[List[float]] = []

        def try_embed_endpoint(batch: List[str]) -> Optional[List[List[float]]]:
            url = f"{self.host}/api/embed"
            payload = {"model": self.model, "input": batch}
            r = self.session.post(url, json=payload, timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            data = r.json()
            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]
            if "data" in data and isinstance(data["data"], list):
                return [row["embedding"] for row in data["data"]]
            return None

        def fallback_embeddings(batch: List[str]) -> List[List[float]]:
            url = f"{self.host}/api/embeddings"
            embs: List[List[float]] = []
            for t in batch:
                payload = {"model": self.model, "prompt": t}
                r = self.session.post(url, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                embs.append(data["embedding"])
                if sleep_s > 0:
                    time.sleep(sleep_s)
            return embs

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            embs = try_embed_endpoint(batch)
            if embs is not None:
                out.extend(embs)
            else:
                out.extend(fallback_embeddings(batch))

            if sleep_s > 0:
                time.sleep(sleep_s)

        return out


def rerank_by_cosine(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    ollama_host: str,
    embed_model: str = "nomic-embed-text",
    text_key_title: str = "title",
    text_key_abstract: str = "abstract",
    id_key: str = "id",
    year_key: str = "year",
    url_key: str = "url",
    pdf_url_key: str = "pdf_url",
    source: str = "openalex",
    top_n: Optional[int] = None,
    abstract_max_chars: int = 2000,
) -> List[RankedCandidate]:
    """
    candidates: list of dicts with at minimum title + abstract (abstract can be empty).

    Returns sorted by cosine similarity(query_emb, candidate_emb) desc.
    """
    embedder = OllamaEmbedder(host=ollama_host, model=embed_model)

    # Build candidate texts (title + truncated abstract)
    cand_texts: List[str] = []
    for c in candidates:
        title = str(c.get(text_key_title) or "").strip()
        abs_text = str(c.get(text_key_abstract) or "").strip()
        if abs_text:
            abs_text = abs_text[:abstract_max_chars]
            combined = f"{title}\n\n{abs_text}" if title else abs_text
        else:
            combined = title
        cand_texts.append(combined)

    # Embed query + candidates
    embs = embedder.embed_batch([query] + cand_texts, batch_size=8, max_chars_per_text=4000)
    if not embs or len(embs) != (1 + len(candidates)):
        raise RuntimeError("Embedding call failed or returned wrong number of vectors.")

    q_emb = embs[0]
    cand_embs = embs[1:]

    ranked: List[RankedCandidate] = []
    for c, e, combined in zip(candidates, cand_embs, cand_texts):
        score = cosine_similarity(q_emb, e)

        ranked.append(
            RankedCandidate(
                id=str(c.get(id_key) or ""),
                title=str(c.get(text_key_title) or ""),
                abstract=str(c.get(text_key_abstract) or ""),
                year=c.get(year_key),
                url=c.get(url_key),
                pdf_url=c.get(pdf_url_key),
                source=source,
                score=score,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    if top_n is not None:
        ranked = ranked[: int(top_n)]
    return ranked
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from collections import Counter

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


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.docs = [self._tokenize(d) for d in docs]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

        self.tf = [Counter(d) for d in self.docs]

        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.df = df

        self.N = len(self.docs)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def score(self, query: str, doc_index: int) -> float:
        tokens = self._tokenize(query)
        score = 0.0
        dl = self.doc_lens[doc_index]
        tf_doc = self.tf[doc_index]

        for term in tokens:
            if term not in self.df:
                continue

            df = self.df[term]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

            tf = tf_doc.get(term, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1) / denom)

        return score
    

def llm_relevance_score(
    query: str,
    title: str,
    abstract: str,
    host: str = "http://localhost:11434",
    model: str = "llama3",
    instruction: str = "High scoring documents are relevant to Natural Language Processing (NLP) in the field of Machine Learning and Artificial Intellegence."
) -> float:
    prompt = f"""

Context: You are a relevance scorer for academic search.

Instructions: {instruction}
Query: "{query}"

Paper:
Title: {title}
Abstract: {abstract}

Rate how relevant this paper is to the query and instructions on a scale from 0.0 to 1.0.
Return ONLY a number.
"""

    r = requests.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    text = r.json().get("response", "").strip()

    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except:
        return 0.0

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

def rerank_hybrid(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    ollama_host: str,
    embed_model: str = "nomic-embed-text",
    alpha: float = 0.25, # weight for cos
    beta: float = 0.25,  # weight for BM25
    gamma: float = 0.25, # weight for metadata
    delta: float = 0.25, # weight for LLM ranking
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

    total_w = alpha + beta + gamma + delta
    embedder = OllamaEmbedder(host=ollama_host, model=embed_model)

    cand_texts: List[str] = []
    cand_llm_ranking: List[str] = []
    for c in candidates:
        if delta != 0:
            cand_llm_ranking.append(
                llm_relevance_score(
                    query=query.strip(),
                    title=c.get(text_key_title),
                    abstract=c.get(text_key_abstract),
                    host=ollama_host,
                    model="llama3"
                )
            )
        else:
            cand_llm_ranking.append(0)
        title = str(c.get(text_key_title) or "").strip()
        abs_text = str(c.get(text_key_abstract) or "").strip()
        if abs_text:
            abs_text = abs_text[:abstract_max_chars]
            combined = f"{title}\n\n{abs_text}" if title else abs_text
        else:
            combined = title
        cand_texts.append(combined)

    embs = embedder.embed_batch([query] + cand_texts)
    if not embs or len(embs) != (1 + len(candidates)):
        raise RuntimeError("Embedding call failed or returned wrong number of vectors.")

    q_emb = embs[0]
    cand_embs = embs[1:]

    bm25 = BM25(cand_texts)

    ranked: List[RankedCandidate] = []
    for idx, (c, e) in enumerate(zip(candidates, cand_embs)):
        cos = cosine_similarity(q_emb, e)
        bm = bm25.score(query, idx)
        ranked.append((c, cos, bm))

    bm_values = [bm for (_, _, bm) in ranked]
    bm_min, bm_max = min(bm_values), max(bm_values)
    
    final: List[RankedCandidate] = []
    for ((c, cos, bm), llm) in zip(ranked,cand_llm_ranking):
        hybrid = alpha/total_w * cos + beta/total_w * ((bm - bm_min) / (bm_max - bm_min)) + delta/total_w * llm

        final.append(
            RankedCandidate(
                id=str(c.get(id_key) or ""),
                title=str(c.get(text_key_title) or ""),
                abstract=str(c.get(text_key_abstract) or ""),
                year=c.get(year_key),
                url=c.get(url_key),
                pdf_url=c.get(pdf_url_key),
                source=source,
                score=hybrid,
            )
        )

    final.sort(key=lambda x: x.score, reverse=True)
    if top_n is not None:
        final = final[: int(top_n)]
    return final
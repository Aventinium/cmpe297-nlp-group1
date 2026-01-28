# rag_local/local_index.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import os
import pickle

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

from rag_local.chunking import Chunk


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _cosine_topk(q: np.ndarray, M: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (idxs, scores) of top-k cosine similarity between q (d,) and M (n,d).
    """
    if M is None or M.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    q = q.astype(np.float32)
    M = M.astype(np.float32)

    q = _l2_normalize(q.reshape(1, -1))[0]
    M = _l2_normalize(M)

    sims = M @ q  # (n,)

    k = min(int(k), sims.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.astype(np.int64), sims[idx].astype(np.float32)


@dataclass
class LocalHit:
    chunk_id: str
    score: float
    text: str
    meta: Dict[str, Any]


class LocalDenseIndex:
    """
    Pure-local dense index:
    - Embeds with sentence-transformers
    - Stores vectors + payloads in memory
    - Optional persistence via pickle
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        persist_path: Optional[str] = None,
        batch_size: int = 64,
    ):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for LocalDenseIndex")

        self.embedding_model = embedding_model or os.getenv(
            "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.embedder = SentenceTransformer(self.embedding_model)

        self.batch_size = int(batch_size)
        self.persist_path = persist_path or os.getenv(
            "RAG_INDEX_PATH", "data/local_index.pkl"
        )

        self.ids: List[str] = []
        self.texts: Dict[str, str] = {}
        self.meta: Dict[str, Dict[str, Any]] = {}
        self.vectors: Optional[np.ndarray] = None  # (n, d)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        arr = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        return arr.astype(np.float32)

    def add_documents(
        self,
        docs: Iterable[Union[Tuple[str, str, Optional[Dict[str, Any]]], Chunk]],
    ) -> None:
        """
        Accepts either:
        - iterable of Chunk objects
        - iterable of (id, text, meta_dict)
        """
        docs = list(docs)
        if not docs:
            return

        new_ids: List[str] = []
        new_texts: List[str] = []

        first = docs[0]

        # Chunk objects
        if isinstance(first, Chunk) or (hasattr(first, "id") and hasattr(first, "text")):
            for c in docs:
                cid = str(getattr(c, "id"))
                txt = str(getattr(c, "text"))
                md: Dict[str, Any] = {}

                if hasattr(c, "source"):
                    md["source"] = getattr(c, "source")
                if hasattr(c, "meta") and isinstance(getattr(c, "meta"), dict):
                    md.update(getattr(c, "meta"))

                self.texts[cid] = txt
                self.meta[cid] = md
                new_ids.append(cid)
                new_texts.append(txt)

        # tuples
        else:
            for cid, txt, md in docs:
                cid = str(cid)
                txt = str(txt)
                md = dict(md or {})

                self.texts[cid] = txt
                self.meta[cid] = md
                new_ids.append(cid)
                new_texts.append(txt)

        # embed in batches
        embs_list: List[np.ndarray] = []
        for i in range(0, len(new_texts), self.batch_size):
            embs_list.append(self._embed_texts(new_texts[i : i + self.batch_size]))

        dim = self.embedder.get_sentence_embedding_dimension()
        embs = (
            np.vstack(embs_list)
            if embs_list
            else np.zeros((0, dim), dtype=np.float32)
        )

        if self.vectors is None:
            self.vectors = embs
            self.ids = new_ids
        else:
            self.vectors = np.vstack([self.vectors, embs])
            self.ids.extend(new_ids)

    def rank(self, query: str, top_k: int = 8) -> List[LocalHit]:
        if self.vectors is None or len(self.ids) == 0:
            return []

        qv = self._embed_texts([query])[0]
        idxs, scores = _cosine_topk(qv, self.vectors, top_k)

        hits: List[LocalHit] = []
        for i, s in zip(idxs.tolist(), scores.tolist()):
            cid = self.ids[i]
            hits.append(
                LocalHit(
                    chunk_id=cid,
                    score=float(s),
                    text=self.texts.get(cid, ""),
                    meta=self.meta.get(cid, {}),
                )
            )
        return hits

    # Back-compat API for rag.py
    def search(self, query: str, k: int = 8) -> List[LocalHit]:
        return self.rank(query, top_k=k)

    def save(self, path: Optional[str] = None) -> str:
        if path is not None:
            self.persist_path = path

        parent = os.path.dirname(self.persist_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "embedding_model": self.embedding_model,
            "ids": self.ids,
            "texts": self.texts,
            "meta": self.meta,
            "vectors": self.vectors,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(payload, f)
        return self.persist_path

    def load(self, path: Optional[str] = None) -> bool:
        if path is not None:
            self.persist_path = path

        if not self.persist_path or not os.path.exists(self.persist_path):
            return False

        with open(self.persist_path, "rb") as f:
            payload = pickle.load(f)

        self.ids = payload.get("ids", [])
        self.texts = payload.get("texts", {})
        self.meta = payload.get("meta", {})
        self.vectors = payload.get("vectors", None)
        return True

import os, uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

def _uuid(s): return str(uuid.uuid5(uuid.NAMESPACE_URL, s))

class DenseIndex:
    def __init__(self, collection="rag_chunks", url=None):
        self.embedder = SentenceTransformer(os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        self.client = QdrantClient(url=url) if url else None
        self.collection = collection
        self.mem = {}
        self.vecs = {}
        if self.client:
            try: self.client.get_collection(collection)
            except:
                self.client.create_collection(collection,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedder.get_sentence_embedding_dimension(),
                        distance=qmodels.Distance.COSINE))

    def add(self, items):
        texts = [p["text"] for _, p in items]
        vecs = self.embedder.encode(texts, convert_to_numpy=True)
        for (cid, payload), v in zip(items, vecs):
            if self.client:
                self.client.upsert(self.collection, [
                    qmodels.PointStruct(id=_uuid(cid), vector=v.tolist(), payload=payload)
                ])
            else:
                self.mem[cid] = payload
                self.vecs[cid] = v

    def search(self, q, k=5):
        qv = self.embedder.encode([q], convert_to_numpy=True)[0]
        if self.client:
            res = self.client.search(self.collection, qv.tolist(), limit=k)
            return [(r.payload, r.score) for r in res]
        scores = []
        for cid, v in self.vecs.items():
            s = float(np.dot(qv, v) / (np.linalg.norm(qv)*np.linalg.norm(v)))
            scores.append((self.mem[cid], s))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

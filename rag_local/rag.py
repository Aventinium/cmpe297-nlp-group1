# rag_local/rag.py
from __future__ import annotations

from typing import List
from rag_local.local_index import LocalDenseIndex, LocalHit
from rag_local.ollama_client import generate


def _build_prompt(query: str, hits: List[LocalHit]) -> str:
    """
    Grounded prompt: use only provided context.
    """
    context_blocks = []
    for h in hits:
        cid = h.chunk_id
        src = h.meta.get("source", "")
        context_blocks.append(f"[{cid} | {src} | score={h.score:.3f}]\n{h.text}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a careful research assistant.
Answer the user's question using ONLY the provided context.
If the context is insufficient, say you don't know.

When you make a claim, cite the chunk ids like [file://...#chunk3].

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""
    return prompt


def rag_answer(query: str, index: LocalDenseIndex, top_k: int = 5) -> str:
    hits = index.search(query, k=top_k)
    prompt = _build_prompt(query, hits)
    return generate(prompt)

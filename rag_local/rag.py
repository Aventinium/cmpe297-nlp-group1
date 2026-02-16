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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rag_local.loaders import load_documents, Document
from rag_local.chunking import chunk_text, Chunk
from rag_local.local_index import SimpleLocalIndex, Embedder
from rag_local.ollama_client import chat as ollama_chat


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
    docs: List[Document] = load_documents(data_dir)

    all_chunks: List[Chunk] = []
    for d in docs:
        doc_text = d.get("text", "")
        source = d.get("source", "unknown")
        doc_id = d.get("doc_id", None)

        chunks = chunk_text(
            doc_text,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc_id,
            source=source,
            metadata=d.get("meta", None),
            include_spans=True,
        )
        all_chunks.extend(chunks)

    index = SimpleLocalIndex(embedder=embedder)
    index.add(all_chunks)

    stats = BuildStats(doc_count=len(docs), chunk_count=len(all_chunks))
    return index, stats


def save_index(index: SimpleLocalIndex, path: Union[str, Path]) -> None:
    index.save(path)


def load_index(path: Union[str, Path], *, embedder: Embedder) -> SimpleLocalIndex:
    return SimpleLocalIndex.load(path, embedder=embedder)


def _format_retrieved_context(results: Sequence[Dict[str, Any]], *, max_chars: int = 6000) -> str:
    """
    Build a stable context block with source tags [S1], [S2], ... and a char budget.
    """
    parts: List[str] = []
    used = 0
    for i, r in enumerate(results, start=1):
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        src = r.get("source", "")
        chunk_id = r.get("chunk_id", "")
        score = float(r.get("score", 0.0))

        header = f"[S{i}] score={score:.3f} chunk_id={chunk_id}"
        if src:
            header += f" source={src}"
        block = header + "\n" + txt + "\n"

        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 200:
                parts.append(block[:remain])
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()


def answer_query(
    *,
    query: str,
    index: SimpleLocalIndex,
    model: str,
    system_prompt: str,
    top_k: int = 5,
    max_context_chars: int = 6000,
) -> Dict[str, Any]:
    """
    Full RAG loop:
      1) retrieve top_k chunks from local index
      2) build prompt with citations [S1], [S2], ...
      3) call LLM
      4) return answer + sources
    """
    q = (query or "").strip()
    if not q:
        return {"answer": "(Empty query.)", "sources": [], "used_top_k": 0}

    results = index.search(q, top_k=top_k)
    context = _format_retrieved_context(results, max_chars=max_context_chars)

    rag_rules = (
        "You are an NLP tutor.\n"
        "Use the SOURCES to answer.\n"
        "Rules:\n"
        "- If the sources do not contain the answer, say so and ask a clarifying question.\n"
        "- Cite sources like [S1], [S2] for factual claims.\n"
        "- Be clear and step-by-step.\n"
    )

    user_content = f"SOURCES:\n{context}\n\nQUESTION:\n{q}\n"

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "system", "content": rag_rules},
        {"role": "user", "content": user_content},
    ]

    reply = ollama_chat(messages, model=model)

    sources = []
    for i, r in enumerate(results, start=1):
        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": r.get("chunk_id", ""),
                "score": float(r.get("score", 0.0)),
            }
        )

    return {"answer": reply or "(No response.)", "sources": sources, "used_top_k": len(results)}

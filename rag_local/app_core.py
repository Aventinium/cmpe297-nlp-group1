"""rag_local.app_core

Shared backend functions for:
  - CLI chatbot (rag_local/chat.py)
  - Streamlit UI (streamlit_app.py)

Streamlit should remain a thin UI layer; this module keeps all RAG/LLM logic
in rag_local/ so the CLI and GUI do not diverge.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from rag_local.config import AppConfig
from rag_local.embedders import make_embedder
from rag_local.ollama_client import chat as ollama_chat
from rag_local.rag import build_index, load_index, save_index


Role = Literal["system", "user", "assistant"]
Message = Dict[str, str]


def cfg_with_overrides(cfg: dict, **overrides) -> dict:
    """
    Merge overrides into cfg (shallow). This keeps Streamlit flexible and prevents
    signature mismatch errors when we add new UI controls.
    """
    out = dict(cfg or {})
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def init_embedder(cfg: AppConfig):
    """Create the embedder used by the local JSON index."""

    return make_embedder(
        backend=getattr(cfg, "embed_backend", "ollama"),
        model=getattr(cfg, "embed_model", "nomic-embed-text"),
        host=getattr(cfg, "ollama_host", "http://localhost:11434"),
    )


def init_index(cfg: AppConfig, *, force_rebuild: bool = False) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Load (or build) the local JSON index.

    Returns (index, meta). If RAG is disabled, index is None.
    """

    rag_enabled = bool(getattr(cfg, "rag_enabled", False))
    if not rag_enabled:
        return None, {"rag_enabled": False}

    embedder = init_embedder(cfg)
    data_dir = Path(getattr(cfg, "data_dir", "rag_local/Data")).resolve()
    index_path = Path(getattr(cfg, "index_path", "rag_local/Data/.index/local_index.json")).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "rag_enabled": True,
        "data_dir": str(data_dir),
        "index_path": str(index_path),
        "built": False,
        "loaded": False,
    }

    if index_path.exists() and not force_rebuild:
        index = load_index(index_path, embedder=embedder)
        meta["loaded"] = True
        return index, meta

    index, stats = build_index(
        data_dir=data_dir,
        embedder=embedder,
        chunk_size=int(getattr(cfg, "chunk_size", 800)),
        overlap=int(getattr(cfg, "overlap", 200)),
    )
    save_index(index, index_path)
    meta.update(
        {
            "built": True,
            "doc_count": int(getattr(stats, "doc_count", 0)),
            "chunk_count": int(getattr(stats, "chunk_count", 0)),
        }
    )
    return index, meta


def _trim_history(history: List[Message], *, max_messages: int) -> List[Message]:
    if max_messages <= 0:
        return []
    if len(history) <= max_messages:
        return list(history)
    return list(history[-max_messages:])


def _format_retrieved_context(results: Sequence[Dict[str, Any]], *, max_chars: int = 6000) -> str:
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


def _answer_with_rag(*, query: str, index: Any, cfg: AppConfig) -> Tuple[str, List[Dict[str, Any]]]:
    q = (query or "").strip()
    if not q:
        return "(Empty query.)", []

    results = index.search(q, top_k=int(getattr(cfg, "top_k", 5)))
    context = _format_retrieved_context(results, max_chars=int(getattr(cfg, "max_context_chars", 6000)))

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
        {"role": "system", "content": getattr(cfg, "system_prompt", "You are a helpful assistant.").strip()},
        {"role": "system", "content": rag_rules},
        {"role": "user", "content": user_content},
    ]

    reply = ollama_chat(messages, model=getattr(cfg, "model", "llama3.1:8b"))
    reply_text = reply or "(No response.)"

    sources: List[Dict[str, Any]] = []
    for i, r in enumerate(results, start=1):
        snippet = (r.get("text") or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400].rstrip() + "…"
        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": r.get("chunk_id", ""),
                "score": float(r.get("score", 0.0)),
                "source": r.get("source", ""),
                "snippet": snippet,
            }
        )

    return reply_text, sources


def answer_turn(
    *,
    history: List[Message],
    user_text: str,
    cfg: AppConfig,
    index: Optional[Any] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Answer one user turn, returning (assistant_text, sources)."""

    q = (user_text or "").strip()
    if not q:
        return "(Empty message.)", []

    if bool(getattr(cfg, "rag_enabled", False)) and index is not None:
        return _answer_with_rag(query=q, index=index, cfg=cfg)

    # Non-RAG: normal chat completion with history.
    max_hist = int(getattr(cfg, "max_history_turns", 12))
    trimmed = _trim_history(history, max_messages=max_hist)

    messages = [{"role": "system", "content": getattr(cfg, "system_prompt", "You are a helpful assistant.")}] + trimmed
    messages.append({"role": "user", "content": q})

    reply = ollama_chat(messages, model=getattr(cfg, "model", None))
    return (reply or "(No response.)"), []

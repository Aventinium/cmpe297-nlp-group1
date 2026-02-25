from __future__ import annotations

from typing import Dict, List, Literal

from rag_local.config import get_config
from rag_local.ollama_client import chat
from rag_local.embedders import make_embedder
from rag_local.rag import build_index, load_index, save_index, answer_query
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppresses TensorFlow spam

Role = Literal["system", "user", "assistant"]
Message = Dict[str, str]


def _append_and_trim(history: List[Message], msg: Message, max_messages: int) -> None:
    history.append(msg)
    if max_messages <= 0:
        history.clear()
        return
    if len(history) > max_messages:
        del history[:-max_messages]


def main() -> None:
    cfg = get_config()

    history: List[Message] = []
    rag_enabled = bool(getattr(cfg, "rag_enabled", True))

    # RAG objects (lazily created)
    index = None

    if rag_enabled:
        embed_backend = getattr(cfg, "embed_backend", "ollama")
        embed_model = getattr(cfg, "embed_model", "nomic-embed-text")
        ollama_host = getattr(cfg, "ollama_host", "http://localhost:11434")

        embedder = make_embedder(backend=embed_backend, model=embed_model, host=ollama_host)
        print("[EMBED]", embed_backend, embedder.__class__.__name__, getattr(embedder, "model", None))

        index_path = Path(getattr(cfg, "index_path", "rag_local/Data/.index/local_index.json")).resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path.exists():
            index = load_index(index_path, embedder=embedder)
            print(f"[RAG] Loaded index: {index_path}")
        else:
            index, stats = build_index(
                data_dir=Path(getattr(cfg, "data_dir", "rag_local/Data")),
                embedder=embedder,
                chunk_size=int(getattr(cfg, "chunk_size", 800)),
                overlap=int(getattr(cfg, "overlap", 200)),
            )
            save_index(index, index_path)
            print(f"[RAG] Built index: docs={stats.doc_count} chunks={stats.chunk_count} -> {index_path}")
            
        # ---- Verify embeddings stored ----
        if index is not None and getattr(index, "chunks", None):
            c0 = index.chunks[0]
            print(f"[EMBED] stored_in_chunk={'embedding' in c0} "f"dim={len(c0.get('embedding', []))} "f"keys={list(c0.keys())}")
        
    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue

            if user.lower() in {"exit", "quit"}:
                print("Bye.")
                return

            # baseline history tracking (still useful even with RAG for follow-ups)
            _append_and_trim(
                history,
                {"role": "user", "content": user},
                max_messages=int(getattr(cfg, "max_history_turns", 12)),
            )

            reply, _sources = answer_turn(history=history, user_text=user, cfg=cfg, index=index)

            _append_and_trim(
                history,
                {"role": "assistant", "content": reply},
                max_messages=int(getattr(cfg, "max_history_turns", 12)),
            )

            print(f"\nBot: {reply}\n")

        except KeyboardInterrupt:
            print("\nBye.")
            return


if __name__ == "__main__":
    main()

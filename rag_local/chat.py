from __future__ import annotations

from typing import Dict, List, Literal

from rag_local.config import get_config
from rag_local.app_core import init_index, answer_turn


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

    # RAG index (loaded/built once at startup if enabled)
    index, meta = init_index(cfg)
    if meta.get("rag_enabled"):
        if meta.get("built"):
            print(
                f"[RAG] Built index: docs={meta.get('doc_count')} chunks={meta.get('chunk_count')} -> {meta.get('index_path')}"
            )
        else:
            print(f"[RAG] Loaded index: {meta.get('index_path')}")

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

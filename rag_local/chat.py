from __future__ import annotations

from typing import Dict, List, Literal, Optional
import inspect

from .ollama_client import chat
from rag_local.config import get_config


Role = Literal["system", "user", "assistant"]
Message = Dict[str, str]  # {"role": "...", "content": "..."}


def _append_and_trim(history: List[Message], msg: Message, max_messages: int) -> None:
    """
    Append a message (user/assistant) and trim history deterministically.

    Notes:
    - `history` should NOT include the system message.
    - `max_messages` counts user+assistant messages only.
    """
    history.append(msg)

    if max_messages <= 0:
        history.clear()
        return

    if len(history) > max_messages:
        del history[:-max_messages]


def _build_messages(system_prompt: str, history: List[Message]) -> List[Message]:
    """
    Build the final message list sent to the model:
    [system] + history
    """
    return [{"role": "system", "content": system_prompt}] + list(history)


def _call_chat(messages: List[Message], model: Optional[str] = None) -> str:
    """
    Call ollama_client.chat with a best-effort signature match.

    Supports common signatures:
    - chat(messages)
    - chat(messages, model="...")
    - chat(model="...", messages=messages)

    Returns:
    - reply string (never None)
    """
    try:
        sig = inspect.signature(chat)
        params = sig.parameters

        # Case A: chat(messages) only
        if len(params) == 1:
            reply = chat(messages)

        # Case B: chat(messages, model=...)
        elif "model" in params:
            reply = chat(messages, model=model) if model else chat(messages)

        # Case C: chat(model=..., messages=...)
        elif "messages" in params:
            kwargs = {"messages": messages}
            if model and "model" in params:
                kwargs["model"] = model
            reply = chat(**kwargs)

        else:
            # Fallback: try simplest call
            reply = chat(messages)

    except Exception:
        # If introspection fails for any reason, fallback to simplest call
        reply = chat(messages)

    return reply if reply else "(No response.)"


def main() -> None:
    cfg = get_config()

    # Keep only user/assistant turns here; system prompt is prepended each call.
    history: List[Message] = []

    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue

            if user.lower() in {"exit", "quit"}:
                print("Bye.")
                return

            # Add user message to history
            _append_and_trim(
                history,
                {"role": "user", "content": user},
                max_messages=getattr(cfg, "max_history_turns", 12),
            )

            # Build messages for model call
            messages = _build_messages(
                system_prompt=getattr(cfg, "system_prompt", "You are a helpful assistant."),
                history=history,
            )

            # Call model
            reply = _call_chat(messages, model=getattr(cfg, "model", None))

            # Add assistant reply to history
            _append_and_trim(
                history,
                {"role": "assistant", "content": reply},
                max_messages=getattr(cfg, "max_history_turns", 12),
            )

            print(f"\nBot: {reply}\n")

        except KeyboardInterrupt:
            print("\nBye.")
            return


if __name__ == "__main__":
    main()

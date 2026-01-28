from .ollama_client import chat

SYSTEM_PROMPT = "You are a helpful assistant."


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Chatbot ready. Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                print("Bye.")
                return

            messages.append({"role": "user", "content": user})
            reply = chat(messages)
            if not reply:
                reply = "(No response.)"
            messages.append({"role": "assistant", "content": reply})

            print(f"\nBot: {reply}\n")

        except KeyboardInterrupt:
            print("\nBye.")
            return


if __name__ == "__main__":
    main()

from agent import build_agent_executor


def main():
    print("Medical Multi-Tool Agent")
    print("Ask about heart disease, cancer, or diabetes statistics,")
    print("or ask general medical questions. Type 'exit' to quit.\n")

    agent_executor = build_agent_executor()

    # AgentExecutor expects chat_history as a list of (role, content)
    # message tuples/objects. We keep a simple running list here so the
    # agent remembers earlier turns in the conversation.
    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        result = agent_executor.invoke(
            {"input": user_input, "chat_history": chat_history}
        )
        answer = result["output"]

        print(f"\nAgent: {answer}\n")

        chat_history.append(("human", user_input))
        chat_history.append(("ai", answer))


if __name__ == "__main__":
    main()

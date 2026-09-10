from agent import build_agent, run_agent

def main():

    print("Medical Multi-Tool Agent")
    print(
        "Ask about heart disease, cancer, or diabetes statistics,"
    )
    print(
        "or ask general medical questions."
    )
    print("Type 'exit' to quit.\n")

    agent = build_agent()

    chat_history = []

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:

            answer = run_agent(
                agent,
                user_input,
                chat_history,
            )

            print(f"\nAgent: {answer}\n")

            chat_history.append(
                ("user", user_input)
            )

            chat_history.append(
                ("assistant", answer)
            )

        except Exception as error:

            print(
                f"\nError: {error}\n"
            )


if __name__ == "__main__":
    main()
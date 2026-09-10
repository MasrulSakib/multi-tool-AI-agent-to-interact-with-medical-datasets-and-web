import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Reads the .env file in this folder and loads GROQ_API_KEY, MODEL_NAME, etc.
# into the environment, so os.getenv() below can see them.
load_dotenv()


def get_llm(temperature: float = 0):
    """
    Create a fresh Groq LLM client.

    temperature=0 by default because we want consistent, predictable
    answers for things like "how many patients have diabetes" -- not
    creative variation.
    """
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

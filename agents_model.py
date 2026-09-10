import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import OpenAIChatCompletionsModel, set_tracing_disabled

load_dotenv()


def get_agents_model():
    """
    Create an OpenAI Agents SDK model backed by Groq.

    No OpenAI model/API usage is required here.
    Groq provides an OpenAI-compatible Chat Completions endpoint.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and fill it in."
        )

    # We don't need OpenAI tracing for this assignment.
    set_tracing_disabled(True)

    groq_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    return OpenAIChatCompletionsModel(
        model=os.getenv(
            "MODEL_NAME",
            "openai/gpt-oss-120b",
        ),
        openai_client=groq_client,
    )
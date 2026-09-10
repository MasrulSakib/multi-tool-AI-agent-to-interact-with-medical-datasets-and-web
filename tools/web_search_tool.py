import os
from langchain_core.tools import tool
from tavily import TavilyClient


@tool("MedicalWebSearchTool")
def medical_web_search_tool(question: str) -> str:
    """
    Use this tool to answer GENERAL medical knowledge questions -- things
    like definitions, symptoms, causes, or treatments/cures for a disease
    or condition. This tool searches the web and does not look at our
    patient datasets.

    Do NOT use this tool for questions about statistics or numbers from
    our datasets (e.g. "how many patients have diabetes") -- use the
    matching DB tool for those instead.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY is not set. Add it to your .env file."

    client = TavilyClient(api_key=api_key)

    # include_answer=True asks Tavily to generate a short direct answer
    # from the search results, in addition to the raw results themselves.
    results = client.search(query=question, include_answer=True, max_results=5)

    answer = results.get("answer")
    sources = results.get("results", [])

    response_lines = []
    if answer:
        response_lines.append(answer)

    if sources:
        response_lines.append("\nSources:")
        for source in sources:
            response_lines.append(f"- {source['title']}: {source['url']}")

    if not response_lines:
        return "No results found for that question."

    return "\n".join(response_lines)

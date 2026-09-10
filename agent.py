from agents import Agent, Runner, function_tool
from agents_model import get_agents_model

from tools.cancer_tool import cancer_db_tool
from tools.diabetes_tool import diabetes_db_tool
from tools.heart_disease_tool import heart_disease_db_tool
from tools.web_search_tool import medical_web_search_tool


SYSTEM_PROMPT = """
You are a medical dataset assistant.

You have access to four tools:

1. HeartDiseaseDBTool
   Use ONLY for statistics and data from the heart disease dataset.

2. CancerDBTool
   Use ONLY for statistics and data from the cancer dataset.

3. DiabetesDBTool
   Use ONLY for statistics and data from the diabetes dataset.

4. MedicalWebSearchTool
   Use ONLY for general medical knowledge from the web.

ROUTING RULES:

- If the user asks for statistics, counts, averages, percentages,
  comparisons, filters, or numerical information from one of the
  provided datasets, use the corresponding database tool.

- Heart disease dataset questions -> HeartDiseaseDBTool.

- Cancer dataset questions -> CancerDBTool.

- Diabetes dataset questions -> DiabetesDBTool.

- If the user asks for general medical knowledge such as:
  definitions, symptoms, causes, risk factors, prevention,
  treatments, or general medical explanations, use
  MedicalWebSearchTool.

- Never use a database tool to answer general medical knowledge.

- Never use MedicalWebSearchTool to answer statistics from
  the provided datasets.

- Always answer the user in clear, concise natural language.

- Do not invent statistics that are not returned by a database tool.

- Medical information is educational and should not be presented
  as a personal diagnosis.
"""


# Wrap existing LangChain tools for OpenAI Agents SDK

@function_tool
def HeartDiseaseDBTool(question: str) -> str:
    """
    Query the heart disease SQLite dataset.

    Use this only for statistics, counts, averages,
    comparisons, and other data-related questions.
    """
    return heart_disease_db_tool.invoke(question)


@function_tool
def CancerDBTool(question: str) -> str:
    """
    Query the cancer SQLite dataset.

    Use this only for statistics, counts, averages,
    comparisons, and other data-related questions.
    """
    return cancer_db_tool.invoke(question)


@function_tool
def DiabetesDBTool(question: str) -> str:
    """
    Query the diabetes SQLite dataset.

    Use this only for statistics, counts, averages,
    comparisons, and other data-related questions.
    """
    return diabetes_db_tool.invoke(question)


@function_tool
def MedicalWebSearchTool(question: str) -> str:
    """
    Search the web for general medical knowledge.

    Use this for definitions, symptoms, causes,
    prevention and treatment information.
    """
    return medical_web_search_tool.invoke(question)


def build_agent():
    """
    Build the main OpenAI Agents SDK agent.
    """

    model = get_agents_model()

    agent = Agent(
        name="Medical Multi-Tool Agent",
        instructions=SYSTEM_PROMPT,
        model=model,
        tools=[
            HeartDiseaseDBTool,
            CancerDBTool,
            DiabetesDBTool,
            MedicalWebSearchTool,
        ],
    )

    return agent


def run_agent(agent, user_input: str, chat_history=None):
    """
    Run the OpenAI Agents SDK agent.

    chat_history is converted into a simple input history
    so the CLI can maintain conversation context.
    """

    if chat_history is None:
        chat_history = []

    input_items = []

    for role, content in chat_history:
        input_items.append(
            {
                "role": role,
                "content": content,
            }
        )

    input_items.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    result = Runner.run_sync(
        agent,
        input_items,
    )

    return result.final_output
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_config import get_llm
from tools.cancer_tool import cancer_db_tool
from tools.diabetes_tool import diabetes_db_tool
from tools.heart_disease_tool import heart_disease_db_tool
from tools.web_search_tool import medical_web_search_tool

SYSTEM_PROMPT = """You are a helpful medical data assistant with access to four tools:

1. heart_disease_db_tool - for statistics/data questions about heart disease patients
2. cancer_db_tool - for statistics/data questions about cancer patients
3. diabetes_db_tool - for statistics/data questions about diabetes patients
4. medical_web_search_tool - for general medical knowledge (definitions, symptoms, causes, cures)

Routing rule:
- If the question asks about counts, averages, comparisons, or any numbers
  from our patient datasets, use the matching DB tool.
- If the question asks what something IS, what causes it, its symptoms, or
  how it's treated/cured, use the web search tool.

Always answer in plain, clear language. If a question doesn't match any
tool well, use your best judgment to pick the closest one.
"""


def build_agent_executor() -> AgentExecutor:
    """
    Assembles the Main Agent and returns an AgentExecutor ready to run.
    """
    llm = get_llm()

    tools = [
        heart_disease_db_tool,
        cancer_db_tool,
        diabetes_db_tool,
        medical_web_search_tool,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True)

import sqlite3

from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.tools import tool

from llm_config import get_llm


def create_database_agent(
    db_path: str,
    schema_description: str,
):
    """
    Create a LangChain AgentExecutor for one SQLite database.
    """

    @tool
    def execute_sql(sql_query: str) -> str:
        """
        Execute a read-only SQLite SELECT query against the database.
        """

        sql = sql_query.strip()

        # Remove accidental markdown fences.
        sql = sql.replace("```sql", "").replace("```", "").strip()

        # Safety: only SELECT statements.
        if not sql.upper().startswith("SELECT"):
            return (
                "ERROR: Only SELECT queries are allowed. "
                "The query was rejected."
            )

        # Prevent multiple SQL statements.
        if ";" in sql.rstrip(";"):
            return (
                "ERROR: Multiple SQL statements are not allowed."
            )

        connection = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
        )

        try:
            cursor = connection.cursor()

            cursor.execute(sql)

            columns = [
                description[0]
                for description in cursor.description
            ]

            rows = cursor.fetchall()

            if not rows:
                return "Query executed successfully. No rows were returned."

            # Keep the returned context reasonably small.
            preview_rows = rows[:50]

            return (
                f"Columns: {columns}\n"
                f"Rows: {preview_rows}\n"
                f"Total rows returned: {len(rows)}"
            )

        except Exception as error:
            return f"SQL execution error: {error}"

        finally:
            connection.close()

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You are a database analysis agent.

Your job is to answer questions using ONLY the SQLite
database provided to you.

DATABASE SCHEMA:

{schema_description}

Rules:

1. Convert the user's question into a valid SQLite SELECT query.
2. Use the execute_sql tool to execute the query.
3. Never modify the database.
4. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
5. Do not invent columns.
6. Do not invent statistics.
7. Base your final answer strictly on the SQL result.
8. Return a concise natural-language answer.
""",
            ),
            MessagesPlaceholder(
                variable_name="chat_history",
                optional=True,
            ),
            (
                "human",
                "{input}",
            ),
            MessagesPlaceholder(
                variable_name="agent_scratchpad",
            ),
        ]
    )

    tools = [execute_sql]

    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
    )


def answer_question_from_db(
    question: str,
    db_path: str,
    schema_description: str,
) -> str:
    """
    Run a question through the LangChain database agent.
    """

    agent_executor = create_database_agent(
        db_path=db_path,
        schema_description=schema_description,
    )

    result = agent_executor.invoke(
        {
            "input": question,
            "chat_history": [],
        }
    )

    return result["output"]
import sqlite3
from llm_config import get_llm


def ask_llm_for_sql(question: str, schema_description: str) -> str:
    """
    Step 1: Turn a plain-English question into a SQL SELECT query.
    """
    prompt = f"""You are a SQL expert. Given a database schema and a question,
write ONE SQLite query that answers the question.

Schema:
{schema_description}

Rules:
- Only write a SELECT query. Never write INSERT, UPDATE, DELETE, or DROP.
- Return ONLY the raw SQL query. No explanation, no markdown formatting,
  no ```sql fences.

Question: {question}

SQL query:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    sql_query = response.content.strip()

    # In case the model wraps the query in markdown code fences anyway,
    # strip those off so sqlite3 doesn't choke on them.
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    return sql_query


def run_sql_query(db_path: str, sql_query: str):
    """
    Step 2: Run the SQL query against the database and return the rows.

    Only SELECT queries are allowed here -- this is read-only data, and an
    LLM-generated query should never be allowed to modify the database.
    """
    if not sql_query.strip().upper().startswith("SELECT"):
        raise ValueError(f"Only SELECT queries are allowed. Got: {sql_query}")

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(sql_query)
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    connection.close()

    return columns, rows


def ask_llm_to_explain_result(question: str, sql_query: str, columns, rows) -> str:
    """
    Step 3: Turn the raw SQL result into a natural-language answer.
    """
    # Keep the result small in case the query returns a lot of rows --
    # we only need enough for the LLM to summarize it accurately.
    preview_rows = rows[:20]

    prompt = f"""A user asked a question about a medical dataset. Here is the
question, the SQL query that was run, and the result. Write a short, clear,
natural-language answer to the question based on the result.

Question: {question}
SQL query: {sql_query}
Columns: {columns}
Result rows (up to 20 shown): {preview_rows}
Total rows returned: {len(rows)}

Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content.strip()


def answer_question_from_db(question: str, db_path: str, schema_description: str) -> str:
    """
    Runs the full 3-step pipeline and returns a natural-language answer.
    Also used to build the answer returned by each tool's @tool function.
    """
    sql_query = ask_llm_for_sql(question, schema_description)

    try:
        columns, rows = run_sql_query(db_path, sql_query)
    except Exception as error:
        return (
            f"I tried to run this SQL query but it failed: {sql_query}\n"
            f"Error: {error}"
        )

    return ask_llm_to_explain_result(question, sql_query, columns, rows)

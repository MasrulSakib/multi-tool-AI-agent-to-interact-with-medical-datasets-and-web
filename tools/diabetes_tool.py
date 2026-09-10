from pathlib import Path
from langchain_core.tools import tool
from tools.db_utils import answer_question_from_db

DB_PATH = str(Path(__file__).parent.parent / "data" / "diabetes.db")

SCHEMA_DESCRIPTION = """
Table: diabetes_records
Columns:
    gender               TEXT     -- 'Male', 'Female', or 'Other'
    age                  REAL     -- patient age, ranges from 0 to 80
    hypertension         INTEGER  -- 1 = has hypertension, 0 = does not
    heart_disease        INTEGER  -- 1 = has heart disease, 0 = does not
    smoking_history      TEXT     -- e.g. 'never', 'current', 'former', 'No Info'
    bmi                  REAL     -- Body Mass Index
    HbA1c_level          REAL     -- average blood sugar over past 2-3 months (%)
    blood_glucose_level  INTEGER  -- blood glucose level at time of measurement
    diabetes             INTEGER  -- 1 = diagnosed with diabetes, 0 = not diagnosed
"""


@tool("DiabetesDBTool")
def diabetes_db_tool(question: str) -> str:
    """
    Use this tool to answer questions about STATISTICS or DATA from the
    diabetes dataset -- things like counts, averages, comparisons, or
    filters over real patient records (age, BMI, blood glucose level,
    HbA1c level, diabetes status, etc.).

    Do NOT use this tool for general medical questions like "what is
    HbA1c" -- use the web search tool for those instead.
    """
    return answer_question_from_db(question, DB_PATH, SCHEMA_DESCRIPTION)

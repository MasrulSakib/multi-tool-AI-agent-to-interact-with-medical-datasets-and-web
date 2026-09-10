from pathlib import Path
from langchain_core.tools import tool
from tools.db_utils import answer_question_from_db

DB_PATH = str(Path(__file__).parent.parent / "data" / "heart_disease.db")

# This describes the table so the LLM knows what columns exist when it
# writes SQL. Column meanings come straight from the dataset's own
# documentation (the UCI Cleveland heart disease attributes).
SCHEMA_DESCRIPTION = """
Table: heart_disease_records
Columns:
    age       INTEGER  -- patient age in years
    sex       INTEGER  -- 1 = male, 0 = female
    cp        INTEGER  -- chest pain type (0-3)
    trestbps  INTEGER  -- resting blood pressure (mm Hg)
    chol      INTEGER  -- serum cholesterol (mg/dl)
    fbs       INTEGER  -- fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    restecg   INTEGER  -- resting ECG result (0-2)
    thalach   INTEGER  -- maximum heart rate achieved
    exang     INTEGER  -- exercise-induced angina (1 = yes, 0 = no)
    oldpeak   REAL     -- ST depression induced by exercise relative to rest
    slope     INTEGER  -- slope of the peak exercise ST segment (0-2)
    ca        INTEGER  -- number of major vessels colored by flourosopy (0-3)
    thal      INTEGER  -- thalassemia result (0-3)
    target    INTEGER  -- 1 = has heart disease, 0 = does not
"""


@tool("HeartDiseaseDBTool")
def heart_disease_db_tool(question: str) -> str:
    """
    Use this tool to answer questions about STATISTICS or DATA from the
    heart disease dataset -- things like counts, averages, comparisons,
    or filters over real patient records (age, blood pressure, cholesterol,
    chest pain type, whether they have heart disease, etc.).

    Do NOT use this tool for general medical questions like "what causes
    heart disease" -- use the web search tool for those instead.
    """
    return answer_question_from_db(question, DB_PATH, SCHEMA_DESCRIPTION)

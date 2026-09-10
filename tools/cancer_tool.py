from pathlib import Path
from langchain_core.tools import tool
from tools.db_utils import answer_question_from_db

DB_PATH = str(Path(__file__).parent.parent / "data" / "cancer.db")

SCHEMA_DESCRIPTION = """
Table: cancer_records
Columns:
    Age               INTEGER  -- patient age, ranges from 20 to 80
    Gender            INTEGER  -- 0 = male, 1 = female
    BMI               REAL     -- Body Mass Index, ranges from 15 to 40
    Smoking           INTEGER  -- 1 = smoker, 0 = non-smoker
    GeneticRisk       INTEGER  -- 0 = low, 1 = medium, 2 = high
    PhysicalActivity  REAL     -- hours per week of physical activity (0-10)
    AlcoholIntake     REAL     -- alcohol units consumed per week (0-5)
    CancerHistory     INTEGER  -- 1 = has personal history of cancer, 0 = no
    Diagnosis         INTEGER  -- 1 = diagnosed with cancer, 0 = not diagnosed
"""


@tool("CancerDBTool")
def cancer_db_tool(question: str) -> str:
    """
    Use this tool to answer questions about STATISTICS or DATA from the
    cancer prediction dataset -- things like counts, averages, comparisons,
    or filters over real patient records (age, BMI, smoking status,
    genetic risk, cancer diagnosis, etc.).

    Do NOT use this tool for general medical questions like "what are the
    symptoms of cancer" -- use the web search tool for those instead.
    """
    return answer_question_from_db(question, DB_PATH, SCHEMA_DESCRIPTION)

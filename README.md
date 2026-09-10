# Medical Multi-Tool Agent

A single AI agent that answers questions about three medical datasets
(Heart Disease, Cancer, Diabetes) and, separately, general medical
knowledge questions — by routing each question to the right tool.

## How it works

You ask a question. The agent picks one of 4 tools:

| Question type                                          | Tool used              |
| ------------------------------------------------------ | ---------------------- |
| "How many patients have heart disease?"                | `HeartDiseaseDBTool`   |
| "What's the average BMI of diagnosed cancer patients?" | `CancerDBTool`         |
| "How many diabetic patients have high blood pressure?" | `DiabetesDBTool`       |
| "What are the symptoms of diabetes?"                   | `MedicalWebSearchTool` |

The three DB tools each work the same way:

1. Your question is turned into a SQL query by the LLM
2. That SQL query runs against the real SQLite database
3. The result is turned back into a plain-English answer by the LLM

The web search tool uses Tavily to search the web for general medical
knowledge and does not touch the datasets at all.

## Project structure

```
medical-multi-tool-agent/
├── data/
│   ├── raw/                    <- put the 3 downloaded Kaggle CSVs here
│   ├── prepare_data.py         <- converts CSVs to SQLite databases
│   ├── heart_disease.db        <- created by prepare_data.py
│   ├── cancer.db               <- created by prepare_data.py
│   └── diabetes.db             <- created by prepare_data.py
├── tools/
│   ├── db_utils.py             <- shared text-to-SQL logic used by all 3 DB tools
│   ├── heart_disease_tool.py   <- HeartDiseaseDBTool
│   ├── cancer_tool.py          <- CancerDBTool
│   ├── diabetes_tool.py        <- DiabetesDBTool
│   └── web_search_tool.py      <- MedicalWebSearchTool
├── llm_config.py                <- one place that sets up the Groq LLM
├── agent.py                     <- builds the Main Agent (AgentExecutor + 4 tools)
├── main.py                      <- command-line chat loop
├── list_models.py               <- helper if a Groq model gets retired
├── requirements.txt
└── .env.example
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API keys

- **Groq** (free): https://console.groq.com — create an API key
- **Tavily** (free): https://tavily.com — create an API key

Copy `.env.example` to `.env` and fill in both keys:

```bash
cp .env.example .env
```

### 3. Download the datasets

Download these 3 CSVs from Kaggle and place them in `data/raw/` with
these exact names:

| Save as                      | Download from                                                            |
| ---------------------------- | ------------------------------------------------------------------------ |
| `data/raw/heart_disease.csv` | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset        |
| `data/raw/cancer.csv`        | https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset |
| `data/raw/diabetes.csv`      | https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset |

### 4. Build the databases

```bash
python data/prepare_data.py
```

This creates `data/heart_disease.db`, `data/cancer.db`, and `data/diabetes.db`.

### 5. Run the agent

```bash
python main.py
```

Try asking:

- "How many patients in the heart disease dataset have the disease?"
- "What's the average cholesterol level for patients over 50?"
- "How many cancer patients have a family history of cancer?"
- "What causes type 2 diabetes?"

Type `exit` to quit.

## Key concepts (for anyone new to LangChain agents)

- **Tool**: a Python function the LLM can choose to call, decorated with
  `@tool`. The docstring is what the LLM reads to decide _when_ to use it
  — that's why each tool's docstring explicitly says what it's for and
  what it's _not_ for.
- **Agent**: the LLM plus a prompt that tells it which tools exist. It
  decides which tool(s) to call and in what order.
- **AgentExecutor**: the loop that actually runs the agent — it calls
  the LLM, runs whatever tool the LLM picked, feeds the tool's result
  back to the LLM, and repeats until the LLM gives a final answer instead
  of another tool call.
- **create_tool_calling_agent**: a LangChain helper that wires an LLM +
  tools + prompt together into an agent, using the LLM provider's native
  tool-calling / function-calling feature under the hood.

## Troubleshooting

**`ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'`**
In LangChain 1.x, `AgentExecutor` and `create_tool_calling_agent` moved
out of the main `langchain` package into a separate `langchain-classic`
package. This project already imports from `langchain_classic.agents`
and pins `langchain-classic==1.0.8` in requirements.txt, so a fresh
`pip install -r requirements.txt` should avoid this.

**`groq.NotFoundError: 404 model_not_found`**
Groq periodically retires older models. Run `python list_models.py` to
see which models your API key can currently use, then update
`MODEL_NAME` in your `.env` file to match.

**A DB tool errors out or returns nothing**
Make sure you've run `python data/prepare_data.py` after placing all
3 CSVs in `data/raw/` — the tools look for the `.db` files in `data/`
and will fail if they don't exist yet.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
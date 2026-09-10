import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY is not set. Add it to your .env file first.")
else:
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    models = response.json()["data"]

    print("Models available to your Groq API key:\n")
    for model in models:
        print(f" - {model['id']}")

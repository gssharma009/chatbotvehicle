import json
from groq import Groq

# Groq client (api key provided by Render environment variable)
client = Groq(api_key=None)  # Keep as None, Render injects API key

def load_context():
    try:
        with open("context.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def ask_llm(question: str) -> str:
    context_data = load_context()

    prompt = f"""
Answer in Hinglish unless user specifically requests Hindi or English.

User Question:
{question}

Relevant Reference:
{context_data}
"""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

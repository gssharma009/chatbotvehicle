# app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class Q(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: Q):
    try:
        prompt = f"Answer in Hinglish.\nUser: {data.question}"

        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        answer = chat_completion.choices[0].message["content"]

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

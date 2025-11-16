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
    language: str = "english"   # default

@app.post("/ask")
async def ask_question(data: Q):
    try:
        # Language enforcement prompt
        lang = data.language.lower().strip()

        if lang == "english":
            system_prompt = (
                "You MUST answer strictly in pure English. "
                "Do NOT use Hindi or Hinglish."
            )
        elif lang == "hindi":
            system_prompt = (
                "आप केवल शुद्ध हिंदी में ही जवाब दें। "
                "इंग्लिश या हिंग्लिश का बिल्कुल प्रयोग न करें।"
            )
        else:
            system_prompt = (
                "You MUST respond strictly in English."
            )

        # Call Groq
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.question}
            ],
            temperature=0.3
        )

        answer = chat_completion.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

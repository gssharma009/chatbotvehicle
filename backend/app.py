# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import answer_query, health_check
import os

app = FastAPI(title="Vehicle RAG Chatbot")

class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Vehicle RAG Chatbot live – use /ask or /health"}

@app.get("/health")
async def health():
    return health_check()

@app.post("/ask")
async def ask(query: Query):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # This line is the only important one – it will ALWAYS return something
    answer = answer_query(query.question) or "No answer generated."

    return {"answer": answer}
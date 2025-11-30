# backend/app.py – FINAL VERSION (CORS + guaranteed response)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import answer_query, health_check

app = FastAPI(title="Vehicle RAG Chatbot")

# FIX CORS — allow your frontend (Netlify, Vercel, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                  # Change to your real domain later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    lang: str = "en-US"  # ← Add this line

# Then pass lang to your LLM prompt

@app.get("/")
def root():
    return {"message": "Vehicle RAG Chatbot – ready"}

@app.get("/health")
def health():
    return health_check()

@app.post("/ask")
def ask(query: Query):
    if not query.question or not query.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    # ← THIS LINE WAS WRONG BEFORE
    answer = answer_query(query.question.strip(), query.lang)   # ← ADD query.lang HERE

    if not answer:
        answer = "No relevant information found in the manual."

    return {"answer": answer}
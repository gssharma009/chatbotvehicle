from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import answer_query, health_check  # Your model.py
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="Bilingual RAG Chatbot", version="1.0")

# CORS first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# Routes BEFORE any uvicorn.run
@app.get("/")
def root():
    return {"message": "Bilingual RAG Chatbot - Ready! Try /health or POST /ask"}

@app.get("/health")
def health():
    return health_check()

@app.post("/ask")
def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    results = answer_query(request.question)
    return results  # Flat { "answer": ... } for frontend

# Debug endpoint (remove later)
@app.post("/ask-test")
def ask_test():
    return {"results": {"answer": "Backend working! Endpoint /ask is live.", "source": "debug"}}

# Render-specific: Bind to $PORT
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info", reload=False)
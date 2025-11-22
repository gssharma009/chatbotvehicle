from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import answer_query

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    results = answer_query(request.question)
    return {"results": results}

@app.get("/")
def root():
    return {"message": "FastAPI + MiniLM-L3 + FAISS (Render-friendly)"}

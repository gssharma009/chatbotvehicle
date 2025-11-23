from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import answer_query, health_check
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    results = answer_query(request.question)
    return {"results": results}

@app.get("/health")
def health():
    return health_check()

@app.get("/")
def root():
    return {"message": "Bilingual RAG Chatbot - Ready!"}

# For Render: Bind to $PORT
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
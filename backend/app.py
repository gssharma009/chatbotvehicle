from fastapi import FastAPI
from pydantic import BaseModel
from model import ask_llm

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_endpoint(data: Query):
    answer = ask_llm(data.question)
    return {"answer": answer}

@app.get("/")
def home():
    return {"status": "Backend running OK!"}

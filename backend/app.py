import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import ask_llm

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: Query):
    answer = ask_llm(data.question)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"status": "OK", "message": "Backend running!"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000)

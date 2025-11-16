from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import ask_llm

class Q(BaseModel):
    question: str

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
def ask_bot(data: Q):
    try:
        answer = ask_llm(data.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

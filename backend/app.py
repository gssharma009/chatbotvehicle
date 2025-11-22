from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import ask_llm

app = FastAPI()

# CORS CONFIG
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all
    allow_credentials=True,
    allow_methods=["*"],        # ⬅ OPTIONS allowed
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "Backend is running"}

@app.options("/ask")
def preflight():
    return {"status": "ok"}

@app.post("/ask")
async def ask(data: dict):
    question = data.get("question", "")
    answer = ask_llm(question)
    return {"answer": answer}

@app.get("/hi")
def hi():
    return {"message": "hello from backend!"}

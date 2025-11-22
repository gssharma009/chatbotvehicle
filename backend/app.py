from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import answer_query, health_check

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}

@app.post("/ask")
async def ask(req: Query):
    question = req.question.strip()
    if not question:
        return {"answer": "Please send a non-empty question."}
    return answer_query(question)

@app.get("/health")
def health():
    return health_check()

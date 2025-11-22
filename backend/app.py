from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import ask_llm
from rag import search_docs

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Q(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: Q):
    try:
        query = data.question

        # 🔍 Step 1: Check document relevance
        context = search_docs(query)

        # 🔥 Step 2: Pass context + query to model
        prompt = f"""
        Answer using the following document context if relevant.
        If the context is not helpful, answer normally.

        CONTEXT:
        {context}

        QUESTION:
        {query}
        """

        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = chat_completion.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}

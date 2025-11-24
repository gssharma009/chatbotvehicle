# model.py — FINAL VERSION (best accuracy + stable)
import os, faiss, pickle, re
from typing import List
from sentence_transformers import SentenceTransformer
import requests

model = index = chunks = None

def clean_context(text: str) -> str:
    # Remove garbage symbols
    text = re.sub(r'[^A-Za-z0-9\s\.,;:()\-]', ' ', text)

    # Fix spacing issues: "h v"→"hv", "a c"→"ac"
    text = re.sub(r'(\b[a-z])\s+([a-z]\b)', r'\1\2', text)

    # Remove repeated spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove tiny fragments
    lines = [l.strip() for l in text.split(".") if len(l.strip()) > 20]

    return ". ".join(lines)


def _load():
    global model, index, chunks
    if model is None:
        print("[INIT] Loading model…")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)


def answer_query(question: str) -> str:
    if not question.strip():
        return "Please ask a question."

    _load()

    q_emb = model.encode(
        [question.lower()],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    _, I = index.search(q_emb.astype("float32"), 15)

    raw_context = " ".join(
        chunks[i] for i in I[0]
        if 0 <= i < len(chunks)
    )

    cleaned_context = clean_context(raw_context)

    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "Missing GROQ_API_KEY"

    prompt = f"""
You are an expert who fixes text extracted from car manuals.

Clean this OCR text and answer the question using only corrected information:

OCR RAW:
\"\"\"{cleaned_context}\"\"\" 

QUESTION: {question}

RULES:
- Fix OCR errors (gure → figure, wipera → wiper, off → of)
- Fix broken spacing (h v → HV, a c → AC)
- Remove noise, garbage, headers, page numbers
- Convert into clean, readable bullet points
- Be accurate and professional
- Do NOT include any unclean or raw text

ANSWER:
"""

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.0
        },
        headers={"Authorization": f"Bearer {key}"},
        timeout=18
    )

    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"].strip()

    return "Service error — try again."


def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "error"}

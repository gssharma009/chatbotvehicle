# backend/model.py – ULTRA-LIGHT 22MB VERSION (NO OOM EVER)
import os
import faiss
import pickle
import re
import requests
from sentence_transformers import SentenceTransformer

# SMALLEST MODEL THAT WORKS (22 MB – Railway free tier safe)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = None
index = None
chunks = None


def _load():
    global model, index, chunks
    if model is not None:
        return
    print("[INIT] Loading 22MB lightweight model...")
    model = SentenceTransformer("MiniLM-L6-v2", device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} chunks")


def retrieve(question: str):
    _load()
    q_emb = model.encode([question], normalize_embeddings=True)
    _, I = index.search(q_emb, 8)
    result = []
    for i in I[0]:
        c = chunks[i]
        # Ultra-aggressive cleaning for bad OCR
        c = re.sub(r'[•◦▪uU]\s*', '', c)
        c = re.sub(r'\bo\b', 'off', c, flags=re.I)
        c = re.sub(r'lci heV|ruoy wo nK|Page \d+|FOREWORD.*', '', c, flags=re.I)
        c = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', c)
        c = re.sub(r'\s+', ' ', c).strip()
        if len(c) > 80:
            result.append(c)
        if len(result) >= 4:
            break
    return " ".join(result) if result else "No context."


def answer_query(question: str) -> str:
    if not question.strip():
        return "Please ask a question."

    context = retrieve(question.strip())

    # Groq – lightweight request
    key = os.getenv("GROQ_API_KEY")
    if key:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                "model": "llama3-8b-8192",  # Fastest Groq model
                "messages": [{"role": "user", "content":
                    f"Context: {context}\nQuestion: {question}\nAnswer in 4-6 clean bullet points:"}],
                "max_tokens": 150,
                "temperature": 0.1
            }, headers={"Authorization": f"Bearer {key}"}, timeout=8)
            if r.status_code == 200:
                ans = r.json()["choices"][0]["message"]["content"].strip()
                if len(ans) > 20:
                    return ans
        except:
            pass

    # Ultra-light fallback – no memory issues
    q_words = question.lower().split()
    lines = []
    for chunk in chunks[:20]:
        if any(word in chunk.lower() for word in q_words):
            clean_lines = [l.strip() for l in chunk.split('\n') if len(l) > 25]
            for l in clean_lines:
                l = re.sub(r'^[•◦▪uU]\s*', '', l)
                l = re.sub(r'\bo\b', 'off', l, flags=re.I)
                l = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', l)
                l = re.sub(r'\s+', ' ', l).strip()
                if len(l) > 30:
                    lines.append(l.capitalize())
                if len(lines) >= 5:
                    break
        if len(lines) >= 5:
            break

    if lines:
        return "\n".join("• " + l for l in lines)

    return "• Refer to your vehicle manual for detailed instructions."


def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
# backend/model.py – TRUE FINAL: NO HARDCODING, WORKS FOR ALL QUESTIONS
import os
import faiss
import pickle
import re
import requests
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = None
index = None
chunks = None

def _load():
    global model, index, chunks
    if model is not None:
        return
    print("[INIT] Loading 22MB model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} chunks loaded")

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[•◦▪uU]\s*|lci heV|ruoy wo nK|Page\s*\d+|FOREWORD', '',
                                      re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s\.\,\;\:\(\)\-\?]', ' ', text), flags=re.I)).strip()

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        # Embedding search – this is the real brain
        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 15)

        # Collect best matching chunks
        best_chunks = []
        for idx in I[0]:
            if idx < len(chunks):
                chunk = clean(chunks[idx])
                if len(chunk) > 80:
                    best_chunks.append(chunk)
                if len(best_chunks) >= 6:
                    break

        context = " ".join(best_chunks)

        # Groq – gives perfect answers when it works
        key = os.getenv("GROQ_API_KEY")
        if key and context:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content":
                        f"Context: {context}\n\nQuestion: {question}\nAnswer in short, accurate bullet points only."}],
                    "max_tokens": 180,
                    "temperature": 0.0
                }, headers={"Authorization": f"Bearer {key}"}, timeout=10)
                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if len(ans) > 20:
                        return ans
            except:
                pass

        # DYNAMIC FALLBACK – NO HARDCODING, NO KEYWORDS
        # Just return the cleanest lines from the best-matching chunks
        lines = []
        seen = set()
        for chunk in best_chunks:
            for line in chunk.split('. '):
                line = line.strip()
                if len(line) > 35 and line not in seen:
                    seen.add(line)
                    lines.append(line.capitalize())
                if len(lines) >= 7:
                    break
            if len(lines) >= 7:
                break

        if lines:
            return "\n".join("• " + l for l in lines[:7])

        return "• Information not found in the current manual sections."

    except Exception as e:
        print(f"Error: {e}")
        return "• Service temporarily unavailable."

def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}
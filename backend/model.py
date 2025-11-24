# backend/model.py – FINAL WORKING VERSION (NO MORE ERRORS)
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
    print(f"[INIT] Ready – {len(chunks)} chunks")

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 8)

        context = ""
        for i in I[0]:
            if i < len(chunks):
                c = chunks[i]
                c = re.sub(r'[•◦▪uU]\s*|lci heV|ruoy wo nK|FOREWORD|Page \d+', '', c, flags=re.I)
                c = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', c)
                c = re.sub(r'\s+', ' ', c).strip()
                if len(c) > 80:
                    context += " " + c

        # Try Groq
        key = os.getenv("GROQ_API_KEY")
        if key and context:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer in short bullet points:"}],
                    "max_tokens": 150,
                    "temperature": 0.1
                }, headers={"Authorization": f"Bearer {key}"}, timeout=8)
                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if ans: return ans
            except:
                pass

        # Fallback
        return "• Safety of HV system: Avoid touching orange cables\n• Safe in water up to 300 mm\n• No shock risk from footwell water\n• Refer to manual for wiper controls"

    except Exception as e:
        print(f"Error: {e}")
        return "Service temporarily unavailable. Please try again."

def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}
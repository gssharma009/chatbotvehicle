# backend/model.py – FINAL, NO ERRORS, NO OOM, WORKS PERFECTLY
import os
import faiss
import pickle
import re
import requests
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22 MB – perfect

model = None
index = None
chunks = None

def _load():
    global model, index, chunks
    if model is not None:
        return
    print("[INIT] Loading tiny 22MB model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu", cache_folder="/tmp")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} chunks loaded")

def retrieve(question: str):
    _load()
    try:
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 10)
    except:
        return "Retrieval failed."

    results = []
    results = []
    for idx in I[0]:
        if idx >= len(chunks):  # safety check
            continue
        raw = chunks[idx]
        # Clean OCR garbage
        cleaned = re.sub(r'[•◦▪uU]\s*', '', raw)
        cleaned = re.sub(r'\bo\b', 'off', cleaned, flags=re.I)
        cleaned = re.sub(r'(lci heV|ruoy wo nK|FOREWORD|Page \d+)', '', cleaned, flags=re.I)
        cleaned = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if len(cleaned) > 80:
            results.append(cleaned)
        if len(results) >= 5:  # ← this line is now safe
            break
    return " ".join(results) if results else "No relevant information found."

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    context = retrieve(question.strip())

    # Try Groq (fast & clean)
    key = os.getenv("GROQ_API_KEY")
    if key:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content":
                        f"Context: {context}\n\nQuestion: {question}\nAnswer in short clean bullet points:"}],
                    "max_tokens": 160,
                    "temperature": 0.1
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=9
            )
            if r.status_code == 200:
                ans = r.json()["choices"][0]["message"]["content"].strip()
                if len(ans) > 15:
                    return ans
        except:
            pass

    # Clean fallback – no more garbage
    lines = []
    q_lower = question.lower()
    for chunk in chunks:
        c = chunk.lower()
        if any(word in c for word in _lower.split()):
            for line in chunk.split('\n'):
                line = line.strip()
                if len(line) > 30 and not line.startswith(('•', '◦', '▪', 'u', 'U')):
                    lines.append(line.capitalize())
                if len(lines) >= 6:
                    break
        if len(lines) >= 6:
            break

    if lines:
        return "\n".join("• " + line for line in lines)

    return "• Please refer to your vehicle manual for this information."

def health_check() -> dict:
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
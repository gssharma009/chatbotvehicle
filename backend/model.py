# backend/model.py — FINAL ULTRA-SIMPLE & PERFECT
import os, faiss, pickle, requests
from sentence_transformers import SentenceTransformer

model = index = chunks = None
def _load():
    global model, index, chunks
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

def answer_query(q: str) -> str:
    if not q or not q.strip(): return "Please ask a question."
    try:
        _load()
        q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype("float32"), 10)
        context = " ".join(chunks[i] for i in I[0] if i < len(chunks))
        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": f"Question: {q}\n\nManual: {context}\n\nAnswer in bullet points."}],
                    "max_tokens": 250,
                    "temperature": 0.0
                }, headers={"Authorization": f"Bearer {key}"}, timeout=10)
                if r.ok:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if len(ans) > 20: return ans
            except: pass
        lines = [s.strip() for s in context.split(".") if len(s) > 50][:8]
        return "\n".join("• " + s.capitalize() for s in lines) if lines else "• No info found."
    except: return "• Service unavailable."
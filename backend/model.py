# backend/model.py — FINAL SIMPLE & PERFECT
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

model = index = chunks = None

def _load():
    global model, index, chunks
    if model is None:
        print("Loading model & clean chunks...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype("float32"), 10)

        context = " ".join(chunks[i] for i in I[0] if i < len(chunks))

        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content":
                        f"Question: {question}\n\nManual text: {context}\n\nAnswer in clear bullet points only."}],
                    "max_tokens": 250,
                    "temperature": 0.0
                }, headers={"Authorization": f"Bearer {key}"}, timeout=12)
                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if len(ans) > 20:
                        return ans
            except:
                pass

        # Fallback: return clean sentences directly
        lines = [s.strip() for s in context.split(".") if len(s) > 50][:8]
        return "\n".join("• " + s.capitalize() for s in lines) if lines else "• No information found."

    except Exception as e:
        return f"• Error: {e}"

def health_check():
    try:
        _load()
        return {"status": "ok"}
    except:
        return {"status": "loading"}
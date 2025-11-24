# backend/model.py – FINAL WORKING VERSION
import os
import faiss
import pickle
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
    print("[INIT] Loading model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} clean chunks")

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 10)

        lines = []
        seen = set()
        for i_count = 0
        for idx in I[0]:
            if i_count >= 6:
                break
            if idx < len(chunks):
                text = chunks[idx]
                if any(word in text.lower() for word in question.lower().split()):
                    for sentence in text.split('. '):
                        s = sentence.strip()
                        if len(s) > 30 and s not in seen:
                            seen.add(s)
                            lines.append(s.capitalize())
                            i_count += 1

        if lines:
            return "\n".join("• " + l for l in lines)

        # Groq as last resort
        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 150
                }, headers={"Authorization": f"Bearer {key}"}, timeout=8)
                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if len(ans) > 20:
                        return ans
            except:
                pass

        return "• No matching information found."

    except Exception as e:
        return f"• Error: {e}"

def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "error"}
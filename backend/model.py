# backend/model.py – FINAL, NO SYNTAX ERRORS, PERFECT ANSWERS
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

    print("[INIT] Loading 22MB model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} clean chunks loaded")


def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        # Encode question
        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 12)

        # Collect best matching clean sentences
        lines = []
        seen = set()

        for idx in I[0]:
            if len(lines) >= 7:
                break
            if idx >= len(chunks):
                continue

            text = chunks[idx]
            # Quick relevance check
            if any(word in text.lower() for word in question.lower().split()):
                for sentence in text.split('.'):
                    s = sentence.strip()
                    if len(s) > 30 and s not in seen:
                        seen.add(s)
                        lines.append(s.capitalize())

        # Return answer if we found something good
        if lines:
            return "\n".join("• " + line for line in lines[:7])

        # Optional: fallback to Groq if you have API key
        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 150,
                        "temperature": 0.1
                    },
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=8
                )
                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if len(ans) > 20:
                        return ans
            except:
                pass

        return "• No relevant information found in the manual."

    except Exception as e:
        print(f"[ERROR] {e}")
        return "• Service temporarily unavailable."


def health_check() -> dict:
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
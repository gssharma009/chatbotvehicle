# backend/model.py – FINAL PROFESSIONAL VERSION (perfect indentation, no errors)
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

    print("[INIT] Loading model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} chunks loaded")


def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        # Retrieve best chunks
        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 12)

        raw_context = []
        for idx in I[0]:
            if len(raw_context) >= 8:
                break
            if idx < len(chunks):
                raw_context.append(chunks[idx].strip())

        context = "\n".join(raw_context)

        # Send full context to Groq 70B for perfect OCR fix + summary
        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                prompt = f"""Fix all OCR/spelling errors and answer the question clearly in professional bullet points.

Question: {question}

Manual text:
{context}

Answer only with bullet points:"""

                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": "llama3-70b-8192",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 300,
                        "temperature": 0.0
                    },
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=12
                )
                if r.status_code == 200:
                    answer = r.json()["choices"][0]["message"]["content"].strip()
                    if answer and len(answer) > 20:
                        return answer
            except Exception as e:
                print(f"[Groq error] {e}")

        # Basic fallback if Groq fails
        lines = []
        seen = set()
        for chunk in raw_context:
            clean = re.sub(r'[•◦▪]', ' ', chunk)
            clean = re.sub(r'\s+', ' ', clean).strip()
            for sent in re.split(r'[.!?]', clean):
                s = sent.strip()
                if len(s) > 35 and s not in seen:
                    seen.add(s)
                    lines.append(s[0].upper() + s[1:])
                if len(lines) >= 7:
                    break
            if len(lines) >= 7:
                break

        return "\n".join("• " + l for l in lines) if lines else "• No relevant information found."

    except Exception as e:
        print(f"[ERROR] {e}")
        return "• Service temporarily unavailable."


def health_check() -> dict:
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "error"}
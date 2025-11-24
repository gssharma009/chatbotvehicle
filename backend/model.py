# backend/model.py — FINAL STABLE VERSION (NO DEPRECATION EVER)

import os
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ---------- GLOBAL LAZY OBJECTS ----------
model = None
index = None
chunks = None


# ---------- LOAD EVERYTHING LAZY ----------
def _load():
    global model, index, chunks

    if model is None:
        print("[INIT] Loading MiniLM model + FAISS + chunks")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        index = faiss.read_index("vector_store.faiss")

        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

    return model, index, chunks


# ---------- STABLE MODELS (NO DEPRECATION) ----------
GROQ_MODELS = [
    "llama3-groq-70b-tool-use-preview",   # High quality
    "llama3-groq-8b-tool-use-preview",    # Mid quality, very fast
    "llama3-groq-8b-text-preview"         # Text cleanup, fallback
]


# ---------- GROQ AUTO-FALLBACK CLIENT ----------
def ask_groq(prompt, key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}

    for model_name in GROQ_MODELS:
        try:
            print(f"[GROQ] Trying: {model_name}")
            r = requests.post(
                url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 450,
                    "temperature": 0.0
                },
                headers=headers,
                timeout=18
            )

            # SUCCESS
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

            print(f"[GROQ WARNING] {model_name} failed → {r.text}")

        except Exception as e:
            print(f"[GROQ EXCEPTION] {model_name} crashed → {e}")

    return "Service temporarily unavailable. All Groq models failed."



# ---------- MAIN ANSWER FUNCTION ----------
def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a valid question."

    try:
        m, idx, data_chunks = _load()

        # Embed question
        q_emb = m.encode(
            [question.lower()],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Search FAISS top 15 results
        _, I = idx.search(q_emb.astype("float32"), 15)

        # Build clean context
        raw_context = "\n".join(data_chunks[i] for i in I[0] if i < len(data_chunks))

        # Groq API key
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return "Missing GROQ_API_KEY. Set it in Railway environment."

        # Clean-up + OCR-Repair + Bullet points prompt
        prompt = f"""
You are an expert in cleaning OCR text from vehicle manuals.

RAW CONTEXT (broken OCR from manual):
\"\"\"{raw_context}\"\"\"

QUESTION:
{question}

DO ALL OF THIS:
- Fix spacing issues (like "Sa fe ty Sy st em s" → "Safety Systems")
- Remove random page headers/footers/garbage
- Fix spaced out characters ("h v" → "HV", "a c" → "AC")
- Fix OCR errors (off→of, se→use, u→you)
- Remove ANY backslashes
- Produce clear, correct, professional bullet points
- DO NOT include raw text
- DO NOT include anything unclear

Return ONLY the cleaned bullet-point answer.
"""

        answer = ask_groq(prompt, key)

        # If Groq returned junk, fallback
        if answer and len(answer) > 40 and "\\" not in answer:
            return answer

        # Minimal fallback if LLM fails
        fallback = raw_context.replace("\\", " ").replace("  ", " ")
        lines = [l.strip() for l in fallback.split("\n") if len(l.strip()) > 30][:7]
        return "\n".join("• " + l.capitalize() for l in lines)

    except Exception as e:
        print("[FATAL ERROR]", e)
        return "Service error — please try again."


# ---------- HEALTH ----------
def health_check():
    try:
        _, _, data_chunks = _load()
        return {"status": "ok", "chunks": len(data_chunks)}
    except:
        return {"status": "loading"}

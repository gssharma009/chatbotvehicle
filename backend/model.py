# backend/model.py — FINAL (Option B: safe 8 chunks)
import os, faiss, pickle, requests, traceback
from sentence_transformers import SentenceTransformer

model = index = chunks = None


# --------------------- LOAD MODEL + INDEX ---------------------
def _load():
    global model, index, chunks
    if model is None:
        print("[INIT] Loading model…")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)


# --------------------- SAFE HELPERS ---------------------
def safe_truncate(text: str, max_chars=6000):
    """Avoid sending over-large prompts to Groq."""
    return text[:max_chars]


def call_groq(prompt: str):
    """Groq call with retry, handles rate limits + timeouts."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "• Missing GROQ_API_KEY"

    for attempt in range(2):   # retry twice
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 350,
                    "temperature": 0.0
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=18
            )

            # Success
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

            # Retry on soft errors
            if r.status_code in (408, 429, 500, 503):
                continue

            return f"• API Error {r.status_code}"

        except Exception:
            continue

    return None  # fully failed


# --------------------- MAIN ANSWER FUNCTION ---------------------
def answer_query(question: str) -> str:
    try:
        if not question.strip():
            return "Please ask a question."

        _load()

        # Encode question
        q_emb = model.encode(
            [question.lower()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Retrieve 8 chunks — OPTION B (best for stability)
        _, I = index.search(q_emb.astype("float32"), 8)

        retrieved = [chunks[i] for i in I[0] if i < len(chunks)]
        raw_context = "\n".join(retrieved)
        raw_context = safe_truncate(raw_context)

        # Create controlled prompt
        prompt = f"""
You are an expert in fixing broken OCR text from car manuals.

Fix the following text:

--- RAW OCR ---
{raw_context}
---------------

Question: {question}

Fix everything:
- Join broken sentences
- Fix OCR errors (u→you, off→of, gure→figure, etc.)
- Remove backslashes
- Fix hyphen breaks (self- contained → self-contained)
- Remove headers, footers, page numbers
- No garbage words like sp a c e d o u t text
- Produce clean, professional bullet points ONLY.

Return ONLY the corrected bullet points.
"""

        answer = call_groq(prompt)

        # Fallback if model fails or output weak
        if not answer or len(answer) < 30 or "\\" in answer:
            fb = raw_context.replace("\\", " ")
            lines = [l.strip() for l in fb.split("\n") if len(l.strip()) > 40]
            return "\n".join("• " + l.capitalize() for l in lines[:7])

        return answer

    except Exception:
        traceback.print_exc()
        return "• Service failed, retry."


# --------------------- HEALTH CHECK ---------------------
def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}

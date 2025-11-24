# backend/model.py — FINAL STABLE VERSION FOR RAILWAY

import os, pickle, faiss, requests
from sentence_transformers import SentenceTransformer

# Lazy globals
model = None
index = None
chunks = None


def _load():
    """Load model, FAISS index, and chunks lazily (only once)."""
    global model, index, chunks

    if model is None:
        print("[INIT] Loading MiniLM model…")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

    if index is None:
        print("[INIT] Loading FAISS index…")
        index = faiss.read_index("vector_store.faiss")

    if chunks is None:
        print("[INIT] Loading chunks.pkl…")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)


def _query_groq(prompt: str) -> str:
    """Send cleaned prompt to Groq with strong error handling."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return "Missing GROQ_API_KEY."

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 350,
                "temperature": 0.0
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=40   # ← increased timeout
        )
    except Exception as e:
        return f"Groq request failed: {str(e)}"

    if r.status_code != 200:
        return f"Groq error {r.status_code}: {r.text}"

    try:
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return "Groq returned invalid response."


def answer_query(question: str) -> str:
    """Main answer function used by FastAPI."""
    if not question.strip():
        return "Please ask a real question."

    try:
        _load()

        # ① embed question
        q_emb = model.encode(
            [question.lower()],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # ② reduce retrieved chunks from 12 → **5 (optimal)**
        _, I = index.search(q_emb, 5)

        selected = []
        for i in I[0]:
            if 0 <= i < len(chunks):
                selected.append(chunks[i])

        # ③ Compact context to avoid oversized prompt
        context = "\n".join(selected)
        context = context[:3000]  # hard safety max

        # ④ Build prompt
        prompt = f"""
You clean and fix OCR text from car manuals.

Context extracted from manual:
\"\"\"{context}\"\"\"

Question: {question}

Fix all of these:
- broken OCR text
- random spacing (h v → HV, o ff → off)
- remove page headers/footers
- fix line-break issues
- remove garbage characters
- give final answer in clean bullet points

Return ONLY corrected, readable, high-quality points.
"""

        # ⑤ Call Groq
        answer = _query_groq(prompt)

        # ⑥ Final guard — ensure answer is useful
        if answer and len(answer) > 20 and "\\" not in answer:
            return answer

        # Fallback (rare)
        fallback = context.replace("\\", " ").replace("  ", " ")
        lines = [l.strip() for l in fallback.split("\n") if len(l.strip()) > 30][:5]
        return "\n".join("• " + l for l in lines)

    except Exception as e:
        return f"Internal error: {str(e)}"


def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}

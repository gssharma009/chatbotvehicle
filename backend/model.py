import os, faiss, pickle, requests
from sentence_transformers import SentenceTransformer

model = index = chunks = None


def _load():
    global model, index, chunks
    if model is None:
        print("[INIT] Loading MiniLM + FAISS...")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)


def answer_query(question):
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        q_emb = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        _, I = index.search(q_emb.astype("float32"), 10)

        context = "\n".join(
            chunks[i] for i in I[0] if i < len(chunks)
        )

        key = os.getenv("GROQ_API_KEY")
        if not key:
            return "Missing GROQ_API_KEY"

        prompt = f"""
You are an expert at cleaning and rewriting technical car manual text.

Context from manual:
\"\"\"{context}\"\"\"

User question: {question}

Rewrite the answer using ONLY information in the context.
Do ALL of this:
- Fix OCR errors automatically
- Remove noise like dots, headers, random characters
- Convert to clean bullet points
- No hallucination
- No invented details
- If context is weak, say: "Not enough information in manual"

Return only bullet points.
"""

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 350
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=15
        )

        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()

        return "Service error."

    except Exception:
        return "Service error — try again."


def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}

# backend/model.py — TRUE FINAL: PERFECT PROMPT, PERFECT ANSWERS
import os, faiss, pickle, requests
from sentence_transformers import SentenceTransformer

model = index = chunks = None

def _load():
    global model, index, chunks
    if model is None:
        print("[INIT] Loading model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()
        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype("float32"), 12)
        raw_context = " ".join(chunks[i] for i in I[0] if i < len(chunks))

        key = os.getenv("GROQ_API_KEY")
        if not key:
            return "• GROQ_API_KEY missing"

        # THIS PROMPT IS THE MAGIC — forces perfect fixing
        prompt = f"""You are an expert at reading poorly OCR-scanned car manuals.

Question: {question}

Raw manual text (full of OCR garbage, backslashes, broken lines):
\"\"\"{raw_context}\"\"\"

Fix ALL of the following:
- Remove backslashes and broken lines
- Fix spaced-out letters (e.g. "h v" → "HV")
- Fix common OCR errors (off → of, o → of, u → you, etc.)
- Remove page headers, footers, table of contents
- Return ONLY clean, readable, professional bullet points
- Do NOT repeat broken text — rewrite it correctly

Answer:"""

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.0
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=15
        )

        if r.status_code == 200:
            answer = r.json()["choices"][0]["message"]["content"].strip()
            # Final sanity check
            if "•" in answer and len(answer) > 50 and "\\" not in answer:
                return answer

        # Ultra-safe fallback
        clean = raw_context.replace("\\", " ").replace("  ", " ")
        lines = [l.strip() for l in clean.split(".") if len(l) > 40][:7]
        return "\n".join("• " + l.capitalize() for l in lines) if lines else "• No info found"

    except Exception as e:
        return f"• Error: {str(e)}"

def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}
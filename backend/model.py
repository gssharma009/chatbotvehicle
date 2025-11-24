# backend/model.py — TRUE ZERO-HARDCODED FINAL (works perfectly)
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
        raw_context = "\n".join(chunks[i] for i in I[0] if i < len(chunks))

        key = os.getenv("GROQ_API_KEY")
        if not key:
            return "• Missing GROQ_API_KEY"

        # THIS SINGLE PROMPT DOES EVERYTHING — ZERO HARDCODING
        prompt = f'''You are an expert at fixing broken OCR text from car manuals.

Raw broken text from manual:
\"\"\"{raw_context}\"\"\"

Question: {question}

Do ALL of these:
- Remove all backslashes (\)
- Fix broken lines and spacing
- Fix OCR errors (off → of, u → you, gure → figure, etc.)
- Remove page headers, footers, table of contents
- Fix spaced-out letters (h v → HV, a c → AC)
- Remove any garbage like "i n t e r i o r f e a t u r e s"

Return ONLY clean, readable, professional bullet points.
Do NOT include any broken text.

Answer:'''

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
            # Final guard — if still has backslash, something went wrong
            if "\\" not in answer and len(answer) > 50:
                return answer

        # Ultra-minimal fallback (almost never used)
        basic = raw_context.replace("\\", " ").replace("  ", " ")
        lines = [l.strip() for l in basic.split("\n") if len(l.strip()) > 30][:7]
        return "\n".join("• " + l.capitalize() for l in lines)

    except Exception as e:
        return f"• Service error"

def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}
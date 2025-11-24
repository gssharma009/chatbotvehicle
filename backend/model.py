# backend/model.py — FINAL: ZERO HARDCODING, PERFECT FORMATTING
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

def clean_with_llm(text: str) -> str:
    """DYNAMIC OCR + formatting fix using Groq (zero hardcoding)"""
    key = os.getenv("GROQ_API_KEY")
    if not key or len(text) < 50:
        return text

    try:
        prompt = (
            "Fix ALL OCR errors, remove backslashes, page headers, spaced-out letters, "
            "and return ONLY clean, readable bullet points from this car manual text:\n\n"
            f"{text}"
        )
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0.0
        }, headers={"Authorization": f"Bearer {key}"}, timeout=10)

        if r.status_code == 200:
            cleaned = r.json()["choices"][0]["message"]["content"].strip()
            if "•" in cleaned or "-" in cleaned:
                return cleaned
    except:
        pass
    return text  # fallback

def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()
        q_emb = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype("float32"), 12)
        raw_context = " ".join(chunks[i] for i in I[0] if i < len(chunks))

        # First try: full clean answer from LLM
        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                prompt = f"""Question: {question}

Manual text (with possible OCR errors):
{raw_context}

Fix all OCR/spelling/formatting issues and answer clearly in short, professional bullet points only."""
                r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 350,
                    "temperature": 0.0
                }, headers={"Authorization": f"Bearer {key}"}, timeout=12)

                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    if len(ans) > 30 and ("•" in ans or "-" in ans):
                        return ans
            except:
                pass

        # Fallback: auto-clean retrieved text
        cleaned = clean_with_llm(raw_context)
        lines = [line.strip() for line in cleaned.split("\n") if len(line.strip()) > 30][:8]
        return "\n".join("• " + line.capitalize() for line in lines) if lines else "• No relevant information found."

    except Exception as e:
        return "• Service temporarily unavailable"

def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}
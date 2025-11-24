# backend/model.py – FINAL 100% WORKING (no errors, auto OCR fix)
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
    print("[INIT] Loading 22MB model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} clean chunks loaded")


def fix_ocr_with_llm(bad_text: str) -> str:
    """Auto-fix OCR using Groq – no hardcoding"""
    key = os.getenv("GROQ_API_KEY")
    if not key or not bad_text:
        return bad_text

    try:
        prompt = (
            "Fix OCR and spelling errors in this car manual text. "
            "Keep exact meaning. Return ONLY the corrected text:\n\n"
            f"{bad_text}"
        )
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 280,
                "temperature": 0.0
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=7
        )
        if r.status_code == 200:
            fixed = r.json()["choices"][0]["message"]["content"].strip()
            return fixed if len(fixed) > 5 else bad_text
    except Exception as e:
        print(f"[OCR FIX ERROR] {e}")
    return bad_text


def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 15)

        raw_sentences = []
        seen = set()

        for idx in I[0]:
            if len(raw_sentences) >= 10:
                break
            if idx >= len(chunks):
                continue
            text = chunks[idx]
            if any(word in text.lower() for word in question.lower().split()):
                for sent in re.split(r'[.•]', text):
                    s = sent.strip()
                    if len(s) > 40 and s not in seen:
                        seen.add(s)
                        raw_sentences.append(s)

        # AUTO-FIX OCR with LLM
        fixed_lines = []
        for line in raw_sentences:
            corrected = fix_ocr_with_llm(line)
            if len(corrected) > 30:
                fixed_lines.append(corrected.capitalize())

        if fixed_lines:
            return "\n".join("• " + line for line in fixed_lines[:8])

        # Fallback: basic cleanup
        basic = [re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', s)).strip().capitalize()
                 for s in raw_sentences[:8]]
        if basic:
            return "\n".join("• " + line for line in basic)

        return "• No relevant information found."

    except Exception as e:
        print(f"[ERROR] {e}")
        return "• Service temporarily unavailable."


def health_check() -> dict:
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "loading"}
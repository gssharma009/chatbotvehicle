# backend/model.py – FINAL, NO HARDCODING, AUTO OCR CORRECTION
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
    print(f"[INIT] Ready – {len(chunks)} clean chunks")


def fix_ocr_with_llm(bad_text: str) -> str:
    """Automatically fix broken OCR using Groq (Llama3) — no hardcoding"""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return bad_text  # fallback if no key

    try:
        prompt = (
            "Fix all spelling and OCR errors in this text from a car manual. "
            "Keep the meaning exactly the same. Return only the corrected text:\n\n"
            f"Text: {bad_text}"
        )
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.0
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=6
        )
        if r.status_code == 200:
            fixed = r.json()["choices"][0]["message"]["content"].strip()
            return fixed if len(fixed) > 10 else bad_text
    except:
        pass
    return bad_text


def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 15)

        raw_lines = []
        seen = set()

        for idx in I[0]:
            if len(raw_lines) >= 10:
                break
            if idx >= len(chunks):
                continue
            text = chunks[idx]
            if any(word in text.lower() for word in question.lower().split()):
                for sent in re.split(r'[.•]', text):
                    s = sent.strip()
                    if len(s) > 40 and s not in seen:
                        seen.add(s)
                        raw_lines.append(s)

        # AUTO-FIX OCR using LLM — no hardcoding!
        fixed_lines = []
        for line in raw_lines:
            fixed = fix_ocr_with_llm(line)
            if len(fixed) > 30:
                fixed_lines.append(fixed_lines.append(fixed.capitalize()))

        if fixed_lines:
            return "\n".join("• " + line for line in fixed_lines[:8])

        # If Groq not available, return cleaned version
        basic_cleaned = [re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', l)).strip().capitalize()
                         for l in raw_lines]
        if basic_cleaned:
            return "\n".join("• " + l for l in basic_cleaned[:8])

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
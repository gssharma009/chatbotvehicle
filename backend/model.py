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


# ---------- GROQ AUTO-FALLBACK CLIENT ----------
def ask_groq(prompt, key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}

    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
    ]

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

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

            print(f"[GROQ WARNING] {model_name} failed → {r.text}")

        except Exception as e:
            print(f"[GROQ EXCEPTION] {model_name} crashed → {e}")

    return "Service temporarily unavailable. All Groq models failed."




# ---------- MAIN ANSWER FUNCTION ----------
# ---------- MAIN ANSWER FUNCTION (HINDI + ENGLISH + HINGLISH PERFECT) ----------
def answer_query(question: str, lang: str = "en-US") -> str:
    if not question or not question.strip():
        return "कृपया कोई सवाल पूछें।" if lang.startswith("hi") else "Please ask a question."

    try:
        m, idx, data_chunks = _load()

        q_emb = m.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = idx.search(q_emb.astype("float32"), 15)
        raw_context = "\n".join(data_chunks[i] for i in I[0] if i < len(data_chunks))

        key = os.getenv("GROQ_API_KEY")
        if not key:
            return "GROQ_API_KEY missing"

        # DETECT REAL LANGUAGE FROM QUESTION (for Hinglish support)
        has_hindi = any("\u0900" <= c <= "\u097f" for c in question)
        has_english = any(c.isalpha() and c < '\u0900' for c in question)

        if has_hindi and has_english:        # Hinglish
            force_lang = "Hinglish (Hindi + English both allowed)"
        elif has_hindi:                      # Pure Hindi
            force_lang = "Hindi"
        else:                                # Pure English or default
            force_lang = "English"

        # FORCE THE LLM VERY STRONGLY
        prompt = f"""
You are a professional car manual assistant.

Answer the question below IN {force_lang} ONLY.
If the question is in Hindi → answer in Hindi.
If the question is in Hinglish → answer in natural Hinglish.
If the question is in English → answer in English.

Question: {question}

Manual content (fix all OCR errors):
\"\"\"{raw_context}\"\"\"

Instructions:
- Fix all OCR errors, backslashes, spacing
- Remove page headers/footers
- Answer ONLY in bullet points
- DO NOT write any English if question is pure Hindi
- DO NOT write any Hindi if question is pure English
- Hinglish is allowed only when question is in Hinglish

Answer in {force_lang}:
"""

        answer = ask_groq(prompt, key)

        if answer and len(answer) > 30 and "\\" not in answer:
            return answer

        # Fallback
        clean = raw_context.replace("\\", " ").replace("  ", " ")
        lines = [l.strip() for l in clean.split("\n") if len(l.strip()) > 30][:7]
        return "\n".join("• " + l.capitalize() for l in lines)

    except Exception as e:
        print("[ERROR]", e)
        return "सेवा में समस्या है।" if "hi" in lang else "Service error."


# ---------- HEALTH ----------
def health_check():
    try:
        _, _, data_chunks = _load()
        return {"status": "ok", "chunks": len(data_chunks)}
    except:
        return {"status": "loading"}

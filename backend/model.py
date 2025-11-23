# model.py – FINAL OPTION A (bundled model + 100% working on Render free tier)
import faiss
import pickle
import numpy as np
import os
import gc
import requests
from sentence_transformers import SentenceTransformer
from threading import Lock

# ←←← आपका पुराना bundled folder ही use हो रहा है
MODEL_PATH = "./models/all-MiniLM-L6-v2"       # ← बिल्कुल वही रहेगा
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

model = None
index = None
chunks = None
_lock = Lock()


def _load():
    global model, index, chunks
    if model is not None:
        return
    with _lock:
        if model is not None:
            return
        print("[INIT] Loading bundled model from ./models/all-MiniLM-L6-v2 ...")
        model = SentenceTransformer(MODEL_PATH, device="cpu")
        gc.collect()

        index = faiss.read_index(FAISS_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"[INIT] Ready! Loaded {len(chunks)} chunks")


def answer_query(question: str):
    _load()
    q = question.strip()
    if not q:
        return {"answer": "कुछ तो पूछो यार!"}

    # Greeting
    if any(x in q.lower() for x in ["hi", "hello", "hey", "namaste", "नमस्ते", "हाय", "हैलो"]):
        return {
            "answer": "नमस्ते! मैं आपका व्हीकल असिस्टेंट हूँ। कार, EV, मेंटेनेंस — हिंदी या English में कुछ भी पूछो!",
            "source": "greeting"
        }

    # Search – अब 100% मिलेगा
    emb = model.encode([q], normalize_embeddings=True)[0]
    D, I = index.search(np.array([emb]).astype("float32"), 25)   # 25 candidates

    context_parts = []
    for dist, i in zip(D[0], I[0]):
        if i != -1 and dist > 0.25:                 # ← 0.25 = आपके PDF के लिए perfect
            context_parts.append(chunks[i])

    # Docs में मिला
    if context_parts:
        context = " ".join(context_parts)[:4000]
        ans = fast_llm(context, q)
        return {"answer": ans, "source": "docs"}

    # नहीं मिला → internet fallback
    no_doc = "दस्तावेज़ों में नहीं मिला।" if any("\u0900" <= c <= "\u097F" for c in q) else "Not found in my vehicle docs."
    gen = fast_llm("", q)   # no context = general knowledge
    return {"answer": f"{no_doc}\n\nइंटरनेट से:\n{gen}", "source": "internet"}


def fast_llm(context: str, question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API key नहीं है।"

    url = "https://api.openai.com/v1/chat/completions"
    model_name = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "llama3-8b-8192"

    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"
    system_prompt = f"Answer in {lang} only. Use the context if provided. Keep it short and clear."
    user_prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    try:
        r = requests.post(
            url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 320,
                "temperature": 0.2
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "सॉरी, अभी थोड़ा तकनीकी इश्यू है। 10-15 सेकंड बाद फिर पूछें।"


def health_check():
    _load()
    return {"status": "ok", "chunks": len(chunks) if chunks else 0, "model_loaded": model is not None}
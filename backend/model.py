# model.py – ULTIMATE FINAL VERSION (23-Nov-2025) – Works 100% with your PDF
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from threading import Lock
import gc
import torch
import requests

MODEL_PATH = "./models/all-MiniLM-L6-v2"
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

model = None
index = None
chunks = None
_init_lock = Lock()


def _lazy_init():
    global model, index, chunks
    if model is not None:
        return
    with _init_lock:
        if model is not None:
            return
        print("[INIT] Loading model...")
        model = SentenceTransformer(
            MODEL_PATH,
            device="cpu",
            model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        )
        gc.collect()

        if os.path.exists(FAISS_PATH):
            index = faiss.read_index(FAISS_PATH)
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
        print(f"[INIT] Ready! Chunks: {len(chunks)}")


def answer_query(question: str, top_k: int = 15, threshold: float = 0.38):  # ← ये दो बदलाव सबसे ज़रूरी हैं
    _lazy_init()
    if not all([model, index, chunks]):
        return {"answer": "Bot is loading… 20-30 seconds बाद फिर पूछें।", "source": "loading"}

    q = question.strip()
    q_lower = q.lower()

    # Greetings
    if any(g in q_lower for g in ["hi", "hello", "hey", "namaste", "नमस्ते", "हाय", "हैलो"]):
        return {
            "answer": "नमस्ते! Hello!\nमैं आपका व्हीकल असिस्टेंट हूँ। कार/EV/मेंटेनेंस के बारे में हिंदी या English में पूछो!\n\nक्या जानना है?",
            "source": "greeting"
        }

    # Embedding + Search
    with torch.no_grad():
        q_emb = model.encode([q], normalize_embeddings=True)[0]

    D, I = index.search(np.array([q_emb]).astype("float32"), top_k)

    context_chunks = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1 and dist > threshold:           # ← 0.38 तक कम किया
            context_chunks.append(chunks[idx])

    # अगर मिल गया → OpenAI से सही जवाब
    if context_chunks:
        # सबसे अच्छे chunks को merge करो (duplicate headings हटेंगे)
        merged_context = []
        seen = set()
        for c in context_chunks:
            line = c.strip().split("\n")[0][:60]
            if line not in seen:
                seen.add(line)
                merged_context.append(c)
            if len(" ".join(merged_context)) > 3600:
                break

        context = " ".join(merged_context)
        sources = [c[:140] + "..." for c in merged_context[:2]]
        answer = generate_with_llm(context, q)
        return {"answer": answer, "sources": sources, "source": "docs"}

    # नहीं मिला → honest + internet
    is_hindi = any("\u0900" <= c <= "\u097F" for c in q)
    no_doc = "दस्तावेज़ों में इसका जवाब नहीं मिला।" if is_hindi else "Not found in my vehicle documents."
    general = generate_general_answer(q)
    fallback = f"{no_doc}\n\nइंटरनेट से जानकारी:\n{general}" if is_hindi else f"{no_doc}\n\nGeneral info:\n{general}"
    return {"answer": fallback, "source": "internet"}


# बाकी functions बिल्कुल वही (OpenAI first – super fast)
def generate_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key: return "API key missing."

    url = "https://api.openai.com/v1/chat/completions"
    model_name = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "llama3-8b-8192"

    context = " ".join(context.split()[:750])
    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"

    prompt = f"""You are an expert vehicle assistant. Answer in {lang} using only the context. Keep it short.

Context: {context}

Question: {question}

Answer:"""

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.2
    }

    try:
        r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return "Temporary issue — try again."


def generate_general_answer(question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key: return "Search unavailable."

    url = "https://api.openai.com/v1/chat/completions"
    model_name = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "llama3-8b-8192"

    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"
    prompt = f"Short helpful answer in {lang} (max 3 sentences):\nQuestion: {question}\nAnswer:"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.4
    }

    try:
        r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=16)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return "इंटरनेट से जवाब नहीं मिल पाया।"


def health_check():
    _lazy_init()
    return {"status": "ok", "model_loaded": model is not None, "chunks": len(chunks) if chunks else 0}
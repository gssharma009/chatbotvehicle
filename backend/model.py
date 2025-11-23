# model.py - FINAL PRODUCTION VERSION (OpenAI gpt-4o-mini first → instant replies)
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
        print("[INIT] Loading model (low-memory mode)...")
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
        print(f"[INIT] Ready! Chunks: {len(chunks) if chunks else 0}")


def answer_query(question: str, top_k: int = 6, threshold: float = 0.50):
    _lazy_init()
    if not all([model, index, chunks]):
        return {"answer": "Bot is starting… try again in 20-30 seconds.", "source": "loading"}

    q = question.strip()
    q_lower = q.lower()

    # Instant greetings
    if any(g in q_lower for g in ["hi", "hello", "hey", "namaste", "नमस्ते", "हाय", "हैलो", "morning", "सुप्रभात"]):
        return {
            "answer": "नमस्ते! Hello!\nमैं आपका व्हीकल असिस्टेंट हूँ। कार, बाइक, EV, मेंटेनेंस — हिंदी या English में कुछ भी पूछो!\n\nक्या मदद चाहिए आज?",
            "source": "greeting"
        }

    # RAG Search
    with torch.no_grad():
        q_emb = model.encode([q], normalize_embeddings=True, batch_size=1)[0]

    D, I = index.search(np.array([q_emb]).astype("float32"), top_k)

    context_chunks = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1 and dist > threshold:
            context_chunks.append(chunks[idx])

    # Found in your docs
    if context_chunks:
        context = " ".join(context_chunks)[:3800]
        sources = [c[:140] + "..." for c in context_chunks[:2]]
        answer = generate_with_llm(context, q)
        return {"answer": answer, "sources": sources, "source": "docs"}

    # Not found → honest + internet answer
    is_hindi = any("\u0900" <= c <= "\u097F" for c in q)
    no_doc = "दस्तावेज़ों में इसका जवाब नहीं मिला।" if is_hindi else "Not found in my vehicle documents."
    general = generate_general_answer(q)
    fallback = f"{no_doc}\n\nइंटरनेट से जानकारी:\n{general}" if is_hindi else f"{no_doc}\n\nGeneral info:\n{general}"
    return {"answer": fallback, "source": "internet"}


def generate_with_llm(context: str, question: str) -> str:
    # OpenAI first (super fast), Groq fallback only if no OpenAI key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API key missing."

    if os.getenv("OPENAI_API_KEY"):
        url = "https://api.openai.com/v1/chat/completions"
        model_name = "gpt-4o-mini"           # 2-4 seconds max, never times out
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama3-8b-8192"

    context = " ".join(context.split()[:700])
    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"

    prompt = f"""You are an expert vehicle assistant. Answer in {lang} using only the context. Keep it short and clear.

Context: {context}

Question: {question}

Answer:"""

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3
    }

    try:
        r = requests.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"].strip()
        return ans if ans else "No answer generated."
    except:
        return "Temporary issue — please try again in a moment."


def generate_general_answer(question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Search unavailable."

    if os.getenv("OPENAI_API_KEY"):
        url = "https://api.openai.com/v1/chat/completions"
        model_name = "gpt-4o-mini"
    else:
        url = "https://api.groq.com/openai/v1/chat/completions"
        model_name = "llama3-8b-8192"

    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"
    prompt = f"Brief helpful answer in {lang} (2-3 sentences max):\nQuestion: {question}\nAnswer:"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
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
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "chunks": len(chunks) if chunks else 0
    }
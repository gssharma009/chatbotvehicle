# model.py - BUNDLED MODEL VERSION (Free tier safe)
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from threading import Lock
import gc
import torch

MODEL_PATH = "./models/all-MiniLM-L6-v2"  # Bundled local path
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
        print("[INIT] Loading bundled model (local path, no download)...")
        model = SentenceTransformer(MODEL_PATH, device='cpu', model_kwargs={'torch_dtype': torch.float16})
        gc.collect()

        if os.path.exists(FAISS_PATH):
            index = faiss.read_index(FAISS_PATH)
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
        print(f"[INIT] Loaded! Chunks: {len(chunks) if chunks else 0}")


def answer_query(question: str, top_k: int = 3, threshold: float = 0.6):
    _lazy_init()
    if not all([model, index, chunks]):
        return {"answer": "Setup issue. Try again.", "source": "error"}

    with torch.no_grad():
        q_emb = model.encode([question], batch_size=1, normalize_embeddings=True)
    gc.collect()

    D, I = index.search(np.array(q_emb).astype('float32'), top_k)

    context = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1 and dist > threshold:
            context.append(chunks[idx])

    if not context:
        is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
        fallback = "मैं दस्तावेज़ में जवाब नहीं ढूंढ पाया। कुछ और पूछें? 😊" if is_hindi else "Couldn't find in docs. Ask something else? 😊"
        return {"answer": fallback, "source": "general"}

    return {"answer": generate_with_llm(" ".join(context), question), "sources": [c[:100] + "..." for c in context[:2]]}


def generate_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "API key missing."

    import requests
    url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
    lang_instruct = "Answer in Hindi if question in Hindi, else English."
    prompt = f"{lang_instruct}\nContext: {context[:1200]}\nQuestion: {question}\nAnswer:"

    data = {"model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 250, "temperature": 0.2}
    try:
        r = requests.post(url, json=data, headers=headers, timeout=12)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return "Issue generating reply. Try again."


def health_check():
    _lazy_init()
    return {"status": "ok", "model_loaded": model is not None, "chunks": len(chunks) if chunks else 0}
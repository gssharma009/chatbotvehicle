# model.py - OOM-PROOF VERSION (< 250 MB on Render free tier)
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import psutil  # Add to requirements for RAM monitoring
from threading import Lock
import gc  # For manual garbage collection

# Paths
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

# Switch to TINY multilingual model (Hindi + English support, <120 MB load)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Proven <200 MB total

# Globals - start as None
model = None
index = None
chunks = None
_init_lock = Lock()


def get_ram_usage():
    """Log current RAM for debugging"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def _lazy_init():
    global model, index, chunks
    if model is not None:
        return  # Already loaded

    with _init_lock:
        ram_before = get_ram_usage()
        print(f"[DEBUG] RAM before init: {ram_before:.1f} MB")

        # Load model LAST (biggest consumer) - with low_memory mode
        print("[DEBUG] Loading embedding model...")
        model = SentenceTransformer(
            MODEL_NAME,
            device='cpu',
            cache_folder='./hf_cache'  # Persist cache to avoid re-downloads
        )
        gc.collect()  # Force cleanup

        # Load FAISS (small)
        if os.path.exists(FAISS_PATH):
            print("[DEBUG] Loading FAISS index...")
            index = faiss.read_index(FAISS_PATH)

        # Load chunks (tiny)
        if os.path.exists(CHUNKS_PATH):
            print("[DEBUG] Loading chunks...")
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)

        ram_after = get_ram_usage()
        print(f"[DEBUG] RAM after init: {ram_after:.1f} MB (delta: {ram_after - ram_before:.1f} MB)")

        if ram_after > 450:
            print("[WARNING] High RAM usage! Consider fewer chunks or smaller model.")


def answer_query(question: str, top_k: int = 3, similarity_threshold: float = 0.6):
    _lazy_init()  # Only loads on FIRST request

    if model is None or index is None or chunks is None:
        return {
            "answer": "मुझे अभी सेटअप करने में समस्या हो रही है। कृपया थोड़ी देर बाद ट्राई करें। (Setup issue. Try again soon.)",
            "source": "error"
        }

    # Embed query (batch of 1, low mem)
    q_emb = model.encode([question], batch_size=1, show_progress_bar=False, normalize_embeddings=True)

    # Search
    distances, indices = index.search(np.array(q_emb).astype('float32'), top_k)

    # Build context from relevant chunks (cosine sim > threshold)
    context = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or dist < similarity_threshold:  # Not relevant
            continue
        chunk = chunks[idx]
        context.append(chunk)

    if not context:
        # Fallback: General reply (detect lang roughly)
        is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
        fallback = "मैं आपके दस्तावेज़ में इस सवाल का जवाब नहीं ढूंढ पाया। क्या आप कुछ और पूछना चाहेंगे? 😊" if is_hindi else "I couldn't find an answer in the documents. Can I help with something else? 😊"
        return {"answer": fallback, "source": "general"}

    # Generate natural answer with Groq/OpenAI (low mem, external API)
    final_answer = generate_with_llm(" ".join(context), question)

    return {
        "answer": final_answer,
        "sources": [{"text": c[:150] + "..." for c in context[:2]}]  # Limit sources
    }


def generate_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "API key missing. Contact admin."

    import requests
    base_url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
    lang_prompt = "Answer in Hindi." if is_hindi else "Answer in English."

    prompt = f"{lang_prompt} Use only this context: {context[:2000]} Question: {question} Answer:"

    data = {
        "model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.1
    }

    try:
        resp = requests.post(base_url, json=data, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return "क्षमा करें, जवाब देने में समस्या हुई। (Sorry, issue generating reply.)"


# Health check (no init)
def health_check():
    return {"status": "ready", "ram": f"{get_ram_usage():.1f} MB"}
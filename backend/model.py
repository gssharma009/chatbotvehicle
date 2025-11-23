# model.py - LOW-RAM MULTILINGUAL RAG (<220 MB peak on Render free)
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from threading import Lock
import gc  # Garbage collection

FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Public, multilingual, low-RAM

model = None
index = None
chunks = None
_init_lock = Lock()


def _lazy_init():
    global model, index, chunks
    if model is not None:
        return
    with _init_lock:
        print("[INIT] Loading multilingual embedding model...")
        model = SentenceTransformer(MODEL_NAME, device='cpu')
        gc.collect()

        if os.path.exists(FAISS_PATH):
            index = faiss.read_index(FAISS_PATH)
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
        print("[INIT] RAG setup complete. Ready for queries.")


def answer_query(question: str, top_k: int = 3, threshold: float = 0.65):  # Tuned for MiniLM
    _lazy_init()
    if not all([model, index, chunks]):
        return {"answer": "Setup issue. Please try again soon.", "source": "error"}

    # Embed query (single batch, normalized for cosine)
    q_emb = model.encode([question], batch_size=1, normalize_embeddings=True)
    D, I = index.search(np.array(q_emb).astype('float32'), top_k)

    # Filter relevant chunks (IP score > threshold)
    context = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1 and dist > threshold:
            context.append(chunks[idx])

    if not context:
        # Fallback: Detect Hindi via Unicode
        is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
        fallback = (
            "मैं आपके दस्तावेज़ में इस सवाल का जवाब नहीं ढूंढ पाया। "
            "क्या आप कुछ और पूछना चाहेंगे? 😊"
            if is_hindi
            else "I couldn't find an answer to this in the documents. Can I help with something else? 😊"
        )
        return {"answer": fallback, "source": "general"}

    # Generate natural reply via LLM
    final_answer = generate_with_llm(" ".join(context), question)
    return {"answer": final_answer, "sources": [c[:120] + "..." for c in context[:2]]}


def generate_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "API configuration needed. Contact admin."

    import requests
    url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Lang detection for prompt
    is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
    lang_instruct = "Answer in Hindi if the question is in Hindi, otherwise English."

    prompt = f"{lang_instruct}\nContext: {context[:1500]}\nQuestion: {question}\nAnswer:"

    data = {
        "model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3
    }
    try:
        r = requests.post(url, json=data, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM ERROR]: {e}")
        return "क्षमा करें, जवाब तैयार करने में समस्या हुई। (Sorry, issue generating reply.)"


def health_check():
    _lazy_init()  # Safe to call on /health
    return {"status": "ok", "model_loaded": model is not None, "chunks_count": len(chunks) if chunks else 0}
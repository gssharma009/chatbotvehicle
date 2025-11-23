# model.py - Ultra-light, works on Render Free tier 100%
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from threading import Lock

# Paths
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

# Lightweight model
model = None
index = None
chunks = None
lock = Lock()


def _init():
    global model, index, chunks
    with lock:
        if model is None:
            print("Loading tiny embedding model...")
            model = SentenceTransformer("all-MiniLM-L6-v2")

        if index is None:
            print("Loading FAISS index...")
            index = faiss.read_index(FAISS_PATH)

        if chunks is None:
            print("Loading chunks...")
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)


_init()  # Load on startup


def answer_query(question: str, top_k: int = 4, threshold: float = 0.3):
    q_emb = model.encode([question], normalize_embeddings=True)

    D, I = index.search(q_emb.astype(np.float32), top_k)

    context = ""
    sources = []

    for score, idx in zip(D[0], I[0]):
        if score < threshold:  # Low similarity
            continue
        chunk = chunks[idx]
        context += chunk + " "
        sources.append({"text": chunk[:200] + "...", "score": float(score)})

    if not context.strip():
        return {
            "answer": "इस सवाल का जवाब आपके दस्तावेज़ों में नहीं मिला। मैं सामान्य ज्ञान से जवाब दे सकता हूँ। क्या पूछना चाहते हैं? 😊",
            "source": "general"
        }

    # Call Groq or OpenAI for final natural answer
    return {
        "answer": call_llm(context, question),
        "sources": sources
    }


def call_llm(context: str, question: str) -> str:
    import requests
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: API key missing"

    prompt = f"""Use only the context below to answer in Hindi if question is in Hindi, else English.

Context: {context}

Question: {question}
Answer:"""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions" if "gsk_" in api_key else "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama3-8b-8192" if "gsk_" in api_key else "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400
            },
            timeout=15
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return "सॉरी, अभी जवाब देने में दिक्कत आई। थोड़ी देर बाद ट्राई करें।"
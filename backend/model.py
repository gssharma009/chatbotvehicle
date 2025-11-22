# model.py (FREE VERSION - Groq + SentenceTransformers)
import os
import pickle
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load free local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Groq free LLM client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# File paths
FAISS_PATH = "vector_store.faiss"
META_PATH = "vector_store.pkl"

_index = None
_metadata = None


def _load_faiss():
    """Lazy-load FAISS index & metadata once."""
    global _index, _metadata

    if _index is None:
        _index = faiss.read_index(FAISS_PATH)

        with open(META_PATH, "rb") as f:
            _metadata = pickle.load(f)


def embed_text_local(text: str):
    """Generate embeddings using local free model."""
    emb = embedder.encode([text])[0]
    return emb.astype("float32")


def search_similar(query: str, top_k=3):
    """Retrieve similar chunks."""
    _load_faiss()
    q = embed_text_local(query)

    distances, indices = _index.search(np.array([q]), top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "text": _metadata[idx]["text"]
        })

    return results


def answer_query(question: str):
    """Main RAG pipeline: Retrieve + Groq LLM answer."""
    hits = search_similar(question, 3)

    context = "\n\n".join(h["text"] for h in hits)

    prompt = f"""
You are a helpful assistant.
Use ONLY the following retrieved context to answer.
If context is insufficient, say: "Not enough information in the documents."

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly.
"""

    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = completion.choices[0].message.content

    return {"answer": answer, "sources": hits}


def health_check():
    return {"status": "ok", "faiss_loaded": _index is not None}

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from threading import Lock

FAISS_PATH = "vector_store.faiss"
META_PATH = "vector_store.pkl"

# Global variables for lazy loading
embed_model = None
index = None
metadata = None
_init_lock = Lock()  # thread-safe initialization


def _lazy_init():
    global embed_model, index, metadata
    with _init_lock:
        if embed_model is None:
            # Load tiny embedding model
            embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        if index is None:
            # Load FAISS index
            index = faiss.read_index(FAISS_PATH)
        if metadata is None:
            # Load document metadata
            with open(META_PATH, "rb") as f:
                metadata = pickle.load(f)


def answer_query(question: str, top_k: int = 3):
    """Return top_k most similar documents for the query."""
    _lazy_init()  # initialize only once on first call

    # Embed the query
    q_vec = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)

    # Search FAISS
    D, I = index.search(q_vec, k=top_k)

    # Collect corresponding metadata
    results = [metadata[i] for i in I[0]]
    return {"answer": " ".join(results), "scores": D[0].tolist()}


def health_check():
    return {"status": "ok"}

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

FAISS_PATH = "vector_store.faiss"
META_PATH = "vector_store.pkl"

# Load tiny embedding model once
embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Load FAISS index
index = faiss.read_index(FAISS_PATH)

# Load document metadata
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

def answer_query(question: str):
    # Embed the query
    q_vec = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_vec, k=3)  # top 3 results
    results = [metadata[i] for i in I[0]]
    return {"answer": " ".join(results)}

def health_check():
    return {"status": "ok"}

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_DB = "vector_store.faiss"
CHUNKS_JSON = "chunks.json"

# Load model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS
index = faiss.read_index(VECTOR_DB)

# Load text chunks
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

def search_docs(query, top_k=3):
    """Search vector DB and return top document chunks"""
    q_emb = embedder.encode(query)
    q_emb = np.array([q_emb]).astype("float32")

    distances, idxs = index.search(q_emb, top_k)

    retrieved = []
    for i in idxs[0]:
        retrieved.append(chunks[i])

    return "\n\n".join(retrieved)

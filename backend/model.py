# model.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

FAISS_PATH = "vector_store.faiss"
META_PATH = "vector_store.pkl"

# load FAISS index
index = faiss.read_index(FAISS_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

# load local embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text_local(text):
    return embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")

def answer_query(question):
    vec = embed_text_local(question)
    D, I = index.search(np.array([vec]), k=5)
    results = [meta[i] for i in I[0]]
    return {"answer": results}

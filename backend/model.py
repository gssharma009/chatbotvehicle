import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

embed_model = None
index = None
chunks = None

def load_everything():
    global embed_model, index, chunks

    if embed_model is None:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if index is None:
        index = faiss.read_index("vector_store.faiss")

    if chunks is None:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

def embed_text(t: str):
    load_everything()
    v = embed_model.encode([t])[0]
    return np.array(v).astype("float32")

def search_similar(query: str, k=3):
    load_everything()
    qv = embed_text(query)
    scores, idxs = index.search(np.array([qv]), k)
    return [chunks[int(i)] for i in idxs[0]]

# Groq LLM Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def ask_llm(question: str):
    docs = search_similar(question, 3)
    doc_text = "\n".join(docs)

    prompt = f"""
Use the following document context to answer:

{doc_text}

Question: {question}

If the answer is not in the context, answer generally.
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

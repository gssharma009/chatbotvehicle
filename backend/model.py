import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector_store.faiss")

# Load text chunks
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def embed_text(text: str):
    vector = embed_model.encode([text])[0]
    return np.array(vector).astype("float32")

def search_similar(query: str, k: int = 3):
    query_emb = embed_text(query)
    scores, idxs = index.search(np.array([query_emb]), k)
    return [chunks[i] for i in idxs[0]]

def ask_llm(question: str) -> str:
    docs = search_similar(question, k=3)
    context = "\n".join(docs)

    prompt = f"""
Use the document context below to answer the question.

Document Context:
{context}

Question:
{question}

If the answer is not found in the document, reply with the closest possible answer.
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return resp.choices[0].message.content

import os
import json
import numpy as np
import faiss
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

index = None
chunks = None

def load_everything():
    global index, chunks

    if index is None:
        index = faiss.read_index("vector_store.faiss")

    if chunks is None:
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)

def embed_text(t: str):
    """
    Uses Groq embedding model so we don't load any local ML model.
    """
    load_everything()
    t = t.replace("\n", " ")
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=[t]
    )
    return np.array(res.data[0].embedding, dtype="float32")

def search_similar(query: str, k=3):
    qv = embed_text(query)
    scores, idxs = index.search(np.array([qv]), k)
    return [chunks[int(i)] for i in idxs[0]]

def ask_llm(question: str):
    docs = search_similar(question, 3)
    doc_text = "\n".join(docs)

    prompt = f"""
Use this document context:

{doc_text}

Question: {question}

If not answerable from context, answer normally.
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

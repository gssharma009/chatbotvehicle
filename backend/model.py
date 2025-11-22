# model.py  (Render backend)

import pickle
import numpy as np
import faiss
from groq import Groq
import os

FAISS_PATH = "vector_store.faiss"
META_PATH = "vector_store.pkl"

# Cache
index = None
documents = None


def load_index():
    global index, documents
    if index is None:
        index = faiss.read_index(FAISS_PATH)

    if documents is None:
        with open(META_PATH, "rb") as f:
            documents = pickle.load(f)


def search_similar(query_vector, k=3):
    load_index()
    scores, idxs = index.search(np.array([query_vector]).astype("float32"), k)
    return [documents[int(i)] for i in idxs[0]]


# --------------------------
# YOU DO NOT embed on Render
# --------------------------
def embed_text_remote(text):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    res = client.embeddings.create(
        model="groq-embedding-3-small",
        input=text
    )

    return np.array(res.data[0].embedding).astype("float32")


def answer_query(question: str):
    v = embed_text_remote(question)
    docs = search_similar(v, 3)
    context = "\n\n".join(docs)

    prompt = f"""
Use ONLY this context:

{context}

Question: {question}
"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"answer": res.choices[0].message.content}


def health_check():
    return {"status": "healthy"}

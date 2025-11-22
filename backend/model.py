import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load embedding model (same as used while creating embeddings)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
faiss_index = faiss.read_index("vector_store.faiss")

# Load stored chunks (text data)
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Groq LLM client
client = Groq(api_key=GROQ_API_KEY)


def retrieve_context(query: str, top_k=3):
    """Retrieve top-k relevant chunks from FAISS."""
    q_emb = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(q_emb), top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx]["text"])

    return "\n".join(results)


def ask_llm(question: str) -> str:
    """Main RAG pipeline → retrieve → combine → send to Groq."""
    doc_context = retrieve_context(question)

    prompt = f"""
You are a helpful assistant.

First prefer answers based ONLY on the document context below.
If the document does not contain the answer, then use general knowledge.

Document Context:
{doc_context}

User Question:
{question}

Answer:
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return completion.choices[0].message.content

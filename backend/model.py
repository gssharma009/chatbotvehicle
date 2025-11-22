import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------------------
# Lazy-loaded globals
# ----------------------------------------
faiss_index = None
chunks = None
embedding_model = None

# ----------------------------------------
# Lazy load embedding model
# ----------------------------------------
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model

# ----------------------------------------
# Lazy load FAISS index and chunks
# ----------------------------------------
def load_faiss_index():
    global faiss_index, chunks
    if faiss_index is None:
        faiss_index = faiss.read_index("vector_store.faiss")
        with open("chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    return faiss_index, chunks

# ----------------------------------------
# Groq client setup
# ----------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ----------------------------------------
# Main LLM query function
# ----------------------------------------
def ask_llm(question: str) -> str:

    # Load FAISS + chunks
    index, chunks_data = load_faiss_index()

    # Create embeddings for user question
    model = get_embedding_model()
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')

    # Search top 3 similar chunks
    top_k = 3
    distances, results = index.search(question_embedding, top_k)

    retrieved_chunks = []
    for idx in results[0]:
        if str(idx) in chunks_data:
            retrieved_chunks.append(chunks_data[str(idx)])

    doc_context = "\n".join(retrieved_chunks)

    # Build prompt
    prompt = f"""
Answer the user's question based on the following document context.
If not answerable from documents, use general knowledge.

Document Context:
{doc_context}

User Question:
{question}
"""

    # Query Groq
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return completion.choices[0].message.content

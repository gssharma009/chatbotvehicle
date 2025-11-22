import os
import json
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

DOC_DIR = "docs"
VECTOR_DB = "vector_store.faiss"
CHUNKS_JSON = "chunks.json"

# Load a FREE local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def pdf_to_text(path):
    """Extract text from a single PDF file"""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    """Split text into small chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_embeddings():
    all_chunks = []
    embeddings = []

    for filename in os.listdir(DOC_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DOC_DIR, filename)
            print("Processing:", filename)

            text = pdf_to_text(path)
            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                emb = model.encode(chunk)
                embeddings.append(emb)

    # Save chunks
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, VECTOR_DB)

    print("✅ Embeddings & FAISS index created successfully!")

if __name__ == "__main__":
    import numpy as np
    build_embeddings()

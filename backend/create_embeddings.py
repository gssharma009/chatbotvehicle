import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

DOCS_FOLDER = "docs/"
CHUNK_SIZE = 500

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def pdf_to_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_embeddings():
    chunks = []
    vectors = []

    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".pdf"):
            print("Processing:", file)
            text = pdf_to_text(os.path.join(DOCS_FOLDER, file))
            parts = chunk_text(text, CHUNK_SIZE)

            chunks.extend(parts)
            vectors.extend(embedder.encode(parts))

    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, "vector_store.faiss")

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Embedding build complete!")

if __name__ == "__main__":
    build_embeddings()

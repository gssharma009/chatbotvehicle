# create_embeddings.py - OPTIMIZED FOR LOW MEM (Run locally)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import pdfplumber
import re

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Matches backend
CHUNK_SIZE = 200  # Smaller chunks for better retrieval, fewer total
CHUNK_OVERLAP = 20
MAX_CHUNKS_TOTAL = 300  # Hard limit to keep FAISS < 1 MB


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_chunks(text: str, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:  # Skip tiny fragments
            chunks.append(clean_text(chunk))
        i += chunk_size - overlap
    return chunks


def load_and_chunk_pdfs(pdf_dir="docs"):
    all_chunks = []
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print(f"Processing {pdf_file.name}")
        with pdfplumber.open(pdf_file) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        chunks = split_into_chunks(clean_text(full_text))
        all_chunks.extend(chunks[:100])  # Limit per PDF
        if len(all_chunks) >= MAX_CHUNKS_TOTAL:
            break
    print(f"Total chunks: {len(all_chunks)} (limited for low mem)")
    return all_chunks


def main():
    model = SentenceTransformer(MODEL_NAME)
    chunks = load_and_chunk_pdfs()
    if not chunks:
        print("No chunks!")
        return

    embeddings = model.encode(chunks, batch_size=16, show_progress_bar=True, normalize_embeddings=True)  # Smaller batch

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Cosine sim (normalized)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved! Chunks: {len(chunks)}, Index size: ~{len(chunks) * dim * 4 / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
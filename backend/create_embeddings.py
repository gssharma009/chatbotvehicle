# create_embeddings.py - USE BUNDLED MODEL (Run locally after Step 1)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import pdfplumber
import re
import torch  # For no_grad

MODEL_PATH = "./models/all-MiniLM-L6-v2"  # Local bundled path
CHUNK_SIZE = 250
CHUNK_OVERLAP = 30
MAX_CHUNKS_TOTAL = 100  # Tiny for free tier


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def split_into_chunks(text: str, chunk_size=250, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 15:
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
        all_chunks.extend(chunks[:40])
        if len(all_chunks) >= MAX_CHUNKS_TOTAL:
            break
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def main():
    print("Loading bundled model (no download)...")
    model = SentenceTransformer(MODEL_PATH, device='cpu')

    chunks = load_and_chunk_pdfs()
    if not chunks:
        print("No PDFs! Add to /docs")
        return

    print("Encoding chunks...")
    with torch.no_grad():
        embeddings = model.encode(chunks, batch_size=4, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    est_mb = len(chunks) * dim * 4 / (1024 ** 2)
    print(f"✅ Saved! Chunks: {len(chunks)}, FAISS est: {est_mb:.1f} MB")
    print("Commit: git add vector_store.faiss chunks.pkl")


if __name__ == "__main__":
    main()
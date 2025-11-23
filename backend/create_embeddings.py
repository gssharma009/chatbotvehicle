# create_embeddings.py - FIXED WITH PUBLIC MULTILINGUAL MODEL (Run locally)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import pdfplumber
import re

# PROVEN PUBLIC MODEL: Multilingual MiniLM-L12-v2 (Hindi/English, <150MB RAM)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 300  # Good for RAG context
CHUNK_OVERLAP = 50
MAX_CHUNKS_TOTAL = 150  # Ultra-conservative for <50 MB FAISS (adjust up if needed)


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_chunks(text: str, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 20:  # Skip fragments
            chunks.append(clean_text(chunk))
        i += chunk_size - overlap
    return chunks


def load_and_chunk_pdfs(pdf_dir="docs"):
    all_chunks = []
    pdf_count = 0
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print(f"Processing {pdf_file.name}")
        with pdfplumber.open(pdf_file) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        chunks = split_into_chunks(clean_text(full_text))
        all_chunks.extend(chunks[:60])  # Limit per PDF
        pdf_count += 1
        if len(all_chunks) >= MAX_CHUNKS_TOTAL:
            break
    print(f"Processed {pdf_count} PDFs → Total chunks: {len(all_chunks)}")
    return all_chunks


def main():
    print("Loading multilingual model (downloads ~120MB first time)...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')

    chunks = load_and_chunk_pdfs()
    if not chunks:
        print("No PDFs found in 'docs' folder! Add some and retry.")
        return

    print("Encoding chunks (low batch for memory safety)...")
    embeddings = model.encode(
        chunks,
        batch_size=8,  # Small batch to avoid local OOM
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]  # 384 dims
    index = faiss.IndexFlatIP(dim)  # Cosine similarity (normalized vectors)
    index.add(embeddings.astype('float32'))

    # Save files
    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    est_size_mb = len(chunks) * dim * 4 / 1024 / 1024
    print(f"✅ Saved! Chunks: {len(chunks)}, Est. FAISS RAM: {est_size_mb:.1f} MB")
    print("Upload 'vector_store.faiss' + 'chunks.pkl' to Render (delete old ones).")


if __name__ == "__main__":
    main()
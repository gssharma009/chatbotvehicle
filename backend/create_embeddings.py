# create_embeddings.py - TINY MODEL VERSION (Run locally)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import pdfplumber
import re

# TINY MODEL: all-MiniLM-L6-v2 (22MB, <120MB RAM, good for Hindi/English RAG)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 250  # Smaller for low mem
CHUNK_OVERLAP = 30
MAX_CHUNKS_TOTAL = 100  # Max for <20 MB FAISS


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_into_chunks(text: str, chunk_size=250, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 15:  # Skip tiny
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
        all_chunks.extend(chunks[:40])  # Limit per PDF
        pdf_count += 1
        if len(all_chunks) >= MAX_CHUNKS_TOTAL:
            break
    print(f"Processed {pdf_count} PDFs → Total chunks: {len(all_chunks)}")
    return all_chunks


def main():
    print("Loading tiny embedding model (~22MB download)...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')

    chunks = load_and_chunk_pdfs()
    if not chunks:
        print("❌ No PDFs in 'docs'! Add some (even sample ones) and retry.")
        return

    print("Encoding with low batch...")
    import torch
    with torch.no_grad():  # Memory saver
        embeddings = model.encode(
            chunks,
            batch_size=4,  # Tiny batch
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    dim = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    est_mb = len(chunks) * dim * 4 / (1024 * 1024)
    print(f"✅ Saved! Chunks: {len(chunks)}, Est. FAISS RAM: {est_mb:.1f} MB")
    print("Upload to Render: Delete old files, add these new ones.")


if __name__ == "__main__":
    main()
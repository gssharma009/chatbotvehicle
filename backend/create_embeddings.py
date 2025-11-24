# create_embeddings.py → NEW & FINAL VERSION (2025 BEST PRACTICES)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import pdfplumber
import re
import torch

# BEST MODEL FOR OCR + HINDI/ENGLISH MIXED PDFs (2025)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 280          # थोड़ा बड़ा chunk → ज्यादा context
CHUNK_OVERLAP = 50
MAX_CHUNKS_TOTAL = 120    # आप चाहें तो बढ़ा सकते हो


def clean_text(text: str) -> str:
    """Remove extra spaces, page numbers, headers, footers, garbage"""
    text = re.sub(r'\f|\r', ' ', text)                    # form feed, carriage return
    text = re.sub(r'\s*\n\s*', '\n', text)                # multiple newlines
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.I)
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s\.\,\;\:\(\)\-\?\%\/]', ' ', text)  # Hindi + English
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text: str, chunk_size=280, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if len(chunk) > 50:  # बहुत छोटे chunks हटाओ
            chunks.append(clean_text(chunk))
        i += chunk_size - overlap
    return chunks


def load_and_chunk_pdfs(pdf_dir="docs"):
    all_chunks = []
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        print("No PDFs found in /docs folder!")
        return all_chunks

    print(f"Found {len(pdf_paths)} PDFs")

    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            if not full_text.strip():
                print(f"  → No text extracted from {pdf_path.name}")
                continue

            chunks = split_into_chunks(clean_text(full_text))
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks added")
        except Exception as e:
            print(f"  → Error reading {pdf_path.name}: {e}")

        if len(all_chunks) >= MAX_CHUNKS_TOTAL:
            print(f"Reached max chunks limit ({MAX_CHUNKS_TOTAL})")
            break

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def main():
    print(f"Loading best multilingual model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    chunks = load_and_chunk_pdfs()
    if not chunks:
        print("No chunks generated. Check /docs folder.")
        return

    print("Generating embeddings...")
    with torch.no_grad():
        embeddings = model.encode(
            chunks,
            batch_size=8,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    # Save files
    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    est_mb = len(chunks) * dim * 4 / (1024 ** 2)
    print(f"ALL DONE!")
    print(f"Chunks: {len(chunks)} | Dimension: {dim} | Est. size: {est_mb:.1f} MB")
    print("Now commit these files:")
    print("   git add vector_store.faiss chunks.pkl")
    print("   git commit -m 'new multilingual embeddings'")
    print("   git push")


if __name__ == "__main__":
    main()
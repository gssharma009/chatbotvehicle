# create_embeddings.py - FINAL VERSION (Run this locally once)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
import pdfplumber
import re

# BEST model for Hindi + English (small, fast, excellent multilingual)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 300MB RAM locally, only ~90MB ONNX later

# Chunking settings
CHUNK_SIZE = 400  # tokens approx
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text: str, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(clean_text(chunk))
        i += chunk_size - overlap
        if i >= len(words) and chunks:
            break
    return chunks


def load_and_chunk_pdfs(pdf_dir="docs"):
    all_chunks = []
    all_metadatas = []

    print("Loading and chunking PDFs...")
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        with pdfplumber.open(pdf_file) as pdf:
            full_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"

        full_text = clean_text(full_text)
        if not full_text:
            continue

        chunks = split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": pdf_file.name,
                "chunk_id": idx,
                "text": chunk
            })

    print(f"Created {len(all_chunks)} chunks from {len(list(Path(pdf_dir).glob('*.pdf')))} PDFs")
    return all_chunks, all_metadatas


def main():
    print("Loading embedding model (this takes 10-20 seconds)...")
    model = SentenceTransformer(MODEL_NAME)

    chunks, metadatas = load_and_chunk_pdfs()
    if not chunks:
        print("No text found in PDFs!")
        return

    print("Generating embeddings...")
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (since normalized = cosine)
    index.add(embeddings)

    # Save everything
    faiss.write_index(index, "vector_store.faiss")

    # Save only the text chunks (not full metadata to save RAM)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    # Optional: save metadata too if you want source tracking
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadatas, f)

    print(f"Done! Created {len(chunks)} chunks → vector_store.faiss + chunks.pkl")
    print("Upload these 2 files to your Render project!")


if __name__ == "__main__":
    main()
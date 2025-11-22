from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import pdfplumber

FAISS_PATH = "vector_store.faiss"
META_PATH = "vector_store.pkl"

# Tiny embedding model to save memory
embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

def load_pdfs(pdf_dir="docs"):
    texts = []
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        with pdfplumber.open(pdf_file) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            texts.append(full_text)
    return texts

def create_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    docs = load_pdfs()
    if not docs:
        print("No PDFs found in docs folder!")
        return

    embeddings = embed_model.encode(docs, convert_to_numpy=True)
    index = create_faiss_index(np.array(embeddings))

    # Save FAISS index and metadata
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"FAISS index and metadata saved. Total docs: {len(docs)}")

if __name__ == "__main__":
    main()

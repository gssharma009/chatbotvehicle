import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")

FAISS_PATH = os.path.join(BASE_DIR, "vector_store.faiss")
META_PATH = os.path.join(BASE_DIR, "vector_store.pkl")


def extract_text_from_pdf(file_path):
    """Extract text from PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_txt = page.extract_text()
                if page_txt:
                    text += page_txt + "\n"
    except Exception as e:
        print(f"[ERROR] Failed to parse PDF {file_path}: {e}")

    return text.strip()


def load_documents(folder_path):
    docs = []
    metadata = []

    allowed_ext = [".txt", ".pdf"]

    if not os.path.exists(folder_path):
        print(f"[ERROR] Docs folder not found: {folder_path}")
        return docs, metadata

    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()

            if ext not in allowed_ext:
                continue

            full_path = os.path.join(root, f)

            try:
                if ext == ".txt":
                    with open(full_path, "r", encoding="utf-8") as file:
                        text = file.read().strip()

                elif ext == ".pdf":
                    text = extract_text_from_pdf(full_path)

                if text:
                    docs.append(text)
                    metadata.append({"filename": f, "path": full_path})
                else:
                    print(f"[WARN] No extractable text found: {full_path}")

            except Exception as e:
                print(f"[ERROR] Cannot read file {full_path}: {e}")

    return docs, metadata


def create_faiss_index(embeddings):
    if embeddings.size == 0:
        raise ValueError("No embeddings created! Check your docs folder.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    print("Loading model (FREE MiniLM)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading documents from:", DOCS_DIR)
    docs, metadata = load_documents(DOCS_DIR)

    print(f"Total documents loaded: {len(docs)}")
    if len(docs) == 0:
        print("[ERROR] No documents found.")
        return

    print("Embedding documents...")
    embeddings = model.encode(docs, convert_to_numpy=True)

    print("Creating FAISS index...")
    index = create_faiss_index(np.array(embeddings))

    print("Saving FAISS index...")
    faiss.write_index(index, FAISS_PATH)

    print("Saving metadata...")
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("\n🚀 Done!")
    print(f"FAISS saved → {FAISS_PATH}")
    print(f"Metadata saved → {META_PATH}")


if __name__ == "__main__":
    main()

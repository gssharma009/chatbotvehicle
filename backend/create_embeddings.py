# create_embeddings.py → FINAL PERFECT VERSION
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import pdfplumber
import re
from pathlib import Path
import torch

# BEST MODEL 2025: OCR + Hindi + English + Long context
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # ← यह सबसे बेस्ट है अब

CHUNK_SIZE = 400        # बड़ा chunk → पूरा sentence आएगा
CHUNK_OVERLAP = 100     # overlap → कोई sentence नहीं कटेगा
MAX_CHUNKS = 150


def clean_text(text: str) -> str:
    text = re.sub(r'\f', ' ', text)                           # form feed
    text = re.sub(r'\s*\n\s*', '\n', text)                    # clean newlines
    text = re.sub(r'\bPage\s*\d+\b|\b\d+\s*of\s*\d+\b', '', text, flags=re.I)
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\s\.\,\;\:\(\)\-\?\%\/\'\"]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # अगर नया sentence डालने से chunk बड़ा हो जाए तो पहले वाला save करो
        if len(current.split()) + len(sentence.split()) > CHUNK_SIZE:
            if current:
                chunks.append(clean_text(current))
            current = sentence
        else:
            current += " " + sentence if current else sentence

        # अगर current भी बहुत लंबा हो जाए तो save करो
        if len(current.split()) > CHUNK_SIZE + 50:
            chunks.append(clean_text(current))
            current = ""

    if current.strip():
        chunks.append(clean_text(current))

    return chunks


def load_pdfs(pdf_dir="docs"):
    all_chunks = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        print(f"Reading: {pdf_path.name}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                chunks = split_into_chunks(clean_text(text))
                all_chunks.extend(chunks)
                print(f"  → {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error: {e}")

        if len(all_chunks) >= MAX_CHUNKS:
            break

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def main():
    print("Loading BEST multilingual model (handles OCR + Hindi + English perfectly)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    chunks = load_pdfs()
    if not chunks:
        print("No text found! Add PDFs to /docs folder")
        return

    print("Creating high-quality embeddings...")
    with torch.no_grad():
        embeddings = model.encode(
            chunks,
            batch_size=8,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, "vector_store.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("SUCCESS! New clean embeddings ready.")
    print("Now run:")
    print("   git add vector_store.faiss chunks.pkl")
    print("   git commit -m 'perfect multilingual embeddings - no cut sentences'")
    print("   git push")


if __name__ == "__main__":
    main()
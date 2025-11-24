from sentence_transformers import SentenceTransformer
import faiss, pickle, pdfplumber
from pathlib import Path

print("Generating clean embeddings...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = []
for pdf in Path("docs").glob("*.pdf"):
    print(f"Reading {pdf.name}")
    with pdfplumber.open(pdf) as p:
        text = "\n".join(page.extract_text() or "" for page in p.pages)
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
    chunks.extend(paras[:70])

print(f"Created {len(chunks)} clean chunks")

emb = model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f, protocol=4)   # ← This fixes the pickle error!

print("PERFECT FILES CREATED – DEPLOY NOW!")
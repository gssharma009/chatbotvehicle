# quick_embeddings.py – RUN THIS (300MB RAM only)
from sentence_transformers import SentenceTransformer
import faiss, pickle, pdfplumber
from pathlib import Path

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = []
for pdf in Path("docs").glob("*.pdf"):
    with pdfplumber.open(pdf) as p:
        text = "\n".join(page.extract_text() or "" for page in p.pages)
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 120]
    chunks.extend(paras[:60])

print(f"Generated {len(chunks)} lightweight chunks")
emb = model.encode(chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("LIGHTWEIGHT EMBEDDINGS READY – deploy now!")
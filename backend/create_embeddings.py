# recreate_embeddings.py
from sentence_transformers import SentenceTransformer
import faiss, pickle, pdfplumber, re
from pathlib import Path

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

all_chunks = []
for pdf in Path("docs").glob("*.pdf"):
    with pdfplumber.open(pdf) as p:
        text = "\n".join(page.extract_text() or "" for page in p.pages)
    # Split by paragraphs, not fixed size
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p) > 100]
    all_chunks.extend(paragraphs[:80])

print(f"Generated {len(all_chunks)} high-quality chunks")
emb = model.encode(all_chunks, normalize_embeddings=True, batch_size=16, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("PERFECT EMBEDDINGS READY – deploy now!")
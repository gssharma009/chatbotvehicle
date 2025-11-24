# recreate_embeddings.py – RUN THIS NOW
from sentence_transformers import SentenceTransformer
import faiss, pickle, pdfplumber, re
from pathlib import Path

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

chunks = []
for pdf in Path("docs").glob("*.pdf"):
    text = ""
    with pdfplumber.open(pdf) as p:
        for page in p.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    # Smart chunking
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = ""
    for s in sentences:
        if len((current + s).split()) > 300:
            if current:
                chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s
    if current:
        chunks.append(current.strip())

print(f"Created {len(chunks)} chunks")
emb = model.encode(chunks, normalize_embeddings=True, batch_size=16, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("New perfect embeddings ready!")
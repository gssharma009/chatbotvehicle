import pdfplumber, re, pickle, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("GENERATING 100% CLEAN EMBEDDINGS...")

def deep_clean(text):
    if not text: return ""
    # Remove everything except real words and punctuation
    text = re.sub(r'[^A-Za-z0-9\s\.\,\;\:\(\)\-\?\'’]', ' ', text)
    text = re.sub(r'\b(off|o|of|offf)\b', 'of', text, flags=re.I)
    text = re.sub(r'\b(u|ur)\b', 'your', text, flags=re.I)
    text = re.sub(r'\bse\b', 'use', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = []
for pdf in Path("docs").glob("*.pdf"):
    print(f"Reading {pdf.name}")
    with pdfplumber.open(pdf) as p:
        full = ""
        for page in p.pages:
            t = page.extract_text()
            if t:
                full += deep_clean(t) + " "
        # Split into proper sentences only
        sentences = [s.strip() + "." for s in full.split(".") if len(s.strip()) > 50]
        chunks.extend(sentences[:120])

print(f"Generated {len(chunks)} PERFECT clean sentences")

emb = model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f, protocol=4)

print("DONE! DEPLOY THESE TWO FILES NOW")
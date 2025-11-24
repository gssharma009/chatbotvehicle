import pdfplumber, re, pickle, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean(text):
    text = re.sub(r'\boff?\b', 'off', text, flags=re.I)
    text = re.sub(r'\bu\b', '', text, flags=re.I)
    text = re.sub(r'lci heV cir tce l E|ruoy wo nK|Page\s*\d+|FOREWORD|Thank you|JSW MG|manal|yor|yo|se\b', '', text, flags=re.I)
    text = re.sub(r'[•◦▪uU]\s*', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

chunks = []
for pdf in Path("docs").glob("*.pdf"):
    with pdfplumber.open(pdf) as p:
        text = ""
        for page in p.pages:
            t = page.extract_text()
            if t:
                text += clean(t) + " "
    sentences = [s.strip() for s in text.split('.') if len(s) > 50]
    chunks.extend(sentences[:80])

print(f"Generated {len(chunks)} CLEAN chunks")

emb = model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f, protocol=4)

print("PERFECT CLEAN EMBEDDINGS READY – DEPLOY NOW!")
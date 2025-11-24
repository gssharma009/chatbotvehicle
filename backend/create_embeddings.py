import pdfplumber, re, pickle, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

print("Creating PERFECT embeddings from CLEAN text...")


def ultra_clean(text: str) -> str:
    if not text:
        return ""
    # Remove all known garbage patterns
    text = re.sub(r'\boff?\b', 'off', text, flags=re.I)
    text = re.sub(r'\bu\b', 'u', text, flags=re.I)
    text = re.sub(r'lci heV cir tce l E|ruoy wo nK|Page\s*\d+|FOREWORD|Thank you.*?economy|JSW MG|manal|yor|yo|se\b', '', text, flags=re.I)
    text = re.sub(r'[•◦▪★✓]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

clean_chunks = []
for pdf in Path("docs").glob("*.pdf"):
    print(f"Processing {pdf.name}...")
    with pdfplumber.open(pdf) as p:
        full_text = ""
        for page in p.pages:
            t = page.extract_text()
            if t:
                full_text += ultra_clean(t) + " "

    # Split into clean, meaningful sentences
    sentences = [s.strip() for s in full_text.split('.') if len(s.strip()) > 40]
    clean_chunks.extend(sentences)

print(f"Generated {len(clean_chunks)} PERFECT clean chunks")

# Create embeddings from CLEAN text
emb = model.encode(clean_chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(clean_chunks, f, protocol=4)

print("\nSUCCESS! DEPLOY THESE FILES NOW:")
print("   vector_store.faiss")
print("   chunks.pkl")
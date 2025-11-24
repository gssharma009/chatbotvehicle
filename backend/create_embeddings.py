import pdfplumber, re, pickle, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

print('Creating PERFECT clean embeddings...')

def clean_text(t):
    t = re.sub(r'[^A-Za-z0-9\s\.\,\;\:\(\)\-\?]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = t.replace(' off ', ' of ').replace(' u ', ' you ').replace(' se ', ' use ')
    t = t.replace(' o ', ' of ').replace(' b ', ' ').replace(' ur ', ' your ')
    return t.strip()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

chunks = []
for pdf in Path('docs').glob('*.pdf'):
    print(f'Processing {pdf.name}')
    with pdfplumber.open(pdf) as p:
        text = ''
        for page in p.pages:
            page_text = page.extract_text()
            if page_text:
                text += clean_text(page_text) + ' '
        # Split into clean sentences
        sentences = [s.strip() for s in text.split('.') if len(s) > 60]
        chunks.extend(sentences)

print(f'Created {len(chunks)} super-clean chunks')

emb = model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype('float32'))

faiss.write_index(index, 'vector_store.faiss')
with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f, protocol=4)

print('PERFECT CLEAN EMBEDDINGS READY — DEPLOY NOW!')
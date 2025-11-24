# create_embeddings.py — IMPROVED + TUNED CHUNKING
from sentence_transformers import SentenceTransformer
import faiss, pickle, re

def normalize(text: str) -> str:
    text = text.lower()

    # Remove repeated headers / footers / page numbers
    text = re.sub(r'page \d+|table of contents|index|chapter \d+', ' ', text)

    # Fix hyphen line breaks: “self-\ncontained” → “self-contained”
    text = re.sub(r'-\s*\n\s*', '', text)

    # Fix “a c” → “ac”, “h v” → “hv”
    text = re.sub(r'\b([a-z])\s+([a-z])\b', r'\1\2', text)

    # Remove garbage characters
    text = re.sub(r'[^a-z0-9\s\.,;:()\-/]', ' ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


print("Loading text…")
with open("manual_clean.txt", "r", encoding="utf-8") as f:
    raw = f.read()

clean = normalize(raw)

# Split into sentences conservatively
sentences = [
    s.strip() + "."
    for s in re.split(r'[.\n]', clean)
    if len(s.strip()) > 25
]

print("Raw sentences:", len(sentences))

# NEW: semantic multi-sentence chunks
chunks = []
CHUNK_SIZE = 3  # 3 sentences per chunk
for i in range(0, len(sentences), CHUNK_SIZE):
    block = " ".join(sentences[i:i+CHUNK_SIZE]).strip()
    if len(block) > 50:
        chunks.append(block)

print("Final chunks:", len(chunks))

# Embed
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(
    chunks,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("DONE — tuned chunks + embeddings saved.")

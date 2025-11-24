from sentence_transformers import SentenceTransformer
import faiss, pickle, re

print("Creating PERFECT embeddings from clean text...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Read the clean text we just made
with open("manual_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split into proper sentences
sentences = [s.strip() + "." for s in text.split(".") if len(s.strip()) > 60]

print(f"Using {len(sentences)} clean, real sentences")

emb = model.encode(sentences, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))

faiss.write_index(index, "vector_store.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(sentences, f)

print("PERFECT embeddings ready — deploy now!")
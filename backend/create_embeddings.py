from sentence_transformers import SentenceTransformer
import pickle, faiss, re

print("[STEP] Loading clean text...")

with open("cleaned_manual_clean.txt", "r", encoding="utf-8") as f:
    paras = [p.strip() for p in f.readlines() if len(p.strip()) > 30]

# Remove duplicate or near-duplicate paragraphs
unique = list(dict.fromkeys(paras))
print(f"[INFO] Using {len(unique)} unique paragraphs (deduped).")

print("[STEP] Loading MiniLM model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("[STEP] Encoding embeddings...")
emb = model.encode(
    unique,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("[STEP] Building FAISS index...")
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb.astype("float32"))

faiss.write_index(index, "vector_store.faiss")

with open("chunks.pkl", "wb") as f:
    pickle.dump(unique, f)

print("[DONE] Embeddings + chunks saved.")

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/knowledge.txt"
DB_FOLDER = "vectordb"
INDEX_FILE = f"{DB_FOLDER}/index.faiss"
DOC_FILE = f"{DB_FOLDER}/docs.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load training data
with open(DATA_FILE, "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f.readlines() if line.strip()]

# Create embedding
embeddings = model.encode(docs)

# Convert to numpy float32
embeddings_np = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

# Save DB folder
os.makedirs(DB_FOLDER, exist_ok=True)

# Save FAISS index
faiss.write_index(index, INDEX_FILE)

# Save docs
with open(DOC_FILE, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("Training complete! FAISS index created.")

# download_model.py - Run this ONCE locally to bundle model in repo
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH = "./models/all-MiniLM-L6-v2"

print("Downloading model (~22 MB)...")
model = SentenceTransformer(MODEL_NAME)

# Save to local folder (bundles all files)
os.makedirs(SAVE_PATH, exist_ok=True)
model.save(SAVE_PATH)

print(f"✅ Model bundled! Folder: {SAVE_PATH}")
print("Files: config.json, pytorch_model.bin, tokenizer.json, etc. (~28 MB total)")
print("Add './models/' to Git (git add models/ && git commit -m 'Bundle model')")
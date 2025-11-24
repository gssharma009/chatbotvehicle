# backend/model.py
import os
import faiss
import pickle
import re
import requests
from sentence_transformers import SentenceTransformer
from threading import Lock

# Lightweight, high-quality model – 90 MB, perfect for Railway + OCR + multilingual
MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"

model = None
index = None
chunks = None
_lock = Lock()

def _load():
    global model, index, chunks
    if model is not None:
        return

    with _lock:
        if model is not None:
            return

        print("[INIT] Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME, device="cpu")

        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        print(f"[INIT] Ready – {len(chunks)} chunks loaded")

def clean_text(text: str) -> str:
    """Remove common OCR artifacts and noise"""
    if not text:
        return ""
    text = re.sub(r'\be\s+lci\s+heV\b', '', text, flags=re.I)
    text = re.sub(r'\bruoy\s+wo\s+nK\b', '', text, flags=re.I)
    text = re.sub(r'\b\d+-\d+[a-zA-Z]?\s+', ' ', text)
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.I)
    text = re.sub(r'[•◦▪!\\]', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\.\s+', '. ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def retrieve_context(question: str, top_k: int = 6) -> str:
    _load()
    q_emb = model.encode([question], normalize_embeddings=True)
    _, indices = index.search(q_emb, top_k * 2)

    selected = []
    for i in indices[0]:
        cleaned = clean_text(chunks[i])
        if len(cleaned) > 80:
            selected.append(cleaned)
        if len(selected) >= top_k:
            break
    return " ".join(selected)

def fast_llm(question: str) -> str:
    context = retrieve_context(question)

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        for groq_model in ["mixtral-8x7b-32768", "llama3-70b-8192"]:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": groq_model,
                        "messages": [{"role": "user", "content":
                            f"""Answer in the same language as the question.
Use clean bullet points. Be concise and professional.
Never repeat the full context.

Context: {context}

Question: {question}

Answer:"""}],
                        "max_tokens": 220,
                        "temperature": 0.1
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=14
                )
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["message"]["content"].strip()
                    if len(answer) > 30:
                        return answer
            except:
                continue

    # Final clean dynamic fallback (zero hardcoded sentences)
    relevant = [
        clean_text(c) for c in chunks[:20]
        if any(word in clean_text(c).lower() for word in
               ["avoid","safe","water","shock","cable","touch","do not","warning","orange","modify","risk","damage"])
    ]
    if relevant:
        return "\n".join("• " + line for line in relevant[:7] if len(line) > 40)

    return "Information is available in the vehicle manual."

# Required exports for app.py
def answer_query(question: str) -> str:
    return fast_llm(question)

def health_check() -> dict:
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
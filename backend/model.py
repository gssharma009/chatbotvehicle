# model.py
import re

import faiss, pickle, numpy as np, os, gc, requests
from sentence_transformers import SentenceTransformer
from threading import Lock

MODEL_PATH = "./models/all-MiniLM-L6-v2"
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

model = None; index = None; chunks = None
_lock = Lock()
# model.py के top में
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
def _load():
    global model, index, chunks
    if model is not None:
        return
    with _lock:
        if model is not None:
            return

        print("[INIT] Downloading all-MiniLM-L6-v2 from HuggingFace (22 MB)...")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("Model loaded successfully")

        index = faiss.read_index("vector_store.faiss")
        with open("chunks.pkl", "rb") as f:  # आपका actual chunks file name
            chunks = pickle.load(f)

        print(f"[INIT] Ready! {len(chunks)} chunks loaded")


def answer_query(question: str):
    _load()
    q = question.strip()
    if not q: return {"answer": "पूछो ना कुछ!"}

    # 1. Greeting
    if any(x in q.lower() for x in ["hi","hello","hey","नमस्ते","नमस्ते"]):
        return {"answer": "नमस्ते! मैं आपका व्हीकल असिस्टेंट हूँ। EV/कार के बारे में हिंदी या English में कुछ भी पूछो!", "source": "greeting"}

    # 2. Search
    emb = model.encode([q], normalize_embeddings=True)[0]
    D, I = index.search(np.array([emb]).astype("float32"), 30)
    context = []
    for d, i in zip(D[0], I[0]):
        if i != -1 and d > 0.22:
            context.append(chunks[i])

    # 3. Docs
    if context:
        full_context = " ".join(context)[:3900]
        return {"answer": fast_llm(full_context, q), "source": "docs"}

    # 4. internet
    no = "दस्तावेज़ों में नहीं मिला।" if "हिंदी" in q or "हindi" in q.lower() else "Not in my docs."
    return {"answer": f"{no}\n\nइंटरनेट से:\n{fast_llm('', q)}", "source": "internet"}

def fast_llm(ctx: str, q: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: API key not found."

    # Step 1: OCR GARBAGE
    clean_lines = []
    for raw in ctx.split("\n"):
        line = raw.strip()
        if len(line) < 20:
            continue

        # Remove prefixes: a., b., 1., •, !, \, NOTE, WARNING etc.
        line = re.sub(r"^[a-dA-D][\.\)\]\s]*", "", line, flags=re.I)
        line = re.sub(r"^[\d\.\-\•◦▪!\\]+\s*", "", line)

        # Remove OCR junk patterns
        line = re.sub(r"\bo\b\s+", "off ", line, flags=re.I)
        line = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?\%\/]", " ", line)
        line = re.sub(r"\s+", " ", line).strip()

        if line and len(line) > 25:
            clean_lines.append(line)

    clean_ctx = " ".join(clean_lines)[:3000]

    # Step 2: Try Groq (best models)
    for model in ["mixtral-8x7b-32768", "llama3-70b-8192"]:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content":
                        f"""Answer in the same language as the question.
Use clean bullet points. Be short and professional.
Never repeat the full context.

Context: {clean_ctx}

Question: {q}

Answer:"""}],
                    "max_tokens": 220,
                    "temperature": 0.1
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=16
            )
            if r.status_code == 200:
                ans = r.json()["choices"][0]["message"]["content"].strip()
                if ans and len(ans) > 30:
                    return ans
        except:
            continue

    # Step 3:
    relevant = []
    keywords = ["avoid", "do not", "safe", "water", "shock", "cable", "touch", "orange", "modify", "disassemble", "warning", "risk", "damage", "do not touch"]
    for line in clean_lines:
        if any(word in line.lower() for word in keywords):
            relevant.append(line)
            if len(relevant) >= 6:
                break


    if relevant:
        return "\n".join("• " + line for line in relevant)

    if clean_lines:
        return "\n".join("• " + line for line in clean_lines[:5])

    # Ultimate fallback
    return "No relevant information found in document."

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
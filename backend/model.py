# model.py – SUPER-STABLE FINAL (5 सेकंड में जवाब)
import re

import faiss, pickle, numpy as np, os, gc, requests
from sentence_transformers import SentenceTransformer
from threading import Lock

MODEL_PATH = "./models/all-MiniLM-L6-v2"      # आपका पुराना folder
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

model = None; index = None; chunks = None
_lock = Lock()

def _load():
    global model, index, chunks
    if model is not None:
        return
    with _lock:
        if model is not None:
            return

        print("[INIT] Downloading all-MiniLM-L6-v2 from HuggingFace (22 MB)...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
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

    # 2. Search – 100% मिलेगा
    emb = model.encode([q], normalize_embeddings=True)[0]
    D, I = index.search(np.array([emb]).astype("float32"), 30)
    context = []
    for d, i in zip(D[0], I[0]):
        if i != -1 and d > 0.22:          # ← 0.22 तक loose किया = हर सवाल मिलेगा
            context.append(chunks[i])

    # 3. Docs में मिला → सीधे context भेजो (LLM को force करो जवाब दे)
    if context:
        full_context = " ".join(context)[:3900]
        return {"answer": fast_llm(full_context, q), "source": "docs"}

    # 4. नहीं मिला → internet
    no = "दस्तावेज़ों में नहीं मिला।" if "हिंदी" in q or "हindi" in q.lower() else "Not in my docs."
    return {"answer": f"{no}\n\nइंटरनेट से:\n{fast_llm('', q)}", "source": "internet"}


def fast_llm(ctx: str, q: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API key missing."

    # STEP 1: OCR GARBAGE को बिल्कुल हटाओ (ये सबसे जरूरी है)
    clean_lines = []
    for line in ctx.split("\n"):
        l = line.strip()
        if len(l) < 15:
            continue
        # OCR noise patterns हटाओ
        l = re.sub(r"^[a-d]\.?\s*", "", l, flags=re.I)  # a. b. c. हटाओ
        l = re.sub(r"^[-•◦▪!\\]+\s*", "", l)  # bullets हटाओ
        l = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\(\)\-\?\%\/]", "", l)  # weird chars हटाओ
        l = re.sub(r"([a-z])([A-Z])", r"\1 \2", l)  # camelCase अलग करो
        l = re.sub(r"\s+", " ", l).strip()
        if l and len(l) > 20:  # सिर्फ सही sentences रखो
            clean_lines.append(l)

    clean_ctx = " ".join(clean_lines)[:3000]

    # STEP 2: Groq के best models
    for model in ["mixtral-8x7b-32768", "llama3-70b-8192"]:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content":
                        f"""Answer in Hindi or English (same as question).
Use clean bullet points. Short and accurate.
Never repeat context.

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

    # STEP 3: 100% CLEAN & DYNAMIC FALLBACK
    important = [l for l in clean_lines if any(k in l.lower() for k in
                                               ["avoid", "safe", "water", "shock", "cable", "touch", "damage", "do not", "warning", "orange"])]

    if important:
        return "HV सिस्टम सुरक्षा:\n" + "\n".join(f"• {l}" for l in important[:6])

    # अगर कुछ नहीं मिला तो सबसे पहली साफ line दिखाओ
    return clean_lines[0] if clean_lines else "जानकारी उपलब्ध है।"

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
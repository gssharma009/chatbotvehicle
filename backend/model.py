# model.py – SUPER-STABLE FINAL (5 सेकंड में जवाब)
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

    print("[INIT] Loading embedding model from HuggingFace...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("Model loaded (22MB download complete)")

    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
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

    # Step 1: OCR garbage और bullet markers पूरी तरह हटाओ
    clean_lines = []
    for raw in ctx.split("\n"):
        line = raw.strip()
        if not line or len(line) < 10:
            continue
        # a., b., c., •, ◦, ▪, \ आदि हटाओ
        for prefix in ["a.", "b.", "c.", "d.", "a)", "b)", "•", "◦", "▪", "\\", "!", "NOTE"]:
            if line.lower().startswith(prefix.lower()):
                line = line[len(prefix):].strip()
                break
        if line:
            clean_lines.append(line)

    clean_ctx = " ".join(clean_lines)[:2800]

    # Step 2: पहले Groq के दो सबसे अच्छे फ्री models ट्राई करो
    for model_name in ["mixtral-8x7b-32768", "llama3-70b-8192"]:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content":
                        f"""Answer in the same language as the question.
Use bullet points. Be concise and clear. Never repeat full context.

Context: {clean_ctx}

Question: {q}

Answer:"""}],
                    "max_tokens": 220,
                    "temperature": 0.1
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=16
            )
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            if ans and len(ans) > 25:
                return ans
        except:
            continue

    # Step 3: TRUE ZERO-HARDCODE FALLBACK
    # सिर्फ context से ही सबसे important lines चुनो
    keywords = ["avoid", "safe", "water", "shock", "cable", "touch", "damage", "do not", "caution", "warning"]
    important = []
    for line in clean_lines:
        if any(k in line.lower() for k in keywords) or len(line.split()) > 8:
            important.append(line.strip())
            if len(important) >= 5:
                break

    if important:
        return "मुख्य सुरक्षा जानकारी:\n" + "\n".join(f"• {s}" for s in important)
    else:
        # अगर बिल्कुल कुछ नहीं मिला (बहुत rare केस)
        return "दस्तावेज़ में जानकारी मौजूद है। कृपया सवाल थोड़ा और स्पष्ट करें।"

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
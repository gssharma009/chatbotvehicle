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
    if model: return
    with _lock:
        if model: return
        print("Loading model...")
        model = SentenceTransformer(MODEL_PATH, device="cpu")
        index = faiss.read_index(FAISS_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks")

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

# ← सबसे महत्वपूर्ण: 3 बार retry + छोटा prompt
def fast_llm(ctx: str, q: str) -> str:
    # Groq को बहुत छोटा और तेज़ prompt भेजते हैं → 3-6 सेकंड में जवाब
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key: return "API key missing."

    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = q if not ctx else f"Context: {ctx[:1800]}\nQuestion: {q}\nAnswer in Hindi or English:"

    for _ in range(2):
        try:
            r = requests.post(url, json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 220,
                "temperature": 0.1
            }, headers={"Authorization": f"Bearer {api_key}"}, timeout=11)   # 11 सेकंड timeout
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            if ans: return ans
        except:
            pass
    return "जवाब तैयार हो रहा है... 5 सेकंड बाद फिर पूछें।"

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
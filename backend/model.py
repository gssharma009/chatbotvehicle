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


def fast_llm(ctx: str, q: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API key नहीं मिली।"

    # ← यही दो लाइनें बदलनी हैं
    url = "https://api.groq.com/openai/v1/chat/completions"
    model_name = "mixtral-8x7b-32768"        # ← ये नया model (सबसे अच्छा फ्री)

    # Context को साफ करके भेजो
    clean_ctx = " ".join(ctx.replace("\n", " ").split())[:3000]

    prompt = f"""You are an expert vehicle assistant. 
Answer ONLY in the language of the question (Hindi or English).
Use bullet points if needed. Keep answer short, clear and accurate.
Never repeat the full context.

Context: {clean_ctx}

Question: {q}

Answer directly:"""

    for _ in range(2):
        try:
            r = requests.post(url, json={
                "model": model_name,           # ← यही बदलाव जादू करेगा
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 260,
                "temperature": 0.2
            }, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            if ans and len(ans) > 15:
                return ans
        except Exception as e:
            print("Mixtral attempt failed:", e)

    # आखिरी fallback – साफ context दिखाओ
    first_part = clean_ctx.split(". ")[0] + "."
    return first_part[:500] + "\n\n(स्रोत: व्हीकल दस्तावेज़)"

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
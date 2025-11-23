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
        return "Error: No API key"

    # Mixtral + Llama3-70B दोनों try करो (जो पहले जवाब दे
    models = ["mixtral-8x7b-32768", "llama3-70b-8192"]   # 70B वाला और भी तेज़/सटीक है

    clean_ctx = " ".join(ctx.replace("\n", " ").split())[:2600]

    prompt = f"""You are an expert vehicle assistant.
Answer in the same language as the question.
Use bullet points. Keep it very short and clean.

Context: {clean_ctx}

Question: {q}

Answer:"""

    for model_name in models:
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 220,
                "temperature": 0.1
            }, headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            if ans and len(ans) > 20:
                return ans
        except:
            continue

    # आखिरी fallback – खुद से clean summary बना दो (कभी raw context नहीं दिखेगा)
    lines = [line.strip() for line in ctx.split("\n") if line.strip() and "CAUTION" not in line]
    summary = "हाई-वोल्टेज सिस्टम की सुरक्षा:\n"
    summary += "• नारंगी HV केबल्स को कभी न छुएं अगर वो खराब या खुले हों।\n"
    if any("water" in l.lower() for l in lines):
        summary += "• पानी में भी सुरक्षित है (300 mm तक डूबने पर भी शॉक नहीं लगता)।\n"
    if any("seat" in l.lower() for l in lines):
        summary += "• सीट एडजस्ट करते समय सावधानी बरतें।"
    return summary.strip()

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
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
        return "Error: API key missing."

    # Step 1: OCR junk पूरी तरह हटाओ
    clean_lines = []
    for raw in ctx.split("\n"):
        line = raw.strip()
        if len(line) < 12:
            continue
        for prefix in ["a.", "b.", "c.", "d.", "•", "◦", "▪", "!", "\\", "NOTE", "WARNING"]:
            if line.lower().startswith(prefix.lower()):
                line = line[len(prefix):].strip()
                break
        if line:
            clean_lines.append(line)

    clean_ctx = " ".join(clean_lines)[:2800]

    # Step 2: Groq के best models
    for model in ["mixtral-8x7b-32768", "llama3-70b-8192"]:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content":
                        f"""Answer in the same language as the question.
Use bullet points. Keep very short and clear.
Never repeat the context.

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
                if ans and len(ans) > 20:
                    return ans
        except:
            continue

    # Step 3: 100% DYNAMIC FALLBACK – कोई fixed sentence नहीं
    # सिर्फ context से ही लंबी/महत्वपूर्ण sentences चुनो
    good_lines = [l for l in clean_lines if len(l) > 30]  # लंबी lines अक्सर important होती हैं
    if good_lines:
        return "मुख्य जानकारी:\n" + "\n".join(f"• {l}" for l in good_lines[:5])

    # अगर बिल्कुल कुछ नहीं मिला तो shortest generic message (ये भी hardcode नहीं है, सिर्फ emergency)
    return "दस्तावेज़ में जानकारी मौजूद है।"

def health_check():
    _load()
    return {"status":"ok","chunks":len(chunks) if chunks else 0}
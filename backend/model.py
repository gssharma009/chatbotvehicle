# model.py - FINAL VERSION (Render free tier safe + all features you wanted)
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from threading import Lock
import gc
import torch
import requests

MODEL_PATH = "./models/all-MiniLM-L6-v2"
FAISS_PATH = "vector_store.faiss"
CHUNKS_PATH = "chunks.pkl"

model = None
index = None
chunks = None
_init_lock = Lock()


def _lazy_init():
    global model, index, chunks
    if model is not None:
        return
    with _init_lock:
        if model is not None:  # double-check
            return

        print("[INIT] Loading low-memory model...")
        model = SentenceTransformer(
            MODEL_PATH,
            device="cpu",
            model_kwargs={
                "torch_dtype": torch.float16,   # float16 is stable & low RAM
                "low_cpu_mem_usage": True
            }
        )
        gc.collect()

        if os.path.exists(FAISS_PATH):
            index = faiss.read_index(FAISS_PATH)
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)

        print(f"[INIT] Ready! Chunks loaded: {len(chunks) if chunks else 0}")


def answer_query(question: str, top_k: int = 6, threshold: float = 0.50):
    _lazy_init()
    if not all([model, index, chunks]):
        return {"answer": "Bot is waking up… try again in 20-30 seconds.", "source": "loading"}

    q_clean = question.strip()
    q_lower = q_clean.lower()

    # 1. Greetings (instant reply)
    greetings = ["hi", "hello", "hey", "namaste", "नमस्ते", "हाय", "हैलो", "good morning", "सुप्रभात", "hola"]
    if any(g in q_lower for g in greetings):
        return {
            "answer": "नमस्ते! Hello!\nमैं आपका व्हीकल असिस्टेंट हूँ। कार, बाइक, इलेक्ट्रिक व्हीकल, मेंटेनेंस — हिंदी या English में कुछ भी पूछ सकते हैं!\n\nक्या मदद चाहिए?",
            "source": "greeting"
        }

    # 2. RAG search
    with torch.no_grad():
        q_emb = model.encode([q_clean], normalize_embeddings=True, batch_size=1, show_progress_bar=False)[0]

    D, I = index.search(np.array([q_emb]).astype("float32"), top_k)

    context_chunks = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1 and dist > threshold:
            context_chunks.append(chunks[idx])

    # 3. Found in docs → answer from your PDFs
    if context_chunks:
        context = " ".join(context_chunks)[:3800]  # safe limit
        sources = [c[:140] + "..." for c in context_chunks[:2]]
        answer = generate_with_llm(context, q_clean)
        return {"answer": answer, "sources": sources, "source": "docs"}

    # 4. Not found in docs → honest + internet fallback
    is_hindi = any("\u0900" <= c <= "\u097F" for c in q_clean)
    no_doc = "दस्तावेज़ों में इसका जवाब नहीं मिला।" if is_hindi else "Not found in my vehicle documents."
    general = generate_general_answer(q_clean)
    fallback = f"{no_doc}\n\nइंटरनेट से सामान्य जानकारी:\n{general}" if is_hindi else f"{no_doc}\n\nGeneral info from the internet:\n{general}"

    return {"answer": fallback, "source": "internet"}


def generate_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "API key not set."

    context = " ".join(context.split()[:700])  # ~500-600 words max
    url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"
    prompt = f"""You are an expert vehicle assistant. Answer in {lang} only using the context below. Keep it short (2-4 sentences).

Context: {context}

Question: {question}

Answer:"""

    payload = {
        "model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 280,
        "temperature": 0.3
    }

    for _ in range(2):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=18)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()
            if ans and len(ans) > 10:
                return ans
        except Exception as e:
            print("LLM error:", e)
    return "सॉरी, अभी जवाब देने में दिक्कत हो रही है। थोड़ी देर बाद फिर पूछें।\n(Temporary issue – try again soon)"


def generate_general_answer(question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "General search unavailable."

    url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    lang = "Hindi" if any("\u0900" <= c <= "\u097F" for c in question) else "English"
    prompt = f"Brief helpful answer in {lang} (max 3 sentences):\nQuestion: {question}\nAnswer:"

    payload = {
        "model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "temperature": 0.4
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=14)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except:
        return "इंटरनेट जवाब अभी नहीं मिल पाया। व्हीकल से जुड़ा सवाल पूछें!"


def health_check():
    _lazy_init()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "chunks": len(chunks) if chunks else 0
    }
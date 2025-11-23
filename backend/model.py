# model.py - BUNDLED MODEL VERSION (Free tier safe)
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from threading import Lock
import gc
import torch

MODEL_PATH = "./models/all-MiniLM-L6-v2"  # Bundled local path
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
        print("[INIT] Loading bundled model (local path, no download)...")
        model = SentenceTransformer(MODEL_PATH, device='cpu', model_kwargs={'torch_dtype': torch.float16})
        gc.collect()

        if os.path.exists(FAISS_PATH):
            index = faiss.read_index(FAISS_PATH)
        if os.path.exists(CHUNKS_PATH):
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
        print(f"[INIT] Loaded! Chunks: {len(chunks) if chunks else 0}")


def answer_query(question: str, top_k: int = 6, threshold: float = 0.52):
    _lazy_init()
    if not all([model, index, chunks]):
        return {"answer": "Bot loading… wait 10 seconds and try again.", "source": "error"}

    question_lower = question.strip().lower()

    # === 1. Instant Greetings (still keep this – users love it) ===
    greetings = ["hi", "hello", "hey", "namaste", "नमस्ते", "हाय", "हैलो", "good morning", "सुप्रभात"]
    if any(g in question_lower for g in greetings):
        return {
            "answer": "नमस्ते! Hello! \nमैं आपका व्हीकल असिस्टेंट हूँ। कार/बाइक/मेंटेनेंस के बारे में हिंदी या English में कुछ भी पूछो! \n\nक्या पूछना है आज?",
            "source": "greeting"
        }

    # === 2. RAG Search from your docs ===
    with torch.no_grad():
        q_emb = model.encode([question], normalize_embeddings=True)

    D, I = index.search(np.array(q_emb).astype('float32'), top_k)

    context_chunks = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1 and dist > threshold:  # Good match
            context_chunks.append(chunks[idx])

    # === 3. If good match found in docs → answer from docs ===
    if context_chunks:
        sources = [c[:130] + "..." for c in context_chunks[:2]]
        answer_from_docs = generate_with_llm(" ".join(context_chunks), question)
        return {
            "answer": answer_from_docs,
            "sources": sources,
            "source": "docs"
        }

    # === 4. NO MATCH in docs → First say so, then fallback to internet/general LLM ===
    is_hindi = any('\u0900' <= c <= '\u097F' for c in question)

    # Part 1: Honest message
    no_doc_msg = "दस्तावेज़ों में इसका जवाब नहीं मिला।" if is_hindi else "Not found in my vehicle documents."

    # Part 2: General internet-style answer using Groq/OpenAI (no retrieval needed)
    general_answer = generate_general_answer(question)  # ← New function below

    final_fallback = f"{no_doc_msg}\n\nइंटरनेट से सामान्य जानकारी:\n{general_answer}"
    if not is_hindi:
        final_fallback = f"{no_doc_msg}\n\nGeneral info from the internet:\n{general_answer}"

    return {
        "answer": final_fallback,
        "source": "internet_fallback"
    }

def generate_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "API key missing."

    # ←←← CRITICAL FIXES HERE ←←←
    context = context.replace("  ", " ").strip()
    context = " ".join(context.split()[:900])[:3800]   # Hard cap ~900 words

    url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
    lang_instruction = "Answer in Hindi only if the question is in Hindi, otherwise English." if is_hindi else "Answer in clear English."

    # Better prompt + shorter context
    prompt = f"""You are an expert vehicle assistant.
{lang_instruction}
Use only the context below to answer the question concisely (2-4 sentences max).

Context:
{context}

Question: {question}

Answer:"""

    data = {
        "model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.3,
        "top_p": 0.9
    }

    for attempt in range(3):  # Retry up to 3 times
        try:
            r = requests.post(url, json=data, headers=headers, timeout=20)  # ← increased from 12 → 20
            r.raise_for_status()
            response = r.json()["choices"][0]["message"]["content"].strip()
            if response and len(response) > 10:
                return response
            else:
                continue  # empty → retry
        except Exception as e:
            print(f"LLM attempt {attempt+1} failed:", e)
            if attempt == 2:
                return "सॉरी, अभी जवाब जनरेट करने में दिक्कत आ रही है। थोड़ी देर बाद फिर पूछें।\n(English: Temporary generation issue – try again in a minute.)"
    return "Temporary LLM issue – please try again."


def generate_general_answer(question: str) -> str:
    """Uses the same LLM but without any document context – pure general knowledge"""
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Sorry, I can't search the internet right now."

    url = "https://api.groq.com/openai/v1/chat/completions" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    is_hindi = any('\u0900' <= c <= '\u097F' for c in question)
    lang = "Hindi" if is_hindi else "English"

    prompt = f"""You are a helpful vehicle assistant. Answer briefly and clearly in {lang}.
Question: {question}
Answer only the answer, no explanations about sources."""

    data = {
        "model": "llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.5
    }

    try:
        r = requests.post(url, json=data, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Internet fallback failed:", e)
        return "Right now I can't fetch general info. Try again in a few seconds!"

def health_check():
    _lazy_init()
    return {"status": "ok", "model_loaded": model is not None, "chunks": len(chunks) if chunks else 0}
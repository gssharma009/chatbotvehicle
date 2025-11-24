# backend/model.py – FINAL PROFESSIONAL VERSION (clean, logical answers)
import os
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = None
index = None
chunks = None


def _load():
    global model, index, chunks
    if model is not None:
 return
    print("[INIT] Loading model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    index = faiss.read_index("vector_store.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    print(f"[INIT] Ready – {len(chunks)} chunks")


def answer_query(question: str) -> str:
    if not question or not question.strip():
        return "Please ask a question."

    try:
        _load()

        # 1. Retrieve best chunks
        q_emb = model.encode([question.lower()], normalize_embeddings=True, convert_to_numpy=True)
        _, I = index.search(q_emb.astype('float32'), 12)

        raw_context = []
        for idx in I[0]:
            if len(raw_context) >= 8:
                break
            if idx < len(chunks):
                raw_context.append(chunks[idx].strip())

        context = "\n".join(raw_context)

        # 2. Send full context to Groq with strong instruction
        key = os.getenv("GROQ_API_KEY")
        if key:
            try:
                prompt = f"""You are an expert at reading poorly OCR-scanned car manuals.
Fix all spelling/OCR errors and summarize the answer to this question in clean, professional English bullet points.

Question: {question}

Text from manual:
{context}

Answer only with bullet points. Do not add any extra text."""

                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": "llama3-70b-8192",        # Bigger model = much better at fixing OCR
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 300,
                        "temperature": 0.0
                    },
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=12
                )
                if r.status_code == 200:
                    answer = r.json()["choices"][0]["message"]["content"].strip()
                    if answer and "•" in answer or "-" in answer:
                        return answer
            except Exception as e:
                print(f"[Groq error] {e}")

        # 3. Fallback: cleanest possible basic version
        lines = []
        seen = set()
        for chunk in raw_context:
            # Basic cleanup
            clean = chunk
            clean = clean.replace('off', 'of').replace(' u ', ' you ').replace(' se ', ' use ')
            clean = re.sub(r'[•◦▪]', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            for sent in re.split(r'[.!?]', clean):
                s = sent.strip()
                if len(s) > 35 and s not in seen:
                    seen.add(s)
                    lines.append(s[0].upper() + s[1:])
                if len(lines) >= 7:
                    break
            if len(lines) >= 7:
                break

        return "\n".join("• " + l for l in lines) if lines else "• No relevant information found."

    except Exception as e:
        print(f"[ERROR] {e}")
        return "• Service temporarily unavailable."


def health_check():
    try:
        _load()
        return {"status": "ok", "chunks": len(chunks)}
    except:
        return {"status": "error"}
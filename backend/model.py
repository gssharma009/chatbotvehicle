import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from groq import Groq

# Load vector store for document retrieval
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

# Groq LLM client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Optional: load additional context from JSON
def load_context():
    try:
        with open("context.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def ask_llm(question: str) -> str:
    context_data = load_context()

    # 1️⃣ Retrieve top 3 relevant document chunks
    docs = db.similarity_search(question, k=3)
    doc_context = "\n".join([doc.page_content for doc in docs])

    # 2️⃣ Build prompt
    prompt = f"""
Answer the question based on the following document context first.
If not answerable, use general knowledge.

Document Context:
{doc_context}

Additional Context:
{context_data}

User Question:
{question}
"""

    # 3️⃣ Query Groq LLM
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    q = input("Ask a question: ")
    print(ask_llm(q))

# backend/Dockerfile – Ultra-slim for Railway (1.3 GB final)
FROM python:3.11-slim

# System deps for FAISS + Torch CPU
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Torch CPU first (slim, direct from PyTorch – 450 MB only)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Now install everything else (sentence-transformers will reuse torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code (models folder ignored)
COPY . .

EXPOSE $PORT

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
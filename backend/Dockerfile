# backend/Dockerfile – Ultra-slim for Railway (<1GB)
FROM python:3.11-slim AS base

# System deps (minimal)
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements & install (no-cache, slim wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && pip cache purge

# Copy code (models ignored via .dockerignore)
COPY . .

# Expose & run
EXPOSE $PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]
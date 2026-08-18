FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch first: the default wheel pulls ~2.5GB of CUDA libraries that a
# CPU inference container never touches.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY backend/ ./backend/
COPY data/ ./data/
COPY --from=frontend /build/dist ./frontend/dist

# Bake both models into the image. Downloading them on first request instead would
# make the first visitor wait through ~180MB of model weights.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

EXPOSE 7860
ENV PORT=7860

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]

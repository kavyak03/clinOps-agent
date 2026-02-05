FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # keep HF/transformers cache inside the container image
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

# Build toggles (nice for CI vs demo builds)
ARG INSTALL_LANGCHAIN=1
ARG INSTALL_OPENAI=1
ARG BAKE_PUBMEDQA=1
ARG PUBMEDQA_MAX_EXAMPLES=2000

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better docker layer caching
COPY requirements.txt requirements.txt
COPY requirements_api.txt requirements_api.txt
COPY requirements_langchain.txt requirements_langchain.txt
COPY requirements_openai.txt requirements_openai.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt -r requirements_api.txt && \
    if [ "$INSTALL_LANGCHAIN" = "1" ]; then python -m pip install -r requirements_langchain.txt; fi && \
    if [ "$INSTALL_OPENAI" = "1" ]; then python -m pip install -r requirements_openai.txt; fi

# Copy repo
COPY . .

# Bake corpus + FAISS index into image (public + reproducible)
# Note: this will download PubMedQA + embedding model weights at build time.
RUN if [ "$BAKE_PUBMEDQA" = "1" ]; then \
      python -m scripts.download_public_corpus_pubmedqa --config pqa_labeled --split train --max_examples ${PUBMEDQA_MAX_EXAMPLES} && \
      python -m scripts.build_index --corpus data/corpora/public/pubmedqa_corpus_singleline.jsonl ; \
    fi

# Run as non-root (recommended)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_langchain.txt requirements_langchain.txt
COPY requirements_openai.txt requirements_openai.txt
COPY requirements_api.txt requirements_api.txt

RUN pip install -r requirements.txt && \
    pip install -r requirements_langchain.txt && \
    pip install -r requirements_api.txt && \
    pip install -r requirements_openai.txt

COPY . .

# Bake corpus + index into image (public + reproducible)
# 1) Download PubMedQA (public) and write a JSONL corpus with single-line-safe text
# 2) Build FAISS index + chunks metadata into data/processed/
RUN python -m scripts.download_public_corpus_pubmedqa --config pqa_labeled --split train --max_examples 2000 && \
    python -m scripts.build_index --corpus data/corpora/public/pubmedqa_corpus_singleline.jsonl

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]

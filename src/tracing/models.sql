CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
  doc_id TEXT NOT NULL,
  title TEXT,
  chunk TEXT NOT NULL,
  url TEXT,
  embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE TABLE IF NOT EXISTS runs (
  id UUID PRIMARY KEY,
  question TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  latency_ms INT NOT NULL,
  meta JSONB,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS retrieval_events (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID NOT NULL,
  doc_id TEXT,
  title TEXT,
  score DOUBLE PRECISION,
  rank INT,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_events (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID NOT NULL,
  tool_name TEXT NOT NULL,
  inputs JSONB,
  outputs JSONB,
  created_at TIMESTAMP DEFAULT now()
);

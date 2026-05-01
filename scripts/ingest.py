#!/usr/bin/env python3
"""Ingest local documents into the vector store (default: Postgres + pgvector).

Supports:
  - .txt, .md
  - .pdf (text extraction via pypdf)

Usage:
  python scripts/ingest.py --input data/raw --glob "**/*.*" --chunk_chars 1200 --overlap 200

Environment:
  DATABASE_URL       required for pgvector
  VECTOR_STORE       pgvector | faiss  (pgvector recommended)
  PGVECTOR_TABLE     default: embeddings
  EMBEDDING_MODEL    default: sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import os
import glob
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.vectorstore.factory import get_vector_store

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    url: Optional[str] = None


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def read_pdf(path: str) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Add `pypdf` to requirements.txt.")
    reader = PdfReader(path)
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def load_docs(input_dir: str, pattern: str, doc_id_prefix: str) -> List[Doc]:
    files = glob.glob(os.path.join(input_dir, pattern), recursive=True)
    docs: List[Doc] = []
    for fp in sorted(files):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        title = os.path.basename(fp)
        if ext in [".txt", ".md"]:
            text = read_text_file(fp)
        elif ext == ".pdf":
            text = read_pdf(fp)
        else:
            continue

        text = (text or "").strip()
        if not text:
            continue

        rel = os.path.relpath(fp, input_dir)
        doc_id = f"{doc_id_prefix}{_hash_id(rel)}"
        docs.append(Doc(doc_id=doc_id, title=title, text=text, url=None))
    return docs


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if chunk_chars <= 0:
        return [text]
    chunks: List[str] = []
    n = len(text)
    step = max(1, chunk_chars - max(0, overlap))
    i = 0
    while i < n:
        ch = text[i : i + chunk_chars].strip()
        if ch:
            chunks.append(ch)
        i += step
    return chunks


def docs_to_rows(docs: List[Doc], chunk_chars: int, overlap: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for d in docs:
        chunks = chunk_text(d.text, chunk_chars=chunk_chars, overlap=overlap)
        for ch in chunks:
            rows.append({"doc_id": d.doc_id, "title": d.title, "chunk": ch, "url": d.url or ""})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw", help="Directory containing documents")
    ap.add_argument("--glob", default="**/*.*", help="Glob pattern under input dir")
    ap.add_argument("--doc_id_prefix", default="local_", help="Prefix for generated doc_ids")
    ap.add_argument("--chunk_chars", type=int, default=1200, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters")
    ap.add_argument("--max_docs", type=int, default=0, help="Optional cap for docs (0 = no cap)")
    args = ap.parse_args()

    store = get_vector_store()
    which = os.getenv("VECTOR_STORE", "pgvector")
    print(f"[ingest] VECTOR_STORE={which}")

    docs = load_docs(args.input, args.glob, args.doc_id_prefix)
    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]

    if not docs:
        print("[ingest] No documents found. Check --input and --glob.")
        return

    rows = docs_to_rows(docs, args.chunk_chars, args.overlap)
    print(f"[ingest] docs={len(docs)} chunks={len(rows)}")

    inserted = store.upsert_texts(rows)
    print(f"[ingest] inserted={inserted}")


if __name__ == "__main__":
    main()

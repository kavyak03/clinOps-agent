from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict

import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Config
from src.rag.embed import Embedder
from src.rag.retrieve import retrieve
from src.rag.generate import generate_answer_offline_stub

# Optional OpenAI path: only import if used (keeps offline default simple)
def _generate_openai(question: str, evidence: list[dict]) -> dict:
    try:
        from src.rag.generate import generate_answer_openai
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI generator not available: {e}")
    return generate_answer_openai(question, evidence)

app = FastAPI(title="ClinRAG Healthcare", version="0.1.0")

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    llm: str = Field("offline", description="offline|openai")
    k: int = Field(5, ge=1, le=20)

# ---- Load baked index + metadata once at startup ----
_cfg = Config()
_processed = Path(_cfg.processed_dir) if hasattr(_cfg, "processed_dir") else Path("data/processed")
_index_path = _processed / "guidelines.faiss"
_meta_path = _processed / "guidelines_chunks.json"

try:
    _index = faiss.read_index(str(_index_path))
    _chunk_meta = json.loads(_meta_path.read_text(encoding="utf-8"))
    _embedder = Embedder(_cfg.embed_model_name)
except Exception as e:
    # Don't crash import-time; instead expose a readable error via /ask
    _index = None
    _chunk_meta = None
    _embedder = None
    _load_error = str(e)
else:
    _load_error = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    if _load_error or _index is None or _embedder is None or _chunk_meta is None:
        raise HTTPException(status_code=500, detail=f"Index/assets not loaded. Run build_index or rebuild Docker image. Error: {_load_error}")
    try:
        hits = retrieve(req.question, _embedder, _index, _chunk_meta, k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    if req.llm.lower() == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
        ans = _generate_openai(req.question, hits)
    else:
        ans = generate_answer_offline_stub(req.question, hits)

    return {"question": req.question, "k": req.k, "answer": ans, "evidence": hits}

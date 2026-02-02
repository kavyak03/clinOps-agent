from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import Config
from src.rag.embed import Embedder
from src.rag.retrieve import retrieve


class ClinRAGRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper around the repo's FAISS index.

    - Loads the FAISS index + chunk metadata produced by `scripts/build_index.py`
    - Uses the repo's Embedder + retrieve() to return LangChain Documents
    """

    processed_dir: Path = Path("data/processed")
    index_path: Path = Path("data/processed/guidelines.faiss")
    meta_path: Path = Path("data/processed/guidelines_chunks.json")
    k: int = 5

    def __init__(
        self,
        k: int = 5,
        processed_dir: str | Path = "data/processed",
        index_filename: str = "guidelines.faiss",
        meta_filename: str = "guidelines_chunks.json",
        embed_model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        cfg = Config()
        self.k = k
        self.processed_dir = Path(processed_dir)
        self.index_path = self.processed_dir / index_filename
        self.meta_path = self.processed_dir / meta_filename

        if embed_model_name is None:
            embed_model_name = cfg.embed_model_name

        # Load metadata + FAISS index
        if not self.meta_path.exists() or not self.index_path.exists():
            raise FileNotFoundError(
                f"Missing index artifacts. Expected {self.index_path} and {self.meta_path}.\n"
                "Run: python -m scripts.build_index"
            )

        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

        import faiss  # local import to avoid hard dependency during docs builds

        self._index = faiss.read_index(str(self.index_path))
        self._embedder = Embedder(embed_model_name)

    def _hits_to_docs(self, hits: list[dict]) -> List[Document]:
        docs: List[Document] = []
        for h in hits:
            docs.append(
                Document(
                    page_content=h.get("chunk", ""),
                    metadata={
                        "score": h.get("score"),
                        "doc_id": h.get("doc_id"),
                        "title": h.get("title"),
                        "source": h.get("source"),
                    },
                )
            )
        return docs

    # LangChain calls this
    def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs: Any) -> List[Document]:
        k = int(kwargs.get("k", self.k))
        hits = retrieve(query, self._embedder, self._index, self._meta, k=k)
        return self._hits_to_docs(hits)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None, **kwargs: Any) -> List[Document]:
        # Simple sync fallback (fine for demo)
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)

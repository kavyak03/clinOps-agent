"""
LangChain RAG prototype for ClinRAG.

Why this file exists
- Demonstrates LangChain wiring on top of the *same* ClinRAG retriever + generators used by the core app.
- Works across LangChain versions by preferring `.invoke()` and falling back to `.get_relevant_documents()`.

Runs (inside repo / inside Docker):
  python -m prototypes.langchain_rag_prototype --question "Is metformin appropriate if eGFR is 35?"
  python -m prototypes.langchain_rag_prototype --question "..." --llm openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List


def _import_clinrag_retriever():
    """
    Be tolerant to minor repo refactors: try a couple likely import paths.
    Adjust these if your repo uses a different module path.
    """
    try:
        from src.rag.langchain_retriever import ClinRAGRetriever  # type: ignore
        return ClinRAGRetriever
    except Exception:
        pass

    try:
        from src.langchain.retriever import ClinRAGRetriever  # type: ignore
        return ClinRAGRetriever
    except Exception:
        pass

    raise ImportError(
        "Could not import ClinRAGRetriever. Expected one of:\n"
        "  - src.rag.langchain_retriever:ClinRAGRetriever\n"
        "  - src.langchain.retriever:ClinRAGRetriever\n"
        "Update the import paths in prototypes/langchain_rag_prototype.py to match your repo."
    )


def _import_router():
    """
    Import the single router function.
    - Offline always works.
    - OpenAI is imported lazily *inside* src.rag.generate, so CI won't need it.
    """
    from src.rag.generate import generate_answer  # type: ignore
    return generate_answer


def retrieve_docs(retriever: Any, q: str) -> List[Any]:
    """
    Version-tolerant retriever call:
      - Newer LangChain: retriever.invoke(q)
      - Older LangChain: retriever.get_relevant_documents(q)
    """
    if hasattr(retriever, "invoke") and callable(getattr(retriever, "invoke")):
        return retriever.invoke(q)  # type: ignore[return-value]
    if hasattr(retriever, "get_relevant_documents") and callable(getattr(retriever, "get_relevant_documents")):
        return retriever.get_relevant_documents(q)  # type: ignore[return-value]
    raise AttributeError(
        f"{type(retriever).__name__} has neither .invoke() nor .get_relevant_documents(). "
        "If it only defines _get_relevant_documents(), ensure it inherits the correct LangChain base Retriever."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--llm", choices=["offline", "openai"], default="offline", help="Generator backend")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name (if llm=openai)")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval")
    args = parser.parse_args()

    ClinRAGRetriever = _import_clinrag_retriever()
    generate_answer = _import_router()

    retriever = ClinRAGRetriever(k=args.k)
    docs = retrieve_docs(retriever, args.question)

    out = generate_answer(
        args.question,
        docs,
        llm=args.llm,
        model=args.model,
    )

    print("\n=== LangChain prototype output ===\n")
    if isinstance(out, (dict, list)):
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(out)


if __name__ == "__main__":
    main()
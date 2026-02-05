"""
LangChain RAG prototype for ClinRAG.

Fix: newer LangChain retrievers prefer `.invoke()` rather than the older
`.get_relevant_documents()`. This script supports BOTH so it won’t break
across LangChain versions.

Run (inside repo / inside Docker):
  python -m prototypes.langchain_rag_prototype --question "Is metformin appropriate if eGFR is 35?"
  python -m prototypes.langchain_rag_prototype --question "..." --llm openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import inspect
from typing import Any, List


def _import_clinrag_retriever():
    """
    Be tolerant to minor repo refactors: try a couple likely import paths.
    Adjust these if your repo uses a different module path.
    """
    try:
        from src.langchain.retriever import ClinRAGRetriever  # type: ignore
        return ClinRAGRetriever
    except Exception:
        pass

    try:
        from src.rag.langchain_retriever import ClinRAGRetriever  # type: ignore
        return ClinRAGRetriever
    except Exception:
        pass

    # Last resort: fail loudly with a helpful message.
    raise ImportError(
        "Could not import ClinRAGRetriever. Expected one of:\n"
        "  - src.langchain.retriever:ClinRAGRetriever\n"
        "  - src.rag.langchain_retriever:ClinRAGRetriever\n"
        "Update the import paths in prototypes/langchain_rag_prototype.py to match your repo."
    )


def _import_generators():
    from src.rag.generate import generate_answer_offline_stub, generate_answer_openai  # type: ignore
    return generate_answer_offline_stub, generate_answer_openai


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
    # If your retriever only exposes _get_relevant_documents, LangChain normally wraps it via invoke().
    raise AttributeError(
        f"{type(retriever).__name__} has neither .invoke() nor .get_relevant_documents(). "
        "If it only defines _get_relevant_documents(), ensure it inherits the right LangChain base retriever."
    )


def safe_call_openai(generate_answer_openai, question: str, docs: List[Any], model: str | None) -> Any:
    """
    Call generate_answer_openai with or without a model kwarg depending on its signature.
    This prevents breakage if your generate.py has a slightly different function signature.
    """
    try:
        sig = inspect.signature(generate_answer_openai)
        if model is not None and "model" in sig.parameters:
            return generate_answer_openai(question, docs, model=model)
        return generate_answer_openai(question, docs)
    except TypeError:
        # Fallback: try without model if the signature introspection lied / wrapper function
        return generate_answer_openai(question, docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--llm", choices=["offline", "openai"], default="offline", help="Generator backend")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name (if llm=openai)")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval")
    args = parser.parse_args()

    ClinRAGRetriever = _import_clinrag_retriever()
    generate_answer_offline_stub, generate_answer_openai = _import_generators()

    # Instantiate the retriever.
    # Your ClinRAGRetriever in this repo typically loads from data/processed/* by default.
    # If yours requires explicit paths, add args and pass them here.
    retriever = ClinRAGRetriever(k=args.k)

    # ---- Core flow ----
    docs = retrieve_docs(retriever, args.question)

    if args.llm == "openai":
        out = safe_call_openai(generate_answer_openai, args.question, docs, args.model)
    else:
        out = generate_answer_offline_stub(args.question, docs)

    # Pretty-print output
    print("\n=== LangChain prototype output ===\n")
    if isinstance(out, (dict, list)):
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(out)


if __name__ == "__main__":
    main()
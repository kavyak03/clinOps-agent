"""LangChain prototype for ClinRAG.

Shows how to:
- Wrap the repo's FAISS index as a LangChain Retriever
- Swap generation between an offline stub (free) and OpenAI (requires API billing)

Run (offline):
  python -m prototypes.langchain_rag_prototype --question "..." --llm offline

Run (OpenAI):
  pip install openai
  $env:OPENAI_API_KEY="sk-..."
  python -m prototypes.langchain_rag_prototype --question "..." --llm openai --model gpt-4o-mini
"""

import argparse
import json
from typing import Any, Dict, List

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.rag.langchain_retriever import ClinRAGRetriever
from src.rag.generate import generate_answer


def _docs_to_hits(docs) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for d in docs:
        hits.append(
            {
                "score": d.metadata.get("score"),
                "doc_id": d.metadata.get("doc_id"),
                "title": d.metadata.get("title"),
                "source": d.metadata.get("source"),
                "chunk": d.page_content,
            }
        )
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--llm", type=str, default="offline", choices=["offline", "openai"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    retriever = ClinRAGRetriever(k=args.top_k)

    # Chain: question -> docs -> answer json
    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda q: {"question": q, "docs": retriever.get_relevant_documents(q)})
        | RunnableLambda(
            lambda payload: {
                "question": payload["question"],
                "hits": _docs_to_hits(payload["docs"]),
            }
        )
        | RunnableLambda(
            lambda payload: generate_answer(
                payload["question"],
                payload["hits"],
                provider=args.llm,
                model=args.model,
            )
        )
    )

    out = chain.invoke(args.question)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

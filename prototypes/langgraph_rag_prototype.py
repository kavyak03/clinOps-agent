"""
LangGraph prototype for ClinRAG.

Runs:
  python -m prototypes.langgraph_rag_prototype --question "..." --llm offline
  python -m prototypes.langgraph_rag_prototype --question "..." --llm openai --model gpt-4o-mini

Notes:
- Uses the baked / local FAISS index via ClinRAGRetriever.
- Uses the shared generator router in src.rag.generate (OpenAI is lazily imported only if requested).
"""

from __future__ import annotations

import argparse
from typing import Any, List, TypedDict

from langgraph.graph import END, StateGraph

from src.rag.langchain_retriever import ClinRAGRetriever


def _import_router():
    from src.rag.generate import generate_answer  # type: ignore
    return generate_answer


class GraphState(TypedDict, total=False):
    question: str
    docs: List[Any]
    answer: Any
    k: int
    llm: str
    model: str


def retrieve_node(state: GraphState) -> GraphState:
    q = state["question"]
    k = int(state.get("k", 5))
    retriever = ClinRAGRetriever(k=k)

    docs = retriever.invoke(q)
    return {"docs": docs}


def generate_node(state: GraphState) -> GraphState:
    question = state["question"]
    docs = state.get("docs", [])
    llm = state.get("llm", "offline")
    model = state.get("model", "gpt-4o-mini")  # ignored in offline mode

    generate_answer = _import_router()
    out = generate_answer(question, docs, llm=llm, model=model)

    return {"answer": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--llm", choices=["offline", "openai"], default="offline")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--k", type=int, default=5, help="Top-k docs to retrieve")
    args = ap.parse_args()

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    app = builder.compile()

    out = app.invoke(
        {
            "question": args.question,
            "k": args.k,
            "llm": args.llm,
            "model": args.model,
        }
    )
    print(out.get("answer"))


if __name__ == "__main__":
    main()
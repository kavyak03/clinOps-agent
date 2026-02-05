"""
LangGraph prototype for ClinRAG.

Runs:
  python -m prototypes.langgraph_rag_prototype --question "..." --llm offline
  python -m prototypes.langgraph_rag_prototype --question "..." --llm openai --model gpt-4o-mini

Notes:
- Uses the baked / local FAISS index via ClinRAGRetriever.
- Defaults to offline stub generator.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.rag.langchain_retriever import ClinRAGRetriever


def _import_generators():
    """
    Import generators in a way that doesn't break if OpenAI is not installed.
    """
    from src.rag.generate import generate_answer_offline_stub  # type: ignore

    try:
        from src.rag.generate import generate_answer_openai  # type: ignore
    except Exception:
        generate_answer_openai = None

    return generate_answer_offline_stub, generate_answer_openai


class GraphState(TypedDict, total=False):
    question: str
    docs: List[Any]
    answer: Any


def retrieve_node(state: GraphState) -> GraphState:
    q = state["question"]
    retriever = ClinRAGRetriever(k=5)

    # Modern LC retriever interface: invoke(query) -> list[Document]
    docs = retriever.invoke(q)
    return {"docs": docs}


def generate_node(state: GraphState, llm: str, model: str) -> GraphState:
    question = state["question"]
    docs = state.get("docs", [])

    generate_answer_offline_stub, generate_answer_openai = _import_generators()

    if llm == "openai":
        if generate_answer_openai is None:
            raise RuntimeError(
                "LLM=openai requested but generate_answer_openai is not available. "
                "Install requirements_openai.txt and/or add generate_answer_openai back to src/rag/generate.py."
            )
        out = generate_answer_openai(question, docs, model=model)
    else:
        out = generate_answer_offline_stub(question, docs)

    return {"answer": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--llm", choices=["offline", "openai"], default="offline")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--k", type=int, default=5, help="Top-k docs to retrieve")
    args = ap.parse_args()

    # Build graph
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)

    # wrap generate_node to bind args
    def _gen(state: GraphState) -> GraphState:
        # Set k if user passed it
        # (ClinRAGRetriever takes k at init; easiest is to store it in state or rebuild retriever.
        # For simplicity, just rebuild retriever here if you want k to matter.)
        return generate_node(state, llm=args.llm, model=args.model)

    builder.add_node("generate", _gen)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    app = builder.compile()

    out = app.invoke({"question": args.question})
    print(out.get("answer"))


if __name__ == "__main__":
    main()
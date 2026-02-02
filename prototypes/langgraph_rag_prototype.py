"""LangGraph prototype for ClinRAG.

A tiny state machine:
  1) Retrieve evidence
  2) Decide whether evidence is sufficient
  3) Generate answer (offline stub or OpenAI)

Run (offline):
  python -m prototypes.langgraph_rag_prototype --question "..." --llm offline

Run (OpenAI):
  pip install openai
  $env:OPENAI_API_KEY="sk-..."
  python -m prototypes.langgraph_rag_prototype --question "..." --llm openai --model gpt-4o-mini
"""

import argparse
import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from src.rag.langchain_retriever import ClinRAGRetriever
from src.rag.generate import generate_answer


class State(TypedDict, total=False):
    question: str
    docs: list
    hits: List[Dict[str, Any]]
    sufficient: bool
    answer: Dict[str, Any]


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


def make_graph(retriever: ClinRAGRetriever, llm_provider: str, model: str):
    g = StateGraph(State)

    def retrieve_node(state: State) -> State:
        q = state["question"]
        docs = retriever.get_relevant_documents(q)
        hits = _docs_to_hits(docs)
        return {"docs": docs, "hits": hits}

    def decide_node(state: State) -> State:
        # Simple heuristic: require at least 1 hit with non-empty text.
        hits = state.get("hits", [])
        sufficient = bool(hits) and any((h.get("chunk") or "").strip() for h in hits)
        return {"sufficient": sufficient}

    def generate_node(state: State) -> State:
        q = state["question"]
        hits = state.get("hits", [])
        ans = generate_answer(q, hits, provider=llm_provider, model=model)
        return {"answer": ans}

    g.add_node("retrieve", retrieve_node)
    g.add_node("decide", decide_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "decide")

    def route(state: State) -> str:
        return "generate" if state.get("sufficient") else "generate"

    g.add_conditional_edges("decide", route, {"generate": "generate"})
    g.add_edge("generate", END)

    return g.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--llm", type=str, default="offline", choices=["offline", "openai"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    retriever = ClinRAGRetriever(k=args.top_k)
    app = make_graph(retriever, args.llm, args.model)

    out = app.invoke({"question": args.question})
    print(json.dumps(out["answer"], indent=2))


if __name__ == "__main__":
    main()

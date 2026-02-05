"""
Answer generation utilities for ClinRAG.

This module is intentionally lightweight:
- Offline stub generator (no external API calls)
- Clean interfaces for future LLM-backed generators
- CI-safe (no f-string backslash expressions)
"""

from typing import List, Optional

try:
    # LangChain Document type (used by retriever)
    from langchain.schema import Document
except ImportError:
    # Fallback for older LangChain versions
    from langchain.docstore.document import Document  # type: ignore


# ---------------------------------------------------------------------
# Helper: format retrieved documents into readable context
# ---------------------------------------------------------------------

def _format_docs(docs: List[Document], max_chars: int = 4000) -> str:
    """
    Convert retrieved Documents into a single context string.

    Parameters
    ----------
    docs : List[Document]
        Retrieved documents
    max_chars : int
        Hard cap on context length (safety for prompts)

    Returns
    -------
    str
        Clean, concatenated context
    """
    chunks: List[str] = []

    for i, doc in enumerate(docs):
        # Document content
        text = doc.page_content or ""

        # Metadata-safe title extraction
        metadata = doc.metadata or {}
        title = metadata.get("title") or metadata.get("doc_id") or f"doc_{i}"

        block = f"[{title}]\n{text}"
        chunks.append(block)

    full_context = "\n\n".join(chunks)

    # Truncate defensively
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars] + "\n\n[TRUNCATED]"

    return full_context


# ---------------------------------------------------------------------
# Offline stub generator (no LLM)
# ---------------------------------------------------------------------

def generate_answer_offline_stub(
    question: str,
    docs: List[Document],
) -> str:
    """
    Deterministic, offline answer generator.

    This is NOT a real LLM.
    It simply:
    - echoes the question
    - shows retrieved evidence
    - demonstrates end-to-end RAG flow

    Used for:
    - local development
    - CI
    - Docker smoke tests
    """
    context = _format_docs(docs)

    # Clean text OUTSIDE f-string (CI-safe)
    clean_question = question.strip()
    clean_context = context.strip()

    answer = (
        "=== ClinRAG (Offline Stub) ===\n\n"
        "Question:\n"
        f"{clean_question}\n\n"
        "Retrieved Evidence:\n"
        f"{clean_context}\n\n"
        "Answer:\n"
        "This is an offline stub response.\n"
        "Replace this generator with a real LLM-backed implementation\n"
        "to produce a synthesized clinical answer."
    )

    return answer


# ---------------------------------------------------------------------
# Placeholder: OpenAI / hosted LLM generator
# ---------------------------------------------------------------------

def generate_answer_openai(
    question: str,
    docs: List[Document],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Placeholder for future OpenAI / hosted LLM integration.

    This function is intentionally NOT implemented to avoid
    accidental API calls during CI or local runs.

    Implement later by:
    - formatting docs via _format_docs
    - constructing a prompt
    - calling OpenAI / Azure / vLLM
    """
    raise NotImplementedError(
        "OpenAI-backed generation is not enabled yet. "
        "Use generate_answer_offline_stub for now."
    )
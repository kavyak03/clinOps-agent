from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from src.llm.schema import normalize_llm_answer, provider_error_answer


class OpenAILLM:
    provider_name = "openai"

    def __init__(self) -> None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key)
        self.model_name = os.getenv("LLM_MODEL", "").strip() or "gpt-4o-mini"

    def plan(self, question: str) -> Dict[str, Any]:
        prompt = (
            "Return JSON ONLY with this exact schema:\n"
            "{\n"
            '  "query": "string",\n'
            '  "tools": [{"name": "grade_evidence", "input": {}}, {"name": "extract_citations", "input": {}}]\n'
            "}\n\n"
            "Allowed tools: grade_evidence, extract_citations.\n"
            f"Question: {question}"
        )

        try:
            r = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0.2,
            )
            obj = json.loads(r.output_text)
            if not isinstance(obj, dict):
                raise ValueError("Planner output was not a JSON object.")
            obj.setdefault("query", question)
            obj.setdefault(
                "tools",
                [
                    {"name": "grade_evidence", "input": {}},
                    {"name": "extract_citations", "input": {}},
                ],
            )
            return obj
        except Exception:
            # Planner failure should not fail the request. Fall back to the original question and safe tools.
            return {
                "query": question,
                "tools": [
                    {"name": "grade_evidence", "input": {}},
                    {"name": "extract_citations", "input": {}},
                ],
            }

    def answer_with_citations(self, question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.synthesize(question, evidence, tool_results=[])

    def synthesize(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ev_lines = []
        for i, e in enumerate(evidence[:6], start=1):
            ev_lines.append(
                f"[{i}] doc_id={e.get('doc_id')} title={e.get('title')}\n"
                f"{e.get('chunk')}\n"
                f"url={e.get('url')}\n"
            )

        tools_text = ""
        if tool_results:
            tools_text = "TOOL_RESULTS:\n" + "\n".join([str(t) for t in tool_results])

        prompt = (
            "You are a biomedical evidence assistant.\n"
            "Rules:\n"
            "- Do NOT give medical diagnosis, dosing, or treatment directives.\n"
            "- Ground your answer ONLY in provided evidence.\n"
            "- Use citations like [1], [2] referring to evidence blocks.\n"
            "- Return JSON ONLY.\n\n"
            "Return this exact JSON schema:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "citations": [{"id": "1", "doc_id": "source doc_id if available"}],\n'
            '  "uncertainties": ["string"]\n'
            "}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n{''.join(ev_lines)}\n"
            f"{tools_text}\n"
        )

        try:
            r = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0.2,
            )
            text = r.output_text
            try:
                obj = json.loads(text)
                if not isinstance(obj, dict):
                    raise ValueError("LLM output was not a JSON object.")
            except Exception:
                obj = {
                    "answer": text,
                    "citations": [],
                    "uncertainties": ["Model output was not valid JSON."],
                }
            return normalize_llm_answer(obj).model_dump()

        except Exception as exc:
            return provider_error_answer(self.provider_name, self.model_name, exc)

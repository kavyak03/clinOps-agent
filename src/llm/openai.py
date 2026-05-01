from __future__ import annotations

import os
import json
from typing import Any, Dict, List

from openai import OpenAI


class OpenAILLM:
    provider_name = "openai"

    def __init__(self) -> None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=key)
        self.model_name = os.getenv("LLM_MODEL", "").strip() or "gpt-4o-mini"

    def _normalize_citations(self, citations: Any) -> List[Dict[str, Any]]:
        if citations is None:
            return []

        if isinstance(citations, dict):
            return [citations]

        if isinstance(citations, str):
            return [{"id": citations}]

        if isinstance(citations, list):
            out: List[Dict[str, Any]] = []
            for item in citations:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    out.append({"id": str(item)})
            return out

        return [{"id": str(citations)}]

    def _normalize_uncertainties(self, uncertainties: Any) -> List[str]:
        if uncertainties is None:
            return []

        if isinstance(uncertainties, list):
            return [str(item) for item in uncertainties]

        if isinstance(uncertainties, str):
            return [uncertainties]

        return [str(uncertainties)]

    def _normalize_response_obj(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": str(obj.get("answer", "")),
            "citations": self._normalize_citations(obj.get("citations", [])),
            "uncertainties": self._normalize_uncertainties(obj.get("uncertainties", [])),
        }

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
            text = r.output_text
            obj = json.loads(text)
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
            return {
                "query": question,
                "tools": [
                    {"name": "grade_evidence", "input": {}},
                    {"name": "extract_citations", "input": {}},
                ],
            }

    def answer_with_citations(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
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

            return self._normalize_response_obj(obj)

        except Exception as exc:
            # Keep API from crashing on provider errors such as quota/rate limit.
            return {
                "answer": (
                    "OpenAI provider error. The retrieval pipeline ran, but synthesis "
                    f"could not complete. Error type: {type(exc).__name__}."
                ),
                "citations": [],
                "uncertainties": [
                    "External LLM provider call failed. Check API key, quota, billing, or model access."
                ],
            }
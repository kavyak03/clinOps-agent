from __future__ import annotations

import os
import json
from typing import Any, Dict, List

from anthropic import Anthropic


class AnthropicLLM:
    provider_name = "anthropic"

    def __init__(self) -> None:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=key)
        self.model_name = os.getenv("LLM_MODEL", "").strip() or "claude-3-5-sonnet-latest"

    def plan(self, question: str) -> Dict[str, Any]:
        prompt = (
            "Return JSON ONLY with keys: query (string), tools (list of {name,input}).\n"
            "Allowed tools: grade_evidence, extract_citations.\n"
            f"Question: {question}\n"
        )
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=300,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text  # type: ignore
        try:
            return json.loads(text)
        except Exception:
            return {"query": question, "tools": [{"name": "grade_evidence", "input": {}}, {"name": "extract_citations", "input": {}}]}

    def answer_with_citations(self, question: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.synthesize(question, evidence, tool_results=[])

    def synthesize(self, question: str, evidence: List[Dict[str, Any]], tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ev_lines = []
        for i, e in enumerate(evidence[:6], start=1):
            ev_lines.append(f"[{i}] doc_id={e.get('doc_id')} title={e.get('title')}\n{e.get('chunk')}\nurl={e.get('url')}\n")

        tools_text = ""
        if tool_results:
            tools_text = "TOOL_RESULTS:\n" + "\n".join([str(t) for t in tool_results])

        prompt = (
            "You are a biomedical evidence assistant.\n"
            "Rules:\n"
            "- Do NOT give medical diagnosis, dosing, or treatment directives.\n"
            "- Ground your answer ONLY in provided evidence.\n"
            "- Use citations like [1], [2] referring to evidence blocks.\n"
            "Return JSON ONLY with keys: answer, citations, uncertainties.\n\n"
            f"QUESTION:\n{question}\n\nEVIDENCE:\n{''.join(ev_lines)}\n{tools_text}\n"
        )

        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=700,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text  # type: ignore
        try:
            obj = json.loads(text)
        except Exception:
            obj = {"answer": text, "citations": [], "uncertainties": ["Model output was not valid JSON."]}
        obj.setdefault("citations", [])
        obj.setdefault("uncertainties", [])
        return obj

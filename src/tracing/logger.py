from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text


class TraceLogger:
    """
    Lightweight Postgres-backed tracing logger.

    Stores:
      - runs: one row per API request
      - retrieval_events: retrieved chunks per run
      - tool_events: tool calls per run
    """

    def __init__(self) -> None:
        self.database_url = os.environ["DATABASE_URL"]
        self.engine = create_engine(self.database_url, pool_pre_ping=True)

    @staticmethod
    def _json(value: Optional[Any]) -> str:
        return json.dumps(value if value is not None else {}, default=str)

    def log_run(
        self,
        run_id: str,
        question: str,
        provider: str,
        model: str,
        latency_ms: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        sql = text(
            """
            INSERT INTO runs (id, question, provider, model, latency_ms, meta)
            VALUES (:id, :question, :provider, :model, :latency_ms, CAST(:meta AS jsonb))
            ON CONFLICT (id) DO NOTHING
            """
        )

        with self.engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "id": run_id,
                    "question": question,
                    "provider": provider,
                    "model": model,
                    "latency_ms": latency_ms,
                    "meta": self._json(meta or {}),
                },
            )

    def log_retrieval(self, run_id: str, evidence: List[Dict[str, Any]]) -> None:
        if not evidence:
            return

        sql = text(
            """
            INSERT INTO retrieval_events
                (run_id, doc_id, title, score, rank, metadata)
            VALUES
                (:run_id, :doc_id, :title, :score, :rank, CAST(:metadata AS jsonb))
            """
        )

        with self.engine.begin() as conn:
            for rank, item in enumerate(evidence, start=1):
                metadata = {
                    "url": item.get("url"),
                    "chunk_preview": str(item.get("chunk", ""))[:300],
                    "vector_score": item.get("vector_score"),
                    "rerank_score": item.get("rerank_score"),
                }
                conn.execute(
                    sql,
                    {
                        "run_id": run_id,
                        "doc_id": item.get("doc_id"),
                        "title": item.get("title"),
                        "score": item.get("score"),
                        "rank": rank,
                        "metadata": self._json(metadata),
                    },
                )

    def log_tool(
        self,
        run_id: str,
        tool_name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        sql = text(
            """
            INSERT INTO tool_events
                (run_id, tool_name, inputs, outputs)
            VALUES
                (:run_id, :tool_name, CAST(:inputs AS jsonb), CAST(:outputs AS jsonb))
            """
        )

        with self.engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "inputs": self._json(inputs or {}),
                    "outputs": self._json(outputs or {}),
                },
            )

    def recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        sql = text(
            """
            SELECT id, question, provider, model, latency_ms, meta, created_at
            FROM runs
            ORDER BY created_at DESC
            LIMIT :limit
            """
        )

        with self.engine.begin() as conn:
            rows = conn.execute(sql, {"limit": limit}).mappings().all()

        return [
            {
                "id": str(r["id"]),
                "question": r["question"],
                "provider": r["provider"],
                "model": r["model"],
                "latency_ms": r["latency_ms"],
                "meta": r["meta"],
                "created_at": str(r["created_at"]),
            }
            for r in rows
        ]

    def get_recent_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.recent_runs(limit=limit)

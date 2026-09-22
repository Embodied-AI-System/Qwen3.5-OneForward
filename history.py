"""Durable local history for System One inference requests."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: str | None) -> Any:
    return json.loads(value) if value else None


def _title_for(payload: dict[str, Any]) -> str:
    state = payload.get("state", "")
    if not isinstance(state, str):
        state = _json_dump(state)
    title = " ".join(state.split())
    return (title[:77] + "...") if len(title) > 80 else (title or "Untitled request")


class HistoryStore:
    """Small SQLite store safe for requests running in worker threads."""

    def __init__(self, path: str | Path, *, retention: int = 500) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention = max(1, retention)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_history (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    question_count INTEGER NOT NULL,
                    question_types_json TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    error_json TEXT,
                    metrics_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS history_created_at_idx "
                "ON inference_history(created_at DESC)"
            )
            connection.execute(
                "UPDATE inference_history SET status = 'interrupted', completed_at = ? "
                "WHERE status = 'running'",
                (_utc_now(),),
            )

    def begin(self, payload: dict[str, Any]) -> str:
        entry_id = uuid.uuid4().hex
        questions = payload.get("questions") or {}
        question_types = [
            str(question.get("type", "unknown"))
            for question in questions.values()
            if isinstance(question, dict)
        ]
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO inference_history (
                    id, created_at, status, title, question_count,
                    question_types_json, request_json
                ) VALUES (?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    _utc_now(),
                    _title_for(payload),
                    len(questions),
                    _json_dump(question_types),
                    _json_dump(payload),
                ),
            )
            connection.execute(
                """
                DELETE FROM inference_history
                WHERE id NOT IN (
                    SELECT id FROM inference_history
                    ORDER BY created_at DESC LIMIT ?
                )
                """,
                (self.retention,),
            )
        return entry_id

    def complete(self, entry_id: str, response: dict[str, Any], metrics: dict[str, Any]) -> None:
        self._finish(entry_id, "success", response, None, metrics)

    def fail(self, entry_id: str, error: dict[str, Any], metrics: dict[str, Any]) -> None:
        self._finish(entry_id, "failed", None, error, metrics)

    def _finish(
        self,
        entry_id: str,
        status: str,
        response: dict[str, Any] | None,
        error: dict[str, Any] | None,
        metrics: dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE inference_history
                SET completed_at = ?, status = ?, response_json = ?,
                    error_json = ?, metrics_json = ?
                WHERE id = ?
                """,
                (
                    _utc_now(),
                    status,
                    _json_dump(response) if response is not None else None,
                    _json_dump(error) if error is not None else None,
                    _json_dump(metrics),
                    entry_id,
                ),
            )

    def list(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM inference_history").fetchone()[0])
            rows = connection.execute(
                """
                SELECT id, created_at, completed_at, status, title,
                       question_count, question_types_json, metrics_json
                FROM inference_history ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return {"object": "list", "total": total, "data": [self._summary(row) for row in rows]}

    def get(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inference_history WHERE id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            **self._summary(row),
            "request": _json_load(row["request_json"]),
            "response": _json_load(row["response_json"]),
            "error": _json_load(row["error_json"]),
        }

    def delete(self, entry_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM inference_history WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
            "title": row["title"],
            "question_count": row["question_count"],
            "question_types": _json_load(row["question_types_json"]) or [],
            "metrics": _json_load(row["metrics_json"]) or {},
        }

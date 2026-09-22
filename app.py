"""Choice-only Jev-style HTTP API and OneForward research UI."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference import HFLogitBackend, InputTooLongError
from history import HistoryStore
from media import MAX_ATTACHMENTS, MAX_VIDEO_BYTES, MediaInputError


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"


class MediaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "video"]
    data_url: str = Field(
        min_length=24,
        max_length=(MAX_VIDEO_BYTES * 4 // 3) + 1024,
    )
    name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_data_url_kind(self) -> "MediaInput":
        if not self.data_url.startswith(f"data:{self.type}/"):
            raise ValueError(f"{self.type} attachment has a mismatched data URL")
        return self


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"]
    instructions: Any
    criteria: dict[str, str]

    @field_validator("instructions")
    @classmethod
    def instructions_must_not_be_null(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("instructions must not be null")
        return value

    @field_validator("criteria")
    @classmethod
    def criteria_size(cls, value: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(value) <= 8:
            raise ValueError("choice criteria must contain 2 to 8 options")
        if any(not name.strip() for name in value):
            raise ValueError("choice option names must not be empty")
        return value


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: Any
    model: str | None = None
    media: list[MediaInput] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)
    questions: dict[str, ChoiceQuestion] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_media_mix(self) -> "SystemOneRequest":
        if sum(item.type == "video" for item in self.media) > 1:
            raise ValueError("at most one video is allowed per request")
        return self


def history_request(request: SystemOneRequest) -> dict[str, Any]:
    payload = request.model_dump(mode="json")
    payload["media"] = [
        {
            "type": item.type,
            "name": item.name,
            "mime_type": item.data_url[5:].partition(";")[0],
            "encoded_bytes": len(item.data_url),
            "content_stored": False,
        }
        for item in request.media
    ]
    return payload


def create_app(
    backend: Any | None = None,
    history_store: HistoryStore | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend or await asyncio.to_thread(
            HFLogitBackend.from_environment
        )
        app.state.history = history_store or HistoryStore(
            os.environ.get(
                "JEV_HISTORY_PATH", str(APP_DIR / ".runtime" / "history.sqlite3")
            ),
            retention=int(os.environ.get("JEV_HISTORY_RETENTION", "500")),
        )
        yield

    app = FastAPI(
        title="OneForward",
        version="0.3.0",
        description=(
            "Multimodal Choice-only Jev-style experiment using Qwen3.5-2B next-token "
            "logits, without project-specific training or decoding."
        ),
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return app.state.backend.status()

    @app.get("/v1/history")
    async def list_history(limit: int = 50, offset: int = 0) -> dict[str, Any]:
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(status_code=422, detail="invalid history pagination")
        return await asyncio.to_thread(
            app.state.history.list, limit=limit, offset=offset
        )

    @app.get("/v1/history/{entry_id}")
    async def get_history(entry_id: str) -> dict[str, Any]:
        entry = await asyncio.to_thread(app.state.history.get, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="history entry not found")
        return entry

    @app.delete("/v1/history/{entry_id}", status_code=204)
    async def delete_history(entry_id: str) -> None:
        deleted = await asyncio.to_thread(app.state.history.delete, entry_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="history entry not found")

    @app.post("/v1/systemone")
    async def system_one(request: SystemOneRequest) -> dict[str, Any]:
        started = time.perf_counter()
        history_id = await asyncio.to_thread(
            app.state.history.begin, history_request(request)
        )

        try:
            prepared_media = await asyncio.to_thread(
                app.state.backend.prepare_media, request.media
            )
        except MediaInputError as exc:
            metrics = {"request_ms": (time.perf_counter() - started) * 1000.0}
            await asyncio.to_thread(
                app.state.history.fail,
                history_id,
                {"status_code": 422, "detail": str(exc)},
                metrics,
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        question_items = list(request.questions.items())
        jobs = [
            {
                "instructions": question.instructions,
                "criteria": list(question.criteria.items()),
            }
            for _, question in question_items
        ]
        try:
            batch = await asyncio.to_thread(
                app.state.backend.score_batch,
                request.state,
                jobs,
                media=prepared_media,
            )
        except InputTooLongError as exc:
            metrics = {"request_ms": (time.perf_counter() - started) * 1000.0}
            await asyncio.to_thread(
                app.state.history.fail,
                history_id,
                {"status_code": 422, "detail": str(exc)},
                metrics,
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            metrics = {"request_ms": (time.perf_counter() - started) * 1000.0}
            detail = f"batch inference failed: {exc}"
            await asyncio.to_thread(
                app.state.history.fail,
                history_id,
                {"status_code": 500, "detail": detail},
                metrics,
            )
            raise HTTPException(status_code=500, detail=detail) from exc

        answers: dict[str, Any] = {}
        for batch_index, ((question_id, question), result) in enumerate(
            zip(question_items, batch.decisions)
        ):
            criteria = list(question.criteria.items())
            names = [name for name, _ in criteria]
            selected_name = names[result.selected_index]
            answers[question_id] = {
                "type": "choice",
                "choice": selected_name,
                "confidence": result.confidence,
                "probabilities": {
                    name: probability
                    for name, probability in zip(names, result.probabilities)
                },
                "_debug": {
                    "candidate_mass": result.candidate_mass,
                    "inference_ms": result.inference_ms,
                    "input_tokens": result.input_tokens,
                    "labels": {
                        label: name
                        for label, name in zip(
                            "ABCDEFGH"[: len(names)], names
                        )
                    },
                    "candidate_tokens": list(result.candidate_tokens),
                    "candidate_token_ids": list(result.candidate_token_ids),
                    "raw_logits": list(result.raw_logits),
                    "top_vocabulary": list(result.top_vocabulary),
                    "prompt": result.prompt,
                    "media": list(result.media),
                    "batch_index": batch_index,
                },
            }

        backend_status = app.state.backend.status()
        batch_metrics = {
            "mode": "shared_prefix_batch",
            "batch_size": batch.batch_size,
            "model_forward_count": 1,
            "shared_prefix_tokens": batch.shared_prefix_tokens,
            "preprocess_ms": batch.preprocess_ms,
            "queue_ms": batch.queue_ms,
            "forward_ms": batch.inference_ms,
            "max_input_tokens": batch.max_input_tokens,
            "total_input_tokens": batch.total_input_tokens,
            "padded_input_tokens": batch.padded_input_tokens,
            "padding_tokens": batch.padding_tokens,
        }
        request_ms = (time.perf_counter() - started) * 1000.0
        response = {
            "id": history_id,
            "model": backend_status["model"],
            "answers": answers,
            "usage": {"input_tokens": batch.total_input_tokens, "output_tokens": 0},
            "_compat": {
                "schema_scope": "choice-only",
                "semantic_compatibility": False,
                "confidence_method": "one_minus_normalized_entropy_experimental",
                "requested_model": request.model,
                "prompt_mode": backend_status.get("prompt_mode"),
                "question_execution": "shared_prefix_batch",
                "batch_semantics": (
                    "one padded model forward over questions sharing state and media; "
                    "each question has an independent decision suffix"
                ),
                "multimodal_input": True,
            },
            "_debug": {
                "request_ms": request_ms,
                "batch": batch_metrics,
                "media_count": len(prepared_media),
            },
        }
        history_metrics = {
            "request_ms": request_ms,
            **batch_metrics,
            "questions": {
                question_id: {
                    "type": "choice",
                    "input_tokens": answer["_debug"]["input_tokens"],
                    "candidate_mass": answer["_debug"]["candidate_mass"],
                    "confidence": answer["confidence"],
                }
                for question_id, answer in answers.items()
            },
        }
        await asyncio.to_thread(
            app.state.history.complete, history_id, response, history_metrics
        )
        return response

    return app


app = create_app()

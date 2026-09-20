"""Choice-only Jev-style HTTP API and OneForward research UI."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from inference import HFLogitBackend


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"


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
    questions: dict[str, ChoiceQuestion] = Field(min_length=1)


def create_app(backend: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = backend or await asyncio.to_thread(
            HFLogitBackend.from_environment
        )
        yield

    app = FastAPI(
        title="OneForward",
        version="0.1.0",
        description=(
            "Choice-only Jev-style experiment using Qwen3.5-2B next-token "
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

    @app.post("/v1/systemone")
    async def system_one(request: SystemOneRequest) -> dict[str, Any]:
        started = time.perf_counter()
        answers: dict[str, Any] = {}
        input_tokens = 0

        # Questions intentionally run one-by-one in the MVP. No batching claim.
        for question_id, question in request.questions.items():
            criteria = list(question.criteria.items())
            try:
                result = await asyncio.to_thread(
                    app.state.backend.score_choice,
                    request.state,
                    question.instructions,
                    criteria,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"inference failed for question {question_id!r}: {exc}",
                ) from exc

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
                },
            }
            input_tokens += result.input_tokens

        backend_status = app.state.backend.status()
        return {
            "model": backend_status["model"],
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "_compat": {
                "schema_scope": "choice-only",
                "semantic_compatibility": False,
                "confidence_method": "one_minus_normalized_entropy_experimental",
                "requested_model": request.model,
                "prompt_mode": backend_status.get("prompt_mode"),
            },
            "_debug": {
                "request_ms": (time.perf_counter() - started) * 1000.0,
                "questions_evaluated_sequentially": True,
            },
        }

    return app


app = create_app()

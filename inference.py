"""One-forward Qwen3.5 next-token-logit backend."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core import PromptRenderer, normalized_entropy_confidence


DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-2B"


@dataclass(frozen=True)
class DecisionResult:
    probabilities: tuple[float, ...]
    selected_index: int
    confidence: float
    candidate_mass: float
    raw_logits: tuple[float, ...]
    candidate_tokens: tuple[str, ...]
    candidate_token_ids: tuple[int, ...]
    prompt: str
    input_tokens: int
    inference_ms: float
    top_vocabulary: tuple[dict[str, Any], ...]


class HFLogitBackend:
    """Loads Qwen once and reads only the final-position vocabulary logits."""

    model_name = "qwen3.5-2b-oneforward"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        prompt_mode: str = "chat",
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        expanded_source = os.path.expanduser(model_path)
        self.is_local_model = os.path.isdir(expanded_source)
        self.model_source = (
            os.path.abspath(expanded_source)
            if self.is_local_model
            else model_path
        )
        self.public_model_source = (
            Path(self.model_source).name
            if self.is_local_model
            else self.model_source
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.prompt_mode = prompt_mode
        self._lock = threading.Lock()

        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            local_files_only=self.is_local_model,
        )
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_source,
            local_files_only=self.is_local_model,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device).eval()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.renderer = PromptRenderer(self.tokenizer, mode=prompt_mode)
        self.load_seconds = time.perf_counter() - started

        # Fail during startup if tokenizer/template assumptions are no longer true.
        self.renderer.render(
            "startup validation",
            "Choose the matching option.",
            (("first", "first option"), ("second", "second option")),
        )

    @classmethod
    def from_environment(cls) -> "HFLogitBackend":
        return cls(
            model_path=os.environ.get("JEV_MODEL_PATH", DEFAULT_MODEL_PATH),
            device=os.environ.get("JEV_DEVICE") or None,
            prompt_mode=os.environ.get("JEV_PROMPT_MODE", "chat"),
            temperature=float(os.environ.get("JEV_TEMPERATURE", "1.0")),
        )

    def score_choice(
        self,
        state: Any,
        instructions: Any,
        criteria: Sequence[tuple[str, str]],
    ) -> DecisionResult:
        torch = self.torch
        rendered = self.renderer.render(state, instructions, criteria)
        input_ids = torch.tensor(
            [rendered.input_ids], dtype=torch.long, device=self.device
        )
        attention_mask = torch.ones_like(input_ids)

        # Serialize access: the MVP deliberately evaluates questions independently.
        with self._lock, torch.inference_mode():
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
                use_cache=False,
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - started) * 1000.0

        vocabulary_logits = output.logits[0, -1].float() / self.temperature
        token_index = torch.tensor(
            rendered.candidate_token_ids,
            dtype=torch.long,
            device=vocabulary_logits.device,
        )
        candidate_logits = vocabulary_logits.index_select(0, token_index)
        probabilities = torch.softmax(candidate_logits, dim=0)
        candidate_mass = torch.exp(
            torch.logsumexp(candidate_logits, dim=0)
            - torch.logsumexp(vocabulary_logits, dim=0)
        )
        selected_index = int(torch.argmax(candidate_logits).item())

        top_values, top_ids = torch.topk(
            torch.softmax(vocabulary_logits, dim=0), k=8
        )
        top_vocabulary = tuple(
            {
                "token": self.tokenizer.decode([int(token_id)]),
                "token_id": int(token_id),
                "probability": float(value),
            }
            for value, token_id in zip(top_values.tolist(), top_ids.tolist())
        )
        probability_values = tuple(float(value) for value in probabilities.tolist())

        return DecisionResult(
            probabilities=probability_values,
            selected_index=selected_index,
            confidence=normalized_entropy_confidence(probability_values),
            candidate_mass=float(candidate_mass.item()),
            raw_logits=tuple(float(value) for value in candidate_logits.tolist()),
            candidate_tokens=rendered.candidate_tokens,
            candidate_token_ids=rendered.candidate_token_ids,
            prompt=rendered.prompt,
            input_tokens=len(rendered.input_ids),
            inference_ms=inference_ms,
            top_vocabulary=top_vocabulary,
        )

    def status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": self.model_name,
            "model_source": self.public_model_source,
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "prompt_mode": self.prompt_mode,
            "temperature": self.temperature,
            "load_seconds": self.load_seconds,
            "max_options": 8,
        }

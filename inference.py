"""One-forward Qwen3.5 next-token-logit backend."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core import PromptRenderer, normalized_entropy_confidence
from media import MediaInputError, PreparedMedia, prepare_media


DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-2B"


class InputTooLongError(ValueError):
    pass


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
    media: tuple[dict[str, Any], ...] = ()


class HFLogitBackend:
    """Loads Qwen once and reads only the final-position vocabulary logits."""

    model_name = "qwen3.5-2b-oneforward"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        prompt_mode: str = "chat",
        temperature: float = 1.0,
        max_input_tokens: int = 8192,
        video_fps: float = 1.0,
        max_video_frames: int = 32,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if max_input_tokens < 128:
            raise ValueError("max_input_tokens must be at least 128")
        if video_fps <= 0 or max_video_frames < 1:
            raise ValueError("video sampling configuration must be positive")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

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
        self.max_input_tokens = max_input_tokens
        self.video_fps = video_fps
        self.max_video_frames = max_video_frames
        self._lock = threading.Lock()

        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            self.model_source,
            local_files_only=self.is_local_model,
        )
        self.tokenizer = self.processor.tokenizer
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
            max_input_tokens=int(os.environ.get("JEV_MAX_INPUT_TOKENS", "8192")),
            video_fps=float(os.environ.get("JEV_VIDEO_FPS", "1.0")),
            max_video_frames=int(os.environ.get("JEV_MAX_VIDEO_FRAMES", "32")),
        )

    def prepare_media(self, items: Sequence[Any]) -> tuple[PreparedMedia, ...]:
        if items and self.prompt_mode != "chat":
            raise MediaInputError("multimodal input requires JEV_PROMPT_MODE=chat")
        return prepare_media(
            items,
            video_fps=self.video_fps,
            max_video_frames=self.max_video_frames,
        )

    def score_choice(
        self,
        state: Any,
        instructions: Any,
        criteria: Sequence[tuple[str, str]],
        media: Sequence[PreparedMedia] = (),
    ) -> DecisionResult:
        torch = self.torch
        rendered = self.renderer.render(state, instructions, criteria)
        prompt = rendered.prompt
        if media:
            body = self.renderer.render_body(state, instructions, criteria)
            messages = self.renderer.messages(
                body, [attachment.content_block() for attachment in media]
            )
            video_metadata = [
                attachment.video_metadata
                for attachment in media
                if attachment.video_metadata is not None
            ]
            processor_kwargs: dict[str, Any] = {}
            if video_metadata:
                processor_kwargs = {
                    "do_sample_frames": False,
                    "video_metadata": video_metadata,
                }
            model_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
            )
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = torch.tensor(
                [rendered.input_ids], dtype=torch.long, device=self.device
            )
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        input_token_count = int(input_ids.shape[-1])
        if input_token_count > self.max_input_tokens:
            raise InputTooLongError(
                f"rendered prompt has {input_token_count} tokens; limit is {self.max_input_tokens}"
            )

        # Serialize access: the MVP deliberately evaluates questions independently.
        with self._lock, torch.inference_mode():
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            output = self.model(
                **model_inputs,
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
            prompt=prompt,
            input_tokens=input_token_count,
            inference_ms=inference_ms,
            top_vocabulary=top_vocabulary,
            media=tuple(attachment.public_metadata() for attachment in media),
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
            "max_input_tokens": self.max_input_tokens,
            "multimodal": True,
            "media_limits": {
                "attachments": 8,
                "videos": 1,
                "video_fps": self.video_fps,
                "max_video_frames": self.max_video_frames,
            },
        }

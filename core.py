"""Prompt construction and probability utilities for the logits readout MVP."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence


LETTERS = tuple("ABCDEFGH")


def render_value(value: Any) -> str:
    """Render structured Jev fields deterministically for the prompt."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class RenderedDecision:
    prompt: str
    labels: tuple[str, ...]
    candidate_tokens: tuple[str, ...]
    candidate_token_ids: tuple[int, ...]
    input_ids: tuple[int, ...]


class PromptRenderer:
    """Render a decision prompt and prove every answer is one continuation token."""

    system_prompt = (
        "You are making a single classification decision. Select the option whose "
        "meaning best answers the question about the state. Return exactly one option "
        "letter and nothing else."
    )

    def __init__(self, tokenizer: Any, mode: str = "chat") -> None:
        if mode not in {"chat", "raw"}:
            raise ValueError("prompt mode must be 'chat' or 'raw'")
        self.tokenizer = tokenizer
        self.mode = mode

    def render(
        self,
        state: Any,
        instructions: Any,
        criteria: Sequence[tuple[str, str]],
    ) -> RenderedDecision:
        if not 2 <= len(criteria) <= len(LETTERS):
            raise ValueError(f"choice must have 2 to {len(LETTERS)} options")

        labels = LETTERS[: len(criteria)]
        option_lines = [
            f"{letter}. {name}: {description}"
            for letter, (name, description) in zip(labels, criteria)
        ]
        body = (
            f"State:\n{render_value(state)}\n\n"
            f"Question:\n{render_value(instructions)}\n\n"
            f"Options:\n" + "\n".join(option_lines) + "\n\n"
            f"Return exactly one option letter from {', '.join(labels)}."
        )

        if self.mode == "chat":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": body},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            candidate_tokens = labels
        else:
            prompt = (
                "You are making a single classification decision.\n\n"
                + body
                + "\n\nAnswer:"
            )
            # With a prompt ending in `Answer:`, Qwen tokenizes the natural
            # continuation as the leading-space token " A", not bare "A".
            candidate_tokens = tuple(f" {label}" for label in labels)

        input_ids = tuple(
            self.tokenizer.encode(prompt, add_special_tokens=False)
        )
        candidate_token_ids: list[int] = []
        for token in candidate_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise RuntimeError(
                    f"candidate {token!r} is not one token: {token_ids}"
                )
            full_ids = self.tokenizer.encode(
                prompt + token, add_special_tokens=False
            )
            expected = [*input_ids, token_ids[0]]
            if full_ids != expected:
                raise RuntimeError(
                    "candidate does not extend the rendered prompt by exactly one "
                    f"token: {token!r}"
                )
            candidate_token_ids.append(token_ids[0])

        return RenderedDecision(
            prompt=prompt,
            labels=tuple(labels),
            candidate_tokens=tuple(candidate_tokens),
            candidate_token_ids=tuple(candidate_token_ids),
            input_ids=input_ids,
        )


def normalized_entropy_confidence(probabilities: Sequence[float]) -> float:
    """Experimental confidence: 0 for uniform, 1 for a point mass."""
    if len(probabilities) < 2:
        raise ValueError("at least two probabilities are required")
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    confidence = 1.0 - entropy / math.log(len(probabilities))
    return min(1.0, max(0.0, confidence))

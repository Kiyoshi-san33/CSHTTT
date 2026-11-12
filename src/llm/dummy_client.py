"""LLM giả lập — chạy thử toàn bộ pipeline không cần API key, kết quả tất định theo prompt."""
from __future__ import annotations

import json
import random
import re
import zlib

from ..schemas.prediction import LLMResponse
from .base import BaseLLM

_MATS = ["steel", "aluminium alloy", "cast iron", "rubber", "polyamide", "copper", "ceramic", "glass fibre"]


class DummyLLM(BaseLLM):
    provider = "dummy"
    SUPPORTED_PARAMS = frozenset({"reasoning_effort", "reasoning_format", "think", "top_p", "seed"})

    def __init__(self, cfg: dict, spec: dict):
        spec = {"rpm": 600000, **spec}
        super().__init__(cfg, spec)

    def _complete(self, messages, max_tokens, temperature) -> LLMResponse:
        prompt = messages[-1]["content"]
        text = self._answer(prompt)
        n_in, n_out = len(prompt.split()), len(text.split())
        return LLMResponse(content=text, model_id=self.model, served_model=self.model, provider=self.provider,
                           params=self._effective_params(max_tokens, temperature), prompt_tokens=n_in,
                           completion_tokens=n_out, reasoning_tokens=0, total_tokens=n_in + n_out,
                           finish_reason="stop", request_id=f"dummy-{zlib.crc32(prompt.encode()):08x}")

    def _answer(self, prompt: str) -> str:
        rnd = random.Random(zlib.crc32((self.name + prompt).encode()))
        if "Reply with JSON" in prompt:
            return self._judge(prompt, rnd)
        mats = _MATS[:]
        rnd.shuffle(mats)
        if "comma-separated" in prompt:
            return ", ".join(mats[:5])
        if "Final Verification Questions:" in prompt:
            return "1. Is steel used in this component?\n2. Is rubber suitable for this component?"
        if "Answer:" in prompt and "Question:" in prompt and "Passage" in prompt and "concisely" in prompt:
            return "Steel is commonly used. Rubber is used for seals."
        return "\n".join(f"{i + 1}. {m.title()}: dummy explanation for {m}." for i, m in enumerate(mats[:4]))

    @staticmethod
    def _judge(prompt: str, rnd: random.Random) -> str:
        if '"ranking"' in prompt:
            m = re.search(r"labelled ([A-Z](?:, [A-Z])*)\.", prompt)
            labels = m.group(1).split(", ") if m else ["A"]
            rnd.shuffle(labels)
            return json.dumps({"ranking": labels})
        plaus = [rnd.random() < 0.85 for _ in range(10)]
        if '"material_rating"' in prompt:
            return json.dumps({"plausible": plaus, "incorrect_sentences": [1] if rnd.random() < 0.2 else [],
                               "material_rating": rnd.randint(3, 5), "explanation_rating": rnd.randint(3, 5),
                               "comment": ""})
        return json.dumps({"plausible": plaus, "dont_know": False})

    def list_models(self) -> list[str] | None:
        return [self.model]


from __future__ import annotations

import json
import re
from typing import Any

from config import GEMINI_API_KEY, LLM_MAX_OUTPUT_TOKENS, LLM_MODEL, require_gemini_key


class GeminiClient:
    def __init__(self, model_name: str = LLM_MODEL, api_key: str | None = GEMINI_API_KEY) -> None:
        key = api_key or require_gemini_key()

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        genai.configure(api_key=key)
        self._genai = genai
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ) -> str:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""

        parts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)

        return "\n".join(parts).strip()


def parse_json_from_text(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Cannot parse JSON from model output")

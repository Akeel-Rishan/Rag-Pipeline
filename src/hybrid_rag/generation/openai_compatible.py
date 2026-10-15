"""Small standard-library adapter for an OpenAI-compatible chat endpoint."""

from dataclasses import dataclass, field
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMGenerationError(RuntimeError):
    """The configured language-model endpoint could not produce an answer."""


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2"
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleSettings":
        raw_timeout = os.getenv("LLM_TIMEOUT_SECONDS", "60")
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a number") from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be a finite positive number")
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("LLM_MODEL", "llama3.2"),
            api_key=os.getenv("LLM_API_KEY") or None,
            timeout=timeout,
        )


class OpenAICompatibleLLM:
    """Send two chat messages; retrieval remains outside this adapter."""

    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        if not settings.base_url.strip() or not settings.model.strip():
            raise ValueError("LLM base URL and model must not be blank")
        self.settings = settings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        body = json.dumps({
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        request = Request(
            self.settings.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LLMGenerationError(f"Language-model request failed: {error}") from error
        try:
            answer = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMGenerationError("Language-model response has an unexpected shape") from error
        if not isinstance(answer, str) or not answer.strip():
            raise LLMGenerationError("Language model returned an empty answer")
        return answer.strip()

"""The minimal contract required from a chat language model."""

from typing import Protocol


class LanguageModel(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return an answer for the supplied system and user messages."""
        ...

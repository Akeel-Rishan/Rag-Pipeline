"""HTTP adapter tests without network access or credentials."""

import json
from unittest.mock import patch

import pytest

from hybrid_rag.generation.openai_compatible import (
    LLMGenerationError,
    OpenAICompatibleLLM,
    OpenAICompatibleSettings,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_chat_request_and_response_shape():
    settings = OpenAICompatibleSettings("http://llm.test/v1/", "model", "secret", 4)
    with patch("hybrid_rag.generation.openai_compatible.urlopen",
               return_value=FakeResponse({"choices": [{"message": {"content": " answer "}}]})) as call:
        assert OpenAICompatibleLLM(settings).generate("system", "user") == "answer"
    request = call.call_args.args[0]
    assert request.full_url == "http://llm.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.data)
    assert body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert call.call_args.kwargs["timeout"] == 4


@pytest.mark.parametrize("payload", [{}, {"choices": []},
                                      {"choices": [{"message": {"content": " "}}]}])
def test_bad_response_rejected(payload):
    llm = OpenAICompatibleLLM(OpenAICompatibleSettings())
    with patch("hybrid_rag.generation.openai_compatible.urlopen",
               return_value=FakeResponse(payload)), pytest.raises(LLMGenerationError):
        llm.generate("system", "user")


def test_environment_settings_do_not_expose_key(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "example-model")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12")
    settings = OpenAICompatibleSettings.from_env()
    assert settings.timeout == 12
    assert settings.model == "example-model"
    assert "test-secret" not in repr(settings)


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "not-a-number"])
def test_invalid_timeout_rejected(monkeypatch, timeout):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", timeout)
    with pytest.raises(ValueError):
        OpenAICompatibleSettings.from_env()

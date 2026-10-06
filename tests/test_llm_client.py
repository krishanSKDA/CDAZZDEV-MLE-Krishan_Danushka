from types import SimpleNamespace

import httpx
import openai
import pytest

from common import config
from common.llm_client import LLMClient, LLMError, Provider

KEYS = ("GEMINI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture
def env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_parse_model_spec():
    assert config.parse_model_spec("gemini") == ("gemini", config.PROVIDERS["gemini"]["model"])
    assert config.parse_model_spec("openrouter:meta-llama/x:free") == ("openrouter", "meta-llama/x:free")
    with pytest.raises(ValueError):
        config.parse_model_spec("groq:llama-3.3-70b-versatile")


def test_gemini_is_default_and_cheapest_model(env):
    env.setenv("GEMINI_API_KEY", "g")
    env.setenv("OPENROUTER_API_KEY", "o")
    client = LLMClient()
    assert [p.name for p in client.providers] == ["gemini", "openrouter"]
    assert client.providers[0].model == "gemini-3.1-flash-lite"
    assert "generativelanguage.googleapis.com" in str(client.providers[0].client.base_url)


def test_missing_keys(env):
    with pytest.raises(LLMError):
        LLMClient()
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLMClient(model_spec="gemini:gemini-3.1-flash-lite")


class JsonModeRejecting:
    def __init__(self):
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
            raise openai.BadRequestError("response_format json_object is not supported", response=resp, body=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])


def test_json_mode_rejection_retries_without_response_format():
    completions = JsonModeRejecting()
    client = LLMClient(providers=[Provider("gemini", SimpleNamespace(chat=SimpleNamespace(completions=completions)), "m")])
    assert client.chat([{"role": "user", "content": "json"}], json_mode=True) == '{"ok": true}'
    assert [("response_format" in c) for c in completions.calls] == [True, False]

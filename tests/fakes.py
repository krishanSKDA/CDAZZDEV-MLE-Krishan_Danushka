from types import SimpleNamespace

from common.llm_client import LLMClient, Provider


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append(messages)
        content = self.responses.pop(0) if self.responses else "{}"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def fake_llm(responses) -> tuple[LLMClient, FakeCompletions]:
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(providers=[Provider("fake", client, "fake-model")]), completions

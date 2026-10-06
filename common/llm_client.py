import json
import re
import time
from dataclasses import dataclass
from typing import TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from common import config
from common.logging_utils import get_logger

log = get_logger("llm_client")

T = TypeVar("T", bound=BaseModel)

RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
MAX_ATTEMPTS_PER_PROVIDER = 4
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60.0
CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class Provider:
    name: str
    client: OpenAI
    model: str


class LLMError(RuntimeError):
    pass


def _make_provider(name: str, api_key: str, base_url: str, model: str) -> Provider:
    return Provider(name, OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0), model)


class LLMClient:
    """OpenAI-compatible client over Gemini (primary) / OpenRouter (optional fallback), tried in config.PROVIDER_ORDER.

    model_spec ('provider:model') pins a single provider, e.g. for a specific teacher or judge model."""

    def __init__(self, temperature: float = 0.2, model_spec: str | None = None, providers: list[Provider] | None = None):
        self.temperature = temperature
        if providers is not None:
            self.providers = providers
        elif model_spec:
            name, model = config.parse_model_spec(model_spec)
            key = config.get_secret(config.PROVIDERS[name]["key_env"], required=False)
            if not key:
                raise LLMError(f"{config.PROVIDERS[name]['key_env']} is not set (needed for '{model_spec}')")
            self.providers = [_make_provider(name, key, config.PROVIDERS[name]["base_url"], model)]
        else:
            self.providers = [_make_provider(*p) for p in config.available_providers()]
        if not self.providers:
            raise LLMError("No LLM provider configured. Set GEMINI_API_KEY.")

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float | None = None) -> str:
        temp = self.temperature if temperature is None else temperature
        last_error: Exception | None = None

        for provider in self.providers:
            kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
                try:
                    resp = provider.client.chat.completions.create(
                        model=provider.model, messages=messages, temperature=temp, **kwargs
                    )
                    return resp.choices[0].message.content or ""
                except openai.BadRequestError as exc:
                    last_error = exc
                    if kwargs and "response_format" in str(exc):
                        # provider without JSON mode: prompts already demand JSON, output is still validated
                        log.warning("json mode unsupported; retrying without it", extra={"data": {"provider": provider.name}})
                        kwargs = {}
                        continue
                    log.error("bad request", extra={"data": {"provider": provider.name, "error": str(exc)[:200]}})
                    break
                except RETRYABLE_ERRORS as exc:
                    last_error = exc
                    wait = BACKOFF_BASE_SECONDS ** attempt
                    log.warning("retryable LLM error", extra={"data": {
                        "provider": provider.name, "attempt": attempt, "wait_s": wait, "error": str(exc)[:200]}})
                    time.sleep(wait)
                except openai.APIError as exc:
                    last_error = exc
                    log.error("non-retryable LLM error", extra={"data": {
                        "provider": provider.name, "error": str(exc)[:200]}})
                    break
            log.warning("falling back to next provider", extra={"data": {"failed": provider.name}})

        raise LLMError(f"All providers failed: {last_error}")

    def structured(self, messages: list[dict], schema: type[T], repair_attempts: int = 1,
                   raise_on_provider_error: bool = False) -> T | None:
        """Call the LLM in JSON mode and validate against `schema`. Returns None if validation never succeeds."""
        convo = list(messages)
        for attempt in range(repair_attempts + 1):
            try:
                raw = self.chat(convo, json_mode=True)
            except LLMError as exc:
                if raise_on_provider_error:
                    raise
                log.error("LLM call failed", extra={"data": {"schema": schema.__name__, "error": str(exc)[:200]}})
                return None
            cleaned = CODE_FENCE.sub("", raw).strip()
            try:
                return schema.model_validate_json(cleaned)
            except (ValidationError, json.JSONDecodeError) as exc:
                log.warning("LLM output failed validation", extra={"data": {
                    "schema": schema.__name__, "attempt": attempt, "error": str(exc)[:300], "raw": cleaned[:200]}})
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON failed validation:\n{exc}\nReturn only corrected JSON."},
                ]
        return None

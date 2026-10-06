import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# OpenAI-compatible endpoints; order = fallback priority
PROVIDERS = {
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        # cheapest Gemini text model with a free tier that new API keys can access
        "model": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    },
}
PROVIDER_ORDER = [p.strip() for p in os.getenv("LLM_PROVIDER_ORDER", "gemini,openrouter").split(",") if p.strip()]
DEFAULT_TICKER = os.getenv("TICKER", "NVDA")


def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata

        return userdata.get(name)
    except Exception:
        return None


def get_secret(name: str, required: bool = True) -> str | None:
    value = _colab_secret(name) or os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Missing secret '{name}'. Add it to .env (local) or Colab Secrets.")
    return value


def parse_model_spec(spec: str) -> tuple[str, str]:
    """'gemini:gemini-3.1-flash-lite' -> ('gemini', 'gemini-3.1-flash-lite'); a bare provider name uses its default model."""
    provider, _, model = spec.strip().partition(":")
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}' in model spec '{spec}'; expected one of {list(PROVIDERS)}")
    return provider, model or PROVIDERS[provider]["model"]


def available_providers() -> list[tuple[str, str, str, str]]:
    """(name, api_key, base_url, default_model) for every provider in PROVIDER_ORDER that has a key."""
    out = []
    for name in PROVIDER_ORDER:
        spec = PROVIDERS.get(name)
        key = spec and get_secret(spec["key_env"], required=False)
        if key:
            out.append((name, key, spec["base_url"], spec["model"]))
    return out

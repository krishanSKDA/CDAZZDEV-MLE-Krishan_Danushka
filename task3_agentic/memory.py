import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from task3_agentic.tracing import log_event

CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_MAX_AGE_DAYS = 0


def cache_path(ticker: str, day: date | None = None, cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_DIR) / f"{ticker.upper()}_{(day or date.today()).isoformat()}.json"


def save_brief(ticker: str, report: dict, metadata: dict | None = None, cache_dir: Path | None = None) -> Path:
    path = cache_path(ticker, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker.upper(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metadata": metadata or {},
        "report": report,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log_event("cache_write", ticker=ticker.upper(), path=path.name)
    return path


def load_cached_brief(ticker: str, max_age_days: int = DEFAULT_MAX_AGE_DAYS, cache_dir: Path | None = None) -> dict | None:
    """Returns the newest brief for `ticker` saved within `max_age_days` (0 = today only)."""
    for offset in range(max_age_days + 1):
        path = cache_path(ticker, date.today() - timedelta(days=offset), cache_dir)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log_event("cache_corrupt", ticker=ticker.upper(), path=path.name)
                continue
            log_event("cache_hit", ticker=ticker.upper(), path=path.name)
            return payload
    log_event("cache_miss", ticker=ticker.upper())
    return None

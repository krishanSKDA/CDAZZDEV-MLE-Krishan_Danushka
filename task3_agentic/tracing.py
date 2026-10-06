import functools
import json
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

TRACE_PATH = Path(__file__).parent / "logs" / "agent_trace.jsonl"
OUTPUT_PREVIEW_CHARS = 200

current_agent: ContextVar[str] = ContextVar("current_agent", default="single_agent")
current_run: ContextVar[str] = ContextVar("current_run", default="unscoped")

# tool name -> agents allowed to call it; enforced at execution time, not only via tool binding
TOOL_PERMISSIONS: dict[str, set[str]] = {
    "get_price_data": {"single_agent", "data_analyst"},
    "calculate_volatility": {"single_agent", "data_analyst"},
    "llm_sentiment": {"single_agent", "data_analyst"},
    "get_news": {"single_agent", "research_writer"},
    "web_search": {"single_agent", "research_writer"},
}

_lock = threading.Lock()
_injected_failures: dict[str, int] = {}
_injected_hints: dict[str, str] = {}
DEFAULT_FAILURE_HINT = "use an alternative tool or a different query"


class ToolPermissionError(PermissionError):
    pass


def inject_failure(tool_name: str, times: int = 1, hint: str = DEFAULT_FAILURE_HINT) -> None:
    """Makes the next `times` calls to `tool_name` return a simulated outage, to demonstrate recovery."""
    _injected_failures[tool_name] = times
    _injected_hints[tool_name] = hint


def _preview(value: Any) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    return text[:OUTPUT_PREVIEW_CHARS]


def write_event(record: dict, path: Path | None = None) -> None:
    path = path or TRACE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_event(event: str, **payload) -> None:
    """Non-tool events (handoffs, cache hits, agent messages) share the same trace file."""
    write_event({
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": current_run.get(),
        "agent": current_agent.get(),
        "event": event,
        **{k: _preview(v) if k == "output" else v for k, v in payload.items()},
    })


def traced(fn: Callable) -> Callable:
    tool_name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        agent = current_agent.get()
        start = time.perf_counter()
        status, output = "ok", None
        try:
            allowed = TOOL_PERMISSIONS.get(tool_name)
            if allowed is not None and agent not in allowed:
                status = "blocked"
                raise ToolPermissionError(f"agent '{agent}' is not permitted to call '{tool_name}'")
            if _injected_failures.get(tool_name, 0) > 0:
                _injected_failures[tool_name] -= 1
                status = "injected_failure"
                output = {"error": f"{tool_name} unavailable (simulated upstream outage)",
                          "hint": _injected_hints.get(tool_name, DEFAULT_FAILURE_HINT)}
                return output
            output = fn(*args, **kwargs)
            if isinstance(output, dict) and "error" in output:
                status = "error"
            return output
        except ToolPermissionError as exc:
            output = {"error": str(exc)}
            return output
        except Exception as exc:
            status, output = "exception", {"error": f"{type(exc).__name__}: {exc}"}
            return output
        finally:
            write_event({
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "run_id": current_run.get(),
                "agent": agent,
                "event": "tool_call",
                "tool": tool_name,
                "inputs": {**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs},
                "output": _preview(output),
                "status": status,
                "duration_ms": round((time.perf_counter() - start) * 1000, 1),
            })

    return wrapper


@contextmanager
def agent_scope(agent: str):
    token = current_agent.set(agent)
    try:
        yield
    finally:
        current_agent.reset(token)


@contextmanager
def run_scope(run_id: str | None = None):
    token = current_run.set(run_id or uuid.uuid4().hex[:12])
    try:
        yield current_run.get()
    finally:
        current_run.reset(token)


def read_trace(path: Path | None = None, run_id: str | None = None) -> list[dict]:
    path = path or TRACE_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [r for r in records if run_id is None or r.get("run_id") == run_id]

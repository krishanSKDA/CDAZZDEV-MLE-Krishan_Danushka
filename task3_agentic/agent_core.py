import json
import os
from typing import Callable, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from common import config
from common.llm_client import LLMClient
from common.logging_utils import get_logger
from task3_agentic.tracing import current_agent, log_event

log = get_logger("task3.agent")

T = TypeVar("T", bound=BaseModel)

AGENT_MODEL = os.getenv("AGENT_MODEL")  # 'provider:model' pins one provider; None = config.PROVIDER_ORDER chain
AGENT_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 4
MAX_TOOL_RESULT_CHARS = 3500
MAX_EVIDENCE_ITEM_CHARS = 2500
DISPLAY_CHARS = 300
REPAIR_NUDGE = ("Your previous response could not be processed (malformed tool call). Call exactly one tool with "
                "valid JSON arguments, or reply in plain text if you already have enough information.")

Printer = Callable[[str], None]


def tool_model(tools: list[BaseTool]):
    """Tool-calling chat model over the OpenAI-compatible endpoints (Gemini first by default), with fallbacks."""
    from langchain_openai import ChatOpenAI

    if AGENT_MODEL:
        name, model = config.parse_model_spec(AGENT_MODEL)
        spec = config.PROVIDERS[name]
        endpoints = [(name, config.get_secret(spec["key_env"]), spec["base_url"], model)]
    else:
        endpoints = config.available_providers()
    if not endpoints:
        raise RuntimeError("Set GEMINI_API_KEY to run the agents.")
    models = [ChatOpenAI(model=model, api_key=key, base_url=url, temperature=AGENT_TEMPERATURE,
                         max_retries=LLM_MAX_RETRIES).bind_tools(tools)
              for _, key, url, model in endpoints]
    return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]


def invoke_with_repair(model, messages: list[BaseMessage]) -> AIMessage:
    """Providers occasionally reject malformed tool calls ; nudge once, then degrade to no-tool reply."""
    try:
        return model.invoke(messages)
    except Exception as exc:
        log_event("llm_error", error=str(exc)[:300], recovery="repair_nudge")
        try:
            return model.invoke(messages + [HumanMessage(content=REPAIR_NUDGE)])
        except Exception as exc2:
            log_event("llm_error", error=str(exc2)[:300], recovery="stop_tool_loop")
            return AIMessage(content=f"[LLM unavailable after retry: {type(exc2).__name__}] Proceeding with the evidence gathered so far.")


def execute_tool_calls(ai: AIMessage, tool_map: dict[str, BaseTool]) -> list[ToolMessage]:
    results = []
    for call in ai.tool_calls:
        name = call["name"]
        if name not in tool_map:
            log_event("tool_blocked", tool=name, inputs=call.get("args"))
            output = {"error": f"tool '{name}' is not available to agent '{current_agent.get()}'",
                      "hint": f"available tools: {sorted(tool_map)}"}
        else:
            try:
                output = tool_map[name].invoke(call.get("args") or {})
            except Exception as exc:
                output = {"error": f"invalid arguments: {type(exc).__name__}: {str(exc)[:200]}",
                          "hint": "check the tool signature and retry"}
        content = json.dumps(output, default=str)[:MAX_TOOL_RESULT_CHARS]
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
    return results


def show_message(msg: BaseMessage, agent: str, printer: Printer = print) -> None:
    tag = f"[{agent}]"
    if isinstance(msg, AIMessage):
        if msg.content:
            printer(f"{tag} THINK: {str(msg.content)[:DISPLAY_CHARS]}")
        for call in msg.tool_calls:
            printer(f"{tag} CALL  -> {call['name']}({json.dumps(call.get('args'), default=str)[:DISPLAY_CHARS]})")
    elif isinstance(msg, ToolMessage):
        status = "ERROR" if '"error"' in msg.content[:40] else "OK"
        printer(f"{tag} OBSERVE <- {msg.name} [{status}]: {msg.content[:DISPLAY_CHARS]}")


def run_react(agent: str, system_prompt: str, user_prompt: str, tools: list[BaseTool], max_steps: int,
              printer: Printer = print, model=None) -> list[BaseMessage]:
    model = model or tool_model(tools)
    tool_map = {t.name: t for t in tools}
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    for _ in range(max_steps):
        ai = invoke_with_repair(model, messages)
        messages.append(ai)
        show_message(ai, agent, printer)
        if not ai.tool_calls:
            break
        for tm in execute_tool_calls(ai, tool_map):
            messages.append(tm)
            show_message(tm, agent, printer)
    else:
        printer(f"[{agent}] step budget ({max_steps}) reached; continuing with gathered evidence")
        log_event("step_budget_reached", max_steps=max_steps)
    return messages


def evidence_digest(messages: list[BaseMessage]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            parts.append(f"TOOL {m.name} RESULT: {m.content[:MAX_EVIDENCE_ITEM_CHARS]}")
        elif isinstance(m, AIMessage) and m.content:
            parts.append(f"AGENT NOTE: {str(m.content)[:MAX_EVIDENCE_ITEM_CHARS]}")
    return "\n\n".join(parts) or "(no evidence gathered)"


def tool_results(messages: list[BaseMessage], tool_name: str) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == tool_name:
            try:
                out.append(json.loads(m.content))
            except json.JSONDecodeError:
                continue
    return out


def schema_prompt(template: str, schema: type[BaseModel]) -> str:
    return template.format(schema=json.dumps(schema.model_json_schema(), separators=(",", ":")))


def structured_call(system: str, user: str, schema: type[T], client: LLMClient | None = None) -> T | None:
    client = client or LLMClient(temperature=AGENT_TEMPERATURE, model_spec=AGENT_MODEL)
    result = client.structured([{"role": "system", "content": system}, {"role": "user", "content": user}], schema)
    if result is None:
        log_event("structured_output_failed", schema=schema.__name__)
    return result

import json
import operator
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from common.llm_client import LLMClient
from task3_agentic import prompts
from task3_agentic.agent_core import (
    Printer, evidence_digest, run_react, schema_prompt, structured_call, tool_model, tool_results,
)
from task3_agentic.memory import DEFAULT_MAX_AGE_DAYS, load_cached_brief, save_brief
from task3_agentic.schemas import (
    ClarificationRequest, ClarificationResponse, DataBrief, PriceSnapshot, ResearchReport, VolatilitySnapshot,
)
from task3_agentic.tools import ANALYST_TOOLS, WRITER_TOOLS
from task3_agentic.tracing import agent_scope, log_event, run_scope

ANALYST = "data_analyst"
WRITER = "research_writer"
ANALYST_MAX_STEPS = 6
WRITER_MAX_STEPS = 6
CLARIFY_MAX_STEPS = 4
MAX_CRITIQUE_ROUNDS = 1
MAX_ATTACHED_HEADLINES = 15
FALLBACK_VOL_WINDOW = 60


class PipelineState(TypedDict):
    ticker: str
    query: str
    run_id: str
    brief: dict | None
    research_digest: str
    headlines: list[str]
    clarification_request: dict | None
    clarification_response: dict | None
    critique_rounds: int
    report: dict | None
    handoffs: Annotated[list[dict], operator.add]


def _fallback_brief(ticker: str, messages: list[BaseMessage]) -> DataBrief:
    """Deterministic brief straight from tool outputs, used only if the LLM's structured brief fails validation."""
    price = next((r for r in tool_results(messages, "get_price_data") if "error" not in r), {})
    vol = next((r for r in tool_results(messages, "calculate_volatility") if "error" not in r), {})
    ind = price.get("indicators", {})
    return DataBrief(
        ticker=ticker,
        as_of=price.get("end") or vol.get("as_of") or "unknown",
        price=PriceSnapshot(last_close=price.get("last_close"), period_return=price.get("period_return"),
                            max_drawdown=price.get("max_drawdown"), sma_50=ind.get("sma_50"), sma_200=ind.get("sma_200"),
                            rsi_14=ind.get("rsi_14"), macd_hist=ind.get("macd_hist"),
                            momentum_label=(price.get("momentum_signal") or {}).get("label")),
        volatility=VolatilitySnapshot(**{k: vol.get(k) for k in VolatilitySnapshot.model_fields}),
        fundamentals=price.get("fundamentals", {}),
        key_findings=(price.get("derived_facts") or ["price data unavailable", "volatility data unavailable"])[:6],
        data_gaps=["sentiment (no headlines available to analyst)", "brief built deterministically after LLM validation failure"],
    )


class MultiAgentPipeline:
    def __init__(self, model_factory: Callable = tool_model, llm_client: LLMClient | None = None, printer: Printer = print):
        self.model_factory = model_factory
        self.llm_client = llm_client
        self.printer = printer
        self.graph = self._build()

    def _build(self):
        g = StateGraph(PipelineState)
        g.add_node("analyst_brief", self._analyst_brief)
        g.add_node("writer_research", self._writer_research)
        g.add_node("writer_critique", self._writer_critique)
        g.add_node("analyst_clarify", self._analyst_clarify)
        g.add_node("writer_final", self._writer_final)
        g.add_edge(START, "analyst_brief")
        g.add_edge("analyst_brief", "writer_research")
        g.add_edge("writer_research", "writer_critique")
        g.add_conditional_edges("writer_critique", self._route_critique,
                                {"analyst_clarify": "analyst_clarify", "writer_final": "writer_final"})
        g.add_edge("analyst_clarify", "writer_final")
        g.add_edge("writer_final", END)
        return g.compile()

    def _banner(self, text: str) -> None:
        self.printer(f"\n{'=' * 20} {text} {'=' * 20}")

    def _handoff(self, src: str, dst: str, payload: BaseModel) -> dict:
        body = payload.model_dump()
        self._banner(f"HANDOFF {src} -> {dst} ({type(payload).__name__})")
        self.printer(json.dumps(body, indent=2, default=str))
        log_event("handoff", from_agent=src, to_agent=dst, schema=type(payload).__name__, payload=body)
        return {"from": src, "to": dst, "schema": type(payload).__name__, "payload": body}

    def _structured(self, template: str, user: str, schema: type[BaseModel]):
        return structured_call(schema_prompt(template, schema), user, schema, self.llm_client)

    def _analyst_brief(self, state: PipelineState) -> dict:
        self._banner(f"AGENT A ({ANALYST}): quantitative brief")
        with run_scope(state["run_id"]), agent_scope(ANALYST):
            msgs = run_react(ANALYST, prompts.ANALYST_SYSTEM,
                             f"Ticker: {state['ticker']}\nResearch question: {state['query']}\nBuild the quantitative data brief.",
                             ANALYST_TOOLS, ANALYST_MAX_STEPS, self.printer, self.model_factory(ANALYST_TOOLS))
            brief = self._structured(prompts.BRIEF_SYSTEM, f"Ticker: {state['ticker']}\n\nEvidence:\n{evidence_digest(msgs)}", DataBrief)
            if brief is None:
                brief = _fallback_brief(state["ticker"], msgs)
            handoff = self._handoff(ANALYST, WRITER, brief)
        return {"brief": brief.model_dump(), "handoffs": [handoff]}

    def _writer_research(self, state: PipelineState) -> dict:
        self._banner(f"AGENT B ({WRITER}): qualitative research")
        user = f"Research question: {state['query']}\n\nDATA BRIEF from Agent A:\n{json.dumps(state['brief'], indent=1)}"
        with run_scope(state["run_id"]), agent_scope(WRITER):
            msgs = run_react(WRITER, prompts.WRITER_SYSTEM, user, WRITER_TOOLS, WRITER_MAX_STEPS, self.printer,
                             self.model_factory(WRITER_TOOLS))
        headlines = [h["title"] for r in tool_results(msgs, "get_news") for h in r.get("headlines", [])]
        return {"research_digest": evidence_digest(msgs[2:]), "headlines": headlines}

    def _writer_critique(self, state: PipelineState) -> dict:
        self._banner(f"AGENT B ({WRITER}): critique of the data brief")
        user = (f"DATA BRIEF:\n{json.dumps(state['brief'], indent=1)}\n\nYOUR RESEARCH SO FAR:\n{state['research_digest']}\n\n"
                f"You have {len(state['headlines'])} headlines available to attach.")
        with run_scope(state["run_id"]), agent_scope(WRITER):
            request = self._structured(prompts.CRITIQUE_SYSTEM, user, ClarificationRequest)
            if request is None:
                needs_sentiment = not state["brief"].get("sentiment") and state["headlines"]
                request = ClarificationRequest(
                    question="Score the sentiment of the attached headlines." if needs_sentiment
                    else f"Provide {FALLBACK_VOL_WINDOW}-day annualised volatility to size the hedge.",
                    requested_metrics=["sentiment_score"] if needs_sentiment else [f"vol_{FALLBACK_VOL_WINDOW}d"],
                    reason="deterministic fallback after critique validation failure",
                    attach_headlines=bool(needs_sentiment),
                )
            if request.attach_headlines:
                request.headlines = state["headlines"][:MAX_ATTACHED_HEADLINES]
            handoff = self._handoff(WRITER, ANALYST, request)
        return {"clarification_request": request.model_dump(), "critique_rounds": state["critique_rounds"] + 1,
                "handoffs": [handoff]}

    @staticmethod
    def _route_critique(state: PipelineState) -> str:
        if state["clarification_request"] and state["critique_rounds"] <= MAX_CRITIQUE_ROUNDS:
            return "analyst_clarify"
        return "writer_final"

    def _analyst_clarify(self, state: PipelineState) -> dict:
        self._banner(f"AGENT A ({ANALYST}): answering clarification")
        request_json = json.dumps(state["clarification_request"], indent=1)
        with run_scope(state["run_id"]), agent_scope(ANALYST):
            msgs = run_react(ANALYST, prompts.CLARIFY_SYSTEM,
                             f"Ticker: {state['ticker']}\nClarification request from Research Writer:\n{request_json}",
                             ANALYST_TOOLS, CLARIFY_MAX_STEPS, self.printer, self.model_factory(ANALYST_TOOLS))
            response = self._structured(prompts.CLARIFY_RESPONSE_SYSTEM,
                                        f"Request:\n{request_json}\n\nEvidence:\n{evidence_digest(msgs)}", ClarificationResponse)
            used = [m.name for m in msgs if isinstance(m, ToolMessage)]
            if response is None:
                response = ClarificationResponse(answer="Structured answer failed validation; raw tool results attached.",
                                                 data={name: tool_results(msgs, name) for name in set(used)})
            response.tools_used = used
            handoff = self._handoff(ANALYST, WRITER, response)
        return {"clarification_response": response.model_dump(), "handoffs": [handoff]}

    def _writer_final(self, state: PipelineState) -> dict:
        self._banner(f"AGENT B ({WRITER}): final report")
        user = prompts.FINAL_WRITER_USER.format(
            query=state["query"], brief=json.dumps(state["brief"], indent=1), research=state["research_digest"],
            request=json.dumps(state["clarification_request"], indent=1),
            response=json.dumps(state["clarification_response"], indent=1, default=str),
        )
        with run_scope(state["run_id"]), agent_scope(WRITER):
            report = self._structured(prompts.FINAL_WRITER_SYSTEM, user, ResearchReport)
            log_event("final_report", ok=report is not None)
        if report is not None:
            self.printer(report.model_dump_json(indent=2))
        return {"report": report.model_dump() if report else None}

    def run(self, ticker: str, use_cache: bool = True, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> dict:
        ticker = ticker.upper()
        with run_scope() as run_id:
            if use_cache:
                cached = load_cached_brief(ticker, max_age_days)
                if cached:
                    self._banner(f"CACHE HIT for {ticker} (saved {cached['saved_at']}): skipping all agents and tools")
                    return {"source": "cache", "run_id": run_id, "report": cached["report"], "metadata": cached["metadata"]}
            state = self.graph.invoke({
                "ticker": ticker, "query": prompts.RESEARCH_QUERY.format(ticker=ticker), "run_id": run_id,
                "brief": None, "research_digest": "", "headlines": [], "clarification_request": None,
                "clarification_response": None, "critique_rounds": 0, "report": None, "handoffs": [],
            })
            if state["report"]:
                path = save_brief(ticker, state["report"], {
                    "run_id": run_id, "data_brief": state["brief"],
                    "clarification_request": state["clarification_request"],
                    "clarification_response": state["clarification_response"],
                })
                self.printer(f"\nSaved brief to {path}")
        return {"source": "agents", "run_id": run_id, "report": state["report"], "handoffs": state["handoffs"], "state": state}

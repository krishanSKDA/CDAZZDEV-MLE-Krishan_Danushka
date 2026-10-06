import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from common.llm_client import LLMClient
from task3_agentic import prompts
from task3_agentic.agent_core import (
    Printer, evidence_digest, execute_tool_calls, invoke_with_repair, schema_prompt, show_message, structured_call,
    tool_model,
)
from task3_agentic.schemas import ResearchReport
from task3_agentic.tools import ALL_TOOLS
from task3_agentic.tracing import agent_scope, read_trace, run_scope

AGENT_NAME = "single_agent"
MAX_RESEARCH_STEPS = 10
RECURSION_LIMIT = 60


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ticker: str
    run_id: str
    phase: Literal["research", "followup"]
    steps: int
    report: dict | None


class SingleResearchAgent:
    """ReAct loop as an explicit LangGraph: agent -> tools -> agent ... -> report. Checkpointer = short-term memory."""

    def __init__(self, model=None, llm_client: LLMClient | None = None, printer: Printer = print):
        self.model = model or tool_model(ALL_TOOLS)
        self.llm_client = llm_client
        self.printer = printer
        self.tool_map = {t.name: t for t in ALL_TOOLS}
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("report", self._report)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route, {"tools": "tools", "report": "report", END: END})
        g.add_edge("tools", "agent")
        g.add_edge("report", END)
        return g.compile(checkpointer=MemorySaver())

    def _agent(self, state: AgentState) -> dict:
        system = prompts.SINGLE_AGENT_SYSTEM if state["phase"] == "research" else prompts.FOLLOWUP_SYSTEM
        with run_scope(state["run_id"]), agent_scope(AGENT_NAME):
            ai = invoke_with_repair(self.model, [SystemMessage(content=system)] + state["messages"])
        show_message(ai, AGENT_NAME, self.printer)
        return {"messages": [ai], "steps": state["steps"] + 1}

    def _tools(self, state: AgentState) -> dict:
        with run_scope(state["run_id"]), agent_scope(AGENT_NAME):
            results = execute_tool_calls(state["messages"][-1], self.tool_map)
        for msg in results:
            show_message(msg, AGENT_NAME, self.printer)
        return {"messages": results}

    @staticmethod
    def _route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state["steps"] < MAX_RESEARCH_STEPS:
            return "tools"
        return "report" if state["phase"] == "research" else END

    def _report(self, state: AgentState) -> dict:
        new_messages = []
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            # step budget hit mid-plan: close the dangling calls so the conversation stays valid
            new_messages += [ToolMessage(content='{"error": "skipped: step budget reached"}', tool_call_id=c["id"], name=c["name"])
                             for c in last.tool_calls]
        query = prompts.RESEARCH_QUERY.format(ticker=state["ticker"])
        with run_scope(state["run_id"]), agent_scope(AGENT_NAME):
            report = structured_call(
                schema_prompt(prompts.REPORT_SYSTEM, ResearchReport),
                prompts.REPORT_USER.format(query=query, evidence=evidence_digest(state["messages"])),
                ResearchReport, self.llm_client,
            )
        if report is None:
            new_messages.append(AIMessage(content="Final report failed schema validation; evidence is retained above."))
            return {"messages": new_messages, "report": None}
        new_messages.append(AIMessage(content="FINAL REPORT:\n" + report.model_dump_json(indent=1)))
        return {"messages": new_messages, "report": report.model_dump()}

    def research(self, ticker: str, thread_id: str | None = None) -> dict:
        thread_id = thread_id or uuid.uuid4().hex[:8]
        with run_scope() as run_id:
            state = self.graph.invoke(
                {"messages": [HumanMessage(content=prompts.RESEARCH_QUERY.format(ticker=ticker.upper()))],
                 "ticker": ticker.upper(), "run_id": run_id, "phase": "research", "steps": 0, "report": None},
                {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT},
            )
        return {"thread_id": thread_id, "run_id": run_id, "report": state["report"],
                "tool_calls": _count_tool_calls(run_id), "messages": state["messages"]}

    def followup(self, question: str, thread_id: str) -> dict:
        with run_scope() as run_id:
            state = self.graph.invoke(
                {"messages": [HumanMessage(content=question)], "run_id": run_id, "phase": "followup", "steps": 0},
                {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT},
            )
        answer = next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content), "")
        return {"answer": answer, "run_id": run_id, "tool_calls": _count_tool_calls(run_id),
                "history_length": len(state["messages"])}


def _count_tool_calls(run_id: str) -> int:
    return sum(1 for r in read_trace(run_id=run_id) if r.get("event") == "tool_call")

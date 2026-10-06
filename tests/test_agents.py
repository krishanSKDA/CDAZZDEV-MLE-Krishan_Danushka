import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from fakes import fake_llm
from task3_agentic import memory, tools, tracing
from task3_agentic.multi_agent import MultiAgentPipeline
from task3_agentic.single_agent import SingleResearchAgent

REPORT = {
    "ticker": "TEST",
    "financial_health_summary": "Solid trend with elevated volatility.",
    "top_risks": [
        {"title": f"Risk {i}", "description": "d", "evidence": ["vol 45%"], "likelihood": "medium", "impact": "high"}
        for i in range(3)
    ],
    "hedge_strategy": {"strategy": "collar", "rationale": "r", "implementation": "i", "data_basis": ["30d vol"]},
    "sources": [],
}
BRIEF = {
    "ticker": "TEST", "as_of": "2026-10-05",
    "price": {"last_close": 100.0, "momentum_label": "bullish"},
    "volatility": {"window_days": 30, "annualized_vol": 0.4},
    "key_findings": ["uptrend", "high vol"], "data_gaps": ["sentiment"],
}
CLARIFY_REQ = {"question": "Score sentiment", "requested_metrics": ["sentiment"], "reason": "missing",
               "attach_headlines": True}
CLARIFY_RESP = {"answer": "60d vol is 38%", "data": {"vol_60d": 0.38}, "tools_used": ["calculate_volatility"]}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def invoke(self, messages):
        self.seen.append(messages)
        return self.responses.pop(0) if self.responses else AIMessage(content="done")


def call(name, args, cid):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "trace.jsonl")
    monkeypatch.setattr(memory, "CACHE_DIR", tmp_path / "cache")
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, len(idx))))
    prices = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=idx)
    monkeypatch.setattr(tools, "fetch_ohlcv", lambda *a, **k: prices.copy())
    monkeypatch.setattr(tools, "fetch_info", lambda *a, **k: {"shortName": "Test Co", "trailingPE": 20.0})
    monkeypatch.setattr(tools, "fetch_news", lambda *a, **k: [])
    yield


def test_tools_return_dicts_and_errors_not_exceptions():
    price = tools.get_price_data("TEST", "6mo")
    assert price["rows"] > 100 and "derived_facts" in price and price["fundamentals"]["trailingPE"] == 20.0
    vol = tools.calculate_volatility("TEST", 30)
    assert 0 < vol["annualized_vol"] < 2 and 0 <= vol["vol_percentile_1y"] <= 1
    assert "error" in tools.get_price_data("TEST", "7y")
    assert "error" in tools.get_news("TEST", 5)
    assert "error" in tools.llm_sentiment([])


def test_trace_records_duration_and_truncated_output():
    tools.calculate_volatility("TEST", 30)
    rec = tracing.read_trace()[-1]
    assert rec["tool"] == "calculate_volatility" and rec["status"] == "ok"
    assert rec["duration_ms"] >= 0 and len(rec["output"]) <= tracing.OUTPUT_PREVIEW_CHARS
    assert rec["inputs"] == {"arg0": "TEST", "arg1": 30}


def test_permission_enforced_at_execution():
    with tracing.agent_scope("research_writer"):
        out = tools.get_price_data("TEST")
    assert "not permitted" in out["error"]
    assert tracing.read_trace()[-1]["status"] == "blocked"


def test_injected_failure_then_recovery():
    tracing.inject_failure("calculate_volatility", 1)
    assert "simulated" in tools.calculate_volatility("TEST")["error"]
    assert "annualized_vol" in tools.calculate_volatility("TEST")
    assert [r["status"] for r in tracing.read_trace()] == ["injected_failure", "ok"]


def test_cache_roundtrip():
    assert memory.load_cached_brief("TEST") is None
    memory.save_brief("TEST", REPORT)
    assert memory.load_cached_brief("test")["report"]["ticker"] == "TEST"


def test_single_agent_replans_after_failure_and_answers_followup_from_memory():
    tracing.inject_failure("web_search", 1)
    model = ScriptedModel([
        AIMessage(content="Start with volatility.", tool_calls=[call("calculate_volatility", {"ticker": "TEST", "window": 30}, "c1")]),
        AIMessage(content="Look for commentary.", tool_calls=[call("web_search", {"query": "TEST risks"}, "c2")]),
        AIMessage(content="Search failed; use price data instead.", tool_calls=[call("get_price_data", {"ticker": "TEST", "period": "1y"}, "c3")]),
        AIMessage(content="Enough evidence."),
        AIMessage(content="The 30-day annualised volatility was the figure retrieved earlier."),
    ])
    llm, _ = fake_llm([json.dumps(REPORT)])
    agent = SingleResearchAgent(model=model, llm_client=llm, printer=lambda *_: None)

    result = agent.research("TEST", thread_id="t1")
    assert result["report"]["ticker"] == "TEST"
    assert result["tool_calls"] == 3
    statuses = [r["status"] for r in tracing.read_trace(run_id=result["run_id"]) if r["event"] == "tool_call"]
    assert statuses == ["ok", "injected_failure", "ok"]

    follow = agent.followup("What was the 30-day volatility?", thread_id="t1")
    assert follow["tool_calls"] == 0
    assert follow["history_length"] > len(result["messages"])
    # the follow-up prompt contained the earlier tool observation
    assert any("annualized_vol" in str(m.content) for m in model.seen[-1])


def test_multi_agent_handoffs_critique_loop_restriction_and_cache():
    analyst = ScriptedModel([
        AIMessage(content="", tool_calls=[call("get_price_data", {"ticker": "TEST"}, "a1")]),
        AIMessage(content="", tool_calls=[call("calculate_volatility", {"ticker": "TEST", "window": 30}, "a2")]),
        AIMessage(content="Brief ready."),
    ])
    writer = ScriptedModel([
        AIMessage(content="Try price tool.", tool_calls=[call("get_price_data", {"ticker": "TEST"}, "w1")]),
        AIMessage(content="Not allowed; search instead.", tool_calls=[call("web_search", {"query": "TEST"}, "w2")]),
        AIMessage(content="Research done."),
    ])
    clarifier = ScriptedModel([
        AIMessage(content="", tool_calls=[call("calculate_volatility", {"ticker": "TEST", "window": 60}, "c1")]),
        AIMessage(content="60d vol computed."),
    ])
    models = iter([analyst, writer, clarifier])
    tracing.inject_failure("web_search", 1)
    llm, _ = fake_llm([json.dumps(BRIEF), json.dumps(CLARIFY_REQ), json.dumps(CLARIFY_RESP), json.dumps(REPORT)])
    pipeline = MultiAgentPipeline(model_factory=lambda _tools: next(models), llm_client=llm, printer=lambda *_: None)

    result = pipeline.run("TEST")
    assert result["source"] == "agents" and result["report"]["ticker"] == "TEST"
    assert [(h["from"], h["to"], h["schema"]) for h in result["handoffs"]] == [
        ("data_analyst", "research_writer", "DataBrief"),
        ("research_writer", "data_analyst", "ClarificationRequest"),
        ("data_analyst", "research_writer", "ClarificationResponse"),
    ]
    assert result["handoffs"][2]["payload"]["tools_used"] == ["calculate_volatility"]
    events = tracing.read_trace(run_id=result["run_id"])
    assert any(e["event"] == "tool_blocked" and e["agent"] == "research_writer" for e in events)

    cached = pipeline.run("TEST")
    assert cached["source"] == "cache"
    assert not [e for e in tracing.read_trace(run_id=cached["run_id"]) if e["event"] == "tool_call"]

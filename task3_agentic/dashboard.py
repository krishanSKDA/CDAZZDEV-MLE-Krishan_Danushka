"""Run: streamlit run task3_agentic/dashboard.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_agentic.tracing import TRACE_PATH, read_trace  # noqa: E402

st.set_page_config(page_title="Agent Trace", layout="wide")
st.title("Agent trace explorer")
st.caption(f"Source: {TRACE_PATH}")

records = read_trace()
if not records:
    st.info("No trace records yet. Run the agents first.")
    st.stop()

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"])
runs = (df.groupby("run_id")["ts"].min().sort_values(ascending=False))
# default to the newest multi-agent run (has handoffs), else the newest run with tool calls
with_handoffs = set(df.loc[df["event"] == "handoff", "run_id"])
with_tools = set(df.loc[df["event"] == "tool_call", "run_id"])
default = next((r for r in runs.index if r in with_handoffs), next((r for r in runs.index if r in with_tools), runs.index[0]))
run_id = st.sidebar.selectbox("Run", runs.index, index=list(runs.index).index(default),
                              format_func=lambda r: f"{r} · {runs[r]:%Y-%m-%d %H:%M:%S}")
run = df[df["run_id"] == run_id].sort_values("ts")
tools = run[run["event"] == "tool_call"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tool calls", len(tools))
c2.metric("Errors / blocked", int((tools.get("status", pd.Series(dtype=str)) != "ok").sum()) if not tools.empty else 0)
c3.metric("Handoffs", int((run["event"] == "handoff").sum()))
c4.metric("Total tool time (s)", f"{tools['duration_ms'].sum() / 1000:.1f}" if not tools.empty else "0")

if not tools.empty:
    st.subheader("Timeline")
    tools["start"] = tools["ts"] - pd.to_timedelta(tools["duration_ms"], unit="ms")
    try:
        import altair as alt

        chart = alt.Chart(tools).mark_bar().encode(
            x="start:T", x2="ts:T", y=alt.Y("tool:N", sort=None), color="agent:N",
            tooltip=["agent", "tool", "status", "duration_ms", "output"],
        ).properties(height=60 + 28 * tools["tool"].nunique())
        st.altair_chart(chart, width="stretch")
    except ImportError:
        st.bar_chart(tools.set_index("start")["duration_ms"])

    st.subheader("Latency by tool (ms)")
    st.bar_chart(tools.groupby("tool")["duration_ms"].agg(["mean", "max"]))

st.subheader("Events")
cols = [c for c in ["ts", "agent", "event", "tool", "status", "duration_ms", "inputs", "output"] if c in run.columns]
st.dataframe(run[cols].astype(str), width="stretch", hide_index=True)

handoffs = run[run["event"] == "handoff"]
if not handoffs.empty:
    st.subheader("Agent handoffs")
    for _, h in handoffs.iterrows():
        with st.expander(f"{h['from_agent']} → {h['to_agent']} · {h['schema']}"):
            st.code(json.dumps(h["payload"], indent=2, default=str), language="json")

import math
import os

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from common.llm_client import LLMClient
from task1_financial.data_pipeline import (
    RSI_COL, SMA_LONG_COL, SMA_SHORT_COL, build_summary, fetch_info, fetch_news, fetch_ohlcv,
)
from task1_financial.indicators import add_all_indicators
from task1_financial.llm_reasoning import derive_technical_context, score_headlines
from task3_agentic.tracing import traced

TRADING_DAYS_PER_YEAR = 252
HEDGE_HORIZON_TRADING_DAYS = 63
RECENT_ROWS = 5
MAX_SENTIMENT_HEADLINES = 15
MAX_NEWS = 20
MAX_SEARCH_RESULTS = 8
PERIOD_OFFSETS = {
    "1mo": pd.DateOffset(months=1), "3mo": pd.DateOffset(months=3), "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1), "2y": pd.DateOffset(years=2),
}
FUNDAMENTAL_FIELDS = [
    "marketCap", "trailingPE", "forwardPE", "priceToBook", "profitMargins", "operatingMargins", "revenueGrowth",
    "earningsGrowth", "debtToEquity", "currentRatio", "freeCashflow", "totalCash", "totalDebt", "beta",
    "returnOnEquity", "recommendationKey", "targetMeanPrice",
]

_sentiment_client: LLMClient | None = None


def _round(value, ndigits: int = 4):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return None if math.isnan(f) or math.isinf(f) else round(f, ndigits)


def _sentiment_llm() -> LLMClient:
    global _sentiment_client
    if _sentiment_client is None:
        _sentiment_client = LLMClient(model_spec=os.getenv("SENTIMENT_MODEL"))
    return _sentiment_client


@traced
def get_price_data(ticker: str, period: str = "1y") -> dict:
    """Daily OHLCV for `ticker` over `period` (1mo, 3mo, 6mo, 1y, 2y) with SMA-50/200, RSI-14, MACD,
    Bollinger Bands, derived technical facts, momentum score and key fundamentals (valuation, margins, leverage)."""
    if period not in PERIOD_OFFSETS:
        return {"error": f"unsupported period '{period}'", "hint": f"use one of {list(PERIOD_OFFSETS)}"}
    raw = fetch_ohlcv(ticker.upper())
    if raw.empty:
        return {"error": f"no price data for '{ticker}'", "hint": "check the ticker symbol"}

    df = add_all_indicators(raw)
    window = df[df.index >= df.index[-1] - PERIOD_OFFSETS[period]]
    info = fetch_info(ticker.upper())
    summary = build_summary(ticker.upper(), df, info)
    running_max = window["Close"].cummax()

    recent = window.iloc[-RECENT_ROWS:][["Open", "High", "Low", "Close", "Volume", SMA_SHORT_COL, SMA_LONG_COL, RSI_COL, "macd_hist"]]
    return {
        "ticker": ticker.upper(),
        "company_name": summary["company_name"],
        "sector": summary["sector"],
        "period": period,
        "start": window.index[0].date().isoformat(),
        "end": window.index[-1].date().isoformat(),
        "rows": len(window),
        "last_close": summary["current_price"],
        "period_return": _round(window["Close"].iloc[-1] / window["Close"].iloc[0] - 1),
        "max_drawdown": _round((window["Close"] / running_max - 1).min()),
        "high_52w": summary["high_52w"],
        "low_52w": summary["low_52w"],
        "ytd_return": summary["ytd_return"],
        "indicators": summary["indicators"],
        "momentum_signal": {k: summary["momentum_signal"][k] for k in ("label", "score", "components")},
        "derived_facts": derive_technical_context(df),
        "recent_ohlcv": [{"date": idx.date().isoformat(), **{k: _round(v, 2) for k, v in row.items()}}
                         for idx, row in recent.iterrows()],
        "fundamentals": {k: _round(info.get(k)) for k in FUNDAMENTAL_FIELDS if info.get(k) is not None},
    }


@traced
def calculate_volatility(ticker: str, window: int = 30) -> dict:
    """Annualised historical volatility of daily log returns over a rolling `window` (trading days), plus 1-year
    realised and downside volatility, the current volatility percentile and the implied 1-sigma 90-day move."""
    if window < 5:
        return {"error": "window must be at least 5 trading days", "hint": "use 20, 30 or 60"}
    df = fetch_ohlcv(ticker.upper())
    if df.empty or len(df) <= window:
        return {"error": f"not enough price history for '{ticker}'", "hint": "check the ticker or use a shorter window"}

    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    annualise = math.sqrt(TRADING_DAYS_PER_YEAR)
    rolling = log_ret.rolling(window).std() * annualise
    last_year = log_ret.iloc[-TRADING_DAYS_PER_YEAR:]
    current = rolling.iloc[-1]
    return {
        "ticker": ticker.upper(),
        "as_of": df.index[-1].date().isoformat(),
        "window_days": window,
        "annualized_vol": _round(current),
        "vol_1y": _round(last_year.std() * annualise),
        "downside_vol_1y": _round(last_year[last_year < 0].std() * annualise),
        "vol_percentile_1y": _round((rolling.iloc[-TRADING_DAYS_PER_YEAR:] <= current).mean()),
        "expected_1sigma_move_90d_pct": _round(current * math.sqrt(HEDGE_HORIZON_TRADING_DAYS / TRADING_DAYS_PER_YEAR)),
        "last_close": _round(df["Close"].iloc[-1], 2),
    }


@traced
def get_news(ticker: str, n: int = 10) -> dict:
    """Recent news headlines for `ticker` as a structured list (title, publisher, published time, source)."""
    n = max(1, min(int(n), MAX_NEWS))
    items = fetch_news(ticker.upper(), n)
    if not items:
        return {"error": f"no headlines found for '{ticker}'", "hint": "try web_search for recent commentary"}
    return {
        "ticker": ticker.upper(),
        "count": len(items),
        "headlines": [{"title": i.title, "publisher": i.publisher,
                       "published": i.published.isoformat() if i.published else None, "source": i.source}
                      for i in items],
    }


@traced
def llm_sentiment(headlines: list[str], ticker: str = "") -> dict:
    """Scores each headline with an LLM (sentiment, confidence, reason) and returns a confidence-weighted
    aggregate score in [-1, 1]. Only pass real headlines that were retrieved, never invented ones."""
    headlines = [h for h in headlines if isinstance(h, str) and h.strip()][:MAX_SENTIMENT_HEADLINES]
    if not headlines:
        return {"error": "no headlines supplied", "hint": "obtain real headlines first"}
    scored, agg = score_headlines(_sentiment_llm(), headlines, ticker.upper() or "the company", ticker.upper() or "the company")
    if agg.n_scored == 0:
        return {"error": "sentiment model returned no valid scores", "hint": "retry later"}
    return {
        **agg.model_dump(),
        "per_headline": [{"headline": s.headline[:90], "sentiment": s.sentiment, "confidence": s.confidence}
                         for s in scored],
    }


@traced
def web_search(query: str, max_results: int = 5) -> dict:
    """DuckDuckGo web search for analyst commentary, risks and recent events. Returns title, url and snippet."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    try:
        hits = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:
        return {"error": f"search failed: {type(exc).__name__}: {str(exc)[:120]}",
                "hint": "retry with a shorter query or use get_news"}
    if not hits:
        return {"error": f"no results for '{query}'", "hint": "broaden the query"}
    return {
        "query": query,
        "results": [{"title": h.get("title"), "url": h.get("href"), "snippet": (h.get("body") or "")[:300]}
                    for h in hits],
    }


get_price_data_tool = tool(get_price_data)
calculate_volatility_tool = tool(calculate_volatility)
get_news_tool = tool(get_news)
llm_sentiment_tool = tool(llm_sentiment)
web_search_tool = tool(web_search)

ALL_TOOLS = [get_price_data_tool, get_news_tool, calculate_volatility_tool, llm_sentiment_tool, web_search_tool]
ANALYST_TOOLS = [get_price_data_tool, calculate_volatility_tool, llm_sentiment_tool]
WRITER_TOOLS = [web_search_tool, get_news_tool]

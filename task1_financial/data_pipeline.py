import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yfinance as yf

from common.config import DEFAULT_TICKER
from common.logging_utils import get_logger
from task1_financial.indicators import (
    BB_WINDOW, MACD_FAST, MACD_SIGNAL, MACD_SLOW, RSI_PERIOD, SMA_LONG_WINDOW, SMA_SHORT_WINDOW, add_all_indicators,
)
from task1_financial.schemas import NewsItem

log = get_logger("task1.pipeline")

HISTORY_YEARS = 2
CALENDAR_DAYS_PER_TRADING_DAY = 365 / 252
# fetch extra history so SMA-200 is already valid at the start of the 2-year window
WARMUP_CALENDAR_DAYS = math.ceil(SMA_LONG_WINDOW * CALENDAR_DAYS_PER_TRADING_DAY)
WEEKS_PER_YEAR = 52
DEFAULT_NEWS_COUNT = 10

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
CORPORATE_SUFFIXES = re.compile(r"\b(corporation|corp|incorporated|inc|ltd|limited|plc|holdings|company|co)\b\.?", re.I)

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_MIDLINE = 50
BB_PCT_B_UPPER = 1.0
BB_PCT_B_LOWER = 0.0
MOMENTUM_SIGNAL_THRESHOLD = 2

SMA_SHORT_COL = f"sma_{SMA_SHORT_WINDOW}"
SMA_LONG_COL = f"sma_{SMA_LONG_WINDOW}"
RSI_COL = f"rsi_{RSI_PERIOD}"


@dataclass
class PipelineResult:
    ticker: str
    prices: pd.DataFrame
    news: list[NewsItem]
    summary: dict
    info: dict = field(default_factory=dict)


def _safe_float(value, ndigits: int = 4) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, ndigits)


def fetch_ohlcv(ticker: str, years: int = HISTORY_YEARS) -> pd.DataFrame:
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years) - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)
    try:
        df = yf.Ticker(ticker).history(start=start, interval="1d", auto_adjust=True)
    except Exception as exc:
        log.error("price download failed", extra={"data": {"ticker": ticker, "error": str(exc)}})
        return pd.DataFrame()
    if df is None or df.empty:
        log.warning("no price data returned", extra={"data": {"ticker": ticker}})
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna(subset=["Close"])

    span_days = (df.index[-1] - df.index[0]).days
    if span_days < years * 365:
        log.warning("history shorter than requested", extra={"data": {"ticker": ticker, "span_days": span_days}})
    return df


def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.warning("info lookup failed", extra={"data": {"ticker": ticker, "error": str(exc)}})
        return {}


def _parse_yf_news_item(item: dict) -> NewsItem | None:
    content = item.get("content") or item
    title = content.get("title")
    if not title:
        return None
    provider = content.get("provider") or {}
    url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
    published = content.get("pubDate") or content.get("providerPublishTime")
    if isinstance(published, (int, float)):
        published = datetime.fromtimestamp(published, tz=timezone.utc)
    return NewsItem(
        title=title.strip(),
        publisher=provider.get("displayName") or content.get("publisher"),
        link=url,
        published=published,
        source="yfinance",
    )


def _news_from_yfinance(ticker: str, n: int) -> list[NewsItem]:
    try:
        raw = yf.Ticker(ticker).get_news(count=n) or []
    except Exception as exc:
        log.warning("yfinance news failed", extra={"data": {"error": str(exc)}})
        return []
    return [item for item in map(_parse_yf_news_item, raw) if item]


def _news_from_rss(url: str, source: str) -> list[NewsItem]:
    feed = feedparser.parse(url)
    if feed.get("bozo") and not feed.entries:
        log.warning("rss feed unreadable", extra={"data": {"source": source, "error": str(feed.get("bozo_exception"))}})
        return []
    items = []
    for entry in feed.entries:
        # some feeds ship mis-encoded curly quotes as U+FFFD
        title = (entry.get("title") or "").replace("�", "").strip()
        if not title:
            continue
        publisher = (entry.get("source") or {}).get("title")
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        try:
            published = parsedate_to_datetime(entry["published"]) if entry.get("published") else None
        except (TypeError, ValueError):
            published = None
        items.append(NewsItem(title=title, publisher=publisher, link=entry.get("link"), published=published, source=source))
    return items


def clean_company_name(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = CORPORATE_SUFFIXES.sub("", name).strip(" ,.")
    return cleaned or name


def fetch_news(ticker: str, n: int = DEFAULT_NEWS_COUNT, company_name: str | None = None) -> list[NewsItem]:
    """Merges yfinance, Yahoo Finance RSS and Google News RSS; headlines naming the company rank first."""
    brand = clean_company_name(company_name)
    query = quote_plus(f"{brand or ticker} stock")
    sources = [
        lambda: _news_from_yfinance(ticker, n),
        lambda: _news_from_rss(YAHOO_RSS.format(ticker=ticker), "yahoo_rss"),
        lambda: _news_from_rss(GOOGLE_NEWS_RSS.format(query=query), "google_news_rss"),
    ]
    seen, collected = set(), []
    for fetch in sources:
        for item in fetch():
            key = item.title.lower()
            if key not in seen:
                seen.add(key)
                collected.append(item)

    patterns = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in (ticker, brand) if term]
    is_relevant = lambda item: any(p.search(item.title) for p in patterns)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    collected.sort(key=lambda x: (is_relevant(x), x.published or epoch), reverse=True)

    if len(collected) < n:
        log.warning("fewer headlines than requested", extra={"data": {"requested": n, "got": len(collected)}})
    return collected[:n]


def _ytd_return(close: pd.Series) -> float | None:
    last_date = close.index[-1]
    prior_year = close[close.index.year < last_date.year]
    this_year = close[close.index.year == last_date.year]
    base = prior_year.iloc[-1] if not prior_year.empty else (this_year.iloc[0] if not this_year.empty else None)
    return _safe_float(close.iloc[-1] / base - 1) if base else None


def momentum_signal(row: pd.Series) -> dict:
    """Rule-based score in [-5, 5]: trend (2 votes), MACD (1), RSI regime (1), Bollinger position (1)."""
    close, sma_s, sma_l = row.get("Close"), row.get(SMA_SHORT_COL), row.get(SMA_LONG_COL)
    rsi_v, hist, pct_b = row.get(RSI_COL), row.get("macd_hist"), row.get("bb_pct_b")

    def vote(inputs, bullish, bearish) -> int:
        if not all(pd.notna(x) for x in inputs):
            return 0
        return 1 if bullish() else (-1 if bearish() else 0)

    components = {
        "price_vs_sma200": vote((close, sma_l), lambda: close > sma_l, lambda: close < sma_l),
        "sma50_vs_sma200": vote((sma_s, sma_l), lambda: sma_s > sma_l, lambda: sma_s < sma_l),
        "macd_histogram": vote((hist,), lambda: hist > 0, lambda: hist < 0),
        "rsi_regime": vote((rsi_v,), lambda: RSI_MIDLINE < rsi_v < RSI_OVERBOUGHT,
                           lambda: RSI_OVERSOLD < rsi_v < RSI_MIDLINE),
        # outside the bands = stretched, so mean-reversion pressure opposes the move
        "bollinger_position": vote((pct_b,), lambda: pct_b < BB_PCT_B_LOWER, lambda: pct_b > BB_PCT_B_UPPER),
    }
    score = sum(components.values())
    has = lambda v: pd.notna(v)
    if score >= MOMENTUM_SIGNAL_THRESHOLD:
        label = "bullish"
    elif score <= -MOMENTUM_SIGNAL_THRESHOLD:
        label = "bearish"
    else:
        label = "neutral"

    flags = []
    if has(rsi_v) and rsi_v >= RSI_OVERBOUGHT:
        flags.append("rsi_overbought")
    if has(rsi_v) and rsi_v <= RSI_OVERSOLD:
        flags.append("rsi_oversold")
    return {"label": label, "score": score, "components": components, "flags": flags}


def build_summary(ticker: str, df: pd.DataFrame, info: dict) -> dict:
    if df.empty:
        return {"ticker": ticker, "error": "no price data"}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    window_52w = df[df.index >= df.index[-1] - pd.DateOffset(weeks=WEEKS_PER_YEAR)]
    pe = info.get("trailingPE") or info.get("forwardPE")

    return {
        "ticker": ticker,
        "company_name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector"),
        "currency": info.get("currency"),
        "as_of": df.index[-1].date().isoformat(),
        "current_price": _safe_float(last["Close"], 2),
        "daily_change_pct": _safe_float(last["Close"] / prev["Close"] - 1),
        "high_52w": _safe_float(window_52w["High"].max(), 2),
        "low_52w": _safe_float(window_52w["Low"].min(), 2),
        "pe_ratio": _safe_float(pe, 2),
        "pe_type": "trailing" if info.get("trailingPE") else ("forward" if info.get("forwardPE") else None),
        "ytd_return": _ytd_return(df["Close"]),
        "indicators": {
            SMA_SHORT_COL: _safe_float(last.get(SMA_SHORT_COL), 2),
            SMA_LONG_COL: _safe_float(last.get(SMA_LONG_COL), 2),
            RSI_COL: _safe_float(last.get(RSI_COL), 2),
            f"macd_{MACD_FAST}_{MACD_SLOW}": _safe_float(last.get("macd")),
            f"macd_signal_{MACD_SIGNAL}": _safe_float(last.get("macd_signal")),
            "macd_hist": _safe_float(last.get("macd_hist")),
            f"bb_upper_{BB_WINDOW}": _safe_float(last.get("bb_upper"), 2),
            f"bb_lower_{BB_WINDOW}": _safe_float(last.get("bb_lower"), 2),
            "bb_pct_b": _safe_float(last.get("bb_pct_b")),
            "bb_bandwidth": _safe_float(last.get("bb_bandwidth")),
        },
        "momentum_signal": momentum_signal(last),
        "rows": len(df),
    }


def run_pipeline(ticker: str = DEFAULT_TICKER, news_count: int = DEFAULT_NEWS_COUNT) -> PipelineResult:
    ticker = ticker.upper().strip()
    prices = fetch_ohlcv(ticker)
    if not prices.empty:
        prices = add_all_indicators(prices)
        cutoff = prices.index[-1] - pd.DateOffset(years=HISTORY_YEARS)
        prices = prices[prices.index >= cutoff]
    info = fetch_info(ticker)
    news = fetch_news(ticker, news_count, info.get("shortName"))
    summary = build_summary(ticker, prices, info)
    return PipelineResult(ticker=ticker, prices=prices, news=news, summary=summary, info=info)


if __name__ == "__main__":
    result = run_pipeline(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER)
    print(json.dumps(result.summary, indent=2))
    print(f"\n{len(result.prices)} rows {result.prices.index[0].date()} -> {result.prices.index[-1].date()}")
    for item in result.news:
        print(f"- [{item.source}] {item.title} ({item.publisher})")

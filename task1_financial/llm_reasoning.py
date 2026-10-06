import json
from dataclasses import dataclass

import pandas as pd

from common.llm_client import LLMClient
from common.logging_utils import get_logger
from task1_financial import prompts
from task1_financial.data_pipeline import RSI_COL, SMA_LONG_COL, SMA_SHORT_COL, PipelineResult
from task1_financial.schemas import HeadlineSentiment, SentimentAggregate, TradingSignal

log = get_logger("task1.reasoning")

POLARITY = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
SENTIMENT_LABEL_THRESHOLD = 0.15
SLOPE_LOOKBACK_DAYS = 20
HIST_TREND_LOOKBACK_DAYS = 5
DIVERGENCE_LOOKBACK_DAYS = 20
RSI_DIVERGENCE_GAP = 5.0
BANDWIDTH_LOOKBACK_DAYS = 126
SQUEEZE_PERCENTILE = 0.20
EXPANSION_PERCENTILE = 0.80
FALLBACK_SIGNAL = {"bullish": "Buy", "neutral": "Hold", "bearish": "Sell"}


@dataclass
class AnalysisResult:
    headline_sentiments: list[HeadlineSentiment]
    sentiment: SentimentAggregate
    derived_facts: list[str]
    signal: TradingSignal


def score_headline(client: LLMClient, headline: str, ticker: str, company: str) -> HeadlineSentiment | None:
    messages = [
        {"role": "system", "content": prompts.SENTIMENT_SYSTEM},
        {"role": "user", "content": prompts.SENTIMENT_USER.format(company=company, ticker=ticker, headline=headline)},
    ]
    result = client.structured(messages, HeadlineSentiment)
    if result is None:
        log.error("headline sentiment failed validation", extra={"data": {"headline": headline[:120]}})
        return None
    # keep the source headline verbatim even if the model paraphrased it
    return result.model_copy(update={"headline": headline})


def aggregate_sentiment(results: list[HeadlineSentiment], n_failed: int = 0) -> SentimentAggregate:
    counts = {k: 0 for k in POLARITY}
    for r in results:
        counts[r.sentiment] += 1
    total_conf = sum(r.confidence for r in results)
    score = sum(POLARITY[r.sentiment] * r.confidence for r in results) / total_conf if total_conf else 0.0
    if score > SENTIMENT_LABEL_THRESHOLD:
        label = "positive"
    elif score < -SENTIMENT_LABEL_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    return SentimentAggregate(score=round(score, 4), label=label, n_scored=len(results), n_failed=n_failed, counts=counts)


def score_headlines(client: LLMClient, headlines: list[str], ticker: str, company: str
                    ) -> tuple[list[HeadlineSentiment], SentimentAggregate]:
    scored = [score_headline(client, h, ticker, company) for h in headlines]
    valid = [s for s in scored if s is not None]
    return valid, aggregate_sentiment(valid, n_failed=len(scored) - len(valid))


def _days_since_sign_change(series: pd.Series) -> int | None:
    signs = series.dropna().apply(lambda x: x > 0)
    if signs.empty:
        return None
    changes = signs.ne(signs.shift()).iloc[1:]
    flips = changes[changes]
    return None if flips.empty else int(len(signs) - signs.index.get_loc(flips.index[-1]) - 1)


def derive_technical_context(df: pd.DataFrame) -> list[str]:
    """Turns raw indicator columns into relational facts so the LLM reasons over interactions."""
    if df.empty or len(df) < SLOPE_LOOKBACK_DAYS + 1:
        return ["Insufficient price history for derived technical facts."]

    facts = []
    last = df.iloc[-1]
    close = last["Close"]

    if pd.notna(last[SMA_SHORT_COL]) and pd.notna(last[SMA_LONG_COL]):
        facts.append(f"Price is {close / last[SMA_SHORT_COL] - 1:+.1%} vs SMA-50 and {close / last[SMA_LONG_COL] - 1:+.1%} vs SMA-200.")
        regime = "golden-cross (SMA-50 above SMA-200)" if last[SMA_SHORT_COL] > last[SMA_LONG_COL] else "death-cross (SMA-50 below SMA-200)"
        cross_age = _days_since_sign_change(df[SMA_SHORT_COL] - df[SMA_LONG_COL])
        facts.append(f"Trend regime: {regime}" + (f", last cross {cross_age} sessions ago." if cross_age is not None else ", no cross in window."))
        slope = df[SMA_LONG_COL].iloc[-1] / df[SMA_LONG_COL].iloc[-1 - SLOPE_LOOKBACK_DAYS] - 1
        facts.append(f"SMA-200 {SLOPE_LOOKBACK_DAYS}-session slope: {slope:+.2%} ({'rising' if slope > 0 else 'falling'}).")

    hist = df["macd_hist"]
    if pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-1 - HIST_TREND_LOOKBACK_DAYS]):
        side = "above" if hist.iloc[-1] > 0 else "below"
        delta = abs(hist.iloc[-1]) - abs(hist.iloc[-1 - HIST_TREND_LOOKBACK_DAYS])
        state = "expanding" if delta > 0 else "contracting"
        age = _days_since_sign_change(hist)
        facts.append(f"MACD is {side} its signal line" + (f" for {age} sessions" if age is not None else "")
                     + f"; histogram {state} over the last {HIST_TREND_LOOKBACK_DAYS} sessions.")

    rsi_window = df[RSI_COL].iloc[-DIVERGENCE_LOOKBACK_DAYS:]
    price_window = df["Close"].iloc[-DIVERGENCE_LOOKBACK_DAYS:]
    if rsi_window.notna().all():
        facts.append(f"RSI-14 at {last[RSI_COL]:.1f}, {last[RSI_COL] - rsi_window.iloc[0]:+.1f} points over {DIVERGENCE_LOOKBACK_DAYS} sessions.")
        if close >= price_window.max() and last[RSI_COL] < rsi_window.max() - RSI_DIVERGENCE_GAP:
            facts.append("Bearish divergence: price at a 20-session high while RSI is below its recent peak.")
        elif close <= price_window.min() and last[RSI_COL] > rsi_window.min() + RSI_DIVERGENCE_GAP:
            facts.append("Bullish divergence: price at a 20-session low while RSI holds above its recent trough.")

    if pd.notna(last["bb_pct_b"]):
        bw_rank = df["bb_bandwidth"].iloc[-BANDWIDTH_LOOKBACK_DAYS:].rank(pct=True).iloc[-1]
        width_state = "squeeze (low volatility, breakout risk)" if bw_rank <= SQUEEZE_PERCENTILE else (
            "expanded (high volatility)" if bw_rank >= EXPANSION_PERCENTILE else "normal")
        facts.append(f"Bollinger %B {last['bb_pct_b']:.2f} ({'above upper band' if last['bb_pct_b'] > 1 else 'below lower band' if last['bb_pct_b'] < 0 else 'inside bands'}); "
                     f"bandwidth at {bw_rank:.0%} percentile of 6 months -> {width_state}.")

    ret_20 = close / df["Close"].iloc[-1 - SLOPE_LOOKBACK_DAYS] - 1
    facts.append(f"{SLOPE_LOOKBACK_DAYS}-session price return: {ret_20:+.1%}.")
    return facts


def _fmt(value, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if pct else str(value)


def build_signal_messages(summary: dict, derived_facts: list[str], sentiment: SentimentAggregate) -> list[dict]:
    mom = summary["momentum_signal"]
    user = prompts.SIGNAL_USER.format(
        ticker=summary["ticker"], company=summary["company_name"], as_of=summary["as_of"],
        current_price=_fmt(summary["current_price"]), low_52w=_fmt(summary["low_52w"]),
        high_52w=_fmt(summary["high_52w"]), ytd_return=_fmt(summary["ytd_return"], pct=True),
        pe_ratio=_fmt(summary["pe_ratio"]),
        indicators=json.dumps(summary["indicators"], indent=2),
        derived_facts="\n".join(f"- {f}" for f in derived_facts),
        momentum_score=mom["score"], momentum_label=mom["label"], momentum_components=json.dumps(mom["components"]),
        sentiment_score=sentiment.score, sentiment_label=sentiment.label, sentiment_n=sentiment.n_scored,
    )
    return [{"role": "system", "content": prompts.SIGNAL_SYSTEM}, {"role": "user", "content": user}]


def rule_based_signal(summary: dict, sentiment: SentimentAggregate) -> TradingSignal:
    mom = summary["momentum_signal"]
    return TradingSignal(
        signal=FALLBACK_SIGNAL[mom["label"]],
        conviction="low",
        justification=(f"The LLM signal could not be validated, so this falls back to the rule-based momentum score. "
                       f"The score is {mom['score']} on a -5 to +5 scale, which maps to {mom['label']}. "
                       f"News sentiment is {sentiment.label} at {sentiment.score:+.2f} and is not weighted in this fallback."),
        key_drivers=[f"{k}={v:+d}" for k, v in mom["components"].items() if v][:4] or ["no decisive component"],
        source="rule_fallback",
    )


def generate_signal(client: LLMClient, summary: dict, derived_facts: list[str], sentiment: SentimentAggregate) -> TradingSignal:
    result = client.structured(build_signal_messages(summary, derived_facts, sentiment), TradingSignal)
    if result is None:
        log.error("signal generation failed validation; using rule-based fallback", extra={"data": {"ticker": summary["ticker"]}})
        return rule_based_signal(summary, sentiment)
    return result


def run_analysis(pipeline: PipelineResult, client: LLMClient | None = None) -> AnalysisResult:
    client = client or LLMClient()
    company = pipeline.summary.get("company_name", pipeline.ticker)
    sentiments, aggregate = score_headlines(client, [n.title for n in pipeline.news], pipeline.ticker, company)
    facts = derive_technical_context(pipeline.prices)
    signal = generate_signal(client, pipeline.summary, facts, aggregate)
    return AnalysisResult(headline_sentiments=sentiments, sentiment=aggregate, derived_facts=facts, signal=signal)

import json

import numpy as np
import pandas as pd
import pytest

from fakes import fake_llm
from task1_financial.data_pipeline import build_summary
from task1_financial.indicators import add_all_indicators
from task1_financial.llm_reasoning import (
    aggregate_sentiment, derive_technical_context, generate_signal, score_headline,
)
from task1_financial.schemas import HeadlineSentiment, TradingSignal

VALID_SIGNAL = {
    "signal": "buy",
    "conviction": "Medium",
    "justification": "Trend and momentum agree. Price sits above a rising SMA-200 while MACD expands. "
                     "RSI is elevated but not diverging. Sentiment adds modest support.",
    "key_drivers": ["trend confirmation", "expanding MACD"],
}


@pytest.fixture
def prices():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, len(idx))))
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Volume": 1_000_000}, index=idx)
    return add_all_indicators(df)


def test_headline_valid_response_keeps_original_headline():
    llm, _ = fake_llm([json.dumps({"headline": "paraphrased", "sentiment": "Positive", "confidence": 0.8,
                                   "brief_reason": "Strong demand."})])
    result = score_headline(llm, "Original headline", "NVDA", "NVIDIA")
    assert result.headline == "Original headline"
    assert result.sentiment == "positive"


def test_headline_invalid_then_repaired():
    bad = json.dumps({"headline": "h", "sentiment": "bullish", "confidence": 1.7, "brief_reason": "x y z"})
    good = json.dumps({"headline": "h", "sentiment": "neutral", "confidence": 0.5, "brief_reason": "No impact."})
    llm, calls = fake_llm([bad, good])
    assert score_headline(llm, "h", "NVDA", "NVIDIA").sentiment == "neutral"
    assert "failed validation" in calls.calls[1][-1]["content"]


def test_headline_unrecoverable_returns_none():
    llm, _ = fake_llm(["not json", "still not json"])
    assert score_headline(llm, "h", "NVDA", "NVIDIA") is None


def test_aggregate_is_confidence_weighted():
    items = [
        HeadlineSentiment(headline="a", sentiment="positive", confidence=0.9, brief_reason="abc"),
        HeadlineSentiment(headline="b", sentiment="negative", confidence=0.3, brief_reason="abc"),
        HeadlineSentiment(headline="c", sentiment="neutral", confidence=0.6, brief_reason="abc"),
    ]
    agg = aggregate_sentiment(items, n_failed=1)
    assert agg.score == pytest.approx((0.9 - 0.3) / 1.8, abs=1e-4)
    assert agg.label == "positive"
    assert agg.counts == {"positive": 1, "neutral": 1, "negative": 1}
    assert agg.n_failed == 1


def test_aggregate_empty_is_neutral():
    agg = aggregate_sentiment([])
    assert agg.score == 0 and agg.label == "neutral"


def test_signal_justification_sentence_limit():
    with pytest.raises(ValueError):
        TradingSignal(**{**VALID_SIGNAL, "justification": "Only one sentence."})


def test_generate_signal_normalises_case(prices):
    llm, calls = fake_llm([json.dumps(VALID_SIGNAL)])
    summary = build_summary("TEST", prices, {})
    signal = generate_signal(llm, summary, derive_technical_context(prices), aggregate_sentiment([]))
    assert signal.signal == "Buy" and signal.conviction == "medium" and signal.source == "llm"
    system, user = calls.calls[0]
    assert system["role"] == "system" and user["role"] == "user"
    assert "Derived technical facts" in user["content"]


def test_generate_signal_falls_back_on_validation_failure(prices):
    llm, _ = fake_llm(["{}", "{}"])
    summary = build_summary("TEST", prices, {})
    signal = generate_signal(llm, summary, derive_technical_context(prices), aggregate_sentiment([]))
    assert signal.source == "rule_fallback"
    assert signal.conviction == "low"


def test_derived_facts_handle_short_history():
    df = add_all_indicators(pd.DataFrame({"Close": [1.0, 2.0, 3.0]}))
    assert derive_technical_context(df) == ["Insufficient price history for derived technical facts."]

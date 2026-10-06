# Task 1: LLM-Powered Equity Research Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krishanSKDA/CDAZZDEV-MLE-Krishan_Danushka/blob/main/task1_financial/Task1_Equity_Research.ipynb)

| File | Purpose |
|---|---|
| `indicators.py` | SMA, RSI (Wilder), MACD, Bollinger: pure pandas/numpy functions, no TA-Lib |
| `data_pipeline.py` | 2-year OHLCV (relative dates + SMA-200 warm-up), news from yfinance → Yahoo RSS → Google News RSS, summary dict, momentum score |
| `prompts.py` | All system/user prompt templates |
| `schemas.py` | Pydantic models: `NewsItem`, `HeadlineSentiment`, `SentimentAggregate`, `TradingSignal` (3 to 5 sentence validator) |
| `llm_reasoning.py` | Per-headline scoring, confidence-weighted aggregation, derived technical facts, signal generation with fallback |
| `report.py` | Markdown brief → styled HTML with an embedded matplotlib chart (`outputs/`) |

## Key decisions
- **Momentum signal:** five votes (price vs SMA-200, SMA-50 vs SMA-200, MACD histogram sign, RSI regime, Bollinger
  stretch) summed to [-5, 5]; ≥ 2 bullish, ≤ -2 bearish. All thresholds are named constants.
- **Sentiment aggregate:** Σ(polarity × confidence) / Σ confidence ∈ [-1, 1], so confident headlines count more and
  low-confidence ones count less. Failed validations are excluded and counted in `n_failed`.
- **Reasoning over combinations:** the LLM receives relational facts (crossover age, SMA-200 slope, MACD histogram
  expansion, RSI divergence, bandwidth percentile), and its prompt asks it to weigh where signals confirm or diverge.
- **Validation:** JSON mode + Pydantic; one repair turn with the validation error; then a logged rule-based fallback.

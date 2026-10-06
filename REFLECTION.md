# Reflection

## Architectural decisions

**Shared foundation.** All three tasks use one OpenAI-compatible LLM client running Google `gemini-3.1-flash-lite`,
the cheapest free-tier Gemini model available to new API keys. It handles exponential backoff, JSON mode, and Pydantic
validation with one repair turn. A response that still fails validation is logged and replaced by a deterministic
fallback: a rule-based signal in Task 1, a brief built from raw tool outputs in Task 3. Nothing crashes silently.

**Task 1.** Indicators are pure functions, tested against independent recursive implementations and the StockCharts
Wilder RSI table. The LLM gets derived facts, not raw numbers: crossover age, SMA-200 slope, MACD histogram expansion,
RSI divergence and bandwidth percentile. For NVDA this produced a *Hold*. The model weighed a strong trend against price
sitting above the upper Bollinger band, which is reasoning about combinations rather than repeating values.

**Task 2.** I chose risk-factor extraction because it can be checked by machine. Evidence quotes must be copied
verbatim, so hallucination becomes measurable. Gemini generated 320 examples from a balanced seed grid with distractor
sentences. Each example was validated, giving 97.6% acceptance; the 8 rejects had non-verbatim quotes. There were zero
near-duplicates, and mean pairwise TF-IDF cosine was 0.02. The student is Qwen2.5-3B (a different family), trained with
QLoRA (4-bit NF4, r=16) and loss on assistant tokens only.

**Task 3.** Both agents are explicit LangGraph state graphs, so routing, step budgets and the one-round critique are
visible and testable. The coordination is real: the analyst has no news access, so the writer sends its headlines back
for sentiment scoring. In the live run, the final report quotes the analyst's clarification (sentiment 0.53, 63-day
volatility at the 67th percentile). Tool permissions are enforced at model binding and again at execution time.

## Results and limitations encountered
- **Fine-tuning:** validation loss went 0.0670 → 0.0564 → 0.0562. It decreased every epoch but had nearly converged by
  epoch 3, so more epochs would mainly risk overfitting. Peak GPU memory was 7.64 GB of 15.6 GB, with no OOM.
- **Library drift on Colab:** transformers v5 removed `warmup_ratio`, so I switched to integer warmup steps. Colab's
  preinstalled torchao 0.10 broke the PEFT merge, so I uninstalled it and re-ran only the merge. Both fixes are
  documented in the notebook.
- **Evaluation (32 held-out passages, base vs fine-tuned):**

  | Metric | Base | Fine-tuned |
  |---|---|---|
  | ROUGE-L | 0.60 | 0.88 |
  | Category F1 | 0.24 | 0.94 |
  | Verbatim-quote rate | 64% | 98% |
  | Judge "hallucinated" | 38% | 3% |
  | Manual hallucination rate | — | [X]% |

  Remaining errors are taxonomy-boundary confusions and severity calibration (0.72).
- **Cheapest-model trade-offs:** Flash-Lite agents often skip the reasoning text between tool calls, so replanning
  shows mainly in the trace: `get_news` errors, then the agent calls `web_search`. The same model also acts as Task 2
  teacher and judge. Self-preference bias is offset by reference-grounded checks that use no LLM.
- **Data sources:** yfinance returned no news, so the pipeline falls back to RSS feeds.

## What I would improve with more time
- **Task 1:** backtest the momentum rule and the LLM signal against forward returns, and calibrate sentiment confidence.
- **Task 2:** add real SEC 10-K passages to the test set to measure synthetic-to-real transfer. Compare LoRA ranks
  8/16/32 and try constrained JSON decoding. Use a judge from a different family than the teacher.
- **Task 3:** use a stronger planning model, add per-agent token budgets, and check every cited risk against the trace.

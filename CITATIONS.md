# Citations

All AI assistance and external references used in this repository, in the format required by Section 2.2 of the
assessment. Code was generated with Claude through Claude Code, then reviewed, run and tested by the candidate.

## AI-assisted code

### Shared
```
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Read API keys from Colab userdata or environment, never hardcoded', Date: 2026-10-06            -> common/config.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'JSON structured logger shared across tasks', Date: 2026-10-06                                    -> common/logging_utils.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'OpenAI-compatible client over Gemini/OpenRouter with retry, backoff, provider fallback and Pydantic-validated JSON output with one repair attempt', Date: 2026-10-06 -> common/llm_client.py
```

### Task 1
```
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'SMA, RSI with Wilder smoothing, MACD(12,26,9), Bollinger(20,2) from first principles without TA-Lib', Date: 2026-10-06 -> task1_financial/indicators.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'yfinance OHLCV fetch without hardcoded dates, multi-source news with RSS fallback, summary dict with rule-based momentum signal, null-safe', Date: 2026-10-06 -> task1_financial/data_pipeline.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Pydantic schemas for news, per-headline sentiment, aggregate sentiment and a Buy/Hold/Sell signal with a 3-5 sentence validator', Date: 2026-10-06 -> task1_financial/schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'System/user prompt constants for headline sentiment and indicator-combination signal reasoning', Date: 2026-10-06 -> task1_financial/prompts.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Per-headline LLM sentiment, confidence-weighted aggregation, derived technical facts and validated signal generation with rule-based fallback', Date: 2026-10-06 -> task1_financial/llm_reasoning.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Markdown research brief rendered to styled HTML with embedded matplotlib price/RSI/MACD chart', Date: 2026-10-06 -> task1_financial/report.py
```

### Task 2
```
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Risk-factor extraction schema, taxonomy, grounding check and correct/partial/hallucinated labeller', Date: 2026-10-06 -> task2_genai/domain.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Teacher system prompt and diversity seed grid for synthetic financial disclosures', Date: 2026-10-06 -> task2_genai/data_generation/teacher_prompt.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Resumable teacher data generation with per-example validation and teacher rotation on quota exhaustion', Date: 2026-10-06 -> task2_genai/data_generation/generate.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'TF-IDF near-duplicate removal, diversity metrics, chat-template JSONL formatting, stratified 80/10/10 split', Date: 2026-10-06 -> task2_genai/dataset.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'QLoRA NF4 training with assistant-only loss masking, justified hyperparameters, per-epoch loss table, merge_and_unload and Hub push', Date: 2026-10-06 -> task2_genai/finetune.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Base vs fine-tuned evaluation: ROUGE-L, BERTScore, grounding metrics, LLM-as-judge with Pydantic rubric, manual review sheet', Date: 2026-10-06 -> task2_genai/evaluation.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Perplexity-gated ChromaDB retrieval fallback with before/after re-query', Date: 2026-10-06 -> task2_genai/rag_fallback.py
```

### Task 3
```
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Five research tools wrapping Task 1 code that return dicts and never raise', Date: 2026-10-06 -> task3_agentic/tools.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Tracing decorator writing agent_trace.jsonl with per-agent tool permissions and failure injection', Date: 2026-10-06 -> task3_agentic/tracing.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'ReAct loop helpers: OpenAI-compatible tool-calling model (Gemini first) with fallback, tool execution, malformed-call repair, evidence digest', Date: 2026-10-06 -> task3_agentic/agent_core.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'LangGraph single research agent with MemorySaver follow-up memory', Date: 2026-10-06 -> task3_agentic/single_agent.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Two-agent LangGraph pipeline with Pydantic handoffs, one-round critique loop and persistent cache', Date: 2026-10-06 -> task3_agentic/multi_agent.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Ticker/date JSON cache for research briefs', Date: 2026-10-06 -> task3_agentic/memory.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Handoff and report Pydantic schemas; agent prompts', Date: 2026-10-06 -> task3_agentic/schemas.py, task3_agentic/prompts.py
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Streamlit dashboard for agent_trace.jsonl', Date: 2026-10-06 -> task3_agentic/dashboard.py
```

### Tests, notebooks, docs
```
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Offline unit tests with fake LLM clients and scripted agent models', Date: 2026-10-06 -> tests/
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'Thin Colab notebooks calling the task modules', Date: 2026-10-06 -> */*.ipynb
# AI-ASSISTED: Claude (claude-opus-5-5) via Claude Code, Prompt: 'README drafts', Date: 2026-10-06 -> README.md, */README.md
```

## Teacher-model data generation
The full teacher system prompt is in [task2_genai/README.md](task2_genai/README.md#appendix-a--teacher-system-prompt-verbatim)
and is printed in `Task2_Data.ipynb`. Teacher: Google `gemini-3.1-flash-lite`.
LLM judge: Google `gemini-3.1-flash-lite` (`JUDGE_MODEL`).

## External references
```
# SOURCE: StockCharts ChartSchool, "Relative Strength Index (RSI)" worked example: closing prices and expected RSI values -> tests/test_indicators.py
# SOURCE: J. Welles Wilder, "New Concepts in Technical Trading Systems" (1978): RSI smoothing definition -> task1_financial/indicators.py
# SOURCE: John Bollinger, "Bollinger on Bollinger Bands" (2001): population standard deviation for bands -> task1_financial/indicators.py
# SOURCE: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023): NF4, double quantisation, lr 2e-4, max_grad_norm 0.3, all-linear-layer adapters -> task2_genai/finetune.py
# SOURCE: Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (2020) via the bert-score package -> task2_genai/evaluation.py
# SOURCE: LangGraph documentation, StateGraph / add_messages / MemorySaver patterns -> task3_agentic/single_agent.py, multi_agent.py
```

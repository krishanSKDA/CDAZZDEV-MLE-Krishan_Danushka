# CDAZZDEV: Senior ML Engineer Assessment

All three tasks are implemented: **Financial AI**, **Generative AI fine-tuning**, and **Agentic workflows**.

| Task | Folder | Notebook(s) | Key deliverable |
|---|---|---|---|
| 1: Equity research assistant | [task1_financial](task1_financial/) | [Task1_Equity_Research.ipynb](task1_financial/Task1_Equity_Research.ipynb) | Indicators from first principles, validated LLM sentiment + Buy/Hold/Sell, HTML brief |
| 2: Domain fine-tuning | [task2_genai](task2_genai/) | [Data](task2_genai/Task2_Data.ipynb) · [Fine-tune](task2_genai/Task2_FineTune.ipynb) · [Eval](task2_genai/Task2_Eval.ipynb) | QLoRA Qwen2.5-3B risk-factor extractor, base vs fine-tuned evaluation |
| 3: Multi-agent research | [task3_agentic](task3_agentic/) | [Task3_Agentic.ipynb](task3_agentic/Task3_Agentic.ipynb) | LangGraph agents, typed handoffs, critique loop, memory, `agent_trace.jsonl` |

**Links**
- Fine-tuned model (Hugging Face): `https://huggingface.co/Krishan-1890/qwen2.5-3b-risk-extractor`
- Video walkthrough (YouTube): https://youtu.be/cptu0oQJnA0
- [CITATIONS.md](CITATIONS.md) · [REFLECTION.md](REFLECTION.md)

## Repository layout

```
common/            config (secrets from env / Colab), LLM client (Gemini, optional OpenRouter fallback) with retry, JSON logging
task1_financial/   indicators, data pipeline, prompts, schemas, LLM reasoning, report renderer
task2_genai/       domain spec, teacher data generation, dataset tooling, QLoRA training, evaluation, RAG fallback
task3_agentic/     tools, tracing, single agent, multi-agent pipeline, memory, Streamlit dashboard, logs/
tests/             33 offline unit tests (indicators, LLM validation paths, agents with scripted models, dataset)
```

## Running

**Secrets.** Copy `.env.example` to `.env` (local) or add the same names to Colab Secrets: `GEMINI_API_KEY` (primary
LLM, model `gemini-3.1-flash-lite`), `OPENROUTER_API_KEY` (optional fallback), `HF_TOKEN` (Task 2
push), `WANDB_API_KEY` (optional). No credentials are committed; `common/config.py` is the only place secrets are read.

**LLM choice.** All providers go through OpenAI-compatible endpoints, so the same client and agent code works for each.
The default is `gemini-3.1-flash-lite`, the cheapest Gemini text model with a free tier that new API keys can use
(`gemini-2.5-flash-lite` is cheaper but limited to existing users). Change the model with `GEMINI_MODEL` and the
fallback order with `LLM_PROVIDER_ORDER`.

**Colab.** Open any notebook via its badge; the first cell clones this repository and installs requirements.
Task 2 fine-tuning and evaluation need a **T4 GPU** runtime.

**Local.**
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -r requirements.txt
pytest -q                                            # offline tests, no API key needed
python -m task1_financial.data_pipeline NVDA         # Task 1A from the command line
streamlit run task3_agentic/dashboard.py             # Task 3 trace dashboard (screenshot in task3_agentic/README.md)
```

## Design highlights
- **No silent failures:** every LLM response is Pydantic-validated, failures are logged, repaired once, then degrade to a
  deterministic fallback (rule-based signal, tool-built data brief) instead of crashing.
- **Indicator correctness is tested**, including against the StockCharts Wilder RSI reference table.
- **Task 2 hallucination is machine-checkable:** evidence quotes must be verbatim spans, so grounding is measured
  automatically and then confirmed by manual review.
- **Task 3 tool restrictions are enforced twice:** model tool binding and an execution-time permission check that logs
  `blocked` events.

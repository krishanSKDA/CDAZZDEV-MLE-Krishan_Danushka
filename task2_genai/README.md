# Task 2 — Domain-Specific Fine-Tuning: Financial Risk-Factor Extraction

| Notebook | Runtime | Colab |
|---|---|---|
| Task2_Data.ipynb — use case, generation, diversity, split | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krishanSKDA/CDAZZDEV-MLE-Krishan_Danushka/blob/main/task2_genai/Task2_Data.ipynb) |
| Task2_FineTune.ipynb — QLoRA training, merge, push | T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krishanSKDA/CDAZZDEV-MLE-Krishan_Danushka/blob/main/task2_genai/Task2_FineTune.ipynb) |
| Task2_Eval.ipynb — base vs fine-tuned, judge, manual review, RAG bonus | T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/krishanSKDA/CDAZZDEV-MLE-Krishan_Danushka/blob/main/task2_genai/Task2_Eval.ipynb) |

**Model:** `https://huggingface.co/YOUR_HF_USERNAME/qwen2.5-3b-risk-extractor`

## Problem statement
- **Input:** a financial-disclosure passage plus its document type.
- **Output:** JSON `{"risks": [{category, severity, affected_metric, evidence_quote}], "overall_risk_level"}`, with a
  12-category taxonomy, at most one entry per category, and verbatim evidence quotes.
- **Correct:** valid schema, same category set as the reference, every quote verbatim in the passage, severities within
  one level.
- **Partially correct:** grounded but missing or extra categories, a severity off by two levels, or JSON that does not parse.
- **Hallucinated:** any evidence quote that does not appear in the passage.

## Models
- **Teachers:** Google `gemini-3.1-flash-lite`, the cheapest free-tier Gemini text model open to new API keys. Generation
  is resumable across daily free-tier caps (override with `TEACHER_MODELS`). A small
  teacher produces more invalid examples, so per-example validation filters them, and the acceptance rate is reported.
- **LLM judge:** `gemini-3.1-flash-lite` (`JUDGE_MODEL`). Judge and teacher share a model, so the judge is backed by
  the reference extraction and by the automatic grounding metrics, which do not depend on any LLM.
- **Student:** `Qwen/Qwen2.5-3B-Instruct`, a different model family, so teacher ≠ student.

## Files
| File | Purpose |
|---|---|
| `domain.py` | Taxonomy, output schema, student system prompt, grounding check, correct/partial/hallucinated labeller |
| `data_generation/teacher_prompt.py` | Teacher prompt and seed grid (sector, document type, style, length, risks, distractor) |
| `data_generation/generate.py` | Resumable generation with per-example validation; `raw.jsonl` / `rejected.jsonl` |
| `dataset.py` | TF-IDF near-duplicate removal, diversity metrics, Qwen chat-template JSONL, stratified 80/10/10 split |
| `finetune.py` | Hyperparameters + written justifications, assistant-only loss masking, QLoRA trainer, epoch loss table, merge, push |
| `evaluation.py` | Greedy generation with perplexity, ROUGE-L, BERTScore, structural/grounding metrics, LLM judge, manual review |
| `rag_fallback.py` | Bonus: perplexity-gated ChromaDB retrieval and re-query |

## Appendix A — teacher system prompt (verbatim)
```
You generate training data for a model that extracts MATERIAL risk factors from corporate financial disclosures.

Given a JSON specification, write ONE realistic disclosure passage and its gold-standard extraction.

PASSAGE REQUIREMENTS
- Use the fictional company name given. Never mention real companies.
- Match the sector, document type and writing style requested; the word count must fall inside the requested range.
- Embed a material risk for EXACTLY the requested categories (zero categories = a passage with no material risk, e.g. results commentary or mitigated issues). Express each risk naturally; never write the category label itself.
- If "distractor" is true, add one sentence about an issue that is explicitly resolved, immaterial or fully mitigated. It must NOT appear in the extraction.
- Use concrete details where natural: percentages, amounts, dates, regions, counterparties.

EXTRACTION REQUIREMENTS
- One entry per requested category, in order of first appearance in the passage.
- evidence_quote: a contiguous span copied EXACTLY, character for character, from the passage (8-40 words) that establishes the risk.
- severity: "high" = already occurring or could materially impair revenue, margins or liquidity within a year; "medium" = plausible material impact; "low" = remote, small or partly mitigated.
- affected_metric: the single financial metric most exposed, e.g. revenue, gross margin, operating margin, operating cash flow, liquidity, net income, market share, capital expenditure.
- overall_risk_level: the highest severity among risks, or "none" if there are no risks.

CATEGORY DEFINITIONS
- supply_chain: supplier concentration, component shortages, logistics disruption, manufacturing capacity
- regulatory_legal: regulation, litigation, export controls, antitrust, compliance failures
- competition: competitive or pricing pressure, market-share loss, new entrants
- customer_concentration: dependence on a few customers, distributors or channels
- macroeconomic: demand cyclicality, recession, inflation, consumer or enterprise spending
- fx_interest_rate: currency translation, interest-rate exposure, hedging costs
- liquidity_debt: leverage, covenants, refinancing, cash burn, going-concern doubt
- cybersecurity_operational: cyber incidents, IT outages, operational failures, recalls
- geopolitical: trade tensions, tariffs, sanctions, conflict, country risk
- technology_ip: technology obsolescence, IP disputes, R&D or product-transition execution
- esg_climate: physical climate risk, transition or environmental regulation, ESG controversies
- key_personnel: dependence on executives or scarce talent, labour relations

OUTPUT
Return a single JSON object and nothing else:
{"passage": "<text>", "extraction": {"risks": [{"category": "<category>", "severity": "low|medium|high", "affected_metric": "<metric>", "evidence_quote": "<exact span>"}], "overall_risk_level": "none|low|medium|high"}}
```

Teacher user message template: `Specification:\n{spec}` (the specification is a JSON object sampled from the seed grid).

## Appendix B — student system prompt (used for training and for the base-model baseline)
```
You are a financial risk analyst. Extract the MATERIAL risk factors from the disclosure excerpt.
Categories: supply_chain (supplier concentration, component shortages, logistics disruption, manufacturing capacity); regulatory_legal (regulation, litigation, export controls, antitrust, compliance failures); competition (competitive or pricing pressure, market-share loss, new entrants); customer_concentration (dependence on a few customers, distributors or channels); macroeconomic (demand cyclicality, recession, inflation, consumer or enterprise spending); fx_interest_rate (currency translation, interest-rate exposure, hedging costs); liquidity_debt (leverage, covenants, refinancing, cash burn, going-concern doubt); cybersecurity_operational (cyber incidents, IT outages, operational failures, recalls); geopolitical (trade tensions, tariffs, sanctions, conflict, country risk); technology_ip (technology obsolescence, IP disputes, R&D or product-transition execution); esg_climate (physical climate risk, transition or environmental regulation, ESG controversies); key_personnel (dependence on executives or scarce talent, labour relations).
Rules: at most one entry per category; evidence_quote must be copied verbatim from the excerpt; ignore issues described as resolved, immaterial or fully mitigated; severity is low, medium or high; affected_metric is the financial metric most exposed; overall_risk_level is the highest severity, or none if there are no material risks.
Return only JSON: {"risks": [{"category": ..., "severity": ..., "affected_metric": ..., "evidence_quote": ...}], "overall_risk_level": ...}
```

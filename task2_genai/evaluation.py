import json
import math
import os
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from common.llm_client import LLMClient
from task2_genai.domain import RISK_CATEGORIES, SEVERITY_RANK, is_grounded, label_response, parse_extraction

GEN_MAX_NEW_TOKENS = 512
GEN_BATCH_SIZE = 4
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini:gemini-3.1-flash-lite")
PASSAGE_MARKER = "Excerpt:\n"


def passage_of(record: dict) -> str:
    user = next(m["content"] for m in record["messages"] if m["role"] == "user")
    return user.split(PASSAGE_MARKER, 1)[-1]


def gold_of(record: dict) -> dict:
    return json.loads(next(m["content"] for m in record["messages"] if m["role"] == "assistant"))


def prompt_messages(record: dict) -> list[dict]:
    return [m for m in record["messages"] if m["role"] != "assistant"]


def load_for_inference(model_id: str, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": {"": 0}, "dtype": torch.float16}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return model, tok


def generate(model, tokenizer, messages_list: list[list[dict]], max_new_tokens: int = GEN_MAX_NEW_TOKENS,
             batch_size: int = GEN_BATCH_SIZE) -> list[dict]:
    """Greedy decoding; also returns per-sequence perplexity of the generated tokens (used as a confidence signal)."""
    import torch

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    outputs = []
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, return_dict_in_generate=True, output_scores=True,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        logprobs = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        for i in range(gen.shape[0]):
            ids = gen[i].tolist()
            end = next((j for j, t in enumerate(ids) if t in eos_ids), len(ids) - 1)
            lp = logprobs[i, :end + 1].float()
            mean_lp = float(lp.mean()) if lp.numel() else float("nan")
            outputs.append({"text": tokenizer.decode(ids[:end + 1], skip_special_tokens=True).strip(),
                            "n_tokens": end + 1, "mean_logprob": mean_lp, "perplexity": math.exp(-mean_lp)})
    return outputs


def rouge_l(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def bertscore_f1(preds: list[str], refs: list[str]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(preds, refs, lang="en", batch_size=16, verbose=False)
    return f1.tolist()


def structural_scores(record: dict, pred_text: str) -> dict:
    passage, gold = passage_of(record), gold_of(record)
    pred = parse_extraction(pred_text)
    gold_cats = {r["category"]: r["severity"] for r in gold["risks"]}
    if pred is None:
        return {"json_valid": 0, "tp": 0, "fp": 0, "fn": len(gold_cats), "quotes": 0, "grounded": 0,
                "exact_set": 0, "severity_match": 0, "severity_pairs": 0}
    pred_cats = {r.category: r.severity for r in pred.risks}
    matched = set(gold_cats) & set(pred_cats)
    return {
        "json_valid": 1,
        "tp": len(matched), "fp": len(set(pred_cats) - set(gold_cats)), "fn": len(set(gold_cats) - set(pred_cats)),
        "quotes": len(pred.risks), "grounded": sum(is_grounded(r.evidence_quote, passage) for r in pred.risks),
        "exact_set": int(set(gold_cats) == set(pred_cats)),
        "severity_match": sum(SEVERITY_RANK[gold_cats[c]] == SEVERITY_RANK[pred_cats[c]] for c in matched),
        "severity_pairs": len(matched),
    }


def aggregate_structural(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "json_valid_rate": df["json_valid"].mean(),
        "category_precision": precision,
        "category_recall": recall,
        "category_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_category_set": df["exact_set"].mean(),
        "quote_grounding_rate": df["grounded"].sum() / df["quotes"].sum() if df["quotes"].sum() else float("nan"),
        "severity_accuracy": df["severity_match"].sum() / df["severity_pairs"].sum() if df["severity_pairs"].sum() else float("nan"),
    }


class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="every risk and quote is supported by the passage")
    completeness: int = Field(ge=1, le=5, description="all material risks in the reference are captured")
    severity_calibration: int = Field(ge=1, le=5)
    format_compliance: int = Field(ge=1, le=5, description="valid JSON, allowed categories, one per category")
    verdict: Literal["correct", "partially_correct", "hallucinated"]
    rationale: str = Field(max_length=600)


JUDGE_SYSTEM = f"""You are a strict reviewer grading a model's risk-factor extraction from a financial disclosure.
Allowed categories: {", ".join(RISK_CATEGORIES)}.
Score each criterion 1-5 (5 = perfect):
- faithfulness: every extracted risk is genuinely stated in the passage and each evidence_quote appears verbatim. Any fabricated risk or quote caps this at 2.
- completeness: every material risk in the reference extraction is captured (category-level).
- severity_calibration: severities are reasonable for the language used.
- format_compliance: valid JSON in the required schema, allowed categories, at most one entry per category.
verdict: "hallucinated" if any risk or quote is not supported by the passage; "correct" if faithful, complete and calibrated; otherwise "partially_correct".
Return only JSON: {{"faithfulness": int, "completeness": int, "severity_calibration": int, "format_compliance": int, "verdict": str, "rationale": str}}"""

JUDGE_USER = """PASSAGE:
{passage}

REFERENCE EXTRACTION:
{gold}

CANDIDATE EXTRACTION:
{candidate}"""


def judge_one(client: LLMClient, record: dict, pred_text: str) -> JudgeScore | None:
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(passage=passage_of(record), gold=json.dumps(gold_of(record)),
                                                              candidate=pred_text[:3000])}]
    return client.structured(messages, JudgeScore)


def judge_all(records: list[dict], preds: list[str]) -> pd.DataFrame:
    client = LLMClient(temperature=0.0, model_spec=JUDGE_MODEL)
    rows = []
    for rec, pred in zip(records, preds):
        score = judge_one(client, rec, pred)
        rows.append({"id": rec["id"], **(score.model_dump() if score else {"verdict": "judge_failed"})})
    return pd.DataFrame(rows)


def evaluate_model(name: str, records: list[dict], generations: list[dict]) -> tuple[dict, pd.DataFrame]:
    preds = [g["text"] for g in generations]
    refs = [json.dumps(gold_of(r), ensure_ascii=False) for r in records]
    per_sample = pd.DataFrame([{"id": r["id"], **structural_scores(r, p)} for r, p in zip(records, preds)])
    per_sample["rougeL"] = rouge_l(preds, refs)
    per_sample["perplexity"] = [g["perplexity"] for g in generations]
    per_sample["auto_label"], per_sample["auto_reason"] = zip(*[label_response(passage_of(r), gold_of(r), p)
                                                               for r, p in zip(records, preds)])
    summary = {"model": name, "rougeL_f1": per_sample["rougeL"].mean(),
               **aggregate_structural(per_sample.to_dict("records")),
               "auto_hallucination_rate": (per_sample["auto_label"] == "hallucinated").mean()}
    return summary, per_sample


def review_sheet(records: list[dict], preds: list[str], per_sample: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for rec, pred, (_, row) in list(zip(records, preds, per_sample.iterrows()))[:n]:
        rows.append({"id": rec["id"], "passage": passage_of(rec), "gold": json.dumps(gold_of(rec), indent=1),
                     "prediction": pred, "auto_label": row["auto_label"], "auto_reason": row["auto_reason"]})
    return pd.DataFrame(rows)


def hallucination_rate(labels: dict[str, str]) -> dict:
    counts = pd.Series(labels).value_counts().to_dict()
    n = len(labels)
    return {"n_reviewed": n, "counts": counts, "hallucination_rate_pct": round(100 * counts.get("hallucinated", 0) / n, 1) if n else None}

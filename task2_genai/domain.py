"""Use case: extract material risk factors from financial-disclosure passages into validated JSON."""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

RISK_CATEGORIES: dict[str, str] = {
    "supply_chain": "supplier concentration, component shortages, logistics disruption, manufacturing capacity",
    "regulatory_legal": "regulation, litigation, export controls, antitrust, compliance failures",
    "competition": "competitive or pricing pressure, market-share loss, new entrants",
    "customer_concentration": "dependence on a few customers, distributors or channels",
    "macroeconomic": "demand cyclicality, recession, inflation, consumer or enterprise spending",
    "fx_interest_rate": "currency translation, interest-rate exposure, hedging costs",
    "liquidity_debt": "leverage, covenants, refinancing, cash burn, going-concern doubt",
    "cybersecurity_operational": "cyber incidents, IT outages, operational failures, recalls",
    "geopolitical": "trade tensions, tariffs, sanctions, conflict, country risk",
    "technology_ip": "technology obsolescence, IP disputes, R&D or product-transition execution",
    "esg_climate": "physical climate risk, transition or environmental regulation, ESG controversies",
    "key_personnel": "dependence on executives or scarce talent, labour relations",
}
Category = Literal[tuple(RISK_CATEGORIES)]  # type: ignore[valid-type]
Severity = Literal["low", "medium", "high"]
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
MAX_RISKS = 5

STUDENT_SYSTEM_PROMPT = (
    "You are a financial risk analyst. Extract the MATERIAL risk factors from the disclosure excerpt.\n"
    "Categories: " + "; ".join(f"{k} ({v})" for k, v in RISK_CATEGORIES.items()) + ".\n"
    "Rules: at most one entry per category; evidence_quote must be copied verbatim from the excerpt; ignore issues "
    "described as resolved, immaterial or fully mitigated; severity is low, medium or high; affected_metric is the "
    "financial metric most exposed; overall_risk_level is the highest severity, or none if there are no material risks.\n"
    'Return only JSON: {"risks": [{"category": ..., "severity": ..., "affected_metric": ..., "evidence_quote": ...}], '
    '"overall_risk_level": ...}'
)

USER_TEMPLATE = "Document type: {doc_type}\n\nExcerpt:\n{passage}"

CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


class ExtractedRisk(BaseModel):
    category: Category
    severity: Severity
    affected_metric: str = Field(min_length=2, max_length=60)
    evidence_quote: str = Field(min_length=10)

    @field_validator("category", "severity", mode="before")
    @classmethod
    def _norm(cls, v):
        return v.strip().lower().replace(" ", "_") if isinstance(v, str) else v


class RiskExtraction(BaseModel):
    risks: list[ExtractedRisk] = Field(max_length=MAX_RISKS)
    overall_risk_level: Literal["none", "low", "medium", "high"]

    @field_validator("risks")
    @classmethod
    def _unique_categories(cls, v):
        cats = [r.category for r in v]
        if len(cats) != len(set(cats)):
            raise ValueError("duplicate categories")
        return v


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(QUOTE_CHARS)).strip().lower()


def is_grounded(quote: str, passage: str) -> bool:
    return normalize_text(quote) in normalize_text(passage)


def parse_extraction(text: str) -> RiskExtraction | None:
    cleaned = CODE_FENCE.sub("", text or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return RiskExtraction.model_validate_json(cleaned[start:end + 1])
    except (ValidationError, json.JSONDecodeError, ValueError):
        return None


def build_messages(passage: str, doc_type: str, extraction: dict | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(doc_type=doc_type, passage=passage)},
    ]
    if extraction is not None:
        messages.append({"role": "assistant", "content": json.dumps(extraction, ensure_ascii=False)})
    return messages


def label_response(passage: str, gold: dict, prediction_text: str) -> tuple[str, str]:
    """Success criteria. correct: valid JSON, same category set, all quotes grounded, severities within one level.
    hallucinated: any quote not in the passage, or a category absent from gold whose quote is ungrounded.
    partially_correct: everything else (grounded but missed/extra categories, severity off, or invalid JSON)."""
    pred = parse_extraction(prediction_text)
    if pred is None:
        return "partially_correct", "invalid or non-schema JSON"
    ungrounded = [r.category for r in pred.risks if not is_grounded(r.evidence_quote, passage)]
    if ungrounded:
        return "hallucinated", f"quote not found in passage for: {ungrounded}"
    gold_sev = {r["category"]: r["severity"] for r in gold["risks"]}
    pred_sev = {r.category: r.severity for r in pred.risks}
    if set(gold_sev) != set(pred_sev):
        missing, extra = set(gold_sev) - set(pred_sev), set(pred_sev) - set(gold_sev)
        return "partially_correct", f"missing={sorted(missing)} extra={sorted(extra)}"
    off = [c for c in gold_sev if abs(SEVERITY_RANK[gold_sev[c]] - SEVERITY_RANK[pred_sev[c]]) > 1]
    if off:
        return "partially_correct", f"severity off by >1 level for {off}"
    return "correct", "categories match, quotes grounded, severities calibrated"

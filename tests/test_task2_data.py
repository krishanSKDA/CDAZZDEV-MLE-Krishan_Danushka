from collections import Counter

from task2_genai.data_generation.generate import TeacherOutput, make_specs, validate
from task2_genai.dataset import deduplicate, diversity_report, split_records, to_chat_record
from task2_genai.domain import RISK_CATEGORIES, RiskExtraction, parse_extraction

PASSAGE = ("Kestrel Systems sources 70% of its wafers from a single foundry in Taiwan, and any disruption there could "
           "halt shipments. A pending antitrust review in the EU may delay our acquisition of Lumen Partners. ")


def spec(cats, words=(20, 60)):
    return {"id": "x", "risk_categories": cats, "word_range": list(words)}


def output(risks, level, passage=PASSAGE):
    return TeacherOutput(passage=passage, extraction=RiskExtraction(risks=risks, overall_risk_level=level))


SUPPLY = {"category": "supply_chain", "severity": "high", "affected_metric": "revenue",
          "evidence_quote": "sources 70% of its wafers from a single foundry in Taiwan"}
REG = {"category": "regulatory_legal", "severity": "medium", "affected_metric": "revenue growth",
       "evidence_quote": "A pending antitrust review in the EU may delay our acquisition"}


def test_specs_are_balanced_and_reproducible():
    specs = make_specs(300)
    assert specs == make_specs(300)
    counts = Counter(c for s in specs for c in s["risk_categories"])
    assert set(counts) == set(RISK_CATEGORIES)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(len(set(s["risk_categories"])) == len(s["risk_categories"]) for s in specs)


def test_validate_accepts_good_example():
    assert validate(spec(["supply_chain", "regulatory_legal"]), output([SUPPLY, REG], "high")) is None


def test_validate_rejects_bad_examples():
    assert "categories" in validate(spec(["supply_chain"]), output([SUPPLY, REG], "high"))
    fake = {**SUPPLY, "evidence_quote": "all wafers come from three diversified suppliers"}
    assert "verbatim" in validate(spec(["supply_chain"]), output([fake], "high"))
    assert "overall_risk_level" in validate(spec(["supply_chain"]), output([SUPPLY], "medium"))
    assert "length" in validate(spec(["supply_chain"], (200, 300)), output([SUPPLY], "high"))


def test_parse_extraction_handles_fences_and_rejects_bad_category():
    assert parse_extraction('```json\n{"risks": [], "overall_risk_level": "none"}\n```') is not None
    assert parse_extraction('{"risks": [{"category": "weather", "severity": "low", "affected_metric": "x", '
                            '"evidence_quote": "0123456789"}], "overall_risk_level": "low"}') is None


def _records(n):
    specs = make_specs(n)
    words = ["revenue", "margin", "supplier", "tariff", "covenant", "cyber", "drought", "lawsuit", "churn", "inflation"]
    return [{**s, "passage": " ".join(words[(i + j) % 10] + str(i * 7 + j) for j in range(40)),
             "extraction": {"risks": [], "overall_risk_level": "none"}} for i, s in enumerate(specs)]


def test_dedup_split_and_diversity():
    recs = _records(100)
    dup = {**recs[0], "id": "dup"}
    clean, dropped = deduplicate(recs + [dup])
    assert len(clean) == 100 and dropped[0][:2] == ("dup", recs[0]["id"])
    splits = split_records(clean)
    assert {k: len(v) for k, v in splits.items()} == {"train": 80, "val": 10, "test": 10}
    assert not set(r["id"] for r in splits["train"]) & set(r["id"] for r in splits["test"])
    report = diversity_report(clean)
    assert report["n_examples"] == 100 and len(report["top_keywords"]) > 0


def test_chat_record_roles():
    rec = to_chat_record(_records(1)[0])
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]

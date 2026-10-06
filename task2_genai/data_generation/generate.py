"""Resumable teacher-model data generation.

python -m task2_genai.data_generation.generate --target 320
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from common.llm_client import LLMClient, LLMError
from common.logging_utils import get_logger
from task2_genai.data_generation.teacher_prompt import (
    DISTRACTOR_PROB, DOC_TYPES, LENGTH_BUCKETS, N_RISKS_WEIGHTS, NAME_PREFIXES, NAME_SUFFIXES, SECTORS, STYLES,
    TEACHER_SYSTEM_PROMPT, TEACHER_USER_TEMPLATE,
)
from task2_genai.domain import RISK_CATEGORIES, RiskExtraction, is_grounded

log = get_logger("task2.generate")

DATA_DIR = Path(__file__).parent
SPECS_PATH = DATA_DIR / "specs.jsonl"
RAW_PATH = DATA_DIR / "raw.jsonl"
REJECTED_PATH = DATA_DIR / "rejected.jsonl"
SEED = 42
DEFAULT_TARGET = 320
TEACHER_TEMPERATURE = 0.9
LENGTH_TOLERANCE = 0.25
# 'provider:model' specs, rotated when one hits its free-tier cap; Gemini teacher != Qwen student family
DEFAULT_TEACHERS = "gemini:gemini-3.1-flash-lite"


class TeacherOutput(BaseModel):
    passage: str
    extraction: RiskExtraction


def make_specs(n: int, seed: int = SEED) -> list[dict]:
    """Balanced seed grid: least-used categories are drawn first so every category is well represented."""
    rng = random.Random(seed)
    usage = Counter({c: 0 for c in RISK_CATEGORIES})
    specs = []
    for i in range(n):
        k = rng.choices(list(N_RISKS_WEIGHTS), weights=list(N_RISKS_WEIGHTS.values()))[0]
        ranked = sorted(RISK_CATEGORIES, key=lambda c: (usage[c], rng.random()))
        cats = ranked[:k]
        usage.update(cats)
        bucket = rng.choice(list(LENGTH_BUCKETS))
        specs.append({
            "id": f"ex_{i:04d}",
            "company": f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_SUFFIXES)}",
            "sector": rng.choice(SECTORS),
            "doc_type": rng.choice(DOC_TYPES),
            "style": rng.choice(STYLES),
            "length_bucket": bucket,
            "word_range": list(LENGTH_BUCKETS[bucket]),
            "risk_categories": rng.sample(cats, len(cats)),
            "distractor": rng.random() < DISTRACTOR_PROB,
        })
    return specs


def validate(spec: dict, out: TeacherOutput) -> str | None:
    """Returns a rejection reason, or None if the example is usable."""
    lo, hi = spec["word_range"]
    words = len(out.passage.split())
    if not lo * (1 - LENGTH_TOLERANCE) <= words <= hi * (1 + LENGTH_TOLERANCE):
        return f"length {words} outside {lo}-{hi}"
    got = {r.category for r in out.extraction.risks}
    if got != set(spec["risk_categories"]):
        return f"categories {sorted(got)} != requested {sorted(spec['risk_categories'])}"
    if any(not is_grounded(r.evidence_quote, out.passage) for r in out.extraction.risks):
        return "evidence_quote not verbatim in passage"
    expected_level = max((r.severity for r in out.extraction.risks),
                         key=["low", "medium", "high"].index, default="none")
    if out.extraction.overall_risk_level != expected_level:
        return f"overall_risk_level {out.extraction.overall_risk_level} != max severity {expected_level}"
    return None


def _read_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate(target: int, max_specs: int, teachers: list[str]) -> None:
    specs = make_specs(max_specs)
    if not SPECS_PATH.exists():
        SPECS_PATH.write_text("\n".join(json.dumps(s) for s in specs) + "\n", encoding="utf-8")
    done = _read_ids(RAW_PATH) | _read_ids(REJECTED_PATH)
    accepted = len(_read_ids(RAW_PATH))
    clients = []
    for spec_str in teachers:
        try:
            clients.append((spec_str, LLMClient(temperature=TEACHER_TEMPERATURE, model_spec=spec_str)))
        except LLMError as exc:
            log.warning("teacher skipped", extra={"data": {"teacher": spec_str, "reason": str(exc)}})
    if not clients:
        raise LLMError("no teacher model has an API key configured")
    teacher_idx = 0

    for spec in specs:
        if accepted >= target:
            break
        if spec["id"] in done:
            continue
        prompt_spec = {k: v for k, v in spec.items() if k != "id"}
        messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": TEACHER_USER_TEMPLATE.format(spec=json.dumps(prompt_spec, indent=1))}]
        out = None
        while teacher_idx < len(clients):
            model_name, client = clients[teacher_idx]
            try:
                out = client.structured(messages, TeacherOutput, raise_on_provider_error=True)
                break
            except LLMError:
                teacher_idx += 1
            log.warning("teacher exhausted, rotating", extra={"data": {"failed": model_name}})
        else:
            log.error("all teachers exhausted; rerun later to resume", extra={"data": {"accepted": accepted}})
            return

        reason = "teacher output failed schema validation" if out is None else validate(spec, out)
        if reason:
            _append(REJECTED_PATH, {"id": spec["id"], "reason": reason, "teacher": model_name})
            log.info("rejected", extra={"data": {"id": spec["id"], "reason": reason}})
            continue
        _append(RAW_PATH, {**spec, "teacher": model_name, "passage": out.passage,
                           "extraction": out.extraction.model_dump()})
        accepted += 1
        log.info("accepted", extra={"data": {"id": spec["id"], "accepted": accepted, "teacher": model_name}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--max-specs", type=int, default=600)
    parser.add_argument("--teachers", default=os.getenv("TEACHER_MODELS", DEFAULT_TEACHERS))
    args = parser.parse_args()
    generate(args.target, args.max_specs, [t.strip() for t in args.teachers.split(",") if t.strip()])

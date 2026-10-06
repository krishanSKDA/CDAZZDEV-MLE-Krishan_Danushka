"""Bonus: perplexity-gated retrieval fallback over a ChromaDB store of verified training extractions."""
import json

import numpy as np

from task2_genai.domain import RISK_CATEGORIES, parse_extraction
from task2_genai.evaluation import generate, gold_of, passage_of, prompt_messages

COLLECTION = "risk_examples"
THRESHOLD_QUANTILE = 0.8
TOP_K = 2


def build_store(train_records: list[dict], persist_dir: str | None = None):
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        col.add(
            ids=[r["id"] for r in train_records],
            documents=[passage_of(r) for r in train_records],
            metadatas=[{"doc_type": r["meta"]["doc_type"], "gold": json.dumps(gold_of(r))} for r in train_records],
        )
    return col


def calibrate_threshold(val_perplexities: list[float], quantile: float = THRESHOLD_QUANTILE) -> float:
    """Generations less confident than `quantile` of validation generations trigger retrieval."""
    return float(np.quantile(val_perplexities, quantile))


def retrieve(col, passage: str, k: int = TOP_K) -> list[dict]:
    res = col.query(query_texts=[passage], n_results=k)
    return [{"id": i, "passage": d, "gold": m["gold"], "distance": round(dist, 4)}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def augmented_messages(record: dict, examples: list[dict]) -> list[dict]:
    messages = prompt_messages(record)
    refs = "\n\n".join(f"Reference excerpt {i + 1}:\n{e['passage']}\nVerified extraction {i + 1}:\n{e['gold']}"
                       for i, e in enumerate(examples))
    guide = "\n".join(f"- {k}: {v}" for k, v in RISK_CATEGORIES.items())
    user = (f"Category guide:\n{guide}\n\nSimilar verified examples for calibration (do not copy their risks):\n{refs}\n\n"
            f"Now extract risks ONLY from this target.\n{messages[1]['content']}")
    return [messages[0], {"role": "user", "content": user}]


def answer_with_fallback(model, tokenizer, record: dict, col, threshold: float) -> dict:
    first = generate(model, tokenizer, [prompt_messages(record)])[0]
    invalid = parse_extraction(first["text"]) is None
    triggered = invalid or first["perplexity"] > threshold
    result = {"id": record["id"], "first": first, "triggered": triggered,
              "trigger_reason": "invalid JSON" if invalid else (f"perplexity {first['perplexity']:.3f} > {threshold:.3f}" if triggered else None)}
    if triggered:
        examples = [e for e in retrieve(col, passage_of(record), TOP_K + 1) if e["id"] != record["id"]][:TOP_K]
        result["retrieved"] = [{"id": e["id"], "distance": e["distance"]} for e in examples]
        result["second"] = generate(model, tokenizer, [augmented_messages(record, examples)])[0]
    return result

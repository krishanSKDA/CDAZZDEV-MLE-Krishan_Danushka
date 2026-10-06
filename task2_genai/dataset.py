import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from task2_genai.domain import build_messages

DATA_DIR = Path(__file__).parent / "data"
SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
NEAR_DUP_THRESHOLD = 0.85
TOP_KEYWORDS = 25


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tfidf(texts: list[str]):
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(texts)


def deduplicate(records: list[dict], threshold: float = NEAR_DUP_THRESHOLD) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """Drops later records whose TF-IDF cosine similarity to an earlier kept record exceeds `threshold`."""
    texts = [r["passage"] for r in records]
    sims = cosine_similarity(_tfidf(texts).transform(texts))
    kept, dropped = [], []
    for i in range(len(records)):
        dup_of = next((j for j in kept if sims[i, j] >= threshold), None)
        if dup_of is None:
            kept.append(i)
        else:
            dropped.append((records[i]["id"], records[dup_of]["id"], round(float(sims[i, dup_of]), 3)))
    return [records[i] for i in kept], dropped


def distinct_n(texts: list[str], n: int) -> float:
    grams = [tuple(t.lower().split()[i:i + n]) for t in texts for i in range(len(t.split()) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def diversity_report(records: list[dict]) -> dict:
    texts = [r["passage"] for r in records]
    words = np.array([len(t.split()) for t in texts])
    vec = _tfidf(texts)
    matrix = vec.transform(texts)
    sims = cosine_similarity(matrix)
    upper = sims[np.triu_indices_from(sims, k=1)]
    keyword_scores = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    return {
        "n_examples": len(records),
        "passage_words": {"min": int(words.min()), "p25": float(np.percentile(words, 25)), "median": float(np.median(words)),
                          "p75": float(np.percentile(words, 75)), "max": int(words.max()), "mean": float(words.mean())},
        "pairwise_tfidf_cosine": {"mean": float(upper.mean()), "p95": float(np.percentile(upper, 95)), "max": float(upper.max())},
        "distinct_1": round(distinct_n(texts, 1), 4),
        "distinct_2": round(distinct_n(texts, 2), 4),
        "category_counts": dict(Counter(c for r in records for c in r["risk_categories"]).most_common()),
        "n_risks_counts": dict(sorted(Counter(len(r["risk_categories"]) for r in records).items())),
        "sector_counts": dict(Counter(r["sector"] for r in records).most_common()),
        "doc_type_counts": dict(Counter(r["doc_type"] for r in records).most_common()),
        "style_counts": dict(Counter(r["style"] for r in records).most_common()),
        "severity_counts": dict(Counter(x["severity"] for r in records for x in r["extraction"]["risks"]).most_common()),
        "teacher_counts": dict(Counter(r.get("teacher", "unknown") for r in records).most_common()),
        "distractor_share": round(float(np.mean([r["distractor"] for r in records])), 3),
        "top_keywords": [(vocab[i], round(float(keyword_scores[i]), 2)) for i in keyword_scores.argsort()[::-1][:TOP_KEYWORDS]],
    }


def to_chat_record(record: dict, tokenizer=None) -> dict:
    messages = build_messages(record["passage"], record["doc_type"], record["extraction"])
    out = {"id": record["id"], "messages": messages,
           "meta": {k: record[k] for k in ("sector", "doc_type", "style", "length_bucket", "risk_categories", "distractor")}}
    if tokenizer is not None:
        out["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return out


def split_records(records: list[dict], seed: int = SPLIT_SEED) -> dict[str, list[dict]]:
    """80/10/10, stratified by number of risks so every split covers the 0-4 risk range."""
    strata = [min(len(r["risk_categories"]), 3) for r in records]
    train, rest = train_test_split(records, test_size=VAL_FRAC + TEST_FRAC, random_state=seed, stratify=strata)
    rest_strata = [min(len(r["risk_categories"]), 3) for r in rest]
    val, test = train_test_split(rest, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), random_state=seed, stratify=rest_strata)
    return {"train": train, "val": val, "test": test}


def build_dataset(raw_path: Path, tokenizer=None, out_dir: Path = DATA_DIR) -> dict:
    raw = load_jsonl(raw_path)
    clean, dropped = deduplicate(raw)
    splits = split_records(clean)
    for name, recs in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", [to_chat_record(r, tokenizer) for r in recs])
    write_jsonl(out_dir / "clean.jsonl", clean)
    sizes = {k: len(v) for k, v in splits.items()}
    return {"raw": len(raw), "after_dedup": len(clean), "near_duplicates_dropped": dropped, "split_sizes": sizes}


def summary_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "id": r["id"], "words": len(r["passage"].split()), "n_risks": len(r["risk_categories"]),
        "sector": r["sector"], "doc_type": r["doc_type"], "style": r["style"], "teacher": r.get("teacher"),
    } for r in records])

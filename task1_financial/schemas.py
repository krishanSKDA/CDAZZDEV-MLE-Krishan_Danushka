import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "negative", "neutral"]
Signal = Literal["Buy", "Hold", "Sell"]

MIN_JUSTIFICATION_SENTENCES = 3
MAX_JUSTIFICATION_SENTENCES = 5
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def count_sentences(text: str) -> int:
    return len([s for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()])


class NewsItem(BaseModel):
    title: str
    publisher: str | None = None
    link: str | None = None
    published: datetime | None = None
    source: str


class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str = Field(min_length=3, max_length=400)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


class SentimentAggregate(BaseModel):
    score: float = Field(ge=-1.0, le=1.0, description="confidence-weighted mean polarity")
    label: Sentiment
    n_scored: int
    n_failed: int
    counts: dict[str, int]


class TradingSignal(BaseModel):
    signal: Signal
    conviction: Literal["low", "medium", "high"]
    justification: str
    key_drivers: list[str] = Field(min_length=1, max_length=4)
    source: Literal["llm", "rule_fallback"] = "llm"

    @field_validator("signal", mode="before")
    @classmethod
    def _title(cls, v):
        return v.strip().title() if isinstance(v, str) else v

    @field_validator("conviction", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("justification")
    @classmethod
    def _sentence_count(cls, v: str) -> str:
        n = count_sentences(v)
        if not MIN_JUSTIFICATION_SENTENCES <= n <= MAX_JUSTIFICATION_SENTENCES:
            raise ValueError(f"justification must be {MIN_JUSTIFICATION_SENTENCES}-{MAX_JUSTIFICATION_SENTENCES} "
                             f"sentences, got {n}")
        return v

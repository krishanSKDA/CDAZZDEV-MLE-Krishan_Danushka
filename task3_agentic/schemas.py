from typing import Any, Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]


class PriceSnapshot(BaseModel):
    last_close: float | None = None
    period_return: float | None = None
    max_drawdown: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    macd_hist: float | None = None
    momentum_label: str | None = None


class VolatilitySnapshot(BaseModel):
    window_days: int | None = None
    annualized_vol: float | None = None
    vol_1y: float | None = None
    vol_percentile_1y: float | None = None
    expected_1sigma_move_90d_pct: float | None = None


class SentimentSnapshot(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    label: Literal["positive", "negative", "neutral"]
    n_headlines: int


class DataBrief(BaseModel):
    """Agent A -> Agent B handoff."""
    ticker: str
    as_of: str
    price: PriceSnapshot
    volatility: VolatilitySnapshot
    fundamentals: dict[str, float | str | None] = Field(default_factory=dict)
    sentiment: SentimentSnapshot | None = None
    key_findings: list[str] = Field(min_length=2, max_length=6)
    data_gaps: list[str] = Field(default_factory=list)


class ClarificationRequest(BaseModel):
    """Agent B -> Agent A critique."""
    question: str
    requested_metrics: list[str] = Field(min_length=1, max_length=4)
    reason: str
    attach_headlines: bool = False
    headlines: list[str] = Field(default_factory=list)


class ClarificationResponse(BaseModel):
    """Agent A -> Agent B answer to the critique."""
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    tools_used: list[str] = Field(default_factory=list)


class Risk(BaseModel):
    title: str
    description: str
    evidence: list[str] = Field(min_length=1)
    likelihood: Level
    impact: Level


class HedgeStrategy(BaseModel):
    strategy: str
    rationale: str
    implementation: str
    data_basis: list[str] = Field(min_length=1)


class ResearchReport(BaseModel):
    ticker: str
    financial_health_summary: str
    top_risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy: HedgeStrategy
    sources: list[str] = Field(default_factory=list)

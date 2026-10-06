import base64
import io
from datetime import datetime, timezone
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Template

from task1_financial.data_pipeline import (
    RSI_COL, RSI_OVERBOUGHT, RSI_OVERSOLD, SMA_LONG_COL, SMA_SHORT_COL, PipelineResult,
)
from task1_financial.indicators import RSI_PERIOD
from task1_financial.llm_reasoning import AnalysisResult

OUTPUT_DIR = Path(__file__).parent / "outputs"
CHART_LOOKBACK_DAYS = 252
TOP_HEADLINES = 3
CHART_DPI = 130

MARKDOWN_TEMPLATE = Template("""# {{ s.company_name }} ({{ s.ticker }}): Equity Research Brief
*As of {{ s.as_of }} · Generated {{ generated }}*

## Company Snapshot
| Metric | Value |
|---|---|
| Price | {{ s.current_price }} {{ s.currency or "" }} |
| 52-week range | {{ s.low_52w }} to {{ s.high_52w }} |
| YTD return | {{ pct(s.ytd_return) }} |
| P/E ({{ s.pe_type or "n/a" }}) | {{ s.pe_ratio if s.pe_ratio is not none else "n/a" }} |
| Sector | {{ s.sector or "n/a" }} |

## Technical Outlook
Momentum score **{{ s.momentum_signal.score }}** ({{ s.momentum_signal.label }}).

{% for f in a.derived_facts %}- {{ f }}
{% endfor %}
{{ chart }}

## News Sentiment
Aggregate score **{{ "%+.2f"|format(a.sentiment.score) }}** ({{ a.sentiment.label }}) from {{ a.sentiment.n_scored }} headlines:
{{ a.sentiment.counts.positive }} positive, {{ a.sentiment.counts.neutral }} neutral, {{ a.sentiment.counts.negative }} negative.

| Headline | Sentiment | Confidence | Reason |
|---|---|---|---|
{% for h in top %}| {{ h.headline }} | {{ h.sentiment }} | {{ "%.2f"|format(h.confidence) }} | {{ h.brief_reason }} |
{% endfor %}

## Recommendation: {{ a.signal.signal }} ({{ a.signal.conviction }} conviction)
{{ a.signal.justification }}

**Key drivers:** {{ a.signal.key_drivers|join(", ") }}{% if a.signal.source == "rule_fallback" %} *(rule-based fallback)*{% endif %}

---
**Risk disclaimer.** This brief is generated automatically by a language model from public market data and news
headlines. It is for educational purposes only, is not investment advice, and has not been reviewed by a licensed
analyst. Technical indicators are backward-looking, LLM outputs can be wrong, and past performance does not guarantee
future results. Do your own research and consult a qualified financial adviser before making investment decisions.
""")

HTML_TEMPLATE = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ ticker }} Research Brief</title>
<style>
:root{--ink:#1b2430;--muted:#5b6675;--line:#dfe3ea;--accent:#1f4e79;--bg:#ffffff;--panel:#f5f7fa}
body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);max-width:900px;margin:32px auto;padding:0 16px;line-height:1.5}
h1{color:var(--accent);border-bottom:3px solid var(--accent);padding-bottom:6px;font-size:1.6rem}
h2{color:var(--accent);margin-top:1.6em;font-size:1.15rem;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:.6em 0}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--panel)}
img{max-width:100%;border:1px solid var(--line);border-radius:6px}
hr{border:none;border-top:1px solid var(--line);margin:2em 0 1em}
em{color:var(--muted)}
</style></head><body>{{ body }}</body></html>""")


def render_chart(pipeline: PipelineResult) -> str:
    df = pipeline.prices.iloc[-CHART_LOOKBACK_DAYS:]
    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(
        3, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    ax_price.fill_between(df.index, df["bb_lower"], df["bb_upper"], color="#9db7d5", alpha=0.25, label="Bollinger (20, 2σ)")
    ax_price.plot(df.index, df["Close"], color="#1b2430", lw=1.4, label="Close")
    ax_price.plot(df.index, df[SMA_SHORT_COL], color="#d9822b", lw=1.1, label="SMA 50")
    ax_price.plot(df.index, df[SMA_LONG_COL], color="#1f4e79", lw=1.1, label="SMA 200")
    ax_price.set_title(f"{pipeline.ticker}: price, trend and volatility bands (1 year)", loc="left", fontsize=11)
    ax_price.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)

    ax_rsi.plot(df.index, df[RSI_COL], color="#6a4c93", lw=1.1)
    ax_rsi.axhline(RSI_OVERBOUGHT, color="#b23a48", ls="--", lw=0.8)
    ax_rsi.axhline(RSI_OVERSOLD, color="#2e7d32", ls="--", lw=0.8)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel(f"RSI {RSI_PERIOD}", fontsize=9)

    colors = ["#2e7d32" if v >= 0 else "#b23a48" for v in df["macd_hist"].fillna(0)]
    ax_macd.bar(df.index, df["macd_hist"], color=colors, width=1.0, alpha=0.6)
    ax_macd.plot(df.index, df["macd"], color="#1b2430", lw=1.0, label="MACD")
    ax_macd.plot(df.index, df["macd_signal"], color="#d9822b", lw=1.0, label="Signal")
    ax_macd.set_ylabel("MACD", fontsize=9)
    ax_macd.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    for ax in (ax_price, ax_rsi, ax_macd):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_markdown(pipeline: PipelineResult, analysis: AnalysisResult, chart_ref: str) -> str:
    top = sorted(analysis.headline_sentiments, key=lambda h: h.confidence, reverse=True)[:TOP_HEADLINES]
    return MARKDOWN_TEMPLATE.render(
        s=pipeline.summary, a=analysis, top=top, chart=f"![Technical chart]({chart_ref})",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        pct=lambda v: "n/a" if v is None else f"{v:+.1%}",
    )


def write_report(pipeline: PipelineResult, analysis: AnalysisResult, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_b64 = render_chart(pipeline)
    stem = f"{pipeline.ticker}_{pipeline.summary['as_of']}"

    chart_path = output_dir / f"{stem}_chart.png"
    chart_path.write_bytes(base64.b64decode(chart_b64))

    md_path = output_dir / f"{stem}_brief.md"
    md_path.write_text(build_markdown(pipeline, analysis, chart_path.name), encoding="utf-8")

    html_body = markdown.markdown(build_markdown(pipeline, analysis, f"data:image/png;base64,{chart_b64}"),
                                  extensions=["tables"])
    html_path = output_dir / f"{stem}_brief.html"
    html_path.write_text(HTML_TEMPLATE.render(ticker=pipeline.ticker, body=html_body), encoding="utf-8")
    return {"markdown": md_path, "html": html_path, "chart": chart_path}

from task2_genai.domain import RISK_CATEGORIES

TEACHER_SYSTEM_PROMPT = """You generate training data for a model that extracts MATERIAL risk factors from corporate financial disclosures.

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
{categories}

OUTPUT
Return a single JSON object and nothing else:
{{"passage": "<text>", "extraction": {{"risks": [{{"category": "<category>", "severity": "low|medium|high", "affected_metric": "<metric>", "evidence_quote": "<exact span>"}}], "overall_risk_level": "none|low|medium|high"}}}}""".format(
    categories="\n".join(f"- {k}: {v}" for k, v in RISK_CATEGORIES.items())
)

TEACHER_USER_TEMPLATE = "Specification:\n{spec}"

SECTORS = [
    "semiconductors", "regional banking", "clinical-stage biotech", "apparel retail", "passenger airline",
    "oil and gas exploration", "enterprise SaaS", "automotive manufacturing", "office REIT", "consumer packaged foods",
    "medical devices", "wireless telecom", "copper mining", "property and casualty insurance", "container shipping",
]
DOC_TYPES = [
    "10-K Item 1A risk factors", "10-Q MD&A excerpt", "earnings call prepared remarks", "earnings call analyst Q&A answer",
    "8-K material event disclosure", "bond prospectus risk section", "annual report CEO letter",
    "investor day transcript", "S-1 IPO prospectus risk factors", "sustainability report",
]
STYLES = ["dense legal boilerplate", "plain-English executive commentary", "hedged forward-looking language",
          "numbers-heavy analytical prose"]
LENGTH_BUCKETS = {"short": (60, 110), "medium": (120, 200), "long": (220, 320)}
N_RISKS_WEIGHTS = {0: 0.08, 1: 0.22, 2: 0.30, 3: 0.25, 4: 0.15}
DISTRACTOR_PROB = 0.35
NAME_PREFIXES = ["Arden", "Bluefin", "Corvane", "Delmar", "Eastwick", "Fenwick", "Granite", "Halcyon", "Ironvale",
                 "Juniper", "Kestrel", "Lumen", "Meridian", "Northgate", "Orchard", "Pinecrest", "Quillon", "Redwater",
                 "Solace", "Tidewell", "Umber", "Vantor", "Westmere", "Yarrow", "Zephyr"]
NAME_SUFFIXES = ["Holdings", "Group", "Inc.", "Corporation", "Technologies", "Industries", "Partners", "Systems"]

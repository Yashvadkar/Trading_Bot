"""High-Speed Financial Text Sanitizer & Ticker Extraction Engine.

Extracts legitimate stock symbols, strips false positives, and computes
lexicon-based sentiment scores without requiring heavy local LLMs.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# Common English words and financial acronyms that collide with valid stock tickers
FALSE_POSITIVE_TICKERS = {
  "A",
  "AI",
  "ALL",
  "AM",
  "AN",
  "AND",
  "ANY",
  "ARE",
  "AS",
  "AT",
  "BE",
  "BIG",
  "BY",
  "CAN",
  "CAR",
  "CEO",
  "CFO",
  "CTO",
  "DO",
  "FOR",
  "GO",
  "HE",
  "IF",
  "IN",
  "IS",
  "IT",
  "ME",
  "MY",
  "NO",
  "NOT",
  "NOW",
  "OF",
  "OK",
  "ON",
  "ONE",
  "OR",
  "OUT",
  "PM",
  "RAW",
  "RUN",
  "SEE",
  "SO",
  "THE",
  "TO",
  "TOP",
  "UP",
  "US",
  "USA",
  "USD",
  "WAR",
  "WE",
  "WIN",
  "YOU",
  "SEC",
  "FED",
  "GDP",
  "CPI",
  "API",
  "LLM",
  "GPU",
  "CPU",
  "RAM",
  "ETF",
  "IPO",
  "SPAC",
  "EPS",
  "NAV",
  "ATH",
  "FOMO",
  "DD",
  "YOLO",
  "EOD",
  "Q1",
  "Q2",
  "Q3",
  "Q4",
  "FY",
}

# Financial domain lexicons with directional weights
BULLISH_KEYWORDS = {
    "surge": 1.5, "surging": 1.5, "jump": 1.2, "rally": 1.4, "rallies": 1.4,
    "beat": 1.5, "beats": 1.5, "outperform": 1.4, "outperformed": 1.4,
    "record": 1.2, "growth": 1.0, "profit": 1.2, "profitable": 1.2,
    "upgrade": 1.6, "upgraded": 1.6, "buy": 1.0, "strong": 1.0,
    "expansion": 1.1, "breakout": 1.5, "fda approval": 2.0, "approved": 1.5,
    "contract": 1.2, "partnership": 1.2, "dividend": 1.1, "guidance raised": 1.8,
    "acquisition": 1.3, "bullish": 1.5, "insider buying": 1.7, "patent": 1.2
}

BEARISH_KEYWORDS = {
    "plunge": -1.6, "plunging": -1.6, "drop": -1.2, "crash": -1.8, "crashes": -1.8,
    "miss": -1.5, "misses": -1.5, "underperform": -1.4, "underperformed": -1.4,
    "loss": -1.3, "losses": -1.3, "downgrade": -1.6, "downgraded": -1.6,
    "sell": -1.0, "weak": -1.1, "recession": -1.5, "inflation": -1.0,
    "guidance cut": -1.8, "investigation": -1.9, "lawsuit": -1.7, "fraud": -2.5,
    "sec probe": -2.2, "bankruptcy": -2.8, "default": -2.0, "layoffs": -1.2,
    "bearish": -1.5, "insider selling": -1.4, "warning": -1.5, "debt": -1.1
}


@dataclass(frozen=True)
class ScoredMention:
    ticker: str
    mention_count: int
    sentiment_score: float
    catalysts: list[str]


def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_tickers_with_sentiment(text: str) -> list[ScoredMention]:
    """Finds valid stock tickers ($TICKER or standalone uppercase) and measures contextual sentiment."""
    text_clean = clean_html_text(text)

    # 1. Look for explicit cash-tags: $NVDA, $AAPL
    cashtag_pattern = re.compile(r"\$([A-Z]{1,5})\b")
    # 2. Look for potential standalone uppercase tickers: "Apple (AAPL) today announced"
    paren_pattern = re.compile(r"\(([A-Z]{1,5})\)")

    raw_matches = cashtag_pattern.findall(text_clean) + paren_pattern.findall(text_clean)

    ticker_counts: dict[str, int] = {}
    for sym in raw_matches:
        sym_clean = sym.upper()
        if sym_clean not in FALSE_POSITIVE_TICKERS and len(sym_clean) <= 5:
            ticker_counts[sym_clean] = ticker_counts.get(sym_clean, 0) + 1

    # Text lowercased for sentiment lexicon matching
    text_lower = text_clean.lower()
    sentences = re.split(r"[.!?\n]", text_clean)

    results: list[ScoredMention] = []

    for sym, count in ticker_counts.items():
        # Contextual sentiment: evaluate sentences mentioning this symbol
        sym_sentences = [
            s.lower() for s in sentences
            if f"${sym.lower()}" in s.lower() or f" {sym.lower()} " in s.lower() or f"({sym.lower()})" in s.lower()
        ]

        if not sym_sentences:
            sym_sentences = [text_lower]

        sentiment_sum = 0.0
        catalysts: list[str] = []

        for s in sym_sentences:
            for kw, weight in BULLISH_KEYWORDS.items():
                if kw in s:
                    sentiment_sum += weight
                    if kw not in catalysts:
                        catalysts.append(kw)
            for kw, weight in BEARISH_KEYWORDS.items():
                if kw in s:
                    sentiment_sum += weight
                    if kw not in catalysts:
                        catalysts.append(kw)

        # Normalize score into [-1.0, +1.0]
        raw_score = sentiment_sum / (len(sym_sentences) + 1.0)
        final_score = float(max(-1.0, min(1.0, raw_score / 2.0)))

        results.append(ScoredMention(
            ticker=sym,
            mention_count=count,
            sentiment_score=final_score,
            catalysts=catalysts[:5]
        ))

    # Sort descending by mention count and positive sentiment
    results.sort(key=lambda x: (x.mention_count, abs(x.sentiment_score)), reverse=True)
    return results

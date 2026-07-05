from __future__ import annotations

import re
from typing import Dict, List


_CONCEPT_WORDS: Dict[str, List[str]] = {
    "ranking": ["top", "highest", "lowest", "best", "worst", "most", "least", "fastest", "slowest", "strongest", "poor", "overpriced"],
    "growth": ["growth", "growing", "declining", "trend", "pattern", "patterns", "over time", "daily", "weekly", "monthly", "seasonal", "seasonality", "holiday", "holidays", "black friday", "month-over-month", "quarter-over-quarter", "fastest growth"],
    "share": ["share", "percentage", "portion", "mix", "concentrated", "concentration"],
    "basket": ["basket", "baskets", "bundle", "bundles", "bundled", "bought together", "commonly bundled", "cross-sell", "cross sell", "recommended", "recommendation", "same order"],
    "contrast": ["but", "versus", "vs", "difference", "compared", "low", "high", "few", "many", "relative to", "overpriced"],
    "relationship": ["relationship", "correlation", "correlate", "impact", "effect", "influence", "affect", "cause", "causal", "drive", "drives", "driver", "drivers", "reduce", "reduces", "lower", "lowers", "harsher", "harsh"],
    "repeat": ["repeat", "return", "retain", "retains", "retention", "loyal", "again", "dormant", "churn", "cohort", "cohorts", "acquired", "acquisition", "lifetime", "long term", "first", "second"],
    "geography": ["state", "states", "city", "cities", "region", "regions", "location", "locations", "area", "areas", "geographic", "where"],
    "demand": ["orders", "order volume", "demand", "customers", "purchased", "buy", "sales", "volume", "convert", "conversion", "spikes", "sell", "sold", "best seller", "best sellers", "rarely sold"],
    "revenue": ["revenue", "sales value", "spend", "payment value", "payment amount", "order value", "amount", "value", "gmv"],
    "price": ["price", "pricing", "premium", "cheap", "expensive", "aov", "order value", "discount", "elasticity", "price war", "price wars"],
    "quality": ["review", "reviews", "rating", "ratings", "score", "quality", "harsher", "harsh", "1-star", "one-star"],
    "delivery": ["delivery", "deliver", "delivered", "delay", "delays", "ship", "ships", "shipping", "serve", "service", "slow", "fast", "late"],
    "supply": ["seller", "sellers", "supply", "presence", "coverage"],
    "coverage": ["underpenetrated", "penetration", "coverage"],
    "risk": ["risk", "risky", "crisis", "crises", "trust", "brand trust", "damage", "dependent", "dependency", "dependence", "underperform", "underperformance", "clustered", "cluster"],
    "payment": ["payment", "payments", "payment method", "payment methods", "payment type", "payment types", "credit card", "boleto", "voucher", "debit card", "installment", "installments"],
    "anomaly": ["anomaly", "anomalies", "unusual", "suspicious", "fraud", "fake", "duplicate", "duplicates", "rapidly", "rapid", "excessive", "spike", "spikes"],
    "data_quality": ["missing", "null", "duplicate rows", "duplicates rows", "inconsistent", "invalid", "negative", "impossible", "broken", "data quality"],
}


def _contains(text: str, phrase: str) -> bool:
    pattern = rf"(?<!\w){re.escape(phrase.lower())}(?!\w)"
    return re.search(pattern, text) is not None


def has_concept(text: str, concept: str) -> bool:
    lowered = (text or "").lower()
    for item in _CONCEPT_WORDS.get(concept, []):
        if _contains(lowered, item):
            return True
        if not item.endswith("s") and _contains(lowered, f"{item}s"):
            return True
    return False


def infer_capability_signals(text: str) -> Dict[str, bool | str | None]:
    lowered = (text or "").lower()
    level = None
    plural_map = {"state": "states", "city": "cities", "region": "regions", "location": "locations", "area": "areas"}
    for candidate in ("state", "city", "region", "location", "area"):
        if _contains(lowered, candidate) or _contains(lowered, plural_map[candidate]):
            level = candidate
            break

    return {
        "asks_ranking": has_concept(lowered, "ranking"),
        "asks_growth": has_concept(lowered, "growth"),
        "asks_share": has_concept(lowered, "share"),
        "asks_basket": has_concept(lowered, "basket"),
        "asks_contrast": has_concept(lowered, "contrast"),
        "asks_relationship": has_concept(lowered, "relationship"),
        "asks_repeat": has_concept(lowered, "repeat"),
        "asks_geography": has_concept(lowered, "geography"),
        "asks_demand": has_concept(lowered, "demand"),
        "asks_revenue": has_concept(lowered, "revenue"),
        "asks_price": has_concept(lowered, "price"),
        "asks_quality": has_concept(lowered, "quality"),
        "asks_delivery": has_concept(lowered, "delivery"),
        "asks_supply": has_concept(lowered, "supply"),
        "asks_coverage": has_concept(lowered, "coverage"),
        "asks_payment": has_concept(lowered, "payment"),
        "asks_risk": has_concept(lowered, "risk"),
        "geography_level": level,
    }

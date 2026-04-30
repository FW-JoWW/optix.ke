from __future__ import annotations

import re
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from utils.numeric_parsing import normalize_numeric_token


UNIT_HINT_GROUPS = (
    ({"hp", "horsepower", "engine_hp"}, {"kw", "kilowatt", "kilowatts", "max_power_kw"}),
    ({"kg", "kilogram", "kilograms", "weight_kg"}, {"lb", "lbs", "pound", "pounds", "weight_lbs"}),
    ({"km", "kilometer", "kilometers", "distance_km"}, {"mile", "miles", "mi", "distance_miles"}),
    ({"kph", "kmh", "km_per_hour"}, {"mph", "miles_per_hour"}),
)

STRUCTURAL_HINTS = {"make", "model", "modle", "id", "sku", "category", "trim", "series"}
ECONOMIC_HINTS = {
    "sales",
    "revenue",
    "price",
    "profit",
    "cost",
    "margin",
    "discount",
    "spend",
    "ads",
    "ad",
    "marketing",
    "gmv",
    "quantity",
    "volume",
}
BEHAVIORAL_HINTS = {
    "retention",
    "churn",
    "click",
    "clicks",
    "session",
    "sessions",
    "engagement",
    "onboarding",
    "signup",
    "conversion",
    "conversions",
    "user",
    "users",
}


def _tokenize(column_name: str) -> set[str]:
    lowered = str(column_name).strip().lower()
    parts = re.split(r"[^a-z0-9]+", lowered)
    return {part for part in parts if part}


def _numeric_pair(df: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    frame = df[[x_column, y_column]].copy()
    frame[x_column] = pd.to_numeric(frame[x_column].map(normalize_numeric_token), errors="coerce")
    frame[y_column] = pd.to_numeric(frame[y_column].map(normalize_numeric_token), errors="coerce")
    return frame.dropna()


def _constant_ratio_stats(frame: pd.DataFrame, x_column: str, y_column: str) -> Dict[str, float]:
    if frame.empty:
        return {"median_ratio": np.nan, "ratio_cv": np.nan, "scaled_residual_cv": np.nan}

    safe = frame[(frame[x_column] != 0) & frame[x_column].notna() & frame[y_column].notna()].copy()
    if safe.empty:
        return {"median_ratio": np.nan, "ratio_cv": np.nan, "scaled_residual_cv": np.nan}

    ratios = (safe[y_column] / safe[x_column]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return {"median_ratio": np.nan, "ratio_cv": np.nan, "scaled_residual_cv": np.nan}

    median_ratio = float(ratios.median())
    ratio_mean = float(ratios.mean()) or np.nan
    ratio_std = float(ratios.std(ddof=0)) if len(ratios) > 1 else 0.0
    ratio_cv = abs(ratio_std / ratio_mean) if ratio_mean not in {0.0, np.nan} else np.inf

    scaled_residual = safe[y_column] - (safe[x_column] * median_ratio)
    denom = float(safe[y_column].std(ddof=0)) or 1.0
    scaled_residual_cv = float(scaled_residual.std(ddof=0) / denom) if len(safe) > 1 else 0.0

    return {
        "median_ratio": median_ratio,
        "ratio_cv": float(ratio_cv),
        "scaled_residual_cv": float(abs(scaled_residual_cv)),
    }


def _has_unit_hint(x_column: str, y_column: str) -> bool:
    x_tokens = _tokenize(x_column)
    y_tokens = _tokenize(y_column)
    for left_group, right_group in UNIT_HINT_GROUPS:
        if (x_tokens & left_group and y_tokens & right_group) or (x_tokens & right_group and y_tokens & left_group):
            return True
    return False


def _has_any_hint(columns: Iterable[str], hints: set[str]) -> bool:
    for column in columns:
        if _tokenize(column) & hints:
            return True
    return False


def classify_relationship(
    x_column: str,
    y_column: str,
    stats_output: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    dataframe = metadata.get("dataframe")
    relationship_type = "unknown"
    triggered_rules: list[str] = []
    details: Dict[str, Any] = {}

    coefficient = None
    stats_payload = stats_output.get("stats", {}) if isinstance(stats_output, dict) else {}
    if isinstance(stats_payload, dict):
        coefficient = stats_payload.get("coefficient")
    if coefficient is None:
        coefficient = ((stats_output.get("effect_size") or {}).get("value") if isinstance(stats_output, dict) else None)
    if coefficient is None and isinstance(stats_output, dict):
        coefficient = stats_output.get("value")
    abs_corr = abs(float(coefficient)) if coefficient is not None else 0.0

    temporal_signal = (stats_output.get("temporal_signal") or {}) if isinstance(stats_output, dict) else {}
    if temporal_signal.get("applicable") and (
        temporal_signal.get("best_lag") not in {None, 0} or temporal_signal.get("timestamp_column")
    ):
        relationship_type = "temporal"
        triggered_rules.append("temporal_signal")

    if relationship_type == "unknown" and _has_any_hint([x_column, y_column], STRUCTURAL_HINTS):
        relationship_type = "structural"
        triggered_rules.append("structural_name_hint")

    if isinstance(dataframe, pd.DataFrame) and x_column in dataframe.columns and y_column in dataframe.columns:
        frame = _numeric_pair(dataframe, x_column, y_column)
        if len(frame) >= 5:
            ratio_stats = _constant_ratio_stats(frame, x_column, y_column)
            details.update(ratio_stats)
            unit_hint = _has_unit_hint(x_column, y_column)
            median_ratio = ratio_stats["median_ratio"]
            near_identity_scale = np.isfinite(median_ratio) and abs(float(median_ratio) - 1.0) <= 0.02
            if relationship_type == "unknown" and abs_corr >= 0.999 and ratio_stats["scaled_residual_cv"] <= 1e-3 and near_identity_scale:
                relationship_type = "duplicate_feature"
                triggered_rules.append("near_duplicate_scaled_feature")
            elif relationship_type == "unknown" and abs_corr >= 0.98 and (
                unit_hint
                or (
                    np.isfinite(ratio_stats["ratio_cv"])
                    and ratio_stats["ratio_cv"] <= 0.03
                    and ratio_stats["scaled_residual_cv"] <= 0.03
                    and not near_identity_scale
                )
            ):
                relationship_type = "unit_conversion"
                triggered_rules.append("unit_conversion_pattern")
                if unit_hint:
                    triggered_rules.append("unit_name_hint")

    if relationship_type == "unknown" and _has_any_hint([x_column, y_column], ECONOMIC_HINTS):
        relationship_type = "economic"
        triggered_rules.append("economic_name_hint")
    elif relationship_type == "unknown" and _has_any_hint([x_column, y_column], BEHAVIORAL_HINTS):
        relationship_type = "behavioral"
        triggered_rules.append("behavioral_name_hint")

    return {
        "relationship_type": relationship_type,
        "triggered_rules": triggered_rules,
        "details": details,
    }

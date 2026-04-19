from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List

import pandas as pd

from utils.numeric_parsing import normalize_numeric_token
from utils.openai_runtime import get_openai_client

ALLOWED_ACTIONS = {
    "impute_median",
    "impute_mode",
    "forward_fill",
    "backward_fill",
    "remove_duplicates",
    "drop_column",
    "investigate_or_cap",
    "convert_to_numeric",
    "convert_to_datetime",
    "standardize_categories",
    "review_only",
}


def _numeric_like_ratio(series: pd.Series) -> float:
    parsed = pd.to_numeric(series.map(normalize_numeric_token), errors="coerce")
    non_null = int(series.notna().sum())
    if non_null == 0:
        return 0.0
    return float(parsed.notna().sum() / non_null)


def _datetime_like_ratio(series: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    non_null = int(series.notna().sum())
    if non_null == 0:
        return 0.0
    return float(parsed.notna().sum() / non_null)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _missing_pattern_profile(series: pd.Series) -> Dict[str, Any]:
    missing_idx = series[series.isna()].index.tolist()
    if not missing_idx:
        return {
            "missing_count": 0,
            "leading_missing": 0,
            "trailing_missing": 0,
            "prev_next_same_ratio": 0.0,
        }

    values = series.tolist()
    prev_next_same = 0
    comparable = 0

    for idx in missing_idx:
        prev_val = values[idx - 1] if idx > 0 else pd.NA
        next_val = values[idx + 1] if idx < len(values) - 1 else pd.NA
        if pd.notna(prev_val) and pd.notna(next_val):
            comparable += 1
            if str(prev_val).strip().lower() == str(next_val).strip().lower():
                prev_next_same += 1

    leading_missing = 0
    for value in values:
        if pd.isna(value):
            leading_missing += 1
        else:
            break

    trailing_missing = 0
    for value in reversed(values):
        if pd.isna(value):
            trailing_missing += 1
        else:
            break

    return {
        "missing_count": len(missing_idx),
        "leading_missing": leading_missing,
        "trailing_missing": trailing_missing,
        "prev_next_same_ratio": round(prev_next_same / comparable, 4) if comparable else 0.0,
    }


def _label_quality_profile(series: pd.Series) -> Dict[str, Any]:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return {
            "whitespace_issues": False,
            "case_inconsistency": False,
        }

    stripped = non_null.str.strip()
    normalized = stripped.str.lower()
    return {
        "whitespace_issues": bool((non_null != stripped).any()),
        "case_inconsistency": normalized.nunique() < stripped.nunique(),
    }


def build_column_profiles(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    total_rows = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        unique_count = int(non_null.nunique()) if not non_null.empty else 0
        label_profile = _label_quality_profile(series)
        profiles[col] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": round(float(series.isna().mean()), 4),
            "unique_count": unique_count,
            "unique_ratio": round(unique_count / total_rows, 4),
            "numeric_like_ratio": round(_numeric_like_ratio(series), 4),
            "datetime_like_ratio": round(_datetime_like_ratio(series), 4),
            "sample_values": non_null.astype(str).head(5).tolist(),
            "missing_pattern": _missing_pattern_profile(series),
            "label_quality": label_profile,
        }
    return profiles


def _deterministic_action(issue: Dict[str, Any], profile: Dict[str, Any]) -> tuple[str, str]:
    issue_type = issue.get("issue_type")
    missing_ratio = profile.get("missing_ratio", 0.0)
    unique_ratio = profile.get("unique_ratio", 0.0)
    numeric_like_ratio = profile.get("numeric_like_ratio", 0.0)
    datetime_like_ratio = profile.get("datetime_like_ratio", 0.0)
    missing_pattern = profile.get("missing_pattern", {})
    label_quality = profile.get("label_quality", {})

    if issue_type == "duplicate_rows":
        return "remove_duplicates", "Duplicate rows reduce data quality and should be removed."
    if issue_type == "outliers":
        return "investigate_or_cap", "Extreme numeric values can distort downstream analysis."
    if issue_type == "constant_column":
        return "drop_column", "A constant column adds no analytical signal."
    if issue_type == "numeric_as_object":
        return "convert_to_numeric", "The column is mostly numeric-like and should be normalized to numeric."
    if issue_type == "datetime_as_object":
        return "convert_to_datetime", "The column is mostly datetime-like and should be parsed as datetime."
    if issue_type == "inconsistent_labels":
        return "standardize_categories", "Whitespace or case variants should be standardized for consistent grouping."
    if issue_type == "high_cardinality":
        return "review_only", "High-cardinality columns may need domain-specific handling rather than automatic cleaning."

    if issue_type == "missing_values":
        if missing_ratio >= 0.75 and unique_ratio >= 0.75:
            return "drop_column", "The column is mostly missing and mostly unique, so dropping is safer than filling."
        if missing_pattern.get("prev_next_same_ratio", 0.0) >= 0.6 and missing_ratio <= 0.4:
            return "forward_fill", "Neighboring values are stable around gaps, so forward fill is appropriate."
        if datetime_like_ratio >= 0.8 and missing_ratio <= 0.3:
            return "forward_fill", "Datetime-like columns with short gaps are often best forward-filled to preserve sequence context."
        if numeric_like_ratio >= 0.8:
            return "impute_median", "Numeric columns should use median imputation to reduce sensitivity to skew."
        if label_quality.get("whitespace_issues") or label_quality.get("case_inconsistency"):
            return "standardize_categories", "Standardize labels before imputing or grouping categorical values."
        return "impute_mode", "Categorical columns default to mode imputation when no safer sequence signal exists."

    return "review_only", "No safe automatic cleaning action was inferred."


def _needs_llm(issue: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    if issue.get("issue_type") != "missing_values":
        return False

    missing_ratio = profile.get("missing_ratio", 0.0)
    numeric_like_ratio = profile.get("numeric_like_ratio", 0.0)
    datetime_like_ratio = profile.get("datetime_like_ratio", 0.0)
    prev_next_same_ratio = profile.get("missing_pattern", {}).get("prev_next_same_ratio", 0.0)
    return (
        0.05 <= missing_ratio <= 0.5
        and (
            0.5 <= numeric_like_ratio < 0.8
            or 0.5 <= datetime_like_ratio < 0.8
            or 0.35 <= prev_next_same_ratio < 0.6
        )
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _llm_recommendation(issue: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any] | None:
    client = get_openai_client()
    if client is None:
        return None

    prompt = f"""
You are a schema-grounded data cleaning assistant.
Choose exactly one cleaning action for the issue below.
Base your decision only on the provided issue metadata and column profile.
Do not assume any specific industry or dataset domain.

Allowed actions:
{sorted(ALLOWED_ACTIONS)}

Issue:
{json.dumps(_to_jsonable(issue), ensure_ascii=True)}

Column profile:
{json.dumps(_to_jsonable(profile), ensure_ascii=True)}

Rules:
- Use only the given column name.
- Prefer conservative actions over risky guessing.
- Use forward_fill or backward_fill only when row order plausibly carries context.
- Use impute_median only for numeric or numeric-like columns.
- Use impute_mode only for categorical-like columns.
- Use drop_column only when the column is mostly unusable.
- If automatic cleaning is not clearly safe, return review_only.
- Output JSON only.

Return exactly:
{{
  "recommended_action": "one allowed action",
  "explanation": "short grounded explanation"
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = _extract_json_object(response.choices[0].message.content or "")
    except Exception:
        return None

    action = parsed.get("recommended_action")
    explanation = parsed.get("explanation")
    if action not in ALLOWED_ACTIONS or not isinstance(explanation, str) or not explanation.strip():
        return None

    return {
        "recommended_action": action,
        "explanation": explanation.strip(),
    }


def recommend_cleaning_issues(
    detected_issues: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    column_profiles = build_column_profiles(df)
    recommendations: List[Dict[str, Any]] = []
    llm_used = False

    for issue in detected_issues.get("detected_issues", []):
        col = issue.get("column")
        profile = column_profiles.get(col, {}) if col is not None else {}
        action, explanation = _deterministic_action(issue, profile)
        source = "rules"

        if col is not None and _needs_llm(issue, profile):
            llm_result = _llm_recommendation(issue, profile)
            if llm_result is not None:
                action = llm_result["recommended_action"]
                explanation = llm_result["explanation"]
                source = "llm_validated"
                llm_used = True

        recommendations.append({
            "column": col,
            "issue_type": issue.get("issue_type"),
            "severity": issue.get("severity", "medium"),
            "explanation": explanation,
            "recommended_action": action,
            "reasoning_source": source,
        })

    return {
        "issues": recommendations,
        "column_profiles": column_profiles,
        "cleaning_reasoning_status": "live_llm" if llm_used else "rules_only",
    }

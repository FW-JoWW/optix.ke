from __future__ import annotations

import re
from typing import Any, Iterable, List


LABEL_PRIORITY = [
    "product_category_name_english",
    "product_category_name",
    "product_name",
    "name",
    "title",
    "label",
    "city",
    "state",
    "description",
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _candidate_columns(dataframe: Any, token: str) -> List[str]:
    if dataframe is None or not hasattr(dataframe, "columns"):
        return []
    token_lower = token.lower()
    candidates: List[str] = []
    for column in list(dataframe.columns):
        column_name = str(column)
        lowered = column_name.lower()
        if lowered == token_lower:
            candidates.append(column_name)
        elif lowered.endswith("_id") and token_lower:
            candidates.append(column_name)
        elif token_lower in lowered and len(token_lower) >= 4:
            candidates.append(column_name)
        elif any(marker in lowered for marker in ("name", "category", "city", "state", "label", "title")):
            candidates.append(column_name)
    return candidates


def _series_equals(series: Any, token: str) -> Any:
    try:
        return series.astype(str) == token
    except Exception:
        try:
            return series.map(str) == token
        except Exception:
            return None


def _humanize_from_dataframe(token: str, dataframe: Any) -> str | None:
    if dataframe is None or not hasattr(dataframe, "columns"):
        return None

    for column in _candidate_columns(dataframe, token):
        try:
            series = dataframe[column]
        except Exception:
            continue
        mask = _series_equals(series, token)
        if mask is None:
            continue
        try:
            matches = dataframe.loc[mask]
        except Exception:
            continue
        if matches is None or getattr(matches, "empty", False):
            continue
        for label_column in LABEL_PRIORITY:
            for actual_column in list(getattr(matches, "columns", [])):
                actual_name = str(actual_column)
                lowered = actual_name.lower()
                if actual_name == column:
                    continue
                if label_column in lowered:
                    try:
                        raw_value = matches.iloc[0][actual_name]
                    except Exception:
                        continue
                    text = _normalize_text(raw_value)
                    if text and text.lower() not in {"none", "nan", "null"}:
                        return text
        try:
            row = matches.iloc[0]
        except Exception:
            continue
        for actual_column in list(getattr(matches, "columns", [])):
            actual_name = str(actual_column)
            if actual_name == column:
                continue
            try:
                raw_value = row[actual_name]
            except Exception:
                continue
            text = _normalize_text(raw_value)
            if text and text.lower() not in {"none", "nan", "null"}:
                return text
    return None


def humanize_identifier(value: Any, dataframe: Any = None) -> str:
    text = _normalize_text(value)
    if not text:
        return text

    dataframe_label = _humanize_from_dataframe(text, dataframe)
    if dataframe_label:
        return dataframe_label

    return text


def humanize_text(value: Any, dataframe: Any = None) -> str:
    text = _normalize_text(value)
    if not text:
        return text

    token_pattern = re.compile(r"\b[a-z0-9][a-z0-9_-]{5,}\b", re.IGNORECASE)
    tokens = sorted({match.group(0) for match in token_pattern.finditer(text)}, key=len, reverse=True)
    output = text
    for token in tokens:
        humanized = humanize_identifier(token, dataframe=dataframe)
        if humanized and humanized != token:
            output = re.sub(rf"\b{re.escape(token)}\b", humanized, output)
    return output


def humanize_columns(values: Iterable[Any], dataframe: Any = None) -> List[str]:
    if isinstance(values, str):
        values = [values]
    return [humanize_text(item, dataframe=dataframe) for item in values if _normalize_text(item)]


def suggestion_impact_percent(suggestion: Any) -> int | None:
    if not isinstance(suggestion, dict):
        return None

    candidate_keys = ("impact_percent", "impact", "priority_score", "confidence")
    for key in candidate_keys:
        value = suggestion.get(key)
        if isinstance(value, (int, float)):
            numeric = float(value)
            if key == "confidence" and numeric <= 1:
                numeric *= 100
            elif key in {"impact", "priority_score"} and numeric <= 1:
                numeric *= 100
            return max(0, min(100, round(numeric)))
        if isinstance(value, dict):
            for nested_key in ("percent", "score", "value"):
                nested = value.get(nested_key)
                if isinstance(nested, (int, float)):
                    numeric = float(nested)
                    if nested_key == "score" and numeric <= 1:
                        numeric *= 100
                    return max(0, min(100, round(numeric)))

    return None


def format_suggestion_line(suggestion: Any, index: int | None = None, dataframe: Any = None) -> str:
    if not isinstance(suggestion, dict):
        text = humanize_text(suggestion, dataframe=dataframe)
        return f"{index}. {text}" if index is not None else text

    title = humanize_text(suggestion.get("title") or "Next investigation", dataframe=dataframe)
    request = humanize_text(suggestion.get("request") or suggestion.get("description") or "No request provided.", dataframe=dataframe)
    impact = suggestion_impact_percent(suggestion)
    impact_text = f"{impact}% impact" if impact is not None else "impact unknown"
    prefix = f"{index}. " if index is not None else ""
    return f"{prefix}{title} - {request} ({impact_text})"

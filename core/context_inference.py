from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.openai_runtime import get_openai_client

ALLOWED_STRUCTURES = {"flat", "grouped", "hierarchical", "time-series", "unknown"}
ALLOWED_ROLES = {
    "identifier",
    "grouping_key",
    "categorical_feature",
    "numeric_measure",
    "derived_metric",
    "timestamp",
    "unknown",
}
ALLOWED_ACTIONS = {
    "forward_fill",
    "drop_rows",
    "convert_type",
    "standardize_categories",
    "recompute_if_possible",
    "leave_unchanged",
}


def _extract_json_object(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _summarize_profile_for_llm(dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    columns = dataset_profile.get("columns", {}) or {}
    summarized_columns = {}
    for col, info in list(columns.items())[:40]:
        summarized_columns[col] = {
            "inferred_type": info.get("inferred_type"),
            "missing_ratio": info.get("missing_ratio"),
            "unique_count": info.get("unique_count"),
            "numeric_like_ratio": info.get("numeric_like_ratio"),
            "datetime_like_ratio": info.get("datetime_like_ratio"),
            "top_values": (info.get("value_patterns") or [])[:3],
        }

    return {
        "row_count": dataset_profile.get("row_count"),
        "column_count": dataset_profile.get("column_count"),
        "column_names": dataset_profile.get("column_names", [])[:40],
        "columns": summarized_columns,
    }


def _summarize_structural_signals(structural_signals: Dict[str, Any] | None) -> Dict[str, Any]:
    structural_signals = structural_signals or {}
    return {
        "signals": (structural_signals.get("signals") or [])[:20],
        "high_missing_columns": (structural_signals.get("high_missing_columns") or [])[:20],
        "mixed_type_columns": (structural_signals.get("mixed_type_columns") or [])[:20],
        "duplicate_or_similar_columns": (structural_signals.get("duplicate_or_similar_columns") or [])[:20],
        "primary_structure_confidence": structural_signals.get("primary_structure_confidence"),
    }


def _fallback_context(dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    roles: Dict[str, str] = {}
    actions: List[Dict[str, str]] = []
    issues: List[Dict[str, str]] = []

    for col, info in dataset_profile.get("columns", {}).items():
        inferred_type = info.get("inferred_type")
        if inferred_type == "identifier_like":
            roles[col] = "identifier"
        elif inferred_type == "datetime":
            roles[col] = "timestamp"
        elif inferred_type == "numeric":
            roles[col] = "numeric_measure"
        elif inferred_type == "categorical":
            roles[col] = "categorical_feature"
        else:
            roles[col] = "unknown"

        if info.get("missing_ratio", 0.0) > 0:
            issues.append(
                {
                    "column": col,
                    "issue_type": "missing_values",
                    "interpretation": "unknown",
                }
            )
            if roles[col] == "timestamp" and info.get("unique_ratio", 1.0) <= 0.25:
                actions.append({"column": col, "action": "forward_fill"})
            else:
                actions.append({"column": col, "action": "leave_unchanged"})

    return {
        "dataset_structure": "unknown",
        "column_roles": roles,
        "issues": issues,
        "recommended_actions": actions,
        "forbidden_actions": [],
    }


def _validate_context(raw: Dict[str, Any], dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    columns = set(dataset_profile.get("column_names", []))
    validated = {
        "dataset_structure": raw.get("dataset_structure", "unknown"),
        "column_roles": {},
        "issues": [],
        "recommended_actions": [],
        "forbidden_actions": [],
    }

    if validated["dataset_structure"] not in ALLOWED_STRUCTURES:
        validated["dataset_structure"] = "unknown"

    for col, role in (raw.get("column_roles") or {}).items():
        if col in columns and role in ALLOWED_ROLES:
            validated["column_roles"][col] = role

    for item in raw.get("issues") or []:
        if not isinstance(item, dict):
            continue
        col = item.get("column")
        if col is not None and col not in columns:
            continue
        validated["issues"].append(
            {
                "column": col,
                "issue_type": str(item.get("issue_type", "unknown")),
                "interpretation": str(item.get("interpretation", "unknown")),
            }
        )

    for key in ("recommended_actions", "forbidden_actions"):
        for item in raw.get(key) or []:
            if not isinstance(item, dict):
                continue
            col = item.get("column")
            action = item.get("action")
            if col is not None and col not in columns:
                continue
            if action not in ALLOWED_ACTIONS:
                continue
            validated[key].append({"column": col, "action": action})

    for col in columns:
        validated["column_roles"].setdefault(col, "unknown")

    return validated


def infer_context(
    dataset_profile: Dict[str, Any],
    ambiguity_report: Dict[str, Any],
    sample_rows: List[Dict[str, Any]],
    structural_signals: Dict[str, Any] | None = None,
    llm_enabled: bool = True,
) -> Dict[str, Any]:
    fallback = _fallback_context(dataset_profile)

    if not llm_enabled or not ambiguity_report.get("requires_reasoning"):
        fallback["reasoning_status"] = "rules_only"
        return fallback

    client = get_openai_client()
    if client is None:
        fallback["reasoning_status"] = "unavailable"
        return fallback

    llm_profile = _summarize_profile_for_llm(dataset_profile)
    llm_structural_signals = _summarize_structural_signals(structural_signals)

    prompt = f"""
You are a dataset-agnostic schema reasoning engine.
You must infer structure only from the provided dataset profile and sample rows.
Do not assume industry, domain, or column meaning from names alone.
Only use evidence from patterns, values, and structure.

Dataset profile:
{json.dumps(llm_profile, ensure_ascii=True)}

Ambiguity report:
{json.dumps(ambiguity_report, ensure_ascii=True)}

Structural signals:
{json.dumps(llm_structural_signals, ensure_ascii=True)}

Sample rows:
{json.dumps(sample_rows, ensure_ascii=True)}

Allowed dataset_structure values:
flat, grouped, hierarchical, time-series, unknown

Allowed column roles:
identifier, grouping_key, categorical_feature, numeric_measure, derived_metric, timestamp, unknown

Allowed actions:
forward_fill, drop_rows, convert_type, standardize_categories, recompute_if_possible, leave_unchanged

Cleaning principles:
- Use forward_fill only when sparse values plausibly carry forward in row order.
- Use drop_rows only when structural evidence clearly shows a row is invalid noise.
- Use leave_unchanged when evidence for automatic transformation is weak.
- Prefer preserving derived columns unless recomputation is justified.

Return JSON only in exactly this shape:
{{
  "dataset_structure": "...",
  "column_roles": {{
    "<column_name>": "<role>"
  }},
  "issues": [
    {{
      "column": "<column_name>",
      "issue_type": "<type>",
      "interpretation": "<meaning>"
    }}
  ],
  "recommended_actions": [
    {{
      "column": "<column_name>",
      "action": "<generic_action>"
    }}
  ],
  "forbidden_actions": [
    {{
      "column": "<column_name>",
      "action": "<action>"
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        fallback["reasoning_status"] = "unavailable"
        return fallback

    parsed = _extract_json_object(response.choices[0].message.content or "")
    validated = _validate_context(parsed, dataset_profile)
    validated["reasoning_status"] = "live_llm"
    return validated

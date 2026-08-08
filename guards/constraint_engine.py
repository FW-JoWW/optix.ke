from __future__ import annotations

from typing import Any, Dict, List


def enforce_cleaning_constraints(
    actions: List[Dict[str, Any]],
    context: Dict[str, Any],
    allow_row_drops: bool = False,
    max_row_loss_ratio: float = 0.1,
) -> Dict[str, Any]:
    roles = context.get("column_roles", {})
    forbidden_pairs = {
        (item.get("column"), item.get("action"))
        for item in context.get("forbidden_actions", [])
    }

    approved: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    for action in actions:
        col = action.get("column")
        name = action.get("action")
        role = roles.get(col, "unknown")

        reason = None
        if (col, name) in forbidden_pairs:
            reason = "forbidden_by_context_inference"
        elif name == "drop_rows" and not allow_row_drops:
            reason = "row_drop_not_explicitly_allowed"
        elif role in {"identifier", "grouping_key"} and name in {"forward_fill", "standardize_categories"}:
            reason = "protected_role_cannot_be_imputed_or_standardized"
        elif role == "derived_metric" and name != "recompute_if_possible":
            reason = "derived_metrics_must_not_be_modified_directly"

        if reason:
            blocked.append({"action": action, "reason": reason})
        else:
            approved.append(action)

    return {
        "approved_actions": approved,
        "blocked_actions": blocked,
        "allow_row_drops": allow_row_drops,
        "max_row_loss_ratio": max_row_loss_ratio,
    }

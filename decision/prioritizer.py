from __future__ import annotations

from typing import Any, Dict, List


def prioritize_decisions(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [item for item in decisions if (item.get("validity") is not False)]
    type_rank = {
        "strategic": 4,
        "optimization": 3,
        "experiment": 2,
        "investigation": 1,
        "no_action": 0,
    }
    return sorted(
        valid,
        key=lambda item: (
            int((item.get("priority") or {}).get("priority_score", 0)),
            type_rank.get(str(item.get("action_type")), 0),
            int(item.get("confidence_in_action", 0)),
        ),
        reverse=True,
    )

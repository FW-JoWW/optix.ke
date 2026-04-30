from __future__ import annotations

from typing import Any, Dict, List


def prioritize_decisions(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [item for item in decisions if (item.get("validity") is not False)]
    return sorted(
        valid,
        key=lambda item: (
            int((item.get("priority") or {}).get("priority_score", 0)),
            int(item.get("confidence_in_action", 0)),
        ),
        reverse=True,
    )

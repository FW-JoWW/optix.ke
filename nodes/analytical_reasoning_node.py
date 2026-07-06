from __future__ import annotations

from core.analytical_reasoning import build_analytical_reasoning
from state.state import AnalystState


def analytical_reasoning_node(state: AnalystState) -> AnalystState:
    reasoning = build_analytical_reasoning(state)
    print("\n=== ANALYTICAL REASONING COMPLETE ===")
    print(
        {
            "confidence_level": reasoning.get("confidence", {}).get("level"),
            "confidence_score": reasoning.get("confidence", {}).get("score"),
            "primary_story_signature": reasoning.get("traceability", {}).get("primary_story_signature"),
        }
    )
    return state

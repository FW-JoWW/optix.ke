from __future__ import annotations

import json

from decision_engine import run_decision_engine
from constraint_engine import enforce_cleaning_constraints
from state.state import AnalystState


def cleaning_strategy_planner_node(state: AnalystState) -> AnalystState:
    context = state.get("context_inference") or {}
    recommended_actions = context.get("recommended_actions", [])

    constraints = enforce_cleaning_constraints(
        actions=recommended_actions,
        context=context,
        allow_row_drops=bool(state.get("allow_row_drops", False)),
        max_row_loss_ratio=0.1,
    )

    decision_output = run_decision_engine(
        dataset_profile=state.get("analysis_evidence", {}).get("preclean_profile_json", {}),
        structural_signals=state.get("structural_signals", {}),
        inferred_context=context,
        relationship_signals=state.get("relationship_signals", {}),
        user_intent=state.get("intent", {}),
        constraint_rules=constraints,
        candidate_plan=[],
        selected_columns=state.get("selected_columns", []),
    )

    final_plan = [
        {
            "column": item.target,
            "action": item.action_type,
            "severity": "contextual",
            "explanation": item.justification,
        }
        for item in decision_output.cleaning_decisions
        if not item.deferred and item.action_type != "leave_unchanged"
    ]

    state["cleaning_constraints"] = constraints
    state["decision_output"] = decision_output.model_dump()
    state["cleaning_plan"] = final_plan

    print("\n=== CLEANING PLAN GENERATED ===")
    print(json.dumps(
        {
            "cleaning_decisions": [item.model_dump() for item in decision_output.cleaning_decisions],
            "approved_actions": final_plan,
            "blocked_actions": constraints.get("blocked_actions", []),
        },
        indent=2,
    ))

    return state

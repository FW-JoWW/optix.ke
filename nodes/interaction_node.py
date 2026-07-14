from __future__ import annotations

import re
from typing import Any, Dict, List

from state.state import AnalystState
from nodes.guided_mode_node import guided_analysis_strategy_checkpoint


PLAN_TOOL_ALIASES = {
    "correlation": "correlation",
    "correlate": "correlation",
    "ttest": "ttest",
    "t_test": "ttest",
    "t-test": "ttest",
    "anova": "anova",
    "summary": "summary_statistics",
    "summary_statistics": "summary_statistics",
    "summarystatistics": "summary_statistics",
    "outliers": "detect_outliers",
    "detect_outliers": "detect_outliers",
    "categorical": "categorical_analysis",
    "categorical_analysis": "categorical_analysis",
}


def _stringify_plan(plan: List[Any]) -> List[str]:
    lines: List[str] = []
    for step in plan:
        if isinstance(step, dict):
            tool = step.get("tool", "unknown")
            columns = ", ".join(step.get("columns", []))
            lines.append(f"{tool}({columns})" if columns else tool)
        else:
            lines.append(str(step))
    return lines


def _split_columns(raw: str) -> List[str]:
    parts = re.split(r",|\band\b", raw, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


def _parse_plan_text(raw_text: str, valid_columns: List[str]) -> List[Dict[str, Any]]:
    if not raw_text.strip():
        return []

    normalized_columns = {col.lower(): col for col in valid_columns}
    segments = [seg.strip() for seg in re.split(r";|\n", raw_text) if seg.strip()]
    parsed_plan: List[Dict[str, Any]] = []

    for segment in segments:
        match = re.match(r"^\s*([a-zA-Z_ -]+)\s*\((.*?)\)\s*$", segment)
        if match:
            raw_tool = match.group(1).strip().lower().replace(" ", "_")
            tool = PLAN_TOOL_ALIASES.get(raw_tool)
            columns = _split_columns(match.group(2))
        else:
            match = re.match(r"^\s*([a-zA-Z_ -]+)\s+on\s+(.+)$", segment, flags=re.IGNORECASE)
            if not match:
                return []
            raw_tool = match.group(1).strip().lower().replace(" ", "_")
            tool = PLAN_TOOL_ALIASES.get(raw_tool)
            columns = _split_columns(match.group(2))

        if not tool:
            return []

        resolved_columns = []
        for column in columns:
            resolved = normalized_columns.get(column.lower())
            if not resolved:
                return []
            resolved_columns.append(resolved)

        parsed_plan.append({"tool": tool, "columns": resolved_columns})

    return parsed_plan


def _record_human_loop(state: AnalystState, mode: str, action: str, details: Dict[str, Any] | None = None) -> None:
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["human_in_loop"] = {
        "mode": mode,
        "action": action,
        "details": details or {},
    }


def interaction_node(state: AnalystState) -> AnalystState:
    mode = state.get("mode", "autonomous")
    evidence = state.setdefault("analysis_evidence", {})
    plan = evidence.get("analysis_plan", []) or state.get("analysis_plan", [])
    plan_lines = _stringify_plan(plan)
    active_df = state.get("analysis_dataset")
    if active_df is None:
        active_df = state.get("cleaned_data")
    if active_df is None:
        active_df = state.get("dataframe")
    valid_columns = list(active_df.columns) if active_df is not None else []

    state["awaiting_user"] = False

    if mode == "autonomous":
        _record_human_loop(state, mode, "system_autonomy", {"plan_preview": plan_lines})
        print("\n[Agent] Running in autonomous mode.")
        return state

    if mode == "guided":
        return guided_analysis_strategy_checkpoint(state)

    if mode == "collaborative":
        state["awaiting_user"] = True
        state["question_for_user"] = "Collaborative mode is managed by the investigation desk orchestration layer."
        evidence["final_output"] = [
            "Collaborative Mode now runs as an investigation orchestration layer.",
            "Use the collaborative session manager to queue, refine, compare, or challenge tasks.",
        ]
        evidence["collaborative_desk"] = {
            "current_proposed_plan": plan_lines,
            "valid_columns": valid_columns,
            "human_actions": [
                "new investigation",
                "refine task",
                "compare results",
                "challenge finding",
                "accept AI suggestion",
                "finish investigation",
            ],
        }
        _record_human_loop(
            state,
            mode,
            "desk_initialized",
            {
                "plan_preview": plan_lines,
                "valid_columns": valid_columns,
            },
        )
        return state

    _record_human_loop(state, mode, "unknown_mode_defaulted", {"plan_preview": plan_lines})
    return state

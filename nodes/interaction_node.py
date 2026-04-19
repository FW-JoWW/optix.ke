from __future__ import annotations

import re
from typing import Any, Dict, List

from state.state import AnalystState


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
        print("\n[Agent] Proposed Analysis Plan:")
        for line in plan_lines:
            print(f" - {line}")

        user_input = input(
            "\nApprove plan? (yes / modify / stop): "
        ).strip().lower()

        if user_input == "yes":
            _record_human_loop(state, mode, "approved_plan", {"plan_preview": plan_lines})
            return state

        if user_input == "modify":
            new_plan = input(
                "Enter the revised plan using entries like correlation(price, mileage); summary_statistics(price):\n> "
            ).strip()
            parsed_plan = _parse_plan_text(new_plan, valid_columns)
            if parsed_plan:
                evidence["analysis_plan"] = parsed_plan
                state["analysis_plan"] = parsed_plan
                _record_human_loop(
                    state,
                    mode,
                    "modified_plan",
                    {
                        "original_plan": plan_lines,
                        "updated_plan": _stringify_plan(parsed_plan),
                    },
                )
                print("\n[Agent] Updated plan:")
                for line in _stringify_plan(parsed_plan):
                    print(f" - {line}")
                return state

            print("\n[Agent] Could not parse the revised plan. Keeping the original proposed plan.")
            _record_human_loop(state, mode, "invalid_modification_kept_original", {"user_input": new_plan, "plan_preview": plan_lines})
            return state

        if user_input == "stop":
            print("\n[Agent] Execution stopped by user before analysis execution.")
            state["awaiting_user"] = True
            state["question_for_user"] = "Guided mode stopped before tool execution."
            evidence["final_output"] = ["Execution paused in guided mode before tool execution."]
            _record_human_loop(state, mode, "stopped", {"plan_preview": plan_lines})
            return state

        print("\n[Agent] Response not recognized. Keeping the original plan.")
        _record_human_loop(state, mode, "unrecognized_response_kept_original", {"user_input": user_input, "plan_preview": plan_lines})
        return state

    if mode == "collaborative":
        print("\n[Agent] Current proposed plan:")
        for line in plan_lines:
            print(f" - {line}")

        instruction = input(
            "\nTell the system exactly what to do.\nExamples: run correlation(price, mileage); run only summary_statistics(price, quantity); stop\n> "
        ).strip()
        state["user_response"] = instruction
        lowered = instruction.lower()

        if lowered in {"stop", "pause", "cancel"}:
            print("\n[Agent] Execution stopped by user before analysis execution.")
            state["awaiting_user"] = True
            state["question_for_user"] = "Collaborative mode stopped before tool execution."
            evidence["final_output"] = ["Execution paused in collaborative mode before tool execution."]
            _record_human_loop(state, mode, "stopped", {"instruction": instruction, "plan_preview": plan_lines})
            return state

        cleaned_instruction = re.sub(r"^\s*(run|do|analyze)\s+(only\s+)?", "", instruction, flags=re.IGNORECASE)
        if cleaned_instruction.lower() in {"continue", "continue as is", "run plan", "run proposed plan"}:
            _record_human_loop(state, mode, "accepted_proposed_plan", {"instruction": instruction, "plan_preview": plan_lines})
            return state

        parsed_plan = _parse_plan_text(cleaned_instruction, valid_columns)
        if parsed_plan:
            evidence["analysis_plan"] = parsed_plan
            state["analysis_plan"] = parsed_plan
            _record_human_loop(
                state,
                mode,
                "directed_plan",
                {
                    "instruction": instruction,
                    "updated_plan": _stringify_plan(parsed_plan),
                },
            )
            print("\n[Agent] Using collaborative user-directed plan:")
            for line in _stringify_plan(parsed_plan):
                print(f" - {line}")
            return state

        print("\n[Agent] Could not parse the instruction into an executable plan. Keeping the proposed plan.")
        _record_human_loop(state, mode, "instruction_not_parsed_kept_original", {"instruction": instruction, "plan_preview": plan_lines})
        return state

    _record_human_loop(state, mode, "unknown_mode_defaulted", {"plan_preview": plan_lines})
    return state

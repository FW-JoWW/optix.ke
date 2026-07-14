from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, pd.Series):
        return value.copy(deep=True)
    return deepcopy(value)


def _summarize_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "shape": tuple(value.shape),
            "columns": list(value.columns),
        }
    if isinstance(value, pd.Series):
        return {
            "type": "series",
            "length": int(len(value)),
            "name": value.name,
        }
    if isinstance(value, dict):
        keys = list(value.keys())
        return {
            "type": "dict",
            "size": len(value),
            "keys": keys[:12],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "size": len(value),
            "preview": [_summarize_value(item) if isinstance(item, (dict, list, pd.DataFrame, pd.Series)) else item for item in value[:5]],
        }
    return value


def _stage_state_keys(stage: str) -> List[str]:
    if stage == "data_preparation":
        return [
            "cleaning_plan",
            "cleaned_data",
            "analysis_dataset",
            "cleaning_validation",
            "data_validation",
            "data_quality_issues",
            "guided_cleaning_notes",
        ]
    if stage == "business_understanding":
        return [
            "selected_columns",
            "analysis_dataset",
            "intent",
            "column_registry",
            "guided_selection_notes",
        ]
    if stage == "analysis_strategy":
        return [
            "analysis_plan",
            "analysis_evidence",
            "decision_output",
        ]
    if stage == "result_review":
        return [
            "guided_visualization_preferences",
            "analysis_evidence",
        ]
    return []


def capture_guided_stage_snapshot(
    state: Dict[str, Any],
    stage: str,
    version: int,
    summary: Dict[str, Any] | None = None,
    note: str | None = None,
) -> Dict[str, Any]:
    evidence = state.setdefault("analysis_evidence", {})
    snapshots = evidence.setdefault("guided_version_snapshots", {})
    stage_snapshots = snapshots.setdefault(stage, {})
    snapshot_state = {}
    for key in _stage_state_keys(stage):
        if key == "analysis_evidence":
            snapshot_state[key] = _copy_value(state.get(key) or {})
            continue
        if key == "guided_cleaning_notes":
            snapshot_state[key] = _copy_value(evidence.get(key) or [])
            continue
        if key == "guided_selection_notes":
            snapshot_state[key] = _copy_value(evidence.get(key) or [])
            continue
        snapshot_state[key] = _copy_value(state.get(key))

    snapshot = {
        "stage": stage,
        "version": int(version),
        "captured_at": _utc_now(),
        "summary": summary or {},
        "note": note,
        "state": snapshot_state,
    }
    stage_snapshots[int(version)] = snapshot
    state["guided_version_snapshots"] = snapshots
    return snapshot


def restore_guided_stage_snapshot(
    state: Dict[str, Any],
    stage: str,
    version: int,
) -> Dict[str, Any] | None:
    snapshots = (state.get("analysis_evidence") or {}).get("guided_version_snapshots") or {}
    stage_snapshots = snapshots.get(stage) or {}
    snapshot = stage_snapshots.get(int(version))
    if not snapshot:
        return None

    snapshot_state = snapshot.get("state") or {}
    for key, value in snapshot_state.items():
        if key == "guided_cleaning_notes":
            state.setdefault("analysis_evidence", {})[key] = _copy_value(value)
            continue
        if key == "guided_selection_notes":
            state.setdefault("analysis_evidence", {})[key] = _copy_value(value)
            continue
        state[key] = _copy_value(value)

    state.setdefault("analysis_evidence", {})["guided_restored_version"] = {
        "stage": stage,
        "version": int(version),
        "restored_at": _utc_now(),
    }
    state.setdefault("guided_checkpoint_versions", {})[stage] = int(version)
    return snapshot


def diff_guided_stage_snapshots(
    state: Dict[str, Any],
    stage: str,
    left_version: int,
    right_version: int,
) -> Dict[str, Any]:
    snapshots = (state.get("analysis_evidence") or {}).get("guided_version_snapshots") or {}
    stage_snapshots = snapshots.get(stage) or {}
    left = stage_snapshots.get(int(left_version)) or {}
    right = stage_snapshots.get(int(right_version)) or {}
    left_state = left.get("state") or {}
    right_state = right.get("state") or {}

    keys = sorted(set(left_state) | set(right_state))
    changes: List[Dict[str, Any]] = []
    for key in keys:
        left_value = _summarize_value(left_state.get(key))
        right_value = _summarize_value(right_state.get(key))
        if left_value != right_value:
            changes.append(
                {
                    "field": key,
                    "from": left_value,
                    "to": right_value,
                }
            )

    return {
        "stage": stage,
        "from_version": int(left_version),
        "to_version": int(right_version),
        "changes": changes,
        "from_summary": left.get("summary") or {},
        "to_summary": right.get("summary") or {},
    }

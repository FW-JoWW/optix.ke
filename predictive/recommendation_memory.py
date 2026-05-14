from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


MEMORY_PATH = Path("data/recommendation_memory.json")


def load_memory() -> List[Dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def store_recommendation_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    history = load_memory()
    history.append(snapshot)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")
    return {"stored": True, "record_count": len(history[-200:])}


def calibrate_from_memory(target: str) -> Dict[str, Any]:
    history = load_memory()
    relevant = [item for item in history if item.get("target") == target]
    if not relevant:
        return {"memory_adjustment": 0.0, "evidence_count": 0, "note": "No prior recommendation outcome history is available."}

    realized = [item for item in relevant if item.get("realized_uplift") is not None and item.get("predicted_uplift") is not None]
    if not realized:
        return {"memory_adjustment": 0.0, "evidence_count": len(relevant), "note": "Historical recommendations exist, but realized outcomes have not been logged yet."}

    errors = []
    for item in realized:
        predicted = float(item.get("predicted_uplift") or 0.0)
        actual = float(item.get("realized_uplift") or 0.0)
        if predicted == 0:
            continue
        errors.append(abs(actual - predicted) / max(abs(predicted), 1e-6))

    if not errors:
        return {"memory_adjustment": 0.0, "evidence_count": len(realized), "note": "Not enough realized outcome variance for calibration."}

    mean_error = sum(errors) / len(errors)
    adjustment = max(-0.25, min(0.1, 0.05 - mean_error))
    return {
        "memory_adjustment": round(adjustment, 4),
        "evidence_count": len(realized),
        "note": "Recommendation confidence was calibrated using historical predicted-versus-realized uplift.",
    }

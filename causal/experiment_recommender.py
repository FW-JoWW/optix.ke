from __future__ import annotations

from typing import Any, Dict, List


def recommend_next_step(
    causal_grade: str,
    temporal_signal: Dict[str, Any],
    confounders: List[Dict[str, Any]],
    bias_risks: List[str],
    question: str,
) -> str:
    query = (question or "").lower()

    if causal_grade == "STRONG":
        return "Use the result operationally, but continue monitoring for drift and validate on future periods."

    if "selection_bias" in bias_risks or "simpsons_paradox" in bias_risks:
        return "Run a controlled cohort comparison with matched groups to reduce selection and aggregation bias."

    if temporal_signal.get("lag_direction") == "x_precedes_y" and confounders:
        return "Run a geo holdout or staggered rollout experiment to isolate the effect from confounders."

    if temporal_signal.get("lag_direction") == "unclear":
        return "Collect a cleaner time-indexed dataset and test the relationship with a before/after design plus a control group."

    if confounders:
        return "Run an A/B test or quasi-experimental control-group design before making a causal claim."

    if "price" in query or "ads" in query or "campaign" in query:
        return "Run an A/B test if randomization is feasible; otherwise use a geo holdout with a pre-registered success metric."

    return "Collect more controlled data and run a randomized or holdout-based experiment before treating the relationship as causal."

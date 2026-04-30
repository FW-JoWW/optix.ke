from __future__ import annotations

import pandas as pd

from core.insight_validator import validate_insight
from core.recommendation_guard import guard_recommendations
from core.semantic_classifier import classify_relationship
from llm.output_filter import filter_llm_output
from nodes.evidence_interpreter_node import evidence_interpreter_node


def test_unit_conversion_case() -> None:
    df = pd.read_csv("data/Car Dataset 1945-2020.csv", low_memory=False, usecols=["engine_hp", "max_power_kw"])
    df = df.dropna().head(4000)
    coefficient = float(df["engine_hp"].corr(df["max_power_kw"]))
    stats_output = {
        "stats": {"coefficient": coefficient},
        "causal_evidence": {"grade": "LOW"},
        "confounders": [],
    }
    semantic = classify_relationship("engine_hp", "max_power_kw", stats_output, {"dataframe": df})
    validity = validate_insight(stats_output, semantic, {"missing_ratio": 0.0})
    recommendation = guard_recommendations(stats_output, semantic, validity)
    filtered = filter_llm_output(
        raw_text=(
            "INSIGHT:\n- Engine horsepower causes higher power output.\n\n"
            "BUSINESS IMPLICATION:\n- Use this to change pricing.\n\n"
            "RECOMMENDATIONS:\n- Change pricing strategy.\n\n"
            "LIMITATIONS:\n- None."
        ),
        payload={"causal_evidence": stats_output["causal_evidence"], "top_stories": []},
        semantic_output=semantic,
        validation_output=validity,
        recommendation_output=recommendation,
    )

    assert semantic["relationship_type"] == "unit_conversion"
    assert validity["valid"] is False
    assert recommendation["final_recommendation"].lower().startswith("no business action")
    assert "does not represent an independent business insight" in filtered["text"]
    print("CASE 1 OK:", semantic, validity, recommendation)


def test_low_causal_economic_case() -> None:
    df = pd.DataFrame(
        {
            "ads": [10, 12, 15, 14, 17, 19, 22, 20],
            "revenue": [100, 105, 120, 118, 135, 138, 150, 149],
            "seasonality": [1, 1, 2, 2, 3, 3, 4, 4],
            "promotion": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    stats_output = {
        "stats": {"coefficient": 0.86},
        "causal_evidence": {"grade": "LOW"},
        "confounders": [{"column": "seasonality"}, {"column": "promotion"}, {"column": "region"}, {"column": "cohort"}],
        "partial_correlation": {"computed": False},
    }
    semantic = classify_relationship("ads", "revenue", stats_output, {"dataframe": df})
    validity = validate_insight(stats_output, semantic, {"missing_ratio": 0.0})
    recommendation = guard_recommendations(stats_output, semantic, validity)
    filtered = filter_llm_output(
        raw_text=(
            "INSIGHT:\n- Ads cause revenue growth.\n\n"
            "BUSINESS IMPLICATION:\n- Increase marketing budgets immediately.\n\n"
            "RECOMMENDATIONS:\n- Make pricing strategy changes.\n- Launch product changes.\n\n"
            "LIMITATIONS:\n- None."
        ),
        payload={"causal_evidence": stats_output["causal_evidence"], "top_stories": []},
        semantic_output=semantic,
        validation_output=validity,
        recommendation_output=recommendation,
    )

    assert semantic["relationship_type"] == "economic"
    assert validity["valid"] is True
    assert validity["severity"] == "high"
    assert "pricing strategy" in recommendation["blocked_actions"]
    assert "causes" not in filtered["text"].lower()
    assert "run experiment" in filtered["text"].lower() or "investigate further" in filtered["text"].lower()
    print("CASE 2 OK:", semantic, validity, recommendation)


def test_duplicate_feature_suppression() -> None:
    df = pd.DataFrame({"metric_a": [1, 2, 3, 4, 5], "metric_b": [1, 2, 3, 4, 5]})
    state = {
        "business_question": "what is the relationship between metric_a and metric_b",
        "cleaned_data": df.copy(),
        "raw_analysis_dataset": df.copy(),
        "analysis_evidence": {
            "tool_results": {
                "duplicate_test": {
                    "tool": "inferential_analysis",
                    "analysis_category": "numeric_relationship",
                    "method_selected": "spearman",
                    "columns": ["metric_a", "metric_b"],
                    "hypothesis_test": {"decision": "reject_h0", "p_value": 0.0},
                    "effect_size": {"metric": "correlation_r", "value": 1.0, "interpretation": "large"},
                    "interpretation": {"summary": "Perfect relationship"},
                    "assumptions": {"warnings": []},
                    "relationship_evidence": {
                        "stats": {"coefficient": 1.0},
                        "causal_evidence": {"grade": "LOW"},
                        "confounders": [],
                    },
                    "causal_evidence": {"grade": "LOW"},
                    "confounders": [],
                }
            }
        },
        "dataset_profile": {},
        "column_registry": {},
    }
    updated = evidence_interpreter_node(state)
    assert updated["analysis_evidence"]["story_candidates"] == []
    print("CASE 3 OK: duplicate feature suppressed")


if __name__ == "__main__":
    test_unit_conversion_case()
    test_low_causal_economic_case()
    test_duplicate_feature_suppression()
    print("All semantic guardrail tests passed.")

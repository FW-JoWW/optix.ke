from __future__ import annotations

from state.state import AnalystState


def _base_score(story):
    story_type = story.get("type")
    validity = story.get("insight_validity") or {}
    relationship_type = story.get("relationship_type")

    if relationship_type == "duplicate_feature":
        return 0.0
    if relationship_type == "unit_conversion":
        return 0.08
    if validity and not validity.get("valid", True):
        return 0.12

    if story_type == "correlation":
        strength = abs(story.get("value", 0))
        if strength >= 0.9:
            return 1.0
        if strength >= 0.75:
            return 0.85
        if strength >= 0.5:
            return 0.7
        return 0.5

    if story_type in ["group_difference", "categorical_relationship"]:
        p_value = story.get("p_value", 1.0)
        if p_value <= 0.001:
            return 1.0
        if p_value <= 0.01:
            return 0.9
        if p_value <= 0.05:
            return 0.75
        return 0.45

    if story_type == "inferential_relationship":
        effect = abs((story.get("effect_size") or {}).get("value") or story.get("value") or 0)
        p_value = story.get("p_value", 1.0)
        causal_grade = ((story.get("causal_evidence") or {}).get("grade")) or "LOW"
        base = 0.45
        if p_value <= 0.001:
            base += 0.3
        elif p_value <= 0.01:
            base += 0.22
        elif p_value <= 0.05:
            base += 0.15
        if effect >= 0.5:
            base += 0.2
        elif effect >= 0.3:
            base += 0.12
        elif effect >= 0.1:
            base += 0.05
        if causal_grade == "STRONG":
            base += 0.1
        elif causal_grade == "MODERATE":
            base += 0.05
        return min(base, 1.0)

    if story_type in ["inferential_group_difference", "inferential_categorical_association"]:
        p_value = story.get("p_value", 1.0)
        effect_value = abs(((story.get("effect_size") or {}).get("value")) or 0)
        causal_grade = ((story.get("causal_evidence") or {}).get("grade")) or "LOW"
        base = 0.45
        if p_value <= 0.001:
            base += 0.3
        elif p_value <= 0.01:
            base += 0.22
        elif p_value <= 0.05:
            base += 0.15
        if effect_value >= 0.5:
            base += 0.2
        elif effect_value >= 0.2:
            base += 0.1
        if causal_grade == "STRONG":
            base += 0.08
        elif causal_grade == "MODERATE":
            base += 0.04
        return min(base, 1.0)

    if story_type == "grouped_numeric":
        effect_size = abs(story.get("effect_size", 0))
        if effect_size > 10000:
            return 0.85
        if effect_size > 1000:
            return 0.7
        return 0.55

    if story_type == "category_frequency":
        share = story.get("share", 0)
        if share >= 60:
            return 0.8
        if share >= 40:
            return 0.65
        return 0.5

    if story_type == "rare_categories":
        count = story.get("count", 0)
        if count >= 3:
            return 0.7
        return 0.55

    if story_type == "outliers":
        count = story.get("count", 0)
        if count >= 20:
            return 0.9
        if count >= 10:
            return 0.75
        if count >= 5:
            return 0.6
        return 0.4

    if story_type == "summary_numeric":
        return 0.6

    if story_type == "predictive_model":
        metrics = story.get("metrics", {}) or {}
        readiness = story.get("readiness_warnings", []) or []
        penalty = sum(0.12 if item.get("severity") == "high" else 0.06 for item in readiness)
        if story.get("problem_type") == "classification":
            base = min(1.0, 0.4 + float(metrics.get("f1") or 0.0))
        elif story.get("problem_type") == "forecasting":
            mape = float(metrics.get("mape") or 1.0)
            base = max(0.35, min(1.0, 1.0 - mape))
        else:
            base = max(0.35, min(1.0, 0.35 + float(metrics.get("r2") or 0.0)))
        return max(0.25, min(1.0, base - penalty))

    if story_type == "prescriptive_action":
        upside = abs(float(story.get("estimated_upside") or 0.0))
        confidence = str(story.get("confidence") or "low").lower()
        if upside > 10000 or confidence == "high":
            return 0.9
        if upside > 1000 or confidence == "moderate":
            return 0.8
        if upside > 0:
            return 0.7
        return 0.5

    return 0.3


def story_scoring_engine_node(state: AnalystState) -> AnalystState:
    """
    Scores story candidates based on relevance and statistical importance.
    """
    evidence = state.get("analysis_evidence", {})
    candidates = evidence.get("story_candidates", [])
    question = state.get("business_question", "").lower()
    analysis_df = state.get("analysis_dataset")

    mentioned_columns = []
    if analysis_df is not None:
        mentioned_columns = [col for col in analysis_df.columns if col.lower() in question]

    if not candidates:
        print("No story candidates available for scoring.")
        state["analysis_evidence"]["top_stories"] = []
        return state

    print("\n=== SCORING STORY CANDIDATES ===")

    scored_stories = []
    for story in candidates:
        story_columns = []
        if "column" in story:
            story_columns.append(story.get("column"))
        if "columns" in story:
            story_columns.extend(story.get("columns"))
        if "group_column" in story:
            story_columns.append(story.get("group_column"))

        matches = [col for col in story_columns if col in mentioned_columns]
        relevance_multiplier = 1.0
        if matches:
            relevance_multiplier += 0.3
        elif mentioned_columns:
            relevance_multiplier -= 0.15

        base = _base_score(story)
        severity = ((story.get("insight_validity") or {}).get("severity")) or "low"
        if severity == "high":
            base *= 0.7
        elif severity == "medium":
            base *= 0.88
        score = round(min(base * relevance_multiplier, 1.0), 4)
        story["score"] = max(score, 0.0)
        story["score_components"] = {
            "base_score": round(base, 4),
            "relevance_multiplier": round(relevance_multiplier, 4),
            "matched_columns": matches,
        }
        scored_stories.append(story)

    ranked = sorted(scored_stories, key=lambda x: x["score"], reverse=True)
    top_stories = ranked[:5]
    state["analysis_evidence"]["top_stories"] = top_stories

    print("\nTop Stories Selected:")
    for story in top_stories:
        print(
            f"{story['type']} | score={story['score']} | "
            f"matched={story['score_components']['matched_columns']}"
        )

    return state

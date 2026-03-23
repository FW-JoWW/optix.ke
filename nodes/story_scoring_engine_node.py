from state.state import AnalystState


def story_scoring_engine_node(state: AnalystState) -> AnalystState:
    """
    Scores story candidates based on statistical importance.

    Reads:
        state["analysis_evidence"]["story_candidates"]

    Writes:
        state["analysis_evidence"]["top_stories"]
    """

    evidence = state.get("analysis_evidence", {})
    candidates = evidence.get("story_candidates", [])
    question = state.get("business_question", "").lower()
    
    # Extract mentioned columns (simple but effective)
    mentioned_columns = [
        col for col in state.get("analysis_dataset", {}).columns
        if col.lower() in question
    ]
    

    if not candidates:
        print("No story candidates available for scoring.")
        state["analysis_evidence"]["top_stories"] = []
        return state

    print("\n=== SCORING STORY CANDIDATES ===")

    scored_stories = []

    for story in candidates:

        score = 0
        story_type = story.get("type")

        # ------------------------
        # Question relevance boost
        # ------------------------

        relevance_multiplier = 1.0

        story_columns = []

        if "column" in story:
            story_columns.append(story.get("column"))

        if "columns" in story:
            story_columns.extend(story.get("columns"))

        # Check if story uses question-relevant columns
        matches = [col for col in story_columns if col in mentioned_columns]

        if matches:
            relevance_multiplier += 0.3  # boost if relevant
        else:
            relevance_multiplier -= 0.2  # penalize if not relevant

        # ------------------------
        # Domain weighting (THIS is the smart part)
        # ------------------------

        high_value_metrics = ["mpg", "mileage"]
        low_value_metrics = ["price", "tax"]

        for col in story_columns:
            if col in high_value_metrics:
                relevance_multiplier += 0.2
            elif col in low_value_metrics:
                relevance_multiplier -= 0.1

        # Apply final score
        score = score * relevance_multiplier
                
        # ------------------------
        # Correlation scoring
        # ------------------------

        if story_type == "correlation":

            corr = abs(story.get("value", 0))

            if corr >= 0.9:
                score = 1.0
            elif corr >= 0.75:
                score = 0.85
            elif corr >= 0.6:
                score = 0.7
            else:
                score = 0.5

        # ------------------------
        # T-test / group difference
        # ------------------------

        elif story_type == "t_test":

            p = story.get("p_value", 1)

            if p <= 0.001:
                score = 1.0
            elif p <= 0.01:
                score = 0.9
            elif p <= 0.05:
                score = 0.75
            else:
                score = 0.5

        # ------------------
        # Grouped_differnce
        # ------------------
        elif story_type == "group_difference":
        
            p = story.get("p_value", 1)
        
            if p <= 0.001:
                score = 1.0
            elif p <= 0.01:
                score = 0.9
            elif p <= 0.05:
                score = 0.75
            else:
                score = 0.4
        # ------------------------
        # Regression scoring
        # ------------------------

        elif story_type == "regression":

            r2 = story.get("r_squared", 0)

            if r2 >= 0.85:
                score = 1.0
            elif r2 >= 0.7:
                score = 0.85
            elif r2 >= 0.5:
                score = 0.7
            else:
                score = 0.5

        # ------------------------
        # Outlier scoring
        # ------------------------

        elif story_type == "outliers":

            count = story.get("count", 0)

            if count >= 20:
                score = 0.9
            elif count >= 10:
                score = 0.75
            elif count >= 5:
                score = 0.6
            else:
                score = 0.4

        else:

            score = 0.3

        story["score"] = score
        scored_stories.append(story)

    # ------------------------
    # Rank stories
    # ------------------------

    ranked = sorted(scored_stories, key=lambda x: x["score"], reverse=True)

    # Select top insights
    top_stories = ranked[:5]

    state["analysis_evidence"]["top_stories"] = top_stories

    print("\nTop Stories Selected:")
    for story in top_stories:
        print(f"{story['type']} | score={story['score']}")

    return state

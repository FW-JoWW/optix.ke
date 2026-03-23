from state.state import AnalystState


def insight_generator_node(state: AnalystState) -> AnalystState:
    """
    Converts raw statistical outputs into structured insights.
    Filters weak results and ranks meaningful ones.
    """

    tool_results = state.get("tool_results", {})
    insights = state.get("insights", [])

    correlations = []

    for key, result in tool_results.items():

        if key.startswith("correlation"):

            corr = result.get("correlation")

            if corr is None:
                continue

            strength = abs(corr)

            # ignore weak correlations
            if strength < 0.5:
                continue

            col1 = result.get("column_1")
            col2 = result.get("column_2")

            correlations.append({
                "pair": f"{col1} vs {col2}",
                "value": corr,
                "strength": strength
            })

        if key.startswith("ttest"):

            p = result.get("p_value")

            if p is not None and p < 0.05:

                insights.append(
                    f"{result['column']} differs significantly across {result['group_column']} groups (p={p:.3f})."
                )

        if key.startswith("detect_outliers"):

            count = result.get("outlier_count", 0)

            if count > 0:

                insights.append(
                    f"{result['column']} contains {count} potential outlier values."
                )

    # rank correlations
    correlations = sorted(correlations, key=lambda x: x["strength"], reverse=True)

    # keep top 3
    for c in correlations[:3]:

        insights.append(
            f"{c['pair']} shows a strong correlation of {c['value']:.2f}."
        )

    state["insights"] = insights

    print("\n=== FINAL INSIGHTS ===")
    for i in insights:
        print("-", i)

    return state
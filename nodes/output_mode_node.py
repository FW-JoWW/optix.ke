# nodes/output_mode_node.py
from state.state import AnalystState
import pandas as pd

def output_mode_node(state: AnalystState) -> AnalystState:
    """
    Final presentation layer (NO heavy logic here).
    Just formats what row_filter_node already computed.
    """

    evidence = state.setdefault("analysis_evidence", {})
    grouped = evidence.get("grouped_summary")
    aggregation = evidence.get("aggregation")

    # -----------------------------
    # NORMALIZE grouped format
    # -----------------------------
    if isinstance(grouped, pd.DataFrame):
        grouped = grouped.to_dict("records")

    # -----------------------------
    # CASE 1: GROUPED OUTPUT
    # -----------------------------
    if grouped is not None and len(grouped) > 0:

        formatted = []

        # Detect grouping key dynamically
        sample_row = grouped[0]
        keys = list(sample_row.keys())

        # Aggregation case
        if aggregation:
            group_key = [k for k in keys if k != aggregation["column"]][0]
            value_key = aggregation["column"]
            agg_type = aggregation["type"]

            for row in grouped:
                formatted.append(
                    f"{row[group_key]} → {round(row[value_key], 2)} ({agg_type})"
                )
        # Count case
        else:
            group_key = [k for k in keys if k !="count"][0]

            for row in grouped:
                formatted.append(
                    f"{row[group_key]} → {row['count']}"
                )

        evidence["final_output"] = formatted
        evidence["structured_output"] = grouped

        print("\n=== FINAL OUTPUT ===")
        for line in formatted[:10]:
            print(line)

        return state

    # -----------------------------
    # CASE 2: RAW PREVIEW
    # -----------------------------
    df = state.get("analysis_dataset")

    if df is not None:
        preview = df.head(10)
        evidence["final_output"] = preview.to_dict("records")
        evidence["structured_output"] = preview

        print("\n=== FINAL OUTPUT (TABLE PREVIEW) ===")
        print(preview)

    return state

'''# node/output_mode_node.py
from state.state import AnalystState
import pandas as pd


def output_mode_node(state: AnalystState) -> AnalystState:
    """
    Determines how to present results based on intent.
    Converts filtered dataset into final user-facing output.
    """

    df = state.get("analysis_dataset")
    intent = state.get("intent", {})

    if df is None:
        raise ValueError("analysis_dataset not found in state")

    state.setdefault("analysis_evidence", {})

    # -----------------------------
    # CASE 1: FILTER + GROUP BY
    # -----------------------------
    if intent.get("type") == "filter" and intent.get("group_by"):
        group_col = intent["group_by"]

        if group_col in df.columns:
            result = df[group_col].value_counts().reset_index()
            result.columns = [group_col, "count"]

            # Format for human readability
            formatted = [
                f"{row[group_col]} → {row['count']} cars"
                for _, row in result.iterrows()
            ]

            state["analysis_evidence"]["final_output"] = formatted
            state["analysis_evidence"]["structured_output"] = result

            print("\n=== FINAL OUTPUT (GROUPED) ===")
            for line in formatted[:10]:
                print(line)

            return state

    # -----------------------------
    # CASE 2: FILTER ONLY (NO GROUP)
    # -----------------------------
    if intent.get("type") == "filter":
        preview = df.head(10)

        state["analysis_evidence"]["final_output"] = preview.to_dict(orient="records")
        state["analysis_evidence"]["structured_output"] = preview

        print("\n=== FINAL OUTPUT (TABLE PREVIEW) ===")
        print(preview)

        return state

    # -----------------------------
    # DEFAULT FALLBACK
    # -----------------------------
    print("\nNo output mode matched — returning raw dataset preview.")

    preview = df.head(10)

    state["analysis_evidence"]["final_output"] = preview.to_dict(orient="records")
    state["analysis_evidence"]["structured_output"] = preview

    return state
    '''
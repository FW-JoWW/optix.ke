import pandas as pd

from nodes.categorical_analysis_node import categorical_analysis_node


def create_state(df: pd.DataFrame):
    return {
        "dataframe": df,
        "analysis_dataset": df.copy(),
        "dataset_profile": {
            "numeric_columns": ["sales", "units"],
            "categorical_columns": ["region", "channel", "status"],
        },
        "analysis_evidence": {},
        "categorical_analysis_config": {
            "rare_category_threshold": 0.15,
            "high_cardinality_threshold": 5,
        },
    }


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "region": ["North", "north", "South", "South ", None, "East", "East"],
            "channel": ["Retail", "Retail", "Online", "Online", "Retail", "Partner", "Partner"],
            "status": ["Active", "Active", "Inactive", "Inactive", "Active", "Active", "Dormant"],
            "sales": [100, 120, 80, 95, 110, 60, 30],
            "units": [10, 12, 8, 9, 11, 6, 3],
        }
    )

    state = create_state(df)
    state = categorical_analysis_node(state)

    print("\n=== SAMPLE CATEGORICAL ANALYSIS OUTPUT ===")
    print(state["analysis_evidence"]["categorical_analysis"])

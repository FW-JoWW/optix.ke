import pandas as pd
from nodes.intent_parser_node import intent_parser_node
from nodes.validation_repair_node import validation_repair_node
from nodes.row_filter_node import row_filter_node

RUN_DOWNSTREAM_STEPS = True

# -----------------------
# MOCK STATE
# -----------------------
def create_state(df, query):
    return {
        "business_question": query,
        "dataframe": df,
        "cleaned_data": None,
        "enable_llm_reasoning": False,
        "disable_semantic_matcher": True,
        "dataset_profile": {
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "string"]).columns.tolist()
        },
        "column_registry": {col: None for col in df.columns},  # for column_selection_node
        "analysis_evidence": {}
    }

# -----------------------
# SAMPLE DATASET
# -----------------------
def create_test_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        "brand": ["BMW", "Audi", "Toyota", "BMW", "Audi", "Toyota"],
        "price": [25000, 18000, 15000, 30000, 22000, 12000],
        "year": [2018, 2019, 2020, 2021, 2017, 2022]
    }
    return pd.DataFrame(data)

# -----------------------
# RUN TEST CASE
# -----------------------
def run_test(query):
    print("\n" + "="*80)
    print("QUERY:", query)
    df = create_test_dataframe()
    state = create_state(df, query)

    try:
        # STEP 1: Intent Parsing
        state = intent_parser_node(state)
        print("\n[INTENT OUTPUT]")
        print(state.get("intent"))
        print("\n[PARSER STATUS]")
        print("Type:", state.get("intent", {}).get("type"))
        print("Low confidence:", state.get("intent", {}).get("low_confidence"))
        print("Filters detected:", len(state.get("intent", {}).get("filters", [])))

        # STEP 2: Validation / Repair
        state = validation_repair_node(state)
        print("\n[VALIDATED INTENT]")
        print(state.get("intent"))
        print("\n[VALIDATION ISSUES]")
        print(state.get("validation_issues", []))

        if not state.get("intent", {}).get("filters"):
            print("\n[RESULT SUMMARY]")
            print("No executable filters were produced for this query.")
        else:
            print("\n[RESULT SUMMARY]")
            print("Executable filters were produced for this query.")

        if not RUN_DOWNSTREAM_STEPS:
            return

        # STEP 3: Row Filtering
        if state.get("intent", {}).get("filters") and not state.get("skip_filtering"):
            state = row_filter_node(state)
        else:
            print("[INFO] Skipping row filtering (no filters detected)")
            state["analysis_dataset"] = df.copy()

        print("\n[FILTERED DATA]")
        print(state["analysis_dataset"])
        print("\n[FILTERED SHAPE]")
        print(state["analysis_dataset"].shape)

        # STEP 4: Column Selection
        from nodes.column_selection_node import column_selection_node
        state = column_selection_node(state)
        print("\n[SELECTED COLUMNS]")
        print(state.get("selected_columns"))
        print("\n[SELECTED COLUMNS DATA]")
        print(state["analysis_dataset"])
        print("\n[SELECTED SHAPE]")
        print(state["analysis_dataset"].shape)

        # STEP 5: Analysis Evidence
        print("\n[ANALYSIS EVIDENCE]")
        for k, v in state["analysis_evidence"].items():
            print(f"{k}: {v}")

    except Exception as e:
        print("\n[ERROR]")
        print(e)

# -----------------------
# TEST CASES
# -----------------------
if __name__ == "__main__":
    queries = [
        "show me BMW cars under 30k and newer than 2018",
        "show me Audi cars above 18k but not too expensive",
        "cars not BMW and below 20k",
        "BMW under 30k or Audi under 20k",
        "Toyota newer than 2019 and below 16k",
        "cars between 15000 and 25000 and not Audi",
        "BMW or Audi above 20k",
        "cars below 25k and newer than 2018",
        "Audi newer than 2017 and under 23000 or BMW under 28000",
        "cheap cars not Toyota"
    ]

    for q in queries:
        run_test(q)

import pandas as pd
from nodes.intent_parser_node import intent_parser_node
from nodes.llm_reasoning_node import llm_reasoning_node
from nodes.validation_repair_node import validation_repair_node
from nodes.row_filter_node import row_filter_node
from nodes.column_selection_node import column_selection_node

# -----------------------
# MOCK STATE
# -----------------------
def create_state(df, query):
    return {
        "business_question": query,
        "dataframe": df,
        "cleaned_data": None,
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

        # STEP 2: LLM reasoning (optional)
        # state = llm_reasoning_node(state)

        # STEP 3: Validation / Repair
        state = validation_repair_node(state)

        # STEP 4: Row Filtering
        if state.get("intent", {}).get("filters") and not state.get("skip_filtering"):
            state = row_filter_node(state)
        else:
            print("[INFO] Skipping row filtering (no filters detected)")
            state["analysis_dataset"] = df.copy()

        print("\n[FILTERED DATA]")
        print(state["analysis_dataset"])

        # STEP 5: Column Selection
        state = column_selection_node(state)
        print("\n[SELECTED COLUMNS DATA]")
        print(state["analysis_dataset"])

        # STEP 6: Analysis Evidence
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
        "show me bimmers",
        "cheap cars",
        "expensive cars",
        "cars below 15k",
        "cars above 20k but not too expensive",
        "bmw under 30k or audi under 20k",
        "price between 15000 and 15000",
        "bmw newer than 2018 and under 30000",
        "cars not bmw",
        "asdfghjkl",
        "???"
    ]

    for q in queries:
        run_test(q)
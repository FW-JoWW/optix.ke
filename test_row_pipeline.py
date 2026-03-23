# test_row_filter_node.py

import pandas as pd
from state.state import AnalystState
from nodes.row_filter_node import row_filter_node


def run_test(question, df, column_registry):
    print("\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)

    state: AnalystState = {
        "analysis_dataset": df.copy(),
        "business_question": question,
        "column_registry": column_registry
    }

    state = row_filter_node(state)

    print("\n--- RESULT ---")
    print(state["analysis_dataset"])

    print("\n--- EVIDENCE ---")
    print(state["analysis_evidence"]["row_filtering"])


# -----------------------------
# DATASET
# -----------------------------
df = pd.DataFrame({
    "make": ["BMW", "Audi", "Toyota", "BMW", "Honda"],
    "price": [20000, 18000, 15000, 25000, 12000],
    "fuelType": ["Petrol", "Diesel", "Petrol", "Diesel", "Hybrid"]
})

column_registry = {
    "make": {"semantic_role": "categorical_feature"},
    "price": {"semantic_role": "numeric_measure"},
    "fuelType": {"semantic_role": "categorical_feature"},
}

# -----------------------------
# TEST CASES
# -----------------------------

# 1. BETWEEN (your original case, but correct wording)
run_test(
    "Show me cars with price between 10k and 18k",
    df,
    column_registry
)

# 2. GREATER THAN (k format)
run_test(
    "price > 20k",
    df,
    column_registry
)

# 3. LESS THAN
run_test(
    "price < 20000",
    df,
    column_registry
)

# 4. CATEGORICAL FILTER
run_test(
    "show me bmw cars",
    df,
    column_registry
)

# 5. COMBINED FILTER
run_test(
    "bmw cars with price > 20000",
    df,
    column_registry
)

# 6. OR CONDITION
run_test(
    "bmw or audi",
    df,
    column_registry
)

# 7. MIXED (OR + NUMERIC)
run_test(
    "bmw or audi with price < 20000",
    df,
    column_registry
)

# 8. FUZZY MATCH (intentional typo)
run_test(
    "show me bmws",
    df,
    column_registry
)
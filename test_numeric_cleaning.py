# test_numeric_cleaning.py

import pandas as pd
from state.state import AnalystState
from nodes.numeric_cleaning_node import numeric_cleaning_node

# -----------------------------
# Create a deliberately dirty dataset
# -----------------------------
df = pd.DataFrame({
    "price": ["$1,000", "$1,100", "950", "$1,300", "$5,000"],          # US format with $
    "mpg": ["60 mpg", "58.9", "49,6", "62.8", "unknown"],              # mixed US/Euro decimals & text
    "mileage": ["10,000", "20000", "$30,000", "unknown", "5000"],      # commas, $, text
    "engineSize": ["2.0L", "1.6", "1,8", "3.0", "unknown"],            # units, Euro decimal, missing
    "tax": ["£100", "120", "150", "unknown", "200"],                   # symbols, text
    "fuelType": ["Petrol", "Diesel", "Petrol", "Diesel", "Hybrid"],    # categorical (should remain unchanged)
})

# -----------------------------
# Create AnalystState
# -----------------------------
state: AnalystState = {
    "analysis_dataset": df,
    "dataset_profile": {
        "numeric_columns": ["price", "mpg", "mileage", "engineSize", "tax"],
    },
    "column_registry": {
        "price": {"semantic_role": "numeric_measure"},
        "mpg": {"semantic_role": "numeric_measure"},
        "mileage": {"semantic_role": "numeric_measure"},
        "engineSize": {"semantic_role": "numeric_measure"},
        "tax": {"semantic_role": "numeric_measure"},
        "fuelType": {"semantic_role": "categorical_feature"},
    },
    "business_question": "Analyze price, mileage and mpg distribution."
}

# -----------------------------
# Run numeric cleaning
# -----------------------------
state = numeric_cleaning_node(state)

# -----------------------------
# Output results
# -----------------------------
print("\n=== CLEANED DATASET ===")
print(state["analysis_dataset"])

print("\n=== DTYPES ===")
print(state["analysis_dataset"].dtypes)

print("\n=== CLEANING EVIDENCE ===")
print(state["analysis_evidence"]["numeric_cleaning"])
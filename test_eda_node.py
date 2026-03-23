# test_eda_node.py

import pandas as pd
from state.state import AnalystState
from nodes.eda_node import eda_node

# Dummy dataset
df = pd.DataFrame({
    "revenue": [1000, 1100, 950, 1300, 5000],
    "ad_spend": [100, 120, 130, 115, 500],
    "channel": ["email", "social", "email", "social", "email"]
})

state: AnalystState = {"dataset": df, "business_question": "Analyze revenue vs ad spend."}

# Run EDA node
state = eda_node(state)

import pprint
pprint.pprint(state["eda_results"])
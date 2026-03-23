# test_statistical_analysis_node.py

import pandas as pd
from state.state import AnalystState
from nodes.statistical_analysis_node import statistical_analysis_node

# Dummy dataset
df = pd.DataFrame({
    "revenue": [1000, 1100, 950, 1300, 5000],
    "ad_spend": [100, 120, 130, 115, 500],
    "channel": ["email", "social", "email", "social", "email"]
})

state: AnalystState = {"dataset": df, "business_question": "Analyze revenue vs ad spend by channel."}

# Run statistical analysis node
state = statistical_analysis_node(state)

import pprint
pprint.pprint(state["statistical_results"])
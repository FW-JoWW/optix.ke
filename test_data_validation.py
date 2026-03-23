import pandas as pd
from state.state import AnalystState
from nodes.data_validation_node import data_validation_node

# Dummy dataset
df = pd.DataFrame({
    "revenue": [1000, 1100, None, 1300, 50000],
    "ad_spend": [100, 120, 130, None, 500],
    "channel": ["email", "social", "email", "social", "email"]
})

state: AnalystState = {"dataset": df, "business_question": "Analyze revenue vs ad spend."}

# Run node
state = data_validation_node(state)

# Print results
import pprint
pprint.pprint(state["data_validation"])
pprint.pprint(state["clarification_questions"])
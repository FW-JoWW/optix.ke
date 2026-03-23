import re
from state.state import AnalystState

def extract_price_range(query: str):
    match = re.search(r'(\d+)[kK]?\s*(?:to|-)\s*(\d+)[kK]?', query)
    if match:
        low = int(match.group(1)) * (1000 if 'k' in match.group(1).lower() else 1)
        high = int(match.group(2)) * (1000 if 'k' in match.group(2).lower() else 1)
        return low, high
    return None

def intent_parser_node(state: AnalystState) -> AnalystState:
    query = state.get("business_question", "").lower()

    intent = {
        "type": None,
        "filters": [],
        "group_by": None
    }

    # ---- FILTER DETECTION ----
    if "price" in query and ("between" in query or "to" in query):
        price_range = extract_price_range(query)
        if price_range:
            intent["type"] = "filter"
            intent["filters"].append({
                "column": "price",
                "operator": "between",
                "value": price_range
            })

    # ---- GROUPING DETECTION ----
    if "make" in query:
        intent["group_by"] = "make"

    state["intent"] = intent

    print("\n=== PARSED INTENT ===")
    print(intent)

    return state
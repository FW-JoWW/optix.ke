import pandas as pd
from nodes.intent_parser_node import intent_parser_node
from nodes.llm_reasoning_node import llm_reasoning_node
from nodes.validation_repair_node import validation_repair_node
from nodes.row_filter_node import row_filter_node

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
# RUN TEST
# -----------------------
def run_test(query):
    print("\n" + "="*50)
    print("QUERY:", query)

    df = create_test_dataframe()
    state = create_state(df, query)

    try:
        # STEP 1: Intent Parsing (includes LLM now)
        state = intent_parser_node(state)

        print("\n[INTENT OUTPUT]")
        print(state["intent"])

        # STEP 2: LLM reasoning
        #state = llm_reasoning_node(state)

        # STEP : Validation
        state = validation_repair_node(state)

        # STEP : Row Filtering
        if state.get("intent", {}).get("filters") and not state.get("skip_filtering"):
            state = row_filter_node(state)
        else:
            print("[INFO] Skipping row filtering due to empty/invalid filters")
            state["analysis_dataset"] = pd.DataFrame()  # safe empty dataset

        print("\n[FILTERED DATA]")
        print(state["analysis_dataset"])

        print("\n[ANALYSIS EVIDENCE]")
        print(state["analysis_evidence"])

    except Exception as e:
        print("\n[ERROR]")
        print(e)

# -----------------------
# TEST CASES
# -----------------------
if __name__ == "__main__":
    queries = [

    # -------------------------
    # 🧠 FUZZY + HUMAN LANGUAGE
    # -------------------------
    "show me bimmers",
    "cheap cars",
    "expensive cars",
    "cars that are kinda affordable",
    "not too old cars",
    "recent cars",

    # -------------------------
    # 💸 PRICE CHAOS
    # -------------------------
    "cars below 15k",
    "cars under 15000",
    "cars less than 15,000",
    "cars cheaper than 15k",
    "cars above 20k but not too expensive",
    "cars around 20k",

    # -------------------------
    # 🔀 MIXED LOGIC (HUMAN STYLE)
    # -------------------------
    "bmw under 30k or audi under 20k",
    "bmw and audi below 25k",
    "bmw or audi or toyota under 20000",
    "bmw but cheap",
    "audi but not too old",

    # -------------------------
    # 🤯 CONTRADICTIONS (REAL USERS DO THIS)
    # -------------------------
    "cars above 20k and below 15k",
    "cheap expensive cars",
    "bmw under 10k but above 30k",

    # -------------------------
    # 🧩 PARTIAL / INCOMPLETE
    # -------------------------
    "bmw price",
    "audi year",
    "toyota cost",
    "cars price",
    "just show me cars",

    # -------------------------
    # 🔤 TYPO / DIRTY INPUT
    # -------------------------
    "bmww",
    "auid",
    "toyata",
    "bmw wit price 20k",
    "audii below 20000",

    # -------------------------
    # 🔢 RANGE EDGE CASES
    # -------------------------
    "price between 15000 and 15000",
    "price between 30000 and 10000",
    "price equal 20000",
    "price exactly 20000",

    # -------------------------
    # 🧠 IMPLICIT INTENT
    # -------------------------
    "which cars can i afford with 20k",
    "best cars for 15000",
    "what cars are in my budget 20000",
    "cars i can buy under 18k",

    # -------------------------
    # 🔀 COMPLEX COMBINATIONS
    # -------------------------
    "bmw newer than 2018 and under 30000",
    "audi between 15000 and 25000 but newer than 2018",
    "toyota or audi between 12000 and 22000",
    "bmw under 30k and not older than 2020",

    # -------------------------
    # 🧨 EDGE SEMANTIC FAILURES
    # -------------------------
    "cars not bmw",
    "anything except audi",
    "non toyota cars",
    "cars that are not expensive",

    # -------------------------
    # 🧪 EMPTY / NONSENSE
    # -------------------------
    "",
    "asdfghjkl",
    "???",
]

    for q in queries:
        run_test(q)
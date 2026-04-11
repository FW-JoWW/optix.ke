import pandas as pd

from nodes.intent_parser_node import intent_parser_node
from nodes.validation_repair_node import validation_repair_node
from nodes.row_filter_node import row_filter_node
from nodes.initialize_analysis_evidence_node import initialize_analysis_evidence_node
from nodes.analysis_planner_node import analysis_planner_node
from nodes.tool_executor_node import tool_executor_node


def create_test_dataframe():
    return pd.DataFrame(
        {
            "brand": [
                "BMW", "BMW", "BMW", "Audi", "Audi", "Audi",
                "Toyota", "Toyota", "Toyota", "Honda", "Honda", "Honda"
            ],
            "price": [
                25000, 30000, 32000, 18000, 22000, 24000,
                15000, 16000, 17000, 14000, 15500, 16500
            ],
            "year": [
                2018, 2020, 2021, 2017, 2019, 2020,
                2019, 2020, 2022, 2018, 2021, 2022
            ],
            "mileage": [
                60000, 50000, 45000, 80000, 70000, 65000,
                40000, 35000, 30000, 55000, 42000, 38000
            ],
            "fuelType": [
                "Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "Diesel",
                "Hybrid", "Petrol", "Hybrid", "Petrol", "Hybrid", "Petrol"
            ],
        }
    )


def create_state(df, query):
    return {
        "business_question": query,
        "dataframe": df,
        "cleaned_data": None,
        "enable_llm_reasoning": False,
        "disable_semantic_matcher": True,
        "dataset_profile": {
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(include=["object", "string"]).columns.tolist(),
        },
        "column_registry": {
            col: {"semantic_role": "feature"} for col in df.columns
        },
        "analysis_evidence": {},
        "intent": {},
    }


def run_test(query):
    print("\n" + "=" * 100)
    print("QUERY:", query)

    df = create_test_dataframe()
    state = create_state(df, query)

    try:
        state = intent_parser_node(state)
        print("\n[INTENT OUTPUT]")
        print(state.get("intent"))

        state = validation_repair_node(state)
        print("\n[VALIDATED INTENT]")
        print(state.get("intent"))
        print("\n[VALIDATION ISSUES]")
        print(state.get("validation_issues", []))

        if state.get("intent", {}).get("filters"):
            state = row_filter_node(state)
        else:
            state["analysis_dataset"] = df.copy()
            state["raw_analysis_dataset"] = df.copy()
            print("\n[INFO] No row filtering needed for this query")

        print("\n[ANALYSIS DATA SHAPE]")
        print(state.get("analysis_dataset").shape if state.get("analysis_dataset") is not None else None)
        print("\n[ANALYSIS DATA PREVIEW]")
        print(state.get("analysis_dataset").head(10))

        state = initialize_analysis_evidence_node(state)
        state = analysis_planner_node(state)

        print("\n[OUTPUT MODE]")
        print(state.get("output_mode"))
        print("\n[ANALYSIS PLAN]")
        print(state.get("analysis_evidence", {}).get("analysis_plan"))

        state = tool_executor_node(state)

        print("\n[TOOL RESULTS]")
        print(state.get("analysis_evidence", {}).get("tool_results"))

    except Exception as e:
        print("\n[ERROR]")
        print(e)


if __name__ == "__main__":
    queries = [
        "relationship between price and mileage",
        "correlation between price and year",
        "does brand affect price",
        "compare price by brand",
        "difference in price between fuelType groups",
        "average price and mileage",
        "summary statistics for price and mileage",
        "compare mileage by brand",
        "impact of year on price",
        "outliers in price",
        "distribution of brand",
        "mode of fuelType",
        "cardinality of brand",
        "rare fuelType categories",
        "relationship between brand and fuelType",
        "average price by fuelType",
    ]

    for q in queries:
        run_test(q)

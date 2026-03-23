import pandas as pd

from nodes.data_intake_node import data_intake_node
from nodes.dataset_profiler_node import dataset_profiler_node
from nodes.column_semantic_classifier_node import column_semantic_classifier_node
from nodes.column_selection_node import column_selection_node
from nodes.numeric_cleaning_node import numeric_cleaning_node


def run_pipeline():

    state = {}

    # -----------------------------
    # dataset path
    # -----------------------------
    state["dataset_path"] = "data/data_set.csv"

    # -----------------------------
    # business question
    # -----------------------------
    state["business_question"] = "compair fuel type and mpg and mileage and price?"

    print("\n==============================")
    print("RUNNING ANALYST PIPELINE")
    print("==============================")

    # Step 1
    state = data_intake_node(state)

    # Step 2
    state = dataset_profiler_node(state)

    # Step 3
    state = column_semantic_classifier_node(state)

    # Step 4
    state = column_selection_node(state)

    # Step 5
    state = numeric_cleaning_node(state)

    print("\n==============================")
    print("FINAL OUTPUT")
    print("==============================")

    print("\nSelected Columns:")
    print(state["selected_columns"])

    print("\nAnalysis Dataset:")
    print(state["analysis_dataset"].head())


if __name__ == "__main__":
    run_pipeline()
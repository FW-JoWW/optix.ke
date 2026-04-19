import pandas as pd
from sqlalchemy import create_engine
from state.state import AnalystState
from typing import Optional
from utils.structure_normalizer import choose_best_dataframe

def load_csv(state: AnalystState, file_path: str) -> AnalystState:
    """
    Load a CSV file into the state.
    """
    standard_df = pd.read_csv(file_path)
    raw_df = pd.read_csv(file_path, header=None)
    normalized = choose_best_dataframe(standard_df, raw_df)
    state["dataframe"] = normalized.dataframe
    state["dataset_path"] = file_path
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["structure_normalization"] = {
        "applied": normalized.applied,
        "strategy": normalized.strategy,
        "details": normalized.details,
    }
    return state


def load_excel(state: AnalystState, file_path: str) -> AnalystState:
    """
    Load an Excel file into the state.
    """
    standard_df = pd.read_excel(file_path)
    raw_df = pd.read_excel(file_path, header=None)
    normalized = choose_best_dataframe(standard_df, raw_df)
    state["dataframe"] = normalized.dataframe
    state["dataset_path"] = file_path
    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["structure_normalization"] = {
        "applied": normalized.applied,
        "strategy": normalized.strategy,
        "details": normalized.details,
    }
    return state


def load_sql_table(state: AnalystState, connection_string: str, table_name: str) -> AnalystState:
    """
    Load a SQL table into the state. (Postgres example)
    """
    engine = create_engine(connection_string)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    state["dataframe"] = df
    state["dataset_path"] = f"SQL:{table_name}"
    return state


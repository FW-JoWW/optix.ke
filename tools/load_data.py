import pandas as pd
from sqlalchemy import create_engine
from state.state import AnalystState
from typing import Optional

def load_csv(state: AnalystState, file_path: str) -> AnalystState:
    """
    Load a CSV file into the state.
    """
    df = pd.read_csv(file_path)
    state["dataframe"] = df
    state["dataset_path"] = file_path
    return state


def load_excel(state: AnalystState, file_path: str) -> AnalystState:
    """
    Load an Excel file into the state.
    """
    df = pd.read_excel(file_path)
    state["dataframe"] = df
    state["dataset_path"] = file_path
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


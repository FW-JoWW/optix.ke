from state.state import AnalystState
import pandas as pd

def correlation_tool(df: pd.DataFrame, columns: list) -> dict:
    
    if len(columns) < 2:
        return {"error": "Correlation requires two columns"}

    col1, col2 = columns[0], columns[1]
    
    correlation = df[col1].corr(df[col2])
    
    return {
        "tool": "correlation",
        "column_1": col1,
        "column_2": col2,
        "correlation": float(correlation)
    }


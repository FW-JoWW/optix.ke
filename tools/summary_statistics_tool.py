import pandas as pd

def summary_statistics_tool(df: pd.DataFrame, columns: list) -> dict:

    if len(columns) < 1:
        return {"error": "Summary statistics requires at least one column"}

    results = []

    for col in columns:

        series = df[col].dropna()

        results.append({
            "column": col,
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std_dev": float(series.std()),
            "variance": float(series.var()),
            "min": float(series.min()),
            "max": float(series.max())
        })

    return {
        "tool": "summary_statistics",
        "results": results
    }
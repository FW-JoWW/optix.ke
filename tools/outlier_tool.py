
import pandas as pd

def outlier_tool(df: pd.DataFrame, columns: list):

    if len(columns) < 1:
        return {"error": "Outlier detection requires a column"}

    col = columns[0]

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    return {
        "type": "outlier_detection",
        "column": col,
        "outlier_count": int(len(outliers)),
        "lower_bound": float(lower),
        "upper_bound": float(upper)
    }


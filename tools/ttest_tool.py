import pandas as pd
from scipy import stats

def ttest_tool(df: pd.DataFrame, columns: list):

    if len(columns) < 2:
        return {"error": "T-test requires numeric column and group column"}

    value_col, group_col = columns[0], columns[1]

    groups = df[group_col].dropna().unique()

    if len(groups) != 2:
        return {"error": "T-test requires exactly 2 groups"}

    g1 = df[df[group_col] == groups[0]][value_col]
    g2 = df[df[group_col] == groups[1]][value_col]

    t_stat, p_value = stats.ttest_ind(g1, g2)

    return {
        "type": "t_test",
        "column": value_col,
        "group_column": group_col,
        "groups": list(groups),
        "t_stat": float(t_stat),
        "p_value": float(p_value)
    }


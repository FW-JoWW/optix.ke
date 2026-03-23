from scipy.stats import f_oneway

def anova_tool(df, numeric_col, categorical_col):

    groups = df.groupby(categorical_col)[numeric_col].apply(list)

    if len(groups) < 2:
        return None

    f_stat, p_value = f_oneway(*groups)

    return {
        "tool": "anova",
        "numeric_column": numeric_col,
        "categorical_column": categorical_col,
        "f_statistic": float(f_stat),
        "p_value": float(p_value)
    }
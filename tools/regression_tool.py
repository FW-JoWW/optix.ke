from sklearn.linear_model import LinearRegression
from state.state import AnalystState

def regression_tool(state: AnalystState, x_col: str, y_col: str) -> AnalystState:

    df = state["cleaned_data"] if state.get("cleaned_data") is not None else state["dataframe"]

    X = df[[x_col]].dropna()
    y = df[y_col].dropna()

    X = X.loc[y.index]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    result = {
        "tool": "regression",
        "x_column": x_col,
        "y_column": y_col,
        "r_squared": float(r2)
    }

    if state.get("tool_results") is None:
        state["tool_results"] = []

    state["tool_results"].append(result)

    return state
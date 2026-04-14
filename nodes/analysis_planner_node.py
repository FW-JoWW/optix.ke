from typing import Dict, List

from state.state import AnalystState


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _pairwise_numeric_plans(columns: List[str], tool: str) -> List[Dict]:
    plans: List[Dict] = []
    for idx, col1 in enumerate(columns):
        for col2 in columns[idx + 1:]:
            plans.append({"tool": tool, "columns": [col1, col2]})
    return plans


def _numeric_by_categorical_plans(
    numeric_columns: List[str],
    categorical_columns: List[str],
    unique_counts: Dict[str, int],
) -> List[Dict]:
    plans: List[Dict] = []
    for num in numeric_columns:
        for cat in categorical_columns:
            n_unique = unique_counts.get(cat, 0)
            if n_unique == 2:
                tool = "ttest"
            elif n_unique > 2:
                tool = "anova"
            else:
                continue
            plans.append({"tool": tool, "columns": [num, cat]})
    return plans


def analysis_planner_node(state: AnalystState) -> AnalystState:
    """
    Determines:
    - What analysis to run
    - What output mode to use
    """
    intent = state.get("intent", {})
    evidence = state.setdefault("analysis_evidence", {})

    state["output_mode"] = "analysis"

    if evidence.get("grouped_summary") is not None:
        state["output_mode"] = "grouped_summary"

    if intent.get("type") == "filter" and not intent.get("wants_analysis"):
        if not intent.get("group_by"):
            state["output_mode"] = "raw_filter"

    print("\n=== OUTPUT MODE ===")
    print(state["output_mode"])

    if state["output_mode"] in ["raw_filter", "grouped_summary"]:
        evidence["analysis_plan"] = []
        print("\n=== ANALYSIS PLAN ===")
        print("No statistical analysis required.")
        return state

    question = state.get("business_question", "").lower()
    if not intent:
        raise ValueError("Intent missing before analysis planning")

    if "cleaned_data" in state and state["cleaned_data"] is not None:
        state["active_dataset"] = "cleaned_data"
    else:
        state["active_dataset"] = "dataframe"

    df = state.get("analysis_dataset")
    if df is None:
        df = state.get(state.get("active_dataset"))

    if df is None:
        raise ValueError("No analysis dataset found.")

    dataset_profile = state.get("dataset_profile", {})
    column_registry = state.get("column_registry", {})
    selected_columns = state.get("selected_columns", list(df.columns))

    numeric_cols = [
        c for c in dataset_profile.get("numeric_columns", [])
        if c in selected_columns
        and column_registry.get(c, {}).get("semantic_role") != "identifier"
    ]
    categorical_cols = [
        c for c in dataset_profile.get("categorical_columns", [])
        if c in selected_columns
        and column_registry.get(c, {}).get("semantic_role") != "identifier"
    ]
    unique_counts = {col: df[col].nunique() for col in df.columns}

    plan: List[Dict] = []

    mentioned_columns = [col for col in selected_columns if col.lower() in question]
    mentioned_numeric = [c for c in mentioned_columns if c in numeric_cols]
    mentioned_categorical = [c for c in mentioned_columns if c in categorical_cols]
    target_numeric = mentioned_numeric or numeric_cols
    target_categorical = mentioned_categorical or categorical_cols

    relationship_words = ["relationship", "correlation", "affect", "impact"]
    comparison_words = ["compare", "difference", "better"]
    categorical_words = [
        "distribution",
        "frequency",
        "mode",
        "cardinality",
        "rare",
        "category",
        "categories",
        "categorical",
        "contingency",
        "chi-square",
        "chi square",
        "independence",
    ]
    grouped_numeric_words = ["average", "mean", "median", "summary", "describe"]
    grouped_numeric_query = bool(mentioned_numeric and mentioned_categorical and _contains_any(question, grouped_numeric_words))

    if _contains_any(question, relationship_words):
        if len(target_numeric) >= 2 and not target_categorical:
            plan.extend(_pairwise_numeric_plans(target_numeric, "correlation"))
        elif target_numeric and target_categorical:
            plan.extend(
                _numeric_by_categorical_plans(
                    target_numeric,
                    target_categorical,
                    unique_counts,
                )
            )
        elif len(target_categorical) >= 2:
            plan.append({"tool": "categorical_analysis", "columns": target_categorical})

    if _contains_any(question, comparison_words):
        if target_numeric and target_categorical:
            plan.extend(
                _numeric_by_categorical_plans(
                    target_numeric,
                    target_categorical,
                    unique_counts,
                )
            )
        elif not target_numeric and len(target_categorical) >= 1:
            plan.append({"tool": "categorical_analysis", "columns": target_categorical})
        else:
            plan.extend(
                _numeric_by_categorical_plans(
                    numeric_cols,
                    categorical_cols,
                    unique_counts,
                )
            )

    if _contains_any(question, ["outlier", "unusual", "anomaly"]):
        for num in numeric_cols:
            plan.append({"tool": "detect_outliers", "columns": [num]})

    if _contains_any(question, grouped_numeric_words):
        if numeric_cols:
            if grouped_numeric_query:
                plan.append({"tool": "categorical_analysis", "columns": target_categorical})
            else:
                plan.append({"tool": "summary_statistics", "columns": numeric_cols})
        if target_categorical and not grouped_numeric_query:
            plan.append({"tool": "categorical_analysis", "columns": target_categorical})

    if _contains_any(question, categorical_words):
        categorical_targets = target_categorical or categorical_cols[:2]
        if categorical_targets:
            plan.append({"tool": "categorical_analysis", "columns": categorical_targets})

    if not plan:
        if len(target_categorical) >= 1 and not target_numeric:
            plan.append({"tool": "categorical_analysis", "columns": target_categorical})
        elif target_numeric and target_categorical:
            plan.extend(
                _numeric_by_categorical_plans(
                    target_numeric,
                    target_categorical,
                    unique_counts,
                )
            )
        elif len(numeric_cols) >= 2:
            plan.append({"tool": "correlation", "columns": numeric_cols[:2]})

    seen = set()
    unique_plan = []
    for item in plan:
        key = (item["tool"], tuple(item["columns"]))
        if key not in seen:
            unique_plan.append(item)
            seen.add(key)

    evidence["analysis_plan"] = unique_plan

    print("\n=== ANALYSIS PLAN ===")
    print("Dataset shape:", df.shape)
    for p in unique_plan:
        print("-", p)

    return state

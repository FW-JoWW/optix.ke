from typing import Dict, List

from decision_engine import run_decision_engine
from intent_alignment import validate_analysis_plan_against_intent
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


def _dedupe_preserve_order(columns: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for col in columns:
        if col and col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


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

    relationship_words = ["relationship", "correlation", "affect", "impact", "cause", "causal", "drive"]
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

    unique_plan = validate_analysis_plan_against_intent(
        question=question,
        intent=intent,
        plan=unique_plan,
        selected_columns=selected_columns,
    )

    decision_output = run_decision_engine(
        dataset_profile=dataset_profile,
        structural_signals=state.get("structural_signals", {}),
        inferred_context=state.get("context_inference", {}),
        relationship_signals=state.get("relationship_signals", {}),
        user_intent={**intent, "query": question},
        constraint_rules=state.get("cleaning_constraints", {}),
        candidate_plan=unique_plan,
        selected_columns=selected_columns,
    )
    state["decision_output"] = decision_output.model_dump()
    evidence["computation_plan"] = decision_output.computation_plan.model_dump()
    unique_plan = [
        {
            "tool": item.tool,
            "columns": item.columns,
            "parameters": item.parameters,
            "computation_refs": item.computation_refs,
        }
        for item in decision_output.analysis_plan.operations
        if item.valid
    ]

    evidence["analysis_plan"] = unique_plan
    evidence["analysis_decisions"] = decision_output.analysis_plan.model_dump()
    evidence["decision_notes"] = decision_output.decision_notes

    required_columns: List[str] = []
    for step in decision_output.computation_plan.steps:
        if step.column:
            required_columns.append(step.column)
        required_columns.extend(step.columns)
        group_by = (step.parameters or {}).get("group_by")
        within = (step.parameters or {}).get("within")
        if group_by:
            required_columns.append(group_by)
        if within:
            required_columns.append(within)
    for item in unique_plan:
        required_columns.extend(item.get("columns", []))

    required_columns = _dedupe_preserve_order(selected_columns + required_columns)
    available_columns = [col for col in required_columns if col in df.columns]
    if available_columns:
        state["selected_columns"] = available_columns
        state["analysis_dataset"] = df[available_columns].copy()

    print("\n=== ANALYSIS PLAN ===")
    print("Dataset shape:", df.shape)
    print("Strategy:", decision_output.analysis_plan.analytical_strategy)
    print("Confidence:", decision_output.analysis_plan.confidence_score)
    print("Computation plan:", decision_output.computation_plan.model_dump())
    for p in unique_plan:
        print("-", p)

    return state

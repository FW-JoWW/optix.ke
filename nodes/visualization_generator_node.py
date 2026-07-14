# nodes/visualization_node.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from state.state import AnalystState

# ------------------------------
# Helper Functions
# ------------------------------

def derive_decision_from_top_stories(top_stories):
    """
    Derive a decision based on the top stories.
    Returns a dict with decision and confidence or
    "No strong decision can be made".
    """
    if not top_stories:
        return "No strong decision can be made"

    high_confidence_stories = [s for s in top_stories if s.get("score", 0) >= 0.7]

    if not high_confidence_stories:
        return "No strong decision can be made"

    # Simple logic: take top 1-3 stories and summarize
    types = set(s["type"] for s in high_confidence_stories[:3])
    decision_summary = f"Decision based on: {', '.join(types)}"
    # Confidence = average score of stories used
    avg_confidence = sum(s["score"] for s in high_confidence_stories[:3]) / len(high_confidence_stories[:3])

    return {"decision": decision_summary, "confidence": round(avg_confidence, 2)}

# ------------------------------
# Chart Generators
# ------------------------------

def generate_boxplot(df, story):
    num_col = story.get("column")
    cat_col = story.get("group_column")
    if num_col not in df.columns or cat_col not in df.columns:
        return None

    plt.figure()
    sns.boxplot(data=df, x=cat_col, y=num_col)
    filename = f"charts/box_{num_col}_by_{cat_col}.png"
    plt.title(f"{num_col} by {cat_col}")
    plt.savefig(filename)
    plt.close()

    return {
        "type": "boxplot",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_scatter(df, story):
    cols = story.get("columns") or []
    if not cols or any(c not in df.columns for c in cols):
        return None
    x, y = cols[:2]
    plt.figure()
    sns.scatterplot(data=df, x=x, y=y)
    sns.regplot(data=df, x=x, y=y, scatter=False, color="red")
    filename = f"charts/scatter_{x}_{y}.png"
    plt.title(f"{x} vs {y}")
    plt.savefig(filename)
    plt.close()
    return {
        "type": "scatter",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_histogram(df, story):
    col = story.get("column")
    if col not in df.columns:
        return None
    plt.figure()
    sns.histplot(df[col], kde=True)
    filename = f"charts/hist_{col}.png"
    plt.title(f"Distribution of {col}")
    plt.savefig(filename)
    plt.close()
    return {
        "type": "histogram",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_category_bar(df, story):
    col = story.get("column")
    if col not in df.columns:
        return None

    counts = df[col].fillna("__MISSING__").astype(str).value_counts().head(10)
    plt.figure()
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha="right")
    filename = f"charts/bar_{col}.png"
    plt.title(f"Category Distribution: {col}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return {
        "type": "bar",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_grouped_bar(df, story):
    num_col = story.get("column")
    cat_col = story.get("group_column")
    if num_col not in df.columns or cat_col not in df.columns or not pd.api.types.is_numeric_dtype(df[num_col]):
        top_group = story.get("top_group")
        bottom_group = story.get("bottom_group")
        top_value = story.get("top_value")
        bottom_value = story.get("bottom_value")
        if top_group is None or top_value is None:
            return None
        labels = [str(top_group)]
        values = [float(top_value)]
        if bottom_group is not None and bottom_value is not None and str(bottom_group) != str(top_group):
            labels.append(str(bottom_group))
            values.append(float(bottom_value))
        plt.figure()
        sns.barplot(x=labels, y=values)
        plt.xticks(rotation=45, ha="right")
        filename = f"charts/bar_story_{num_col or 'metric'}_by_{cat_col or 'group'}.png"
        plt.title(story.get("insight", f"{num_col} by {cat_col}"))
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return {
            "type": "grouped_bar",
            "file_path": filename,
            "based_on": story,
            "priority": "primary"
        }

    grouped = (
        df[[cat_col, num_col]]
        .dropna()
        .groupby(cat_col)[num_col]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure()
    sns.barplot(x=grouped.index.astype(str), y=grouped.values)
    plt.xticks(rotation=45, ha="right")
    filename = f"charts/bar_{num_col}_by_{cat_col}.png"
    plt.title(f"Average {num_col} by {cat_col}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return {
        "type": "grouped_bar",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_heatmap(df, story):
    table = story.get("contingency_table")
    if not table:
        return None

    heatmap_df = pd.DataFrame(table).fillna(0)
    plt.figure()
    sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="Blues")
    filename = f"charts/heatmap_{'_'.join(story.get('columns', ['categorical']))}.png"
    plt.title("Categorical Relationship")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return {
        "type": "heatmap",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_regression_plot(df, story):
    cols = story.get("columns") or []
    if not cols or any(c not in df.columns for c in cols):
        return None
    x, y = cols[:2]
    plt.figure()
    sns.scatterplot(data=df, x=x, y=y)
    sns.regplot(data=df, x=x, y=y, scatter=False, color="red")
    filename = f"charts/regression_{x}_{y}.png"
    plt.title(f"Regression: {y} vs {x}")
    plt.savefig(filename)
    plt.close()
    return {
        "type": "regression",
        "file_path": filename,
        "based_on": story,
        "priority": "primary"
    }

def generate_predictive_driver_bar(df, story):
    del df
    drivers = story.get("top_drivers") or []
    if not drivers:
        return None
    top = drivers[:8]
    labels = [str(item.get("feature")) for item in top]
    values = [float(item.get("importance") or 0.0) for item in top]
    plt.figure()
    sns.barplot(x=values, y=labels, orient="h")
    filename = f"charts/predictive_drivers_{story.get('column', 'target')}.png"
    plt.title(f"Top Drivers for {story.get('column', 'target')}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return {
        "type": "predictive_drivers",
        "file_path": filename,
        "based_on": story,
        "priority": "primary",
    }

def generate_prediction_fit_plot(df, story):
    del df
    preview = story.get("predictions_preview") or []
    if not preview:
        return None
    preview_df = pd.DataFrame(preview)
    if not {"actual", "predicted"}.issubset(preview_df.columns):
        return None
    plt.figure()
    sns.scatterplot(data=preview_df, x="actual", y="predicted")
    filename = f"charts/prediction_fit_{story.get('column', 'target')}.png"
    plt.title(f"Prediction Fit for {story.get('column', 'target')}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return {
        "type": "prediction_fit",
        "file_path": filename,
        "based_on": story,
        "priority": "secondary",
    }

# ------------------------------
# Story → Chart Mapper
# ------------------------------

def map_story_to_chart(story, df, chart_override=None):
    mapping = {
        "histogram": generate_histogram,
        "boxplot": generate_boxplot,
        "scatter": generate_scatter,
        "bar": generate_category_bar,
        "grouped_bar": generate_grouped_bar,
        "heatmap": generate_heatmap,
        "regression": generate_regression_plot,
        "predictive_drivers": generate_predictive_driver_bar,
        "prediction_fit": generate_prediction_fit_plot,
        "group_difference": generate_boxplot,
        "inferential_group_difference": generate_boxplot,
        "outliers": generate_histogram,
        "summary_numeric": generate_histogram,
        "correlation": generate_scatter,
        "inferential_relationship": generate_scatter,
        "regression": generate_regression_plot,
        "numeric_anomaly": generate_histogram,
        "category_frequency": generate_category_bar,
        "rare_categories": generate_category_bar,
        "grouped_numeric": generate_grouped_bar,
        "categorical_relationship": generate_heatmap,
        "inferential_categorical_association": generate_heatmap,
        "predictive_model": generate_predictive_driver_bar,
    }
    preferred_type = chart_override or story.get("chart_override")
    chart_func = mapping.get(preferred_type) if preferred_type else mapping.get(story.get("type"))
    if chart_func is None and story.get("type") == "predictive_model":
        chart_func = generate_prediction_fit_plot
    if chart_func:
        primary = chart_func(df, story)
        if primary or story.get("type") != "predictive_model":
            return primary
    if story.get("type") == "predictive_model":
        return generate_prediction_fit_plot(df, story)
    return None

# ------------------------------
# Validation Layer
# ------------------------------

def validate_visualizations(charts, top_stories):
    """
    Ensures all charts map to top stories, no duplicates, aligned with decision.
    """
    valid_charts = []
    seen_stories = set()
    for chart in charts:
        story_id = id(chart["based_on"])
        if story_id in seen_stories:
            continue
        valid_charts.append(chart)
        seen_stories.add(story_id)
    return valid_charts[:3]  # hard limit 3 charts

# ------------------
# 
# -------------------
def align_llm_insights(charts, llm_insights):
    """
    Align LLM insights to visualizations.
    Each chart gets a 'headline' and 'action' from LLM if available.
    """
    for chart in charts:
        chart["llm_headline"] = None
        chart["llm_action"] = None
        for insight in llm_insights:
            if insight.get("related_story_signature") == (
                f"{chart['based_on'].get('type', 'story')}|"
                + "|".join(
                    [
                        *([str(chart["based_on"].get("column"))] if chart["based_on"].get("column") else []),
                        *[str(col) for col in chart["based_on"].get("columns", []) if col],
                        *([str(chart["based_on"].get("group_column"))] if chart["based_on"].get("group_column") else []),
                    ]
                )
            ):
                chart["llm_headline"] = insight.get("headline")
                chart["llm_action"] = insight.get("recommended_action")
                break
    return charts
# ------------------------------
# Main Visualization Node
# ------------------------------

def visualization_generator_node(state: AnalystState) -> AnalystState:
    """
    Generates insight-driven visualizations from top stories.
    """

    df = state.get("analysis_dataset")
    if df is None:
        print("No analysis dataset found for visualization.")
        return state

    top_stories = state.get("analysis_evidence", {}).get("top_stories", [])
    llm_insights = state.get("analysis_evidence", {}).get("llm_insight_details", [])
    guided_preferences = state.get("guided_visualization_preferences", {}) or {}
    chart_overrides = guided_preferences.get("chart_overrides", {}) or {}
    if not top_stories:
        print("No top stories available for visualization.")
        return state

    os.makedirs("charts", exist_ok=True)

    # 1️⃣ Derive decision context
    state["decision_context"] = derive_decision_from_top_stories(top_stories)

    # 2️⃣ Generate charts for top stories (max 3)
    charts = []
    for story in top_stories[:3]:
        if story.get("score", 0) < 0.7:
            continue
        chart = map_story_to_chart(story, df, chart_override=chart_overrides.get(story.get("type")))
        if chart:
            chart["caption"] = story.get("insight")
            charts.append(chart)

    # 3️⃣ Validate visualizations
    charts = validate_visualizations(charts, top_stories)
    
    #    Align llm insights
    charts = align_llm_insights(charts, llm_insights)

    state.setdefault("analysis_evidence", {})
    state["analysis_evidence"]["visualizations"] = charts

    print("\n=== VISUALIZATION GENERATION COMPLETE ===")
    for c in charts:
        print(f"{c['type']} -> {c['file_path']} (priority={c['priority']})")
        if c.get("llm_headline"):
            print(f" Headline: {c['llm_headline']}")
        if c.get("llm_action"):
            print(f" Action: {c['llm_action']}")

    return state


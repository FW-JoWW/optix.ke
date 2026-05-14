from __future__ import annotations

import numpy as np
import pandas as pd

from predictive.predictive_engine import run_predictive_analysis
from prescriptive.prescriptive_engine import run_prescriptive_analysis
from analysis_engine import execute_analysis_plan


def test_sales_regression() -> None:
    rng = np.random.default_rng(42)
    rows = 240
    df = pd.DataFrame(
        {
            "marketing_spend": rng.normal(100, 15, rows),
            "price": rng.normal(25, 4, rows),
            "region": rng.choice(["north", "south", "west"], rows),
        }
    )
    df["sales"] = 500 + (df["marketing_spend"] * 3.2) - (df["price"] * 6.5) + rng.normal(0, 20, rows)
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "sales"}},
        {"business_question": "predict future sales", "selected_columns": list(df.columns)},
    )
    assert result["problem_type"] == "regression"
    assert result["chosen_model"]
    assert result["confidence"]["score"] < 100
    assert result["validation_summary"]["cross_validation"]["folds"] >= 3
    print("SALES REGRESSION OK:", result["chosen_model"], result["metrics"])


def test_churn_classification() -> None:
    rng = np.random.default_rng(43)
    rows = 260
    tenure = rng.integers(1, 36, rows)
    tickets = rng.integers(0, 8, rows)
    charge = rng.normal(70, 12, rows)
    churn = ((tickets > 4) | (tenure < 6) | (charge > 82)).astype(int)
    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "support_tickets": tickets,
            "monthly_charge": charge,
            "churn_flag": churn,
        }
    )
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "churn_flag"}},
        {"business_question": "predict churn risk", "selected_columns": list(df.columns)},
    )
    assert result["problem_type"] == "classification"
    assert result["metrics"]["values"]["f1"] is not None
    assert result["validation_summary"]["diagnostics"]["confusion_matrix"]["matrix"]
    print("CHURN CLASSIFICATION OK:", result["chosen_model"], result["metrics"])


def test_demand_forecasting() -> None:
    rng = np.random.default_rng(44)
    rows = 180
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    seasonality = np.sin(np.arange(rows) / 12.0) * 20
    demand = 200 + seasonality + rng.normal(0, 5, rows)
    df = pd.DataFrame({"date": dates, "demand": demand})
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "demand"}},
        {"business_question": "forecast future demand", "selected_columns": list(df.columns)},
    )
    assert result["problem_type"] == "forecasting"
    assert result["metrics"]["values"]["mape"] is not None
    assert result["validation_summary"]["cross_validation"]["folds"] >= 3
    print("DEMAND FORECASTING OK:", result["chosen_model"], result["metrics"])


def test_customer_risk_classification() -> None:
    rng = np.random.default_rng(45)
    rows = 220
    utilization = rng.uniform(0.1, 1.0, rows)
    missed = rng.integers(0, 5, rows)
    balance = rng.normal(4000, 800, rows)
    risk = ((utilization > 0.75) | (missed >= 2) | (balance > 4700)).astype(int)
    df = pd.DataFrame(
        {
            "credit_utilization": utilization,
            "missed_payments": missed,
            "balance": balance,
            "high_risk": risk,
        }
    )
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "high_risk"}},
        {"business_question": "predict customer risk", "selected_columns": list(df.columns)},
    )
    assert result["problem_type"] == "classification"
    print("CUSTOMER RISK OK:", result["chosen_model"], result["metrics"])


def test_budget_optimization_scenario() -> None:
    rng = np.random.default_rng(46)
    rows = 220
    df = pd.DataFrame(
        {
            "ad_spend": rng.normal(120, 20, rows),
            "discount_rate": rng.uniform(0.0, 0.2, rows),
            "region": rng.choice(["east", "west", "central"], rows),
        }
    )
    df["revenue"] = 1000 + (df["ad_spend"] * 4.5) - (df["discount_rate"] * 900) + rng.normal(0, 35, rows)
    predictive = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "revenue"}},
        {"business_question": "predict revenue and optimize budget allocation", "selected_columns": list(df.columns)},
    )
    prescriptive = run_prescriptive_analysis(predictive, "optimize budget allocation for revenue growth")
    assert prescriptive["recommended_actions"]
    assert prescriptive["estimated_upside"] is not None
    assert prescriptive["decision_paths"]
    assert prescriptive["recommended_actions"][0]["estimated_uplift_range"]
    print("BUDGET OPTIMIZATION OK:", prescriptive["estimated_upside"], prescriptive["recommended_actions"][0]["action"])


def test_execute_analysis_plan_runs_predictive_and_prescriptive() -> None:
    rng = np.random.default_rng(47)
    rows = 200
    df = pd.DataFrame(
        {
            "marketing_spend": rng.normal(120, 18, rows),
            "price": rng.normal(30, 5, rows),
            "region": rng.choice(["north", "south", "east"], rows),
        }
    )
    df["revenue"] = 1500 + (df["marketing_spend"] * 5.0) - (df["price"] * 12.0) + rng.normal(0, 25, rows)
    results = execute_analysis_plan(
        df=df,
        plan=[
            {"tool": "predictive_analysis", "columns": list(df.columns), "parameters": {"target_column": "revenue"}},
            {"tool": "prescriptive_analysis", "columns": list(df.columns), "parameters": {"target_column": "revenue"}},
        ],
        state_context={
            "business_question": "predict revenue and recommend how to optimize budget allocation",
            "selected_columns": list(df.columns),
            "analysis_evidence": {},
        },
    )
    assert any((value or {}).get("tool") == "predictive_analysis" for value in results.values())
    assert any((value or {}).get("tool") == "prescriptive_analysis" for value in results.values())
    print("EXECUTION PIPELINE OK:", list(results.keys()))


def test_graceful_failure_on_poor_dataset() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1]})
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "b"}},
        {"business_question": "predict b", "selected_columns": list(df.columns)},
    )
    assert result.get("error") or result.get("readiness_warnings")
    print("POOR DATASET FAILURE OK:", result)


def test_missing_requested_target_fails_honestly() -> None:
    rng = np.random.default_rng(48)
    rows = 160
    df = pd.DataFrame(
        {
            "engine_hp": rng.normal(180, 25, rows),
            "max_power_kw": rng.normal(135, 18, rows),
            "year_from": rng.integers(1950, 2020, rows),
        }
    )
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {}},
        {"business_question": "predict price and recommend pricing actions", "selected_columns": ["engine_hp", "year_from"]},
    )
    assert result.get("error")
    assert "target" in result["error"].lower()
    print("MISSING TARGET FAILURE OK:", result["error"])


def test_weak_model_is_flagged_and_downgraded() -> None:
    rng = np.random.default_rng(49)
    rows = 180
    df = pd.DataFrame(
        {
            "noise_a": rng.normal(0, 1, rows),
            "noise_b": rng.normal(0, 1, rows),
            "noise_c": rng.choice(["x", "y", "z"], rows),
            "target": rng.normal(0, 1, rows),
        }
    )
    result = run_predictive_analysis(
        df,
        {"tool": "predictive_analysis", "parameters": {"target_column": "target"}},
        {"business_question": "predict target", "selected_columns": list(df.columns)},
    )
    assert result["confidence"]["label"] == "low"
    assert result["no_reliable_recommendation"] is True
    assert result["truthfulness_flags"]
    prescriptive = run_prescriptive_analysis(result, "recommend what to do next")
    assert "improve model quality" in prescriptive["recommended_actions"][0]["action"].lower()
    print("WEAK MODEL DOWNGRADE OK:", result["confidence"], prescriptive["recommended_actions"][0]["action"])


if __name__ == "__main__":
    test_sales_regression()
    test_churn_classification()
    test_demand_forecasting()
    test_customer_risk_classification()
    test_budget_optimization_scenario()
    test_execute_analysis_plan_runs_predictive_and_prescriptive()
    test_graceful_failure_on_poor_dataset()
    test_missing_requested_target_fails_honestly()
    test_weak_model_is_flagged_and_downgraded()
    print("All predictive/prescriptive tests passed.")

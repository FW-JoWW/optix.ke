from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def get_candidate_models(problem_type: str) -> Dict[str, object]:
    if problem_type in {"regression", "forecasting"}:
        return {
            "linear_regression": LinearRegression(),
            "random_forest_regressor": RandomForestRegressor(n_estimators=120, random_state=42),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
        }
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest_classifier": RandomForestClassifier(n_estimators=120, random_state=42),
        "gradient_boosting_classifier": GradientBoostingClassifier(random_state=42),
    }

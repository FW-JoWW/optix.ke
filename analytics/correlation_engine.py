from __future__ import annotations

from math import atanh, sqrt, tanh
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


CorrelationKind = Literal[
    "pearson",
    "spearman",
    "kendall",
    "point_biserial",
    "cramers_v",
    "phi",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _shapiro_ok(values: pd.Series) -> bool:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 30:
        return False
    sample = clean if len(clean) <= 5000 else clean.sample(5000, random_state=42)
    if len(sample) < 3:
        return False
    _, p_value = stats.shapiro(sample)
    return bool(p_value >= 0.05)


def _skew_ok(values: pd.Series) -> bool:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return False
    skewness = float(stats.skew(clean, bias=False))
    return abs(skewness) <= 1


def _correlation_ci(r_value: float, n: int) -> Dict[str, Optional[float]]:
    if n <= 3 or abs(r_value) >= 1:
        return {"lower": None, "upper": None}
    z_value = atanh(r_value)
    standard_error = 1 / sqrt(n - 3)
    critical = stats.norm.ppf(0.975)
    lower = tanh(z_value - critical * standard_error)
    upper = tanh(z_value + critical * standard_error)
    return {"lower": float(lower), "upper": float(upper)}


def _cramers_v(contingency: pd.DataFrame, chi2_value: float) -> float:
    total = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    if total == 0 or min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2_value / (total * min_dim)))


def _phi_coefficient(contingency: pd.DataFrame, chi2_value: float) -> float:
    total = contingency.to_numpy().sum()
    if total == 0:
        return 0.0
    return float(np.sqrt(chi2_value / total))


def choose_numeric_method(x: pd.Series, y: pd.Series) -> tuple[CorrelationKind, list[str], bool]:
    warnings: list[str] = []
    x_clean = pd.to_numeric(x, errors="coerce").dropna()
    y_clean = pd.to_numeric(y, errors="coerce").dropna()
    assumptions_ok = _shapiro_ok(x_clean) and _shapiro_ok(y_clean) and _skew_ok(x_clean) and _skew_ok(y_clean)
    paired_n = min(len(x_clean), len(y_clean))

    if paired_n < 30:
        warnings.append("Sample size below 30; robust rank correlation preferred.")
    if paired_n < 15:
        warnings.append("Very small paired sample; Kendall correlation preferred.")
        return "kendall", warnings, False
    if assumptions_ok:
        return "pearson", warnings, True
    warnings.append("Normality or skewness assumptions were not met; rank correlation preferred.")
    return "spearman", warnings, False


def run_smart_correlation(
    x: pd.Series,
    y: pd.Series,
    x_kind: str,
    y_kind: str,
) -> Dict[str, Any]:
    warnings: list[str] = []

    if x_kind == "numeric" and y_kind == "numeric":
        frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
        method, method_warnings, assumptions_ok = choose_numeric_method(frame["x"], frame["y"])
        warnings.extend(method_warnings)
        if len(frame) < 3:
            return {
                "method_used": method,
                "coefficient": None,
                "p_value": None,
                "sample_size": len(frame),
                "test_statistic": None,
                "assumptions_met": False,
                "warnings": warnings + ["At least three paired observations are required."],
                "confidence_interval_95": {"lower": None, "upper": None},
            }
        if method == "pearson":
            statistic, p_value = stats.pearsonr(frame["x"], frame["y"])
        elif method == "kendall":
            statistic, p_value = stats.kendalltau(frame["x"], frame["y"])
        else:
            statistic, p_value = stats.spearmanr(frame["x"], frame["y"])
        ci = _correlation_ci(float(statistic), len(frame)) if method in {"pearson", "spearman"} else {"lower": None, "upper": None}
        return {
            "method_used": method,
            "coefficient": float(statistic),
            "p_value": float(p_value),
            "sample_size": int(len(frame)),
            "test_statistic": float(statistic),
            "assumptions_met": assumptions_ok,
            "warnings": warnings,
            "confidence_interval_95": ci,
        }

    if {x_kind, y_kind} == {"numeric", "binary"}:
        numeric = pd.to_numeric(x if x_kind == "numeric" else y, errors="coerce")
        binary = x if x_kind == "binary" else y
        frame = pd.DataFrame({"numeric": numeric, "binary": binary}).dropna()
        if len(frame["binary"].unique()) != 2:
            warnings.append("Binary correlation requested but the grouping column is not truly binary.")
            return {
                "method_used": "point_biserial",
                "coefficient": None,
                "p_value": None,
                "sample_size": int(len(frame)),
                "test_statistic": None,
                "assumptions_met": False,
                "warnings": warnings,
                "confidence_interval_95": {"lower": None, "upper": None},
            }
        binary_codes = pd.Categorical(frame["binary"]).codes
        statistic, p_value = stats.pointbiserialr(binary_codes, frame["numeric"])
        return {
            "method_used": "point_biserial",
            "coefficient": float(statistic),
            "p_value": float(p_value),
            "sample_size": int(len(frame)),
            "test_statistic": float(statistic),
            "assumptions_met": True,
            "warnings": warnings,
            "confidence_interval_95": _correlation_ci(float(statistic), len(frame)),
        }

    contingency = pd.crosstab(x.astype(str), y.astype(str))
    if contingency.empty:
        return {
            "method_used": "cramers_v",
            "coefficient": None,
            "p_value": None,
            "sample_size": 0,
            "test_statistic": None,
            "assumptions_met": False,
            "warnings": ["No valid observations available for categorical association."],
            "confidence_interval_95": {"lower": None, "upper": None},
        }
    chi2_value, p_value, _, expected = stats.chi2_contingency(contingency)
    if (expected < 5).any():
        warnings.append("Expected cell frequencies below 5 were detected.")
    if contingency.shape == (2, 2):
        coefficient = _phi_coefficient(contingency, float(chi2_value))
        method: CorrelationKind = "phi"
    else:
        coefficient = _cramers_v(contingency, float(chi2_value))
        method = "cramers_v"
    return {
        "method_used": method,
        "coefficient": float(coefficient),
        "p_value": float(p_value),
        "sample_size": int(contingency.to_numpy().sum()),
        "test_statistic": float(chi2_value),
        "assumptions_met": not (expected < 5).any(),
        "warnings": warnings,
        "confidence_interval_95": {"lower": None, "upper": None},
        "contingency_table": contingency.to_dict(),
    }

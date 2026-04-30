from __future__ import annotations

from math import atanh, sqrt, tanh
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from analytics.relationship_scanner import analyze_relationship_evidence
from utils.numeric_parsing import normalize_numeric_token


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _numeric_like_ratio(series: pd.Series) -> float:
    parsed = pd.to_numeric(series.map(normalize_numeric_token), errors="coerce")
    non_null = int(series.notna().sum())
    return float(parsed.notna().sum() / non_null) if non_null else 0.0


def _is_numeric_column(column: str, series: pd.Series, state_context: Dict[str, Any]) -> bool:
    dataset_profile = state_context.get("dataset_profile", {}) or {}
    column_registry = state_context.get("column_registry", {}) or {}
    if column in dataset_profile.get("numeric_columns", []):
        return True
    role = (column_registry.get(column, {}) or {}).get("semantic_role")
    if role in {"numeric_measure", "derived_metric"}:
        return True
    if role in {"categorical_feature", "grouping_key", "timestamp", "identifier"}:
        return False
    return _numeric_like_ratio(series) >= 0.8


def _confidence_label(effect_value: float, thresholds: tuple[float, float]) -> str:
    absolute = abs(effect_value)
    if absolute >= thresholds[1]:
        return "large"
    if absolute >= thresholds[0]:
        return "medium"
    return "small"


def _cohens_d(group1: pd.Series, group2: pd.Series) -> Dict[str, Any]:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return {"metric": "cohens_d", "value": None, "interpretation": "unavailable"}
    var1 = group1.var(ddof=1)
    var2 = group2.var(ddof=1)
    pooled = sqrt((((n1 - 1) * var1) + ((n2 - 1) * var2)) / max(n1 + n2 - 2, 1))
    if pooled == 0:
        return {"metric": "cohens_d", "value": 0.0, "interpretation": "small"}
    effect = float((group1.mean() - group2.mean()) / pooled)
    return {
        "metric": "cohens_d",
        "value": effect,
        "interpretation": _confidence_label(effect, (0.2, 0.5)),
    }


def _eta_squared(groups: List[pd.Series]) -> Dict[str, Any]:
    valid_groups = [group.dropna() for group in groups if len(group.dropna()) > 0]
    if len(valid_groups) < 2:
        return {"metric": "eta_squared", "value": None, "interpretation": "unavailable"}
    grand = pd.concat(valid_groups)
    grand_mean = grand.mean()
    ss_between = sum(len(group) * ((group.mean() - grand_mean) ** 2) for group in valid_groups)
    ss_total = ((grand - grand_mean) ** 2).sum()
    if ss_total == 0:
        return {"metric": "eta_squared", "value": 0.0, "interpretation": "small"}
    effect = float(ss_between / ss_total)
    return {
        "metric": "eta_squared",
        "value": effect,
        "interpretation": _confidence_label(effect, (0.06, 0.14)),
    }


def _rank_biserial(group1: pd.Series, group2: pd.Series, u_statistic: float) -> Dict[str, Any]:
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return {"metric": "rank_biserial", "value": None, "interpretation": "unavailable"}
    effect = float(1 - (2 * u_statistic) / (n1 * n2))
    return {
        "metric": "rank_biserial",
        "value": effect,
        "interpretation": _confidence_label(effect, (0.1, 0.3)),
    }


def _epsilon_squared(h_statistic: float, total_n: int, k_groups: int) -> Dict[str, Any]:
    denominator = max(total_n - k_groups, 1)
    effect = float(max((h_statistic - k_groups + 1) / denominator, 0.0))
    return {
        "metric": "epsilon_squared",
        "value": effect,
        "interpretation": _confidence_label(effect, (0.08, 0.26)),
    }


def _cramers_v(contingency: pd.DataFrame, chi2_value: float) -> Dict[str, Any]:
    total = contingency.to_numpy().sum()
    if total == 0:
        return {"metric": "cramers_v", "value": None, "interpretation": "unavailable"}
    min_dim = min(contingency.shape) - 1
    if min_dim <= 0:
        return {"metric": "cramers_v", "value": None, "interpretation": "unavailable"}
    effect = float(sqrt(chi2_value / (total * min_dim)))
    return {
        "metric": "cramers_v",
        "value": effect,
        "interpretation": _confidence_label(effect, (0.1, 0.3)),
    }


def _mean_confidence_interval(series: pd.Series, confidence: float = 0.95) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    mean_value = _safe_float(clean.mean())
    if n < 2:
        return {"center": mean_value, "lower": None, "upper": None, "level": confidence}
    sem = stats.sem(clean, nan_policy="omit")
    margin = float(stats.t.ppf((1 + confidence) / 2.0, n - 1) * sem)
    return {
        "center": mean_value,
        "lower": float(clean.mean() - margin),
        "upper": float(clean.mean() + margin),
        "level": confidence,
    }


def _correlation_confidence_interval(r_value: float, n: int, confidence: float = 0.95) -> Dict[str, Any]:
    if n <= 3 or abs(r_value) >= 1:
        return {"center": float(r_value), "lower": None, "upper": None, "level": confidence}
    z_value = atanh(r_value)
    standard_error = 1 / sqrt(n - 3)
    critical = stats.norm.ppf((1 + confidence) / 2.0)
    lower = tanh(z_value - critical * standard_error)
    upper = tanh(z_value + critical * standard_error)
    return {"center": float(r_value), "lower": float(lower), "upper": float(upper), "level": confidence}


def _group_statistics(df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for group_name, group_frame in df.groupby(group_col):
        values = pd.to_numeric(group_frame[numeric_col], errors="coerce").dropna()
        summary[str(group_name)] = {
            "count": int(len(values)),
            "mean": _safe_float(values.mean()),
            "median": _safe_float(values.median()),
            "std": _safe_float(values.std(ddof=1)) if len(values) > 1 else None,
            "confidence_interval_95": _mean_confidence_interval(values),
        }
    return summary


def _proportion_summary(contingency: pd.DataFrame) -> Dict[str, Any]:
    total = contingency.to_numpy().sum()
    if total == 0:
        return {"counts": contingency.to_dict(), "proportions": {}}
    proportions = (contingency / total).fillna(0.0)
    return {
        "counts": contingency.to_dict(),
        "proportions": proportions.round(4).to_dict(),
    }


def _assumption_checks_for_groups(df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict[str, Any]:
    warnings: List[str] = []
    subset = df[[numeric_col, group_col]].copy()
    missing_ratio = float(subset.isna().any(axis=1).mean()) if len(subset) else 0.0
    if missing_ratio > 0.2:
        warnings.append("Missing data ratio exceeds 20% for the inferential subset.")

    grouped = {
        str(group): pd.to_numeric(group_df[numeric_col], errors="coerce").dropna()
        for group, group_df in subset.dropna(subset=[group_col]).groupby(group_col)
    }
    sizes = [len(values) for values in grouped.values()]
    if len(sizes) < 2:
        warnings.append("Insufficient valid groups for inferential testing.")
        return {
            "normality": False,
            "equal_variance": False,
            "warnings": warnings,
            "group_sizes": sizes,
        }

    min_size = min(sizes)
    max_size = max(sizes)
    if min_size == 0 or max_size / max(min_size, 1) >= 1.5:
        warnings.append("Group sizes are materially imbalanced.")

    normality_results = []
    for group_name, values in grouped.items():
        if len(values) < 3:
            warnings.append(f"Group '{group_name}' has too few observations for normality testing.")
            normality_results.append(False)
            continue
        if len(values) < 30:
            warnings.append(f"Group '{group_name}' has n < 30; normality test treated as unreliable.")
        skewness = float(stats.skew(values, bias=False)) if len(values) > 2 else 0.0
        if abs(skewness) > 1:
            warnings.append(f"Group '{group_name}' shows |skewness| > 1.")
            normality_results.append(False)
            continue
        sample = values if len(values) <= 5000 else values.sample(5000, random_state=42)
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        normality_results.append(bool(shapiro_p >= 0.05 and len(values) >= 30))

    equal_variance = False
    if all(len(values) >= 2 for values in grouped.values()):
        levene_stat, levene_p = stats.levene(*grouped.values(), center="median")
        equal_variance = bool(levene_p >= 0.05)
        if not equal_variance:
            warnings.append("Levene's test indicates unequal variances.")
    else:
        warnings.append("Some groups have too few observations for a variance test.")

    return {
        "normality": bool(all(normality_results)),
        "equal_variance": equal_variance,
        "warnings": warnings,
        "group_sizes": sizes,
    }


def _assumption_checks_for_numeric_pair(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    warnings: List[str] = []
    subset = df[columns].copy()
    missing_ratio = float(subset.isna().any(axis=1).mean()) if len(subset) else 0.0
    if missing_ratio > 0.2:
        warnings.append("Missing data ratio exceeds 20% for the inferential subset.")

    normality_results = []
    for column in columns:
        values = pd.to_numeric(subset[column], errors="coerce").dropna()
        if len(values) < 3:
            warnings.append(f"{column} has too few observations for normality testing.")
            normality_results.append(False)
            continue
        if len(values) < 30:
            warnings.append(f"{column} has n < 30; normality test treated as unreliable.")
        skewness = float(stats.skew(values, bias=False)) if len(values) > 2 else 0.0
        if abs(skewness) > 1:
            warnings.append(f"{column} shows |skewness| > 1.")
            normality_results.append(False)
            continue
        sample = values if len(values) <= 5000 else values.sample(5000, random_state=42)
        _, shapiro_p = stats.shapiro(sample)
        normality_results.append(bool(shapiro_p >= 0.05 and len(values) >= 30))

    return {
        "normality": bool(all(normality_results)),
        "equal_variance": None,
        "warnings": warnings,
    }


def _effect_interpretation(method: str, effect: Dict[str, Any]) -> str:
    value = effect.get("value")
    if value is None:
        return "Effect size could not be estimated reliably."
    return f"Effect size ({effect['metric']}) is {round(float(value), 4)}, indicating a {effect['interpretation']} practical impact."


def _base_result(method_selected: str, columns: List[str], assumptions: Dict[str, Any], warnings: List[str]) -> Dict[str, Any]:
    return {
        "tool": "inferential_analysis",
        "method_selected": method_selected,
        "columns": columns,
        "assumptions": assumptions,
        "warnings": warnings,
    }


def _relationship_report_to_result(report: Dict[str, Any], analysis_category: str) -> Dict[str, Any]:
    stats_payload = report.get("stats", {}) or {}
    causal_payload = report.get("causal_evidence", {}) or {}
    temporal_signal = report.get("temporal_signal", {}) or {}
    partial_corr = report.get("partial_correlation", {}) or {}
    nonlinear = report.get("nonlinear_signal", {}) or {}
    warnings = list(report.get("warnings", [])) + list(report.get("assumptions", []))
    p_value = stats_payload.get("p_value")
    decision = "reject_h0" if p_value is not None and p_value < 0.05 else "fail_to_reject_h0"
    coefficient = stats_payload.get("coefficient")
    metric = "association"
    if str(report.get("method_used")).startswith(("pearson", "spearman", "kendall", "point_biserial")):
        metric = "correlation_r"
    elif report.get("method_used") == "phi":
        metric = "phi"
    elif report.get("method_used") == "cramers_v":
        metric = "cramers_v"
    effect = {
        "metric": metric,
        "value": coefficient,
        "interpretation": (
            "large" if coefficient is not None and abs(float(coefficient)) >= 0.5
            else "medium" if coefficient is not None and abs(float(coefficient)) >= 0.2
            else "small"
        ),
    }
    result = {
        **_base_result(str(report.get("method_used")), report.get("structured_summary", {}).get("columns") or report.get("structured_summary", {}).get("pair") or [], {"normality": None, "equal_variance": None, "warnings": warnings}, warnings),
        "analysis_category": analysis_category,
        "relationship_evidence": report,
        "estimation": {
            "sample_size": stats_payload.get("sample_size", 0),
            "confidence_interval_95": stats_payload.get("confidence_interval_95"),
            "partial_correlation": partial_corr,
            "temporal_signal": temporal_signal,
            "nonlinear_signal": nonlinear,
        },
        "hypothesis_test": {
            "null_hypothesis": "No meaningful relationship exists between the variables.",
            "alternative_hypothesis": "A meaningful relationship exists between the variables.",
            "test_statistic": stats_payload.get("test_statistic"),
            "p_value": p_value,
            "decision": decision,
        },
        "effect_size": effect,
        "interpretation": {
            "statistical_significance": bool(p_value is not None and p_value < 0.05),
            "practical_significance": effect.get("interpretation"),
            "summary": report.get("relationship_found"),
            "reliability_warnings": warnings,
            "effect_size_meaning": report.get("human_summary"),
        },
        "bias_risks": report.get("bias_risks", []),
        "confounders": report.get("confounders", []),
        "causal_evidence": causal_payload,
        "recommended_next_step": report.get("recommended_next_step"),
        "confidence": report.get("confidence"),
        "human_summary": report.get("human_summary"),
    }
    if stats_payload.get("additional_metrics", {}).get("contingency_table"):
        result["estimation"]["counts"] = stats_payload["additional_metrics"]["contingency_table"]
        if analysis_category == "categorical_association":
            result["hypothesis_test"]["degrees_of_freedom"] = None
    return result


def run_inferential_analysis(
    df: pd.DataFrame,
    task: Dict[str, Any],
    state_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    state_context = state_context or {}
    columns = task.get("columns", [])
    if len(columns) < 2:
        return {"tool": "inferential_analysis", "error": "Inferential analysis requires at least two columns."}

    subset = df[columns].copy()
    numeric_columns = [col for col in columns if _is_numeric_column(col, subset[col], state_context)]
    categorical_columns = [col for col in columns if col not in numeric_columns]
    tool_name = task.get("tool")

    if len(columns) == 2 and tool_name in {"correlation", "chi_square"}:
        relationship_df = state_context.get("reference_dataset")
        if not isinstance(relationship_df, pd.DataFrame) or any(col not in relationship_df.columns for col in columns):
            relationship_df = df
        report = analyze_relationship_evidence(
            df=relationship_df,
            x_col=columns[0],
            y_col=columns[1],
            question=state_context.get("business_question", ""),
            state_context=state_context,
        )
        analysis_category = "categorical_association" if report.get("method_used") in {"cramers_v", "phi"} else "numeric_relationship"
        result = _relationship_report_to_result(report, analysis_category=analysis_category)
        result["columns"] = columns
        return result

    if len(numeric_columns) == 1 and len(categorical_columns) == 1:
        numeric_col = numeric_columns[0]
        group_col = categorical_columns[0]
        subset[numeric_col] = pd.to_numeric(subset[numeric_col], errors="coerce")
        subset = subset.dropna(subset=[numeric_col, group_col])
        groups = list(subset[group_col].dropna().unique())
        if len(groups) < 2:
            return {
                "tool": "inferential_analysis",
                "error": "At least two groups are required for group comparison.",
                "warnings": ["Invalid grouping structure for inferential comparison."],
            }

        assumptions = _assumption_checks_for_groups(subset, numeric_col, group_col)
        warnings = list(assumptions["warnings"])
        grouped_series = [subset[subset[group_col] == group][numeric_col].dropna() for group in groups]
        group_stats = _group_statistics(subset, numeric_col, group_col)

        if len(groups) == 2:
            group1, group2 = grouped_series[0], grouped_series[1]
            if len(group1) < 2 or len(group2) < 2:
                return {
                    "tool": "inferential_analysis",
                    "error": "Insufficient data in one or more groups.",
                    "warnings": warnings + ["Each group needs at least two observations."],
                }
            if assumptions["normality"] and assumptions["equal_variance"]:
                method = "independent_t_test"
                test_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=True, nan_policy="omit")
                effect = _cohens_d(group1, group2)
            else:
                method = "mann_whitney_u"
                test_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")
                effect = _rank_biserial(group1, group2, float(test_statistic))

            decision = "reject_h0" if p_value < 0.05 else "fail_to_reject_h0"
            return {
                **_base_result(method, columns, assumptions, warnings),
                "analysis_category": "two_group_comparison",
                "estimation": {
                    "group_statistics": group_stats,
                    "confidence_intervals_95": {
                        group_name: stats_item["confidence_interval_95"]
                        for group_name, stats_item in group_stats.items()
                    },
                },
                "hypothesis_test": {
                    "null_hypothesis": f"No difference in {numeric_col} across {group_col}.",
                    "alternative_hypothesis": f"A difference exists in {numeric_col} across {group_col}.",
                    "test_statistic": float(test_statistic),
                    "p_value": float(p_value),
                    "decision": decision,
                },
                "effect_size": effect,
                "interpretation": {
                    "statistical_significance": bool(p_value < 0.05),
                    "practical_significance": effect.get("interpretation"),
                    "summary": (
                        "There is evidence of a difference."
                        if p_value < 0.05
                        else "There is not enough evidence to conclude a difference."
                    ),
                    "reliability_warnings": warnings,
                    "effect_size_meaning": _effect_interpretation(method, effect),
                },
            }

        if assumptions["normality"] and assumptions["equal_variance"]:
            method = "one_way_anova"
            test_statistic, p_value = stats.f_oneway(*grouped_series)
            effect = _eta_squared(grouped_series)
        else:
            method = "kruskal_wallis"
            test_statistic, p_value = stats.kruskal(*grouped_series)
            effect = _epsilon_squared(float(test_statistic), int(len(subset)), len(grouped_series))

        decision = "reject_h0" if p_value < 0.05 else "fail_to_reject_h0"
        return {
            **_base_result(method, columns, assumptions, warnings),
            "analysis_category": "multi_group_comparison",
            "estimation": {
                "group_statistics": group_stats,
                "confidence_intervals_95": {
                    group_name: stats_item["confidence_interval_95"]
                    for group_name, stats_item in group_stats.items()
                },
            },
            "hypothesis_test": {
                "null_hypothesis": f"No group mean difference in {numeric_col} across {group_col}.",
                "alternative_hypothesis": f"At least one group differs in {numeric_col} across {group_col}.",
                "test_statistic": float(test_statistic),
                "p_value": float(p_value),
                "decision": decision,
            },
            "effect_size": effect,
            "interpretation": {
                "statistical_significance": bool(p_value < 0.05),
                "practical_significance": effect.get("interpretation"),
                "summary": (
                    "There is evidence that at least one group differs."
                    if p_value < 0.05
                    else "There is not enough evidence that the groups differ."
                ),
                "reliability_warnings": warnings,
                "effect_size_meaning": _effect_interpretation(method, effect),
            },
        }

    if len(numeric_columns) >= 2:
        pair = numeric_columns[:2]
        subset = subset[pair].copy()
        subset[pair[0]] = pd.to_numeric(subset[pair[0]], errors="coerce")
        subset[pair[1]] = pd.to_numeric(subset[pair[1]], errors="coerce")
        subset = subset.dropna()
        if len(subset) < 3:
            return {
                "tool": "inferential_analysis",
                "error": "At least three paired observations are required for correlation.",
                "warnings": ["Insufficient data for correlation testing."],
            }
        assumptions = _assumption_checks_for_numeric_pair(subset, pair)
        warnings = list(assumptions["warnings"])
        if assumptions["normality"]:
            method = "pearson_correlation"
            statistic, p_value = stats.pearsonr(subset[pair[0]], subset[pair[1]])
        else:
            method = "spearman_correlation"
            statistic, p_value = stats.spearmanr(subset[pair[0]], subset[pair[1]])
        effect = {
            "metric": "correlation_r",
            "value": float(statistic),
            "interpretation": _confidence_label(float(statistic), (0.1, 0.3)),
        }
        decision = "reject_h0" if p_value < 0.05 else "fail_to_reject_h0"
        return {
            **_base_result(method, pair, assumptions, warnings),
            "analysis_category": "numeric_relationship",
            "estimation": {
                "correlation_confidence_interval_95": _correlation_confidence_interval(float(statistic), len(subset)),
                "sample_size": int(len(subset)),
            },
            "hypothesis_test": {
                "null_hypothesis": f"No relationship exists between {pair[0]} and {pair[1]}.",
                "alternative_hypothesis": f"A relationship exists between {pair[0]} and {pair[1]}.",
                "test_statistic": float(statistic),
                "p_value": float(p_value),
                "decision": decision,
            },
            "effect_size": effect,
            "interpretation": {
                "statistical_significance": bool(p_value < 0.05),
                "practical_significance": effect.get("interpretation"),
                "summary": (
                    "There is evidence of a relationship between the variables."
                    if p_value < 0.05
                    else "There is not enough evidence of a relationship between the variables."
                ),
                "reliability_warnings": warnings,
                "effect_size_meaning": _effect_interpretation(method, effect),
            },
        }

    if len(categorical_columns) >= 2:
        pair = categorical_columns[:2]
        subset = subset[pair].dropna()
        if subset.empty:
            return {
                "tool": "inferential_analysis",
                "error": "Insufficient categorical data for chi-square testing.",
                "warnings": ["No valid rows remain after dropping missing categorical pairs."],
            }
        contingency = pd.crosstab(subset[pair[0]], subset[pair[1]])
        chi2_value, p_value, degrees_freedom, expected = stats.chi2_contingency(contingency)
        effect = _cramers_v(contingency, float(chi2_value))
        warnings: List[str] = []
        if (expected < 5).any():
            warnings.append("Expected frequencies below 5 were detected; chi-square reliability is reduced.")
        decision = "reject_h0" if p_value < 0.05 else "fail_to_reject_h0"
        assumptions = {
            "normality": None,
            "equal_variance": None,
            "warnings": warnings,
        }
        return {
            **_base_result("chi_square_independence", pair, assumptions, warnings),
            "analysis_category": "categorical_association",
            "estimation": _proportion_summary(contingency),
            "hypothesis_test": {
                "null_hypothesis": f"{pair[0]} and {pair[1]} are independent.",
                "alternative_hypothesis": f"{pair[0]} and {pair[1]} are associated.",
                "test_statistic": float(chi2_value),
                "p_value": float(p_value),
                "decision": decision,
                "degrees_of_freedom": int(degrees_freedom),
            },
            "effect_size": effect,
            "interpretation": {
                "statistical_significance": bool(p_value < 0.05),
                "practical_significance": effect.get("interpretation"),
                "summary": (
                    "There is evidence of association between the categorical variables."
                    if p_value < 0.05
                    else "There is not enough evidence of association between the categorical variables."
                ),
                "reliability_warnings": warnings,
                "effect_size_meaning": _effect_interpretation("chi_square_independence", effect),
            },
        }

    return {
        "tool": "inferential_analysis",
        "error": "No valid inferential path could be selected from the provided columns.",
        "warnings": ["Assumptions were heavily violated or the variable combination is unsupported."],
    }

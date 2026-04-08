# utils/issue_detector.py
import pandas as pd
import numpy as np

def detect_issues(df: pd.DataFrame) -> dict:
    """
    Rule-based deterministic detection of data quality issues.
    Returns a list of detected issues (without LLM interpretation yet).
    """

    issues = []

    total_rows = len(df)

    # ---------------------
    # Build numeric cache
    # ---------------------
    numeric_cache = {}
    
    for col in df.columns:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
        )

        coerced = pd.to_numeric(cleaned, errors="coerce")

        # If ≥80% values are numeric → treat as numeric column
        if coerced.notna().mean() >= 0.8:
            numeric_cache[col] = coerced

    # -----------------------------
    # Missing values
    # -----------------------------
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_percent = (missing_count / total_rows) * 100
            severity = "low"
            if missing_percent > 30:
                severity = "high"
            elif missing_percent > 10:
                severity = "medium"
            issues.append({
                "column": col,
                "issue_type": "missing_values",
                "severity": severity,
                "missing_count": missing_count
            })

    # -----------------------------
    # Duplicate rows
    # -----------------------------
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            "column": None,
            "issue_type": "duplicate_rows",
            "severity": "medium",
            "duplicate_count": dup_count
        })

    # -----------------------------
    # Numeric columns: outliers
    # -----------------------------
    '''numeric_cols = []

    for col in df.columns:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
        )

        coerced = pd.to_numeric(cleaned, errors="coerce")

        if coerced.notna().sum() / len(df) > 0.8:
            numeric_cols.append(col)'''
    #numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col, series in numeric_cache.items():
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = series[(series < lower) | (series > upper)].count()
        if outlier_count > 0:
            issues.append({
                "column": col,
                "issue_type": "outliers",
                "severity": "high" if outlier_count / total_rows > 0.05 else "medium",
                "outlier_count": outlier_count
            })

    # -----------------------------
    # Constant columns
    # -----------------------------
    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            issues.append({
                "column": col,
                "issue_type": "constant_column",
                "severity": "low"
            })

    # -----------------------------
    # High-cardinality categorical
    # -----------------------------
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        if total_rows == 0:
            continue
        unique_ratio = df[col].nunique() / total_rows 
        if unique_ratio > 0.8:
            issues.append({
                "column": col,
                "issue_type": "high_cardinality",
                "severity": "medium"
            })

    # -----------------------------
    # Datatype mismatches (numeric stored as object)
    # -----------------------------
    for col in df.columns:
        if df[col].dtype == "object" and col in numeric_cache:
            '''try:
                pd.to_numeric(df[col])'''
            issues.append({
                "column": col,
                "issue_type": "numeric_as_object",
                "severity": "medium"
            })
            '''except:
                pass'''

    return {"detected_issues": issues}
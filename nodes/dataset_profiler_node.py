import pandas as pd
import numpy as np
import re
from state.state import AnalystState


def clean_numeric_series(series):
    """
    Remove currency symbols, commas, units etc.
    """
    cleaned = series.astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def is_date_like(series):
    """
    Check if column likely contains dates before expensive parsing.
    """
    sample = series.dropna().astype(str).head(50)

    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{2}/\d{2}/\d{4}",
        r"\d{4}/\d{2}/\d{2}"
    ]

    for pattern in date_patterns:
        if sample.str.contains(pattern).any():
            return True

    return False


def dataset_profiler_node(state: AnalystState) -> AnalystState:

    df = state.get("cleaned_data")
    if df is None:
        df = state.get("dataframe")

    if df is None:
        raise ValueError("No dataframe found in state.")

    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    identifier_columns = []

    column_statistics = {}

    total_rows = len(df)

    for col in df.columns:

        series = df[col]
        
        # ---------------------------
        # Identifier detection first
        # ---------------------------

        unique_ratio = series.nunique() / total_rows

        name = col.lower()

        id_keywords = ["id", "_id", "code", "key", "vin", "number"]
        
        looks_like_id_name = any(k in name for k in id_keywords)

        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        is_integer_like = (numeric_series % 1 == 0).all()
        
        #is_integer_like = pd.to_numeric(series, errors="coerce").dropna().apply(float.is_integer).all()
        if unique_ratio > 0.95 and (looks_like_id_name or is_integer_like):
            identifier_columns.append(col)
            column_statistics[col] = {
                "unique_ratio": float(unique_ratio)
            }
            continue

        # -------------------------
        # Numeric detection
        # -------------------------

        coerced_numeric = clean_numeric_series(series)

        numeric_ratio = coerced_numeric.notna().sum() / total_rows

        if numeric_ratio > 0.8:

            numeric_columns.append(col)

            safe_series = coerced_numeric.dropna()

            column_statistics[col] = {
                "min": float(np.nanmin(coerced_numeric)),
                "max": float(np.nanmax(coerced_numeric)),
                "mean": float(np.nanmean(coerced_numeric)),
                "std": float(np.nanstd(coerced_numeric))
            }

            continue

        # -------------------------
        # Datetime detection
        # -------------------------

        if is_date_like(series):

            coerced_datetime = pd.to_datetime(series, errors="coerce")

            datetime_ratio = coerced_datetime.notna().sum() / total_rows

            if datetime_ratio > 0.8:

                datetime_columns.append(col)

                column_statistics[col] = {
                    "min_date": str(coerced_datetime.min()),
                    "max_date": str(coerced_datetime.max())
                }

                continue

        # -------------------------
        # Identifier detection
        # -------------------------

        unique_ratio = series.nunique() / total_rows

        if unique_ratio > 0.95:

            identifier_columns.append(col)

            column_statistics[col] = {
                "unique_ratio": float(unique_ratio)
            }

            continue

        # -------------------------
        # Categorical
        # -------------------------

        categorical_columns.append(col)

        column_statistics[col] = {
            "unique_count": int(series.nunique()),
            "unique_ratio": float(unique_ratio)
        }

    dataset_profile = {
        "row_count": total_rows,
        "column_count": df.shape[1],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "identifier_columns": identifier_columns,
        "missing_values": df.isnull().sum().to_dict(),
        "unique_counts": df.nunique().to_dict(),
        "column_statistics": column_statistics
    }

    state["dataset_profile"] = dataset_profile

    print("\n=== DATASET PROFILE ===")
    print(dataset_profile)

    return state


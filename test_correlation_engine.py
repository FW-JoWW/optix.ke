import pandas as pd

from analytics.correlation_engine import run_smart_correlation


def main() -> None:
    linear = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )
    numeric_result = run_smart_correlation(linear["x"], linear["y"], x_kind="numeric", y_kind="numeric")
    print("\nNUMERIC RESULT")
    print(numeric_result)

    binary = pd.DataFrame(
        {
            "exposed": [0, 0, 0, 1, 1, 1, 1, 0],
            "score": [10, 11, 9, 20, 22, 21, 19, 8],
        }
    )
    point_biserial_result = run_smart_correlation(binary["exposed"], binary["score"], x_kind="binary", y_kind="numeric")
    print("\nPOINT-BISERIAL RESULT")
    print(point_biserial_result)

    categorical = pd.DataFrame(
        {
            "plan": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "channel": ["Web", "Store", "Web", "Store", "Store", "Web", "Partner", "Partner"],
        }
    )
    categorical_result = run_smart_correlation(categorical["plan"], categorical["channel"], x_kind="categorical", y_kind="categorical")
    print("\nCATEGORICAL RESULT")
    print(categorical_result)


if __name__ == "__main__":
    main()

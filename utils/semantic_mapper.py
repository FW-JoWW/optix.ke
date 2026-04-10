# utils/semantic_mapper.py
def map_semantic_filters(query, df, numeric_cols):
    """
    Converts words like 'cheap', 'expensive' into numeric filters
    """
    query = query.lower()

    filters = []

    if df is None or not numeric_cols:
        return filters

    semantic_map = {
        "cheap": ("low", 0.3),
        "affordable": ("low", 0.3),
        "expensive": ("high", 0.7),
        "premium": ("high", 0.7)
    }

    for word, (direction, quantile) in semantic_map.items():
        if word in query:
            for col in numeric_cols:
                if col not in df.columns:
                    continue

                threshold = df[col].quantile(quantile)

                filters.append({
                    "type": "condition",
                    "column": col,
                    "operator": "<" if direction == "low" else ">",
                    "value": float(threshold),
                    "source": "semantic"
                })

            break
        
    return filters
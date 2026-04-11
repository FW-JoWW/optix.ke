# utils/semantic_mapper.py
def map_semantic_filters(query, df, numeric_cols):
    """
    Converts qualitative magnitude words into numeric filters using
    query-mentioned numeric columns when possible.
    """
    query = query.lower()

    filters = []

    if df is None or not numeric_cols:
        return filters

    semantic_map = {
        "cheap": ("low", 0.3),
        "affordable": ("low", 0.3),
        "budget": ("low", 0.3),
        "expensive": ("high", 0.7),
        "premium": ("high", 0.7),
        "high": ("high", 0.7),
        "low": ("low", 0.3)
    }

    def pick_semantic_target():
        available_numeric = [col for col in numeric_cols if col in df.columns]
        mentioned = [
            col for col in available_numeric
            if col.lower() in query or col.lower().replace("_", " ") in query
        ]

        if mentioned:
            return mentioned

        if len(available_numeric) == 1:
            return available_numeric

        return []

    for word, (direction, quantile) in semantic_map.items():
        if word in query:
            negated = any(
                phrase in query for phrase in [f"not {word}", f"not too {word}", f"not very {word}"]
            )

            target_columns = pick_semantic_target()
            if not target_columns:
                break

            for col in target_columns:
                threshold = df[col].quantile(quantile)

                operator = "<" if direction == "low" else ">"
                if negated:
                    operator = ">=" if direction == "low" else "<="

                filters.append({
                    "type": "condition",
                    "column": col,
                    "operator": operator,
                    "value": float(threshold),
                    "source": "semantic"
                })

            break
        
    return filters

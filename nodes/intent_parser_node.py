# nodes/intent_perser_node.py
import re
from state.state import AnalystState
from nodes.llm_reasoning_node import llm_reasoning_node
from utils.semantic_mapper import map_semantic_filters
from core.analytic_capability import infer_capability_signals
from decision_engine import (
    _best_text_match,
    _best_value_metric,
    _business_customer_segment_column,
    _business_entity_column,
    _business_entity_metric,
    _business_geography_column,
    _business_installments_metric,
    _business_review_metric,
    _business_segment_column,
    _business_time_column,
    _preferred_business_metrics,
)

# --- NUMERIC SYNONYMS ---
NUMERIC_SYNONYMS = {
    "under": "<",
    "below": "<",
    "less than": "<",
    "over": ">",
    "above": ">",
    "more than": ">",
    "not equals": "!=",
    "!=": "!=",
    "<=": "<=",
    ">=": ">="
}


def map_numeric_synonyms(query: str):
    for word, op in NUMERIC_SYNONYMS.items():
        query = re.sub(rf"\b{word}\b", op, query, flags=re.IGNORECASE)
    return query

# ------------------
# HELPER
# ------------------
def normalize(text: str):
    return text.lower().replace("_", "").replace(" ", "")


def _dedupe_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _combine_analytic_intents(primary_intent, detected_intents):
    ordered = []
    if primary_intent and primary_intent != "unknown":
        ordered.append(primary_intent)
    for item in detected_intents or []:
        intent_type = item.get("type")
        if intent_type and intent_type != "unknown" and intent_type not in ordered:
            ordered.append(intent_type)
    return ordered


def _resolve_intent_role_columns(query: str, state: AnalystState, df):
    query = (query or "").lower()
    profile = state.get("dataset_profile", {}) or {}
    context = state.get("column_registry", {}) or {}
    signals = infer_capability_signals(query)

    time_column = _business_time_column(query, profile, context, list(df.columns) if df is not None else [])
    order_column = _business_entity_column(query, profile, context, "order")
    customer_column = _business_entity_column(query, profile, context, "customer")
    seller_column = _business_entity_metric(profile, context, "seller")
    category_column = _business_segment_column(profile, context, "category")
    product_column = _business_segment_column(profile, context, "product")
    customer_segment_column = _business_customer_segment_column(profile, context)
    customer_geo_column = _business_geography_column(profile, context, signals.get("geography_level"), owner="customer")
    seller_geo_column = _business_geography_column(profile, context, signals.get("geography_level"), owner="seller")
    geography_column = customer_geo_column or seller_geo_column or _business_geography_column(profile, context, None, owner=None)
    review_metric = _business_review_metric(profile, context)
    business_metrics = _preferred_business_metrics(query, profile, context, state.get("relationship_signals", {}) or {})
    revenue_metric = _best_value_metric(profile, context, state.get("relationship_signals", {}) or {}) or (business_metrics[0] if business_metrics else None)
    price_metric = _best_text_match(query + " price pricing premium", list(df.columns) if df is not None else [], ["price"]) if df is not None else None
    freight_metric = _best_text_match(query + " freight shipping cost", list(df.columns) if df is not None else [], ["freight", "shipping"]) if df is not None else None
    payment_type_column = None
    if df is not None:
        payment_type_column = next(
            (
                col for col in df.columns
                if any(token in str(col).lower() for token in ["payment_type", "payment_types"])
            ),
            None,
        )
        if payment_type_column is None:
            payment_type_column = _best_text_match(query + " payment type method credit card boleto voucher debit", list(df.columns), ["payment", "types"])
    installments_metric = _business_installments_metric(profile, context) or (
        _best_text_match(
            query + " installment installments payment installments",
            list(df.columns) if df is not None else [],
            ["installment"],
        )
        if df is not None
        else None
    )
    size_metrics = []
    for col in (profile.get("numeric_columns", []) or list(df.columns) if df is not None else []):
        lower = str(col).lower()
        if any(token in lower for token in ["weight", "height", "width", "length", "size", "dimension", "volume", "_cm", "_g"]):
            size_metrics.append(col)

    delivered_candidates = []
    estimated_delivery_column = None
    purchase_candidates = []
    status_column = None
    for col in (profile.get("datetime_columns", []) or list(df.columns) if df is not None else []):
        lower = str(col).lower()
        if any(token in lower for token in ["delivered_customer", "delivered", "delivery"]):
            delivered_candidates.append(col)
        if estimated_delivery_column is None and "estimated" in lower:
            estimated_delivery_column = col
        if any(token in lower for token in ["purchase", "approved", "created"]):
            purchase_candidates.append(col)
    delivered_column = None
    if delivered_candidates:
        delivered_candidates.sort(key=lambda col: ("customer" in str(col).lower(), "carrier" not in str(col).lower()), reverse=True)
        delivered_column = delivered_candidates[0]
    purchase_column = None
    if purchase_candidates:
        purchase_candidates.sort(key=lambda col: ("purchase" in str(col).lower(), "approved" in str(col).lower()), reverse=True)
        purchase_column = purchase_candidates[0]
    for col in (profile.get("categorical_columns", []) or list(df.columns) if df is not None else []):
        if "status" in str(col).lower():
            status_column = col
            break

    owner = None
    focus_dimension = None
    owner_candidates = [
        ("geography", geography_column, bool(signals.get("asks_geography"))),
        (
            "customer_segment",
            customer_segment_column,
            any(token in query for token in ["segment", "segments"]),
        ),
        ("seller", seller_column, "seller" in query),
        ("customer", customer_column, "customer" in query),
        ("payment", payment_type_column, any(token in query for token in ["payment method", "payment methods", "payment type", "payment types"])),
        ("category", category_column, "categor" in query),
        ("product", product_column, any(token in query for token in ["product", "products", "sku", "item", "items"])),
    ]
    for candidate_owner, candidate_dimension, matched in owner_candidates:
        if matched and candidate_dimension:
            owner = candidate_owner
            focus_dimension = candidate_dimension
            break

    resolved = {
        "owner": owner,
        "focus_dimension": focus_dimension,
        "time_column": time_column,
        "order_column": order_column,
        "customer_column": customer_column,
        "seller_column": seller_column,
        "category_column": category_column,
        "product_column": product_column,
        "customer_segment_column": customer_segment_column,
        "customer_geo_column": customer_geo_column,
        "seller_geo_column": seller_geo_column,
        "geography_column": geography_column,
        "review_metric": review_metric,
        "revenue_metric": revenue_metric,
        "price_metric": price_metric,
        "freight_metric": freight_metric,
        "payment_type_column": payment_type_column,
        "installments_metric": installments_metric,
        "size_metrics": size_metrics,
        "purchase_column": purchase_column,
        "delivered_column": delivered_column,
        "estimated_delivery_column": estimated_delivery_column,
        "status_column": status_column,
    }

    role_selected = []
    if focus_dimension:
        role_selected.append(focus_dimension)
    if signals.get("asks_growth") and time_column:
        role_selected.append(time_column)
    if signals.get("asks_demand"):
        role_selected.extend([col for col in [order_column, customer_column] if col])
    if signals.get("asks_price"):
        if price_metric or revenue_metric:
            role_selected.append(price_metric or revenue_metric)
    if signals.get("asks_revenue"):
        if revenue_metric:
            role_selected.append(revenue_metric)
    if signals.get("asks_payment"):
        role_selected.extend([
            col
            for col in [
                payment_type_column,
                installments_metric,
                revenue_metric,
                price_metric,
                order_column,
                customer_column,
                customer_geo_column or geography_column,
                category_column,
                status_column,
            ]
            if col
        ])
    if any(term in query for term in ["credit card", "boleto", "voucher", "debit card"]):
        role_selected.extend([col for col in [payment_type_column, customer_column, customer_geo_column or geography_column] if col])
    if "installment" in query:
        role_selected.extend([col for col in [installments_metric, revenue_metric or price_metric, category_column, payment_type_column] if col])
    if "intervention" in query or "need intervention" in query:
        if revenue_metric:
            role_selected.append(revenue_metric)
    if signals.get("asks_quality") and review_metric:
        role_selected.append(review_metric)
    if signals.get("asks_delivery"):
        role_selected.extend([col for col in [purchase_column, delivered_column, estimated_delivery_column, geography_column, seller_column, category_column, review_metric] if col])
    if "freight" in query or "shipping" in query or "cost relative to price" in query or ("cost" in query and "serve" in query) or "expensive to serve" in query or "cost to deliver" in query:
        role_selected.extend([
            col
            for col in [
                freight_metric,
                price_metric or revenue_metric,
                revenue_metric,
                review_metric,
                order_column,
                customer_column,
                time_column,
                geography_column,
                seller_column,
                category_column,
                product_column,
            ]
            if col
        ])
    if any(term in query for term in ["larger", "heavier", "delays", "delay"]):
        role_selected.extend(size_metrics[:4])
        role_selected.extend([col for col in [purchase_column, delivered_column, estimated_delivery_column, review_metric] if col])
    if any(term in query for term in ["late", "early", "estimated", "actual delivery", "delivery gap", "delivery time"]):
        role_selected.extend([col for col in [purchase_column, delivered_column, estimated_delivery_column, order_column, geography_column, seller_column, category_column, review_metric] if col])
    if "distance" in query:
        role_selected.extend([col for col in [seller_geo_column, customer_geo_column, status_column, order_column] if col])
    if any(term in query for term in ["delivered", "canceled", "cancelled", "unavailable", "invoiced", "failure rate", "operational issues", "cancellations increasing"]):
        role_selected.extend([col for col in [status_column, order_column, category_column, seller_column, time_column] if col])
    if "cancel" in query and status_column:
        role_selected.append(status_column)
    if "return" in query and status_column:
        role_selected.append(status_column)
    if "dominate" in query or "dependent" in query or "dependency" in query or "dependence" in query or "overdependent" in query:
        role_selected.extend([col for col in [seller_column, category_column, product_column, order_column, revenue_metric] if col])
    if "poor reviews" in query or "high revenue but poor reviews" in query:
        role_selected.extend([col for col in [revenue_metric, review_metric] if col])
    if any(term in query for term in ["basket", "baskets", "bundle", "bundles", "bundled", "bought together", "commonly bundled", "cross-sell", "cross sell", "same order", "recommend", "recommended"]):
        role_selected.extend([col for col in [order_column, product_column or focus_dimension, category_column, revenue_metric or price_metric] if col])
    if any(term in query for term in ["cohort", "cohorts", "acquired", "acquisition", "retain", "retains", "retention", "bought again", "buy again", "churn", "long term"]):
        role_selected.extend([col for col in [customer_column, order_column, time_column, revenue_metric or price_metric, customer_geo_column or geography_column, customer_segment_column] if col])
    if any(term in query for term in ["anomaly", "anomalies", "unusual", "suspicious", "fraud", "fake", "duplicate", "duplicates", "rapid", "rapidly", "excessive", "spike", "spikes"]):
        role_selected.extend([col for col in [
            order_column,
            customer_column,
            seller_column,
            time_column,
            revenue_metric or price_metric,
            price_metric,
            installments_metric,
            review_metric,
            customer_geo_column or geography_column,
            customer_segment_column,
            category_column,
            payment_type_column,
        ] if col])
    if any(term in query for term in ["missing", "null", "duplicate rows", "inconsistent", "invalid", "negative", "impossible", "broken", "data quality"]):
        role_selected.extend(list(df.columns))
    if "underperform" in query:
        role_selected.extend([col for col in [geography_column, order_column, revenue_metric, review_metric, freight_metric, purchase_column, delivered_column] if col])
    if any(term in query for term in ["crisis", "crises", "brand trust", "damage brand", "trust"]):
        role_selected.extend([col for col in [seller_column, geography_column, review_metric, order_column, status_column, purchase_column, delivered_column] if col])
    if "cluster" in query or "clustered" in query:
        role_selected.extend([col for col in [time_column, order_column, delivered_column, estimated_delivery_column] if col])
    if "premium pricing" in query:
        role_selected.extend([col for col in [price_metric or revenue_metric, review_metric, order_column] if col])
    if any(term in query for term in ["overpriced", "elasticity", "price increases", "price increase", "price wars", "discount", "convert best", "conversion"]):
        role_selected.extend([col for col in [price_metric or revenue_metric, review_metric, order_column, time_column, category_column, seller_column] if col])

    resolved["selected_columns"] = _dedupe_preserve_order(role_selected)
    return resolved

def extract_mentioned_columns(query: str, columns: list[str]):
    normalized_query = normalize(query)
    mentioned = []
    for col in columns:
        if normalize(col) in normalized_query:
            mentioned.append(col)
    return mentioned


def extract_columns_from_intent_clause(query: str, columns: list[str], keywords: list[str]):
    lowered = query.lower()
    match_positions = []
    for keyword in keywords:
        match = re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", lowered)
        if match:
            match_positions.append(match.start())
    if not match_positions:
        return []

    start = min(match_positions)
    boundary_pattern = r"\band\b\s+(summary|statistics|average|mean|median|relationship|correlation|compare|difference|distribution|frequency|predict|forecast|outlier|outliers|trend|breakdown)\b"
    boundary_match = re.search(boundary_pattern, lowered[start + 1 :], flags=re.IGNORECASE)
    end = len(query)
    if boundary_match:
        end = start + 1 + boundary_match.start()
    clause = query[start:end]
    return extract_mentioned_columns(clause, columns)

def get_numeric_columns(df):
    if df is None:
        return []
    return [c for c in df.select_dtypes(include="number").columns if c in df.columns]

def get_temporal_numeric_columns(df):
    numeric_columns = get_numeric_columns(df)
    temporal_keywords = {"year", "date", "time", "month", "day", "week", "quarter"}
    temporal_columns = [
        col for col in numeric_columns
        if any(keyword in col.lower() for keyword in temporal_keywords)
    ]

    if temporal_columns:
        return temporal_columns

    inferred = []
    for col in numeric_columns:
        series = df[col].dropna()
        if series.empty:
            continue
        if series.between(1900, 2100).mean() >= 0.8:
            inferred.append(col)
    return inferred

def classify_analytic_intent(query: str):
    """
    Classifies high-level analytic intent from user query.
    """

    query = query.lower()

    intent_map = {
        "comparison": ["compare", "vs", "versus", "difference", "against", "return more often", "spend most", "cheap items", "expensive ones", "overpriced", "relative to", "price war", "price wars", "spend more", "underperform", "underperformance", "affect cancellation rates", "prefer credit card vs boleto"],
        "temporal": ["trend", "pattern", "patterns", "over time", "per day", "daily", "per week", "weekly", "per month", "growth", "decline", "monthly", "quarterly", "holiday", "holidays", "black friday", "weekend", "weekday", "seasonal", "seasonality", "month-over-month", "quarter-over-quarter", "growing fastest", "first and second purchase", "between first and second purchase", "loyal over time", "becoming more loyal", "dormant customers", "acquisition month"],
        "composition": ["breakdown", "distribution", "percentage", "share", "portion", "repeat purchases", "repeat purchase rate", "buy once", "only buy once", "buy again", "bought again", "retain", "retains", "retention", "cohort", "cohorts", "acquired", "acquisition", "top customers", "product mix", "overdependent", "dependent", "dependency", "dependence", "few categories", "basket", "baskets", "bundle", "bundles", "bundled", "bought together", "commonly bundled", "cross-sell", "cross sell", "same order", "specific payment types", "payment methods are most used"],
        "relationship": ["correlation", "correlate", "relationship", "impact", "effect", "influence", "affect", "cause", "causal", "drive", "drives", "driver", "drivers", "reduce", "reduces", "lower", "lowers", "harsher", "harsh", "elasticity", "correlate with premium categories"],
        "extremes": ["top", "bottom", "highest", "lowest", "max", "min", "best", "worst", "best sellers", "rarely sold", "sell the most"],
        "profiling": ["average", "mean", "median", "summary", "stats", "statistics", "total", "how many", "count", "revenue", "orders", "customers", "profit proxy", "lifetime value", "lifetime value proxy", "repeat behavior", "sell", "sold", "pricing", "volume", "discount", "convert", "recommend", "recommended"],
        "outliers": ["outlier", "outliers", "unusual", "anomaly", "anomalies", "suspicious", "fraud", "fake", "duplicate", "duplicates", "rapid", "rapidly", "excessive"],
        "data_quality": ["missing", "null", "duplicate rows", "inconsistent", "invalid", "negative", "impossible", "broken", "data quality"],
        "investigative": ["why", "drill down", "details", "explain", "scaling efficiently", "lower quality", "crisis", "crises", "brand trust", "damage brand", "clustered"],
        "predictive": ["forecast", "predict", "what if", "estimate"]
    }

    for intent, keywords in intent_map.items():
        if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in keywords):
            return intent

    return "unknown"

def parse_number(value: str):
    value = value.lower().strip()
    if "k" in value:
        return float(value.replace("k", "")) * 1000
    if "m" in value:
        return float(value.replace("m", "")) * 1_000_000
    return float(value)

def infer_numeric_column(query: str, df, raw_value: str | None = None, temporal_only: bool = False):
    if df is None:
        return None

    numeric_columns = get_temporal_numeric_columns(df) if temporal_only else get_numeric_columns(df)
    if not numeric_columns:
        return None

    query_lower = query.lower()
    normalized_columns = {c.lower(): c for c in numeric_columns}

    for key, original in normalized_columns.items():
        if key in query_lower:
            return original

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    if raw_value:
        value_lower = raw_value.lower()
        digits = re.sub(r"[^\d]", "", value_lower)
        if temporal_only and len(digits) == 4:
            year_val = int(digits)
            candidate_columns = []
            for col in numeric_columns:
                series = df[col].dropna()
                if series.empty:
                    continue
                within_range = series.between(year_val - 20, year_val + 20).mean()
                if within_range >= 0.5:
                    candidate_columns.append(col)
            if len(candidate_columns) == 1:
                return candidate_columns[0]

    return None

# ------------------------
# CONDITION BUILDERS
# ------------------------

# ---- NUMERIC PARSING ----

def build_numeric_conditions(query: str, df):
    """
    Build numeric conditions from query after mapping numeric synonyms
    Handles >, <, between, and ignores negation words like 'not'
    """
    conditions = []
    query_mapped = query

    patterns = [
        (r'([\w_]+)\s*>\s*([\d\.kKmM]+)', '>'),
        (r'([\w_]+)\s*<\s*([\d\.kKmM]+)', '<'),
        (r'([\w_]+)\s+(newer than|after)\s+([\d]{4})', '>'),
        (r'([\w_]+)\s+(older than|before)\s+([\d]{4})', '<'),
        (r'([\w_]+)\s+between\s+([\d\.kKmM]+)\s+&&\s+([\d\.kKmM]+)', 'between')
    ]
    # VALIDATE COLUMN BEFORE ADDING
    valid_columns = []
    if df is not None:
        valid_columns = [c.lower() for c in df.columns]

    for pattern, op in patterns:
        matches = re.findall(pattern, query_mapped)
        for match in matches:
            if op == "between":
                col, low, high = match
                if df is not None and col.lower() not in valid_columns:
                    inferred_col = infer_numeric_column(query, df, raw_value=high)
                    if not inferred_col:
                        continue
                    col = inferred_col
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": "between",
                    "value": (parse_number(low), parse_number(high))
                })
            else:
                if len(match) == 3:
                    col, _, val = match
                else:
                    col, val = match
                if df is not None and col.lower() not in valid_columns:
                    inferred_col = infer_numeric_column(query, df, raw_value=val)
                    if not inferred_col:
                        continue
                    col = inferred_col
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": op,
                    "value": parse_number(val)
                })        

    implicit_patterns = [
        (r'\b(?:under|below|less than)\s*([\d\.kKmM]+)\b', '<', False),
        (r'\b(?:over|above|more than)\s*([\d\.kKmM]+)\b', '>', False),
        (r'(^|\s)<\s*([\d\.kKmM]+)\b', '<', False),
        (r'(^|\s)>\s*([\d\.kKmM]+)\b', '>', False),
        (r'\b(?:newer than|after)\s*([\d]{4})\b', '>', True),
        (r'\b(?:older than|before)\s*([\d]{4})\b', '<', True),
        (r'\bbetween\s+([\d\.kKmM]+)\s+&&\s+([\d\.kKmM]+)\b', 'between', None)
    ]

    existing_keys = {
        (c["column"], c["operator"], str(c["value"]))
        for c in conditions
    }

    for pattern, op, temporal_only in implicit_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        for match in matches:
            if op == "between":
                low, high = match
                col = infer_numeric_column(query, df, raw_value=high)
                if not col:
                    continue
                candidate = {
                    "type": "condition",
                    "column": col,
                    "operator": "between",
                    "value": (parse_number(low), parse_number(high))
                }
            else:
                if isinstance(match, tuple):
                    raw_value = match[-1]
                else:
                    raw_value = match
                col = infer_numeric_column(
                    query,
                    df,
                    raw_value=raw_value,
                    temporal_only=bool(temporal_only)
                )
                if not col:
                    continue
                candidate = {
                    "type": "condition",
                    "column": col,
                    "operator": op,
                    "value": parse_number(raw_value)
                }

            key = (candidate["column"], candidate["operator"], str(candidate["value"]))
            if key not in existing_keys:
                conditions.append(candidate)
                existing_keys.add(key)

    return conditions

# ---- CATEGORICAL PARSING ----

def build_categorical_conditions(query: str, df):
    conditions = []
    if df is None:
        return conditions

    categorical_columns = df.select_dtypes(include=["object", "string"]).columns
    query_lower = query.lower()
    query_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query_lower))
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "between", "by", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "the", "to",
        "what", "when", "where", "which", "who", "why", "with",
    }
    contrast_query = any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", query_lower)
        for term in ["vs", "versus", "compare", "compared", "prefer", "prefers", "preference", "difference"]
    )

    def value_matches_query(value_str: str) -> bool:
        value_tokens = [token for token in re.split(r"[_\W]+", value_str) if token]
        if not value_tokens:
            return False
        compact_value = re.sub(r"[\W_]+", "", value_str)
        compact_query = re.sub(r"[\W_]+", "", query_lower)
        phrase_value = re.sub(r"[_\-]+", " ", value_str).strip()

        if len(value_tokens) == 1:
            token = value_tokens[0]
            if token in stopwords or len(token) < 3:
                return False
            return token in query_tokens or bool(compact_value and compact_value in compact_query)

        phrase_pattern = rf"(?<!\w){re.escape(phrase_value)}(?!\w)"
        return bool(re.search(phrase_pattern, query_lower)) or bool(compact_value and compact_value in compact_query)

    for col in categorical_columns:
        non_null = df[col].dropna()
        unique_count = int(non_null.nunique(dropna=True))
        column_mentioned = normalize(col) in normalize(query_lower)
        if unique_count > 1000 and not column_mentioned:
            continue
        values = non_null.unique()
        matched_vals = []
        for value in values:
            value_str = str(value).strip().lower()
            if not value_str:
                continue
            if len(value_str) < 3:
                continue
            if value_str in stopwords:
                continue

            if value_matches_query(value_str):
                matched_vals.append(value)

        if matched_vals:
            if contrast_query and len(matched_vals) >= 2:
                continue
            if len(matched_vals) == 1:
                conditions.append({
                    "type": "condition",
                    "column": col,
                    "operator": "equals",
                    "value": matched_vals[0]
                })
            else:
                # Multiple matches -> ORlogic
                conditions.append({
                    "type": "logic",
                    "operator": "or",
                    "conditions": [
                        {
                            "type": "condition",
                            "column": col,
                            "operator": "equals",
                            "value": v
                        }
                        for v in matched_vals
                    ]
                })

    return conditions

# ------------------
# NAGGATION SUPPORT
# ------------------

def detect_negation(query: str) -> bool:
    """
    Detects whether a query contains negation.
    Only triggers on explicit negation words and ignores numeric comparative words.
    """
    query = query.lower()
    # Explicit negation words
    NAGATION_WORDS = [
        r'\bnot\b',
        r'\bno\b',
        r'\bnever\b',
        r'\bnon\b',
        r'\bwithout\b',
        r'\bexclude\b',
        r'\bexcluding\b',
        r'\bexcept\b'
    ]

    # If any negation word is found, return True
    for word in NAGATION_WORDS:
        if re.search(word, query):
            return True

    return False

def strip_negation_words(query: str) -> str:
    """
    Removes negation words from query AFTER detection,
    so they don't interfere with parsing.
    """
    NEGATION_WORDS = [
        r'\bnot\b',
        r'\bno\b',
        r'\bnever\b',
        r'\bwithout\b',
        r'\bexclude\b',
        r'\bexcluding\b',
        r'\bexcept\b'
    ]

    for pattern in NEGATION_WORDS:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)

    return query.strip()

def extract_negation_scopes(query: str):
    """
    Finds phrases where negation applies.
    Example:
        "not red" → ["red"]
        "exclude toyota" → ["toyota"]
    """
    patterns = [
        r'not\s+(\w+)',
        r'exclude\s+(\w+)',
        r'excluding\s+(\w+)',
        r'without\s+(\w+)'
    ]

    negated_terms = []

    for pattern in patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        negated_terms.extend(matches)

    return negated_terms

def apply_negation_to_condition(cond):
    if cond["operator"] == "equals":
        cond["operator"] = "!="
    elif cond["operator"] == ">":
        cond["operator"] = "<="
    elif cond["operator"] == "<":
        cond["operator"] = ">="
    return cond

def apply_negation(node):
    if node["type"] == "condition":
        return apply_negation_to_condition(node)
    elif node["type"] == "logic":
        return {
            "type": "logic",
            "operator": node["operator"],
            "conditions": [apply_negation(c) for c in node["conditions"]]
        }
    return node

def convert_reasoning_to_ast(reasoning, df):
    if not reasoning:
        return None

    constraints = reasoning.get("constraints", [])
    logic_op = reasoning.get("logic", "and")

    nodes = []

    for c in constraints:
        conf_map = {
            "high": 0.9,
            "medium": 0.75,
            "low": 0.6
        }
        node = {
            "type": "condition",
            "column": c.get("field"),
            "operator": c.get("operator"),
            "value": c.get("value"),
            "confidence": conf_map.get(c.get("confidence"), 0.7)
        }
        nodes.append(node)

    if not nodes:
        return None

    if len(nodes) == 1:
        return nodes[0]

    return {
        "type": "logic",
        "operator": logic_op if logic_op in ["and", "or"] else "and",
        "conditions": nodes
    }

# ----------------------------
# AST BUILDING
# ----------------------------
def build_ast(query: str, df):
    query = query.strip()
    query = map_numeric_synonyms(query)

    # PROTECT BETWEEN
    query = re.sub(
        r'between\s+([\d\.kKmM]+)\s+(and|&&)\s+([\d\.kKmM]+)',
        lambda m: f"between {m.group(1)} && {m.group(3)}",
        query,
        flags=re.IGNORECASE
    )

    # ---- HANDLES OR ----
    or_parts = re.split(r'\s+or\s+', query)
    
    if len(or_parts) > 1:
        nodes = [build_ast(p, df) for p in or_parts if p.strip()]
        nodes = [n for n in nodes if n]

        if not nodes:
            return None
    
        if len(nodes) == 1:
            return nodes[0]
    
        ast_node = {
            "type": "logic",
            "operator": "or",
            "conditions": nodes
        }
        
        return ast_node

    negated_terms = extract_negation_scopes(query)
    query = strip_negation_words(query)

    # ---- BASE CONDITIONS ----     
    conds = []
    conds.extend(build_numeric_conditions(query, df))
    conds.extend(build_categorical_conditions(query, df))

    for cond in conds:
        val = str(cond.get("value", "")).lower()
        col = str(cond.get("column", "")).lower()

        # If value OR column matches a negated term → flip
        if any(term in val or term in col for term in negated_terms):
            apply_negation_to_condition(cond)

    if not conds:
        return None
    
    ast_node = None
    '''if is_negation:
        #conds = [apply_negation_to_condition(c) for c in conds if c["type"] == "condition"]
        for c in conds:
            if c.get("type") == "condition":
                c = apply_negation_to_condition(c)'''

    if len(conds) == 1:
        ast_node = conds[0]
    
    else:
        ast_node = {
            "type": "logic",
            "operator": "and",
            "conditions": conds
        }
    
    return ast_node
    

# -----------------------
# FLAT FILTER EXTRACTOR
# -----------------------
def extract_filters(node):
    """
    Flattens AST into a list of filters while preserving OR blocks.
    """
    if node is None:
        return []

    # Leaf condition
    if node["type"] == "condition":
        return [node]

    filters = []

    if node["type"] == "logic":
        if node["operator"] == "or":
            # PRESERVE OR BLOCK as a single unit
            filters.append(node)
        else:
            # Flatten AND
            for child in node.get("conditions", []):
                filters.extend(extract_filters(child))

    return filters

'''def extract_filters(node):
    if node is None:
        return None

    # Handle leaf nodes(individual conditions)
    if node["type"] == "condition":
        return {
            "type": "condition",
            "column": node.get("column"),
            "operator": node.get("operator"),
            "value": node.get("value")
        }
        
    # Handle logic nodes (AND/OR groups)
    elif node["type"] == "logic":
        return {
            "type": "logic",
            "operator": node.get("operator"),
            # This rebuilds the 'conditions' list by calling this function on each child
            "conditions": [extract_filters(child) for child in node.get("conditions", [])]
        }

    return node'''

def build_final_ast(filters):
    """
    Rebuilds AST from flat filters while preserving OR logic blocks.
    """
    if not filters:
        return None

    logic_nodes = []
    conditions = []

    for f in filters:
        if f.get("type") == "logic":
            logic_nodes.append(f)  # keep OR blocks intact
        else:
            conditions.append({
                "type": "condition",
                "column": f["column"],
                "operator": f["operator"],
                "value": f["value"]
            })

    nodes = []
    nodes.extend(logic_nodes)
    nodes.extend(conditions)

    if not nodes:
        return None

    if len(nodes) == 1:
        return nodes[0]

    return {
        "type": "logic",
        "operator": "and",
        "conditions": nodes
    }

def merge_ast_nodes(base_ast, extra_ast):
    if base_ast and extra_ast:
        return {
            "type": "logic",
            "operator": "and",
            "conditions": [base_ast, extra_ast]
        }
    return extra_ast or base_ast

def estimate_filter_confidence(node):
    if node is None:
        return 0.0

    if node["type"] == "logic":
        child_scores = [estimate_filter_confidence(child) for child in node.get("conditions", [])]
        return min(child_scores) if child_scores else 0.0

    if node.get("confidence") is not None:
        return node["confidence"]

    if node.get("source") == "semantic":
        return 0.65

    if node.get("operator") in {">", "<", ">=", "<=", "between"}:
        return 0.9

    if node.get("operator") in {"equals", "!=", "contains"}:
        return 0.8

    return 0.7

def has_numeric_constraint(filters, numeric_columns):
    return any(
        f.get("type") == "condition" and f.get("column") in numeric_columns
        for f in filters
    )

def query_has_unresolved_numeric_phrase(query: str):
    comparator_patterns = [
        r"\b(?:under|below|less than|over|above|more than)\s*[\d\.kKmM]+\b",
        r"(^|\s)[<>]\s*[\d\.kKmM]+\b",
        r"\bbetween\s+[\d\.kKmM]+\s+(?:and|&&)\s+[\d\.kKmM]+\b"
    ]
    return any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in comparator_patterns)

def query_has_semantic_magnitude(query: str):
    words = ["cheap", "affordable", "budget", "expensive", "premium", "high", "low", "luxury"]
    return any(re.search(rf"\b{word}\b", query, flags=re.IGNORECASE) for word in words)

def should_call_llm_for_intent(state: AnalystState, ast, filters):
    if not state.get("enable_llm_reasoning", True):
        return False

    intent = state.get("intent", {}) or {}
    query = state.get("business_question", "").lower()
    selected_columns = intent.get("selected_columns") or state.get("selected_columns") or []
    group_by = intent.get("group_by")
    aggregate_column = intent.get("aggregate_column")
    analytic_intent = intent.get("analytic_intent")

    if intent.get("low_confidence"):
        return True

    numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
    ambiguous_terms = [
        "cheap",
        "expensive",
        "affordable",
        "premium",
        "budget",
        "luxury",
    ]

    if query_has_unresolved_numeric_phrase(query) and not has_numeric_constraint(filters, numeric_columns):
        return True

    if query_has_semantic_magnitude(query) and not filters:
        return True

    # If the parser already resolved a clear analysis request with explicit columns,
    # defer to deterministic planning instead of paying for an unnecessary LLM pass.
    if analytic_intent in {"relationship", "comparison", "profiling", "outliers", "temporal", "composition"}:
        if len(selected_columns) >= 2:
            return False
        if analytic_intent in {"profiling", "outliers"} and selected_columns:
            return False
        if group_by and aggregate_column:
            return False

    # Filter-only queries with a confident symbolic parse do not need LLM repair.
    if ast is not None and filters:
        return any(term in query for term in ambiguous_terms)

    # If nothing was parsed and no columns were resolved, ask the LLM as a last resort.
    return not selected_columns or any(term in query for term in ambiguous_terms)
'''def build_final_ast(processed_node):
    # If there's nothing there, return None
    if not processed_node:
        return None

    # If it's already a dictionary (the tree), just return it.
    if isinstance(processed_node, dict):
        return processed_node

    # If for some reason you still have a list, wrap it in an AND
    if isinstance(processed_node, list):
        if len(processed_node) == 1:
            return processed_node[0]
        return {
            "type": "logic",
            "operator": "and",
            "conditions": processed_node
        }'''

# ------------------------
# INTENT DETECTION 
# ------------------------
def detect_intents(query: str):
    query = query.lower()

    intents = []

    # --- COMPARISON ---
    if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in ["compare", "vs", "versus", "difference", "affect", "overpriced", "relative to", "price war", "price wars"]):
        intents.append({"type": "comparison", "confidence": 0.8})
    if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in ["correlation", "correlate", "relationship", "elasticity", "drive", "drives", "driver", "drivers", "reduce", "reduces", "lower", "lowers", "harsher", "harsh"]):
        intents.append({"type": "relationship", "confidence": 0.8})

    # --- TEMPORAL ---
    if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in ["over time", "trend", "pattern", "patterns", "monthly", "weekly", "yearly", "daily", "growth", "month-over-month", "quarter-over-quarter", "holiday", "holidays", "black friday", "weekend", "weekday", "seasonal", "seasonality", "declining", "growing fastest", "first and second purchase", "between first and second purchase", "loyal over time", "becoming more loyal", "dormant customers"]):
        intents.append({"type": "temporal", "confidence": 0.75})

    # --- EXTREMES ---
    if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in ["top", "highest", "lowest", "max", "min", "best", "worst", "best sellers", "rarely sold"]):
        intents.append({"type": "extremes", "confidence": 0.8})

    # --- COMPOSITION ---
    if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in ["percentage", "ratio", "breakdown", "share", "repeat purchases", "repeat purchase rate", "buy once", "only buy once", "buy again", "bought again", "retain", "retains", "retention", "cohort", "cohorts", "acquired", "acquisition", "product mix", "overdependent", "basket", "baskets", "bundle", "bundles", "bundled", "bought together", "commonly bundled", "cross-sell", "cross sell", "same order"]):
        intents.append({"type": "composition", "confidence": 0.75})

    # --- PROFILING ---
    if any(re.search(rf"(?<!\\w){re.escape(word)}(?!\\w)", query) for word in ["average", "mean", "distribution", "median", "summary", "statistics", "total", "count", "how many", "revenue", "orders", "customers", "profit proxy", "lifetime value", "lifetime value proxy", "repeat behavior", "sell", "sold", "volume", "pricing", "discount", "convert"]):
        intents.append({"type": "profiling", "confidence": 0.7})

    # --- OUTLIER DETECTION ---
    if any(word in query for word in ["outlier", "outliers", "unusual", "anomaly", "anomalies", "suspicious", "fraud", "fake", "duplicate", "duplicates", "rapid", "rapidly", "excessive"]):
        intents.append({"type": "outliers", "confidence": 0.8})
    if any(word in query for word in ["missing", "null", "duplicate rows", "inconsistent", "invalid", "negative", "impossible", "broken", "data quality"]):
        intents.append({"type": "data_quality", "confidence": 0.85})

    return intents
# ------------------------
# INTENT → OPERATIONS (STEP 3)
# ------------------------
def map_intents_to_operations(intents):
    ops = []

    mapping = {
        "comparison": ["groupby", "aggregate"],
        "temporal": ["time_series"],
        "extremes": ["sort", "limit"],
        "composition": ["ratio"],
        "profiling": ["describe"]
    }

    for intent in intents:
        intent_type = intent.get("type")
        if intent_type in mapping:
            ops.extend(mapping[intent_type])

    # remove duplicates while preserving order
    seen = set()
    final_ops = []
    for op in ops:
        if op not in seen:
            seen.add(op)
            final_ops.append(op)

    return final_ops


def _has_analysis_request(analytic_intent: str, query: str) -> bool:
    analysis_intents = {
        "comparison",
        "relationship",
        "temporal",
        "composition",
        "extremes",
        "profiling",
        "outliers",
        "investigative",
        "predictive",
    }
    if analytic_intent in analysis_intents:
        return True

    query = query.lower()
    analysis_keywords = [
        "relationship",
        "correlation",
        "impact",
        "effect",
        "affect",
        "cause",
        "causal",
        "drive",
        "drives",
        "driver",
        "drivers",
        "reduce",
        "reduces",
        "lower",
        "lowers",
        "harsher",
        "harsh",
        "compare",
        "difference",
        "distribution",
        "summary",
        "statistics",
        "average",
        "mean",
        "median",
        "outlier",
        "outliers",
        "unusual",
        "anomaly",
        "anomalies",
        "suspicious",
        "fraud",
        "fake",
        "duplicate",
        "duplicates",
        "rapid",
        "rapidly",
        "excessive",
        "missing",
        "null",
        "duplicate rows",
        "inconsistent",
        "invalid",
        "negative",
        "impossible",
        "broken",
        "data quality",
        "trend",
        "total",
        "count",
        "how many",
        "revenue",
        "orders",
        "customers",
        "growth",
        "month-over-month",
        "quarter-over-quarter",
        "percentage",
        "share",
        "profit proxy",
        "repeat purchases",
        "repeat purchase rate",
        "buy again",
        "buy once",
        "only buy once",
        "lifetime value",
        "lifetime value proxy",
        "dormant customers",
        "loyal over time",
        "return more often",
        "repeat behavior",
        "sell",
        "sold",
        "volume",
        "seasonal",
        "seasonality",
        "product mix",
        "overdependent",
        "best sellers",
        "rarely sold",
    ]
    return any(re.search(rf"(?<!\\w){re.escape(keyword)}(?!\\w)", query) for keyword in analysis_keywords)

# ------------------------
# MAIN NODE
# ------------------------

def intent_parser_node(state: AnalystState) -> AnalystState:
    original_query = state.get("business_question", "")
    query = original_query.lower()
    
    # Select dataset
    if "cleaned_data" in state and state["cleaned_data"] is not None:
        state["active_dataset"] = "cleaned_data"
    else:
        state["active_dataset"] = "dataframe"
    df = state.get(state.get("active_dataset"))
    
    if "intent" not in state:
        state["intent"] = {}

    state["llm_reasoning"] = None

    analytic_intent = classify_analytic_intent(query)
    ast = build_ast(query, df)
    selected_columns = []
    mentioned_numeric = []
    mentioned_categorical = []
    resolved_role_columns = {}

    if df is not None:
        all_columns = list(df.columns)
        selected_columns = extract_mentioned_columns(query, all_columns)
        resolved_role_columns = _resolve_intent_role_columns(query, state, df)
        selected_columns = _dedupe_preserve_order(selected_columns + resolved_role_columns.get("selected_columns", []))
        numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
        categorical_columns = state.get("dataset_profile", {}).get("categorical_columns", [])
        mentioned_numeric = [col for col in selected_columns if col in numeric_columns]
        mentioned_categorical = [col for col in selected_columns if col in categorical_columns]

        effect_words = ["affect", "effect", "influence"]
        compare_words = ["compare", "difference", "better", "vs", "versus"]
        if (
            any(word in query for word in effect_words)
            and len(mentioned_numeric) == 1
            and len(mentioned_categorical) == 1
            and not any(word in query for word in compare_words)
        ):
            state["original_business_question"] = original_query
            query = f"{query} compare {mentioned_numeric[0]} by {mentioned_categorical[0]}"
            state["business_question"] = query
            analytic_intent = "comparison"

    if df is not None:
        numeric_cols = state.get("dataset_profile", {}).get("numeric_columns", [])
        semantic_filters = []
        if analytic_intent not in {"relationship", "comparison"}:
            semantic_filters = map_semantic_filters(query, df, numeric_cols)
        if semantic_filters:
            semantic_ast = {
                "type": "logic",
                "operator": "and",
                "conditions": semantic_filters
            }
            ast = merge_ast_nodes(ast, semantic_ast)

    filters = extract_filters(ast)
    low_confidence = any(
        estimate_filter_confidence(f) < 0.75 for f in filters
    )

    intent = {
        "query": original_query,
        "type": "ast",
        "analytic_intent": analytic_intent,
        "analytic_intents": [],
        "intent_columns": {},
        "ast": ast,
        "filters": filters,
        "has_filters": bool(filters),
        "wants_analysis": _has_analysis_request(analytic_intent, query),
        "group_by": None,
        "group_by_columns": [],
        "aggregation": None,
        "aggregate_column": None,
        "aggregate_columns": [],

        "intents": [],
        "operations_hint": [],
        "output_mode": None,
        "confidence": None,
        "low_confidence": low_confidence,
        "selected_columns": selected_columns,
        "resolved_role_columns": resolved_role_columns,
    }
    # ------------------------
    # AGGREGATION 
    # ------------------------
    
    aggregation_keywords = {
        "mean": ["average", "mean", "avg"],
        "sum": ["total", "sum"],
        "max": ["max", "maximum", "highest"],
        "min": ["min", "minimum", "lowest"]
    }
    intent_type = intent.get("analytic_intent")
    
    if df is not None:
        numeric_columns = state.get("dataset_profile", {}).get("numeric_columns", [])
        for agg_type, keywords in aggregation_keywords.items():
            if any(word in query for word in keywords):
                intent["aggregation"] = agg_type
                if mentioned_numeric:
                    intent["aggregate_column"] = mentioned_numeric[0]
                    intent["aggregate_columns"] = mentioned_numeric
                else:
                    for col in numeric_columns:
                        if normalize(col) in normalize(query):
                            intent["aggregate_column"] = col
                            intent["aggregate_columns"] = [col]
                            break
                break

        # If no aggregation keyword found, infer from analytic intent
        if not intent["aggregation"]:
            if intent_type in ["comparison", "extremes", "profiling"]:
                # Default to mean for comparison/profiling
                intent["aggregation"] = "mean"
                if mentioned_numeric:
                    intent["aggregate_column"] = mentioned_numeric[0]
                    intent["aggregate_columns"] = mentioned_numeric
                elif resolved_role_columns.get("revenue_metric"):
                    intent["aggregate_column"] = resolved_role_columns["revenue_metric"]
                    intent["aggregate_columns"] = [resolved_role_columns["revenue_metric"]]
                elif numeric_columns:
                    intent["aggregate_column"] = numeric_columns[0]
                    intent["aggregate_columns"] = [numeric_columns[0]]
            elif intent_type == "temporal":
                # For trends, default to sum over time
                intent["aggregation"] = "sum"
                if mentioned_numeric:
                    intent["aggregate_column"] = mentioned_numeric[0]
                    intent["aggregate_columns"] = mentioned_numeric
                elif resolved_role_columns.get("revenue_metric"):
                    intent["aggregate_column"] = resolved_role_columns["revenue_metric"]
                    intent["aggregate_columns"] = [resolved_role_columns["revenue_metric"]]
                elif numeric_columns:
                    intent["aggregate_column"] = numeric_columns[0]
                    intent["aggregate_columns"] = [numeric_columns[0]]

    # ------------------------
    # GROUP BY )
    # ------------------------
    
    if df is not None:
        categorical_columns = state.get("dataset_profile", {}).get("categorical_columns", [])
        if mentioned_categorical:
            intent["group_by"] = mentioned_categorical[0]
            intent["group_by_columns"] = mentioned_categorical
        else:
            for col in categorical_columns:
                if normalize(col) in normalize(query):
                    intent["group_by"] = col
                    intent["group_by_columns"] = [col]
                    break

        # If no group_by detected, infer from analytic intent
        if not intent["group_by"]:
            if intent_type in ["comparison", "composition"]:
                if resolved_role_columns.get("focus_dimension"):
                    intent["group_by"] = resolved_role_columns["focus_dimension"]
                    intent["group_by_columns"] = [resolved_role_columns["focus_dimension"]]
                elif resolved_role_columns.get("geography_column"):
                    intent["group_by"] = resolved_role_columns["geography_column"]
                    intent["group_by_columns"] = [resolved_role_columns["geography_column"]]
                if categorical_columns:
                    if not intent["group_by"]:
                        intent["group_by"] = categorical_columns[0]
                        intent["group_by_columns"] = [categorical_columns[0]]

    if resolved_role_columns.get("revenue_metric") and (
        any(term in query for term in ["revenue", "sales", "spend", "value"])
        and not intent.get("aggregate_column")
    ):
        intent["aggregate_column"] = resolved_role_columns["revenue_metric"]
        intent["aggregate_columns"] = [resolved_role_columns["revenue_metric"]]
    
    '''# --- ENSURE NEGATION APPLIED TO FILTERS ---
    if filters:
        for f in filters:
            if f.get("type") == "condition" and is_negation:
                apply_negation_to_condition(f)
    
    # --- MERGE SEMANTIC FILTERS INTO AST ---
    if df is not None and filters:
        if ast is None:
            ast = {
                "type": "logic",
                "operator": "and",
                "conditions": filters
            }
        else:
            # Combine existing AST with filters
            ast = {
                "type": "logic",
                "operator": "and",
                "conditions": [ast] + filters
            }'''
    
    state["intent"] = intent
    
    # ------------------------
    # APPLY INTENT DETECTION
    # ------------------------
    detected_intents = detect_intents(query)
    state["intent"]["intents"] = detected_intents
    state["intent"]["analytic_intents"] = _combine_analytic_intents(analytic_intent, detected_intents)
    if df is not None:
        all_columns = list(df.columns)
        state["intent"]["intent_columns"] = {
            "aggregation": extract_columns_from_intent_clause(
                query,
                all_columns,
                ["summary", "statistics", "average", "mean", "median", "total", "sum"],
            ),
            "relationship": extract_columns_from_intent_clause(
                query,
                all_columns,
                ["relationship", "correlation", "correlate", "cause", "causal", "drive", "drives", "driver", "drivers", "reduce", "reduces", "lower", "lowers", "harsher", "harsh", "elasticity"],
            ),
            "comparison": extract_columns_from_intent_clause(
                query,
                all_columns,
                ["compare", "difference", "affect", "impact", "effect", "overpriced", "discount", "price wars", "price increases", "convert"],
            ),
            "distribution": extract_columns_from_intent_clause(
                query,
                all_columns,
                ["distribution", "frequency", "mode", "cardinality", "rare"],
            ),
        }
    if selected_columns:
        state["selected_columns"] = selected_columns

    # ------------------------
    # APPLY OPERATION MAPPING
    # ------------------------
    operations_hint = map_intents_to_operations(detected_intents)
    state["intent"]["operations_hint"] = operations_hint

    if should_call_llm_for_intent(state, ast, filters):
        state = llm_reasoning_node(state)
        reasoning = state.get("llm_reasoning")
        llm_ast = convert_reasoning_to_ast(reasoning, df)
        if llm_ast:
            state["intent"]["ast"] = merge_ast_nodes(state["intent"].get("ast"), llm_ast)
        if reasoning:
            llm_group_by = reasoning.get("group_by", []) or []
            if llm_group_by and not state["intent"].get("group_by"):
                state["intent"]["group_by"] = llm_group_by[0]
            merged_group_columns = _dedupe_preserve_order((state["intent"].get("group_by_columns") or []) + llm_group_by)
            if merged_group_columns:
                state["intent"]["group_by_columns"] = merged_group_columns

            llm_aggregation = reasoning.get("aggregation", {}) or {}
            llm_agg_type = llm_aggregation.get("type")
            llm_agg_target = llm_aggregation.get("target")
            if llm_agg_type and llm_agg_type != "none" and not state["intent"].get("aggregation"):
                state["intent"]["aggregation"] = llm_agg_type
            if llm_agg_target and not state["intent"].get("aggregate_column"):
                state["intent"]["aggregate_column"] = llm_agg_target
            merged_aggregate_columns = _dedupe_preserve_order(
                (state["intent"].get("aggregate_columns") or []) + ([llm_agg_target] if llm_agg_target else [])
            )
            if merged_aggregate_columns:
                state["intent"]["aggregate_columns"] = merged_aggregate_columns

            llm_columns = [
                c.get("field")
                for c in reasoning.get("constraints", [])
                if isinstance(c, dict)
            ]
            llm_columns.extend(llm_group_by)
            if llm_agg_target:
                llm_columns.append(llm_agg_target)
            final_selected = _dedupe_preserve_order(
                (state["intent"].get("selected_columns") or []) + llm_columns
            )
            state["intent"]["selected_columns"] = final_selected
            if final_selected:
                state["selected_columns"] = final_selected
    else:
        state["llm_reasoning_status"] = "skipped_symbolic_sufficient"
        print("\n[INFO] Skipping LLM - symbolic parsing sufficient")

    if state["intent"].get("ast"):
        state["intent"]["filters"] = extract_filters(state["intent"]["ast"])
    else:
        state["intent"]["filters"] = []
    state["intent"]["has_filters"] = bool(state["intent"]["filters"])

    state["intent"]["low_confidence"] = any(
        estimate_filter_confidence(f) < 0.75 for f in state["intent"]["filters"]
    )

    final_selected_columns = state["intent"].get("selected_columns") or selected_columns
    if state["intent"].get("group_by"):
        final_selected_columns = _dedupe_preserve_order(final_selected_columns + [state["intent"]["group_by"]])
    if state["intent"].get("group_by_columns"):
        final_selected_columns = _dedupe_preserve_order(final_selected_columns + state["intent"]["group_by_columns"])
    if state["intent"].get("aggregate_column"):
        final_selected_columns = _dedupe_preserve_order(final_selected_columns + [state["intent"]["aggregate_column"]])
    if state["intent"].get("aggregate_columns"):
        final_selected_columns = _dedupe_preserve_order(final_selected_columns + state["intent"]["aggregate_columns"])
    if final_selected_columns:
        state["intent"]["selected_columns"] = final_selected_columns
        state["selected_columns"] = final_selected_columns

    # ------------------------
    # TYPE
    # ------------------------
    
    if intent["aggregation"] and intent["wants_analysis"]:
        intent["type"] = "aggregation"
    elif intent["wants_analysis"]:
        intent["type"] = "analysis"
    elif state["intent"].get("ast"):
        intent["type"] = "filter"
    else:
        intent["type"] = "exploration"
    intent["analysis_type"] = intent.get("analytic_intent", "unknown")


    print("\n=== INTENT PARSER COMPLETE ===")
    print("Filters after llm reasoning:", state["intent"]["filters"])

    return state


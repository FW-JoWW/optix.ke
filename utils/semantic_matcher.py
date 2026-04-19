from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from pathlib import Path
import contextlib
import io

_model = None

QUERY_ALIASES = {
    "sales": ["sale", "sales", "revenue", "amount", "total", "totals", "value"],
    "quantity": ["quantity", "qty", "units", "count", "volume"],
    "price": ["price", "cost", "amount", "value", "rate"],
    "profit": ["profit", "margin", "gain"],
    "date": ["date", "day", "time", "month", "year"],
}


def _model_cached_locally():
    cache_roots = []

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home))

    user_home = Path.home()
    cache_roots.extend(
        [
            user_home / ".cache" / "huggingface",
            user_home / ".cache" / "torch" / "sentence_transformers",
            user_home / ".sentence_transformers",
        ]
    )

    expected_markers = [
        Path("hub") / "models--sentence-transformers--all-MiniLM-L6-v2",
        Path("models--sentence-transformers--all-MiniLM-L6-v2"),
        Path("sentence-transformers_all-MiniLM-L6-v2"),
        Path("all-MiniLM-L6-v2"),
    ]

    for root in cache_roots:
        for marker in expected_markers:
            if (root / marker).exists():
                return True
    return False


def get_model():
    global _model
    if _model is None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    return _model

def semantic_column_match(question, columns, threshold=0.5, fallback_k=3, lexical_only=False):
    """
    Returns columns that semantically match the question.
    """
    if not columns:
        return[]

    normalized_question = question.lower()
    lexical_matches = [
        col for col in columns
        if col.lower() in normalized_question
        or col.lower().replace("_", " ") in normalized_question
    ]
    alias_hits = []
    for term, aliases in QUERY_ALIASES.items():
        if term in normalized_question:
            for col in columns:
                normalized_col = col.lower().replace("_", " ")
                if any(alias in normalized_col for alias in aliases):
                    alias_hits.append(col)
    lexical_matches = list(dict.fromkeys(lexical_matches + alias_hits))
    if lexical_matches:
        return lexical_matches
    if lexical_only:
        return []

    if not _model_cached_locally():
        return columns[:fallback_k]

    try:
        model = get_model()
    except Exception:
        return columns[:fallback_k]

    # embed question
    q_embedding = model.encode([question], convert_to_numpy=True)

    # embed columns
    col_embeddings = model.encode(
        [f"column {c}" for c in columns],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(q_embedding, col_embeddings)[0]

    scored = list(zip(columns, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [c for c, s in scored if s >= threshold]
    if not filtered:
        filtered = [c for c, _ in scored[:fallback_k]]
    
    return filtered

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_column_match(question, columns, threshold=0.5, fallback_k=3):
    """
    Returns columns that semantically match the question.
    """
    if not columns:
        return[]

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
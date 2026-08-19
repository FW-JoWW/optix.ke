from __future__ import annotations

import hashlib
import threading
from typing import Any

import pandas as pd


_CACHE_LOCK = threading.RLock()
_ARTIFACT_CACHE: dict[tuple[str, str], Any] = {}


def dataframe_fingerprint(df: pd.DataFrame | None) -> str:
    if df is None:
        return "none"

    column_names = [str(column) for column in df.columns]
    dtype_names = [str(dtype) for dtype in df.dtypes]
    sample = df.head(12)
    if len(df) > 24:
        sample = pd.concat([df.head(12), df.tail(12)], axis=0)

    try:
        sample_hash = pd.util.hash_pandas_object(sample, index=True)
        digest = hashlib.sha256(sample_hash.to_numpy().tobytes()).hexdigest()[:16]
    except Exception:
        digest = "fallback"

    return "|".join(
        [
            f"{len(df)}x{df.shape[1]}",
            ",".join(column_names),
            ",".join(dtype_names),
            digest,
        ]
    )


def _artifact_key(kind: str, df: pd.DataFrame | None, extra_key: str | None = None) -> tuple[str, str]:
    base = dataframe_fingerprint(df)
    suffix = extra_key.strip() if isinstance(extra_key, str) and extra_key.strip() else ""
    return kind, f"{base}|{suffix}" if suffix else base


def get_cached_artifact(kind: str, df: pd.DataFrame | None, extra_key: str | None = None) -> Any | None:
    key = _artifact_key(kind, df, extra_key)
    with _CACHE_LOCK:
        return _ARTIFACT_CACHE.get(key)


def set_cached_artifact(kind: str, df: pd.DataFrame | None, value: Any, extra_key: str | None = None) -> Any:
    key = _artifact_key(kind, df, extra_key)
    with _CACHE_LOCK:
        _ARTIFACT_CACHE[key] = value
    return value

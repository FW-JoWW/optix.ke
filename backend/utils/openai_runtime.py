from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_clients: Dict[str, OpenAI] = {}


def _operation_timeout(operation: str | None = None) -> httpx.Timeout:
    operation_key = (operation or "default").strip().lower()
    default_read = float(os.getenv("OPENAI_READ_TIMEOUT_SECONDS", "15"))
    default_connect = float(os.getenv("OPENAI_CONNECT_TIMEOUT_SECONDS", "5"))
    operation_read_overrides = {
        "reasoning": float(os.getenv("OPENAI_REASONING_READ_TIMEOUT_SECONDS", "20")),
        "answer_synthesis": float(os.getenv("OPENAI_ANSWER_SYNTHESIS_READ_TIMEOUT_SECONDS", "22")),
        "report": float(os.getenv("OPENAI_REPORT_READ_TIMEOUT_SECONDS", "25")),
        "context_inference": float(os.getenv("OPENAI_CONTEXT_INFERENCE_READ_TIMEOUT_SECONDS", "18")),
        "cleaning": float(os.getenv("OPENAI_CLEANING_READ_TIMEOUT_SECONDS", "18")),
        "insight_generation": float(os.getenv("OPENAI_INSIGHT_GENERATION_READ_TIMEOUT_SECONDS", "22")),
        "default": default_read,
    }
    read_timeout = operation_read_overrides.get(operation_key, default_read)
    return httpx.Timeout(read_timeout, connect=default_connect)


def get_openai_client(operation: str | None = None) -> OpenAI | None:
    """
    Build a shared OpenAI client for this process.
    `trust_env=False` prevents broken proxy env vars from hijacking requests.
    """
    cache_key = (operation or "default").strip().lower() or "default"

    cached = _clients.get(cache_key)
    if cached is not None:
        return cached

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL") or None
    organization = os.getenv("OPENAI_ORG_ID") or None
    project = os.getenv("OPENAI_PROJECT_ID") or None

    http_client = httpx.Client(timeout=_operation_timeout(operation), trust_env=False)

    _clients[cache_key] = OpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        project=project,
        max_retries=2,
        http_client=http_client,
    )
    return _clients[cache_key]


def get_openai_runtime_info() -> Dict[str, Any]:
    return {
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "base_url_configured": bool(os.getenv("OPENAI_BASE_URL")),
        "organization_configured": bool(os.getenv("OPENAI_ORG_ID")),
        "project_configured": bool(os.getenv("OPENAI_PROJECT_ID")),
        "proxy_env_present": any(
            os.getenv(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
        ),
        "trust_env_for_openai": False,
    }



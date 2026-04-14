from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def get_openai_client() -> OpenAI | None:
    """
    Build a shared OpenAI client for this process.
    `trust_env=False` prevents broken proxy env vars from hijacking requests.
    """
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL") or None
    organization = os.getenv("OPENAI_ORG_ID") or None
    project = os.getenv("OPENAI_PROJECT_ID") or None

    http_client = httpx.Client(
        timeout=httpx.Timeout(60.0, connect=10.0),
        trust_env=False,
    )

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        project=project,
        max_retries=2,
        http_client=http_client,
    )
    return _client


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

"""Persistent application-side quota for public, API-key-funded AI features."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Dict

from fastapi import HTTPException


USAGE_PATH = Path(os.environ.get(
    "PUBLIC_AI_USAGE_PATH",
    "/home/paul/.local/share/mujocoweb-ai-usage.json",
))
MONTHLY_LIMIT = int(os.environ.get("PUBLIC_AI_MONTHLY_REQUEST_LIMIT", "300"))
DAILY_LIMIT = int(os.environ.get("PUBLIC_AI_DAILY_REQUEST_LIMIT", "40"))
usage_lock = threading.Lock()


def _read_usage() -> dict:
    try:
        value = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_usage(value: dict) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".mujocoweb-ai-usage-",
        dir=str(USAGE_PATH.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle)
        temporary_path.chmod(0o600)
        temporary_path.replace(USAGE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def claim_public_ai_request(feature: str) -> Dict[str, int]:
    """Atomically reserve one paid AI request or reject it at the hard app cap."""
    now = datetime.now(timezone.utc)
    month = now.strftime("%Y-%m")
    day = now.strftime("%Y-%m-%d")
    with usage_lock:
        usage = _read_usage()
        if usage.get("month") != month:
            usage = {"month": month, "month_requests": 0, "features": {}}
        if usage.get("day") != day:
            usage["day"] = day
            usage["day_requests"] = 0
        month_requests = int(usage.get("month_requests", 0))
        day_requests = int(usage.get("day_requests", 0))
        if month_requests >= MONTHLY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="The public AI assistant has reached its monthly usage limit.",
            )
        if day_requests >= DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="The public AI assistant has reached its daily usage limit. Please return tomorrow.",
            )
        usage["month_requests"] = month_requests + 1
        usage["day_requests"] = day_requests + 1
        features = usage.setdefault("features", {})
        features[feature] = int(features.get(feature, 0)) + 1
        _write_usage(usage)
        return {
            "month_remaining": MONTHLY_LIMIT - usage["month_requests"],
            "day_remaining": DAILY_LIMIT - usage["day_requests"],
        }

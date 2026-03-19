"""
KLING AI API Client - Video Generation.
Supports image-to-video generation with JWT auth.
Ported from sibling production project with Django references removed.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Optional

try:
    import requests
except ImportError:
    requests = None

from .exceptions import KlingNotConfigured, KlingVideoGenerationFailed
import config

logger = logging.getLogger(__name__)

KLING_API_BASE_DEFAULT = "https://api.klingai.com"

KLING_MODEL_V3 = "kling-v3"
KLING_MODEL_V26 = "kling-v2-6"
KLING_MODEL_V25_TURBO = "kling-v2.5-turbo"
KLING_MODEL_O1 = "kling-video-o1"


def _get_api_base() -> str:
    return (os.environ.get("KLING_API_BASE") or config.KLING_API_BASE or KLING_API_BASE_DEFAULT).strip()


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = (api_key or os.environ.get("KLING_API_KEY") or config.KLING_API_KEY or "").strip()
    if not key:
        raise KlingNotConfigured("KLING_API_KEY is not configured.")
    return key


def _get_api_secret(api_secret: Optional[str] = None) -> str:
    return (api_secret or os.environ.get("KLING_API_SECRET") or config.KLING_API_SECRET or "").strip()


def _get_bearer_token(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> str:
    """Get Bearer token. If secret is set, generate JWT; else use key directly."""
    key = _get_api_key(api_key)
    secret = _get_api_secret(api_secret)

    if not secret:
        return key

    try:
        from jose import jwt as jose_jwt
    except ImportError:
        raise KlingNotConfigured(
            "python-jose is required for Kling JWT auth. Install: pip install python-jose"
        )

    payload = {
        "iss": key,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    token = jose_jwt.encode(
        payload, secret, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"}
    )
    return token if isinstance(token, str) else token.decode("utf-8")


def image2video(
    *,
    prompt: str,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    image_tail_url: Optional[str] = None,
    image_tail_base64: Optional[str] = None,
    model: str = None,
    duration: int = 5,
    aspect_ratio: str = "9:16",
    mode: str = "professional",
    negative_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_seconds: int = 30,
) -> str:
    """Submit image-to-video task. Returns task_id for polling."""
    model = model or config.KLING_MODEL
    token = _get_bearer_token(api_key, None)
    if requests is None:
        raise RuntimeError("requests is required for KLING API.")

    if not image_url and not image_base64:
        raise ValueError("Either image_url or image_base64 is required.")

    url = f"{_get_api_base()}/v1/videos/image2video"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    mode_val = "pro" if (mode or "").lower() in ("professional", "pro") else "std"
    payload: dict = {
        "model": model,
        "prompt": (prompt or "").strip(),
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "mode": mode_val,
    }
    if image_url:
        payload["image"] = image_url
    elif image_base64:
        payload["image"] = image_base64
    if image_tail_url:
        payload["image_tail"] = image_tail_url
    elif image_tail_base64:
        payload["image_tail"] = image_tail_base64
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt.strip()

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    if not resp.ok:
        try:
            err_body = resp.json()
            err_obj = err_body.get("error") if isinstance(err_body.get("error"), dict) else {}
            err_msg = (
                err_obj.get("message") or err_body.get("error") or
                err_body.get("message") or err_body.get("detail") or resp.text
            )
        except Exception:
            err_msg = resp.text or resp.reason
        raise KlingVideoGenerationFailed(f"KLING API error {resp.status_code}: {err_msg}")

    data = resp.json()
    task_id = (data.get("task_id") or data.get("data", {}).get("task_id") or "").strip()
    if not task_id:
        raise KlingVideoGenerationFailed(f"KLING API did not return task_id: {data}")
    logger.info(f"KLING image2video task created: {task_id}")
    return task_id


def get_task_status(
    task_id: str,
    *,
    task_type: str = "image2video",
    api_key: Optional[str] = None,
    timeout_seconds: int = 15,
) -> dict:
    """Get status of a video generation task."""
    token = _get_bearer_token(api_key, None)
    if requests is None:
        raise RuntimeError("requests is required for KLING API.")

    url = f"{_get_api_base()}/v1/videos/{task_type}/{task_id}"
    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(url, headers=headers, timeout=timeout_seconds)
    resp.raise_for_status()
    data = resp.json()

    inner = data.get("data", data)
    task_status_raw = inner.get("task_status") or inner.get("status") or ""
    status = task_status_raw.upper() if isinstance(task_status_raw, str) else ""

    status_map = {"COMPLETED": "SUCCESS", "SUCCEED": "SUCCESS", "FAIL": "FAILED"}
    status = status_map.get(status, status)

    video_url = None
    if status == "SUCCESS":
        task_result = inner.get("task_result") or {}
        videos = task_result.get("videos") or []
        if isinstance(videos, list) and videos:
            video_url = videos[0].get("url") if isinstance(videos[0], dict) else videos[0]
        if not video_url:
            urls = inner.get("response") or inner.get("video_url") or inner.get("urls") or []
            if isinstance(urls, list) and urls:
                video_url = urls[0]
            elif isinstance(urls, str):
                video_url = urls

    return {
        "status": status,
        "task_id": task_id,
        "video_url": video_url,
        "error_message": inner.get("error_message") or inner.get("error"),
        "raw": data,
    }


def image2video_sync(
    *,
    prompt: str,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    model: str = None,
    duration: int = 5,
    aspect_ratio: str = "9:16",
    mode: str = "professional",
    negative_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    poll_interval: int = None,
    max_wait: int = None,
) -> bytes:
    """Synchronous image-to-video: submit + poll + download."""
    poll_interval = poll_interval or config.VIDEO_AI_POLL_INTERVAL
    max_wait = max_wait or config.VIDEO_AI_MAX_WAIT

    task_id = image2video(
        prompt=prompt, image_url=image_url, image_base64=image_base64,
        model=model, duration=duration, aspect_ratio=aspect_ratio,
        mode=mode, negative_prompt=negative_prompt, api_key=api_key,
    )

    elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        status = get_task_status(task_id, api_key=api_key)

        if status["status"] == "SUCCESS":
            video_url = status.get("video_url")
            if not video_url:
                raise KlingVideoGenerationFailed("Kling task succeeded but no video URL returned.")
            logger.info(f"Kling video ready, downloading from {video_url[:80]}...")
            resp = requests.get(video_url, timeout=120)
            resp.raise_for_status()
            return resp.content
        elif status["status"] == "FAILED":
            raise KlingVideoGenerationFailed(
                f"Kling video generation failed: {status.get('error_message')}"
            )
        else:
            logger.info(f"Kling task {task_id}: {status['status']} ({elapsed}s elapsed)")

    raise KlingVideoGenerationFailed(f"Kling video generation timed out after {max_wait}s")

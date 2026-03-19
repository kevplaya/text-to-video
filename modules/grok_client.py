"""
Grok xAI API Client - Image Generation & Image-to-Video Generation.
Ported from sibling production project with Django references removed.
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import timedelta
from typing import Optional

import requests

from .exceptions import GrokNotConfigured, GrokImageGenerationFailed, GrokVideoGenerationFailed
import config

logger = logging.getLogger(__name__)

GROK_IMAGE_MODEL = "grok-imagine-image"
GROK_VIDEO_MODEL = "grok-imagine-video"


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = (api_key or os.environ.get("GROK_API_KEY") or config.GROK_API_KEY or "").strip()
    if not key:
        raise GrokNotConfigured(
            "GROK_API_KEY is not configured (set GROK_API_KEY in environment)."
        )
    return key


def generate_image_bytes(
    *,
    prompt: str,
    model: str = GROK_IMAGE_MODEL,
    reference_images: Optional[list] = None,
    aspect_ratio: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """
    Generate an image using Grok Imagine Image (xAI SDK).

    Returns raw image bytes.
    """
    try:
        import xai_sdk
    except ImportError as e:
        raise GrokNotConfigured(
            "xai-sdk is required for Grok image generation. Install: pip install xai-sdk"
        ) from e

    key = _get_api_key(api_key)
    client = xai_sdk.Client(api_key=key)

    kwargs: dict = {
        "prompt": (prompt or "").strip(),
        "model": model,
        "image_format": "base64",
    }
    if aspect_ratio:
        kwargs["aspect_ratio"] = aspect_ratio

    if reference_images:
        data_uris = []
        for img_bytes, mime in reference_images[:3]:
            b64 = base64.b64encode(img_bytes).decode("ascii")
            data_uris.append(f"data:{mime};base64,{b64}")
        if len(data_uris) == 1:
            kwargs["image_url"] = data_uris[0]
        else:
            kwargs["image_urls"] = data_uris

    try:
        response = client.image.sample(**kwargs)
    except Exception as e:
        raise GrokImageGenerationFailed(
            f"Grok image generation failed: {e}",
            raw_text_preview=str(e)[:500],
        ) from e

    img_bytes = getattr(response, "image", None)
    if img_bytes and isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        return bytes(img_bytes)

    img_url = getattr(response, "url", None)
    if img_url:
        logger.info("Grok image API returned URL, downloading: %s", img_url[:80])
        resp = requests.get(img_url, timeout=120)
        resp.raise_for_status()
        return resp.content

    raise GrokImageGenerationFailed("Grok image generation returned no image bytes or URL.")


def image2video(
    *,
    prompt: str,
    image_url: Optional[str] = None,
    image_base64_data_uri: Optional[str] = None,
    duration: int = 6,
    aspect_ratio: str = "9:16",
    resolution: str = "720p",
    model: str = GROK_VIDEO_MODEL,
    api_key: Optional[str] = None,
    poll_interval: int = None,
    max_wait: int = None,
) -> bytes:
    """
    Generate video from image using Grok Imagine Video (xAI SDK).
    Returns video bytes (mp4).
    """
    try:
        import xai_sdk
    except ImportError as e:
        raise GrokNotConfigured(
            "xai-sdk is required for Grok video generation. Install: pip install xai-sdk"
        ) from e

    poll_interval = poll_interval or config.VIDEO_AI_POLL_INTERVAL
    max_wait = max_wait or config.VIDEO_AI_MAX_WAIT

    key = _get_api_key(api_key)

    if not image_url and not image_base64_data_uri:
        raise ValueError("Either image_url or image_base64_data_uri is required.")

    resolved_image = image_url or image_base64_data_uri
    client = xai_sdk.Client(api_key=key, timeout=max_wait)

    try:
        response = client.video.generate(
            prompt=(prompt or "Smooth motion, natural movement").strip(),
            model=model,
            image_url=resolved_image,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            timeout=timedelta(seconds=max_wait),
            interval=timedelta(seconds=poll_interval),
        )
    except TimeoutError as e:
        raise GrokVideoGenerationFailed(
            f"Grok video generation timed out after {max_wait}s"
        ) from e
    except Exception as e:
        raise GrokVideoGenerationFailed(
            f"Grok video generation failed: {e}"
        ) from e

    video_url = getattr(response, "url", None)
    if not video_url:
        raise GrokVideoGenerationFailed("Grok SDK returned a response but no video URL")

    logger.info(f"Grok image2video completed, downloading from {video_url[:80]}...")
    video_resp = requests.get(video_url, timeout=120)
    video_resp.raise_for_status()
    return video_resp.content

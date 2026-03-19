"""
Veo image-to-video via Google GenAI (Gemini API).
Uses GOOGLE_API_KEY for authentication.
Ported from sibling production project.
"""
import logging
import os
import time
from typing import Optional

from .exceptions import VeoNotConfigured, VeoVideoGenerationFailed
import config

logger = logging.getLogger(__name__)

VEO_MODEL = config.VEO_MODEL


def _get_client():
    """Get GenAI client using GOOGLE_API_KEY."""
    from .gemini_client import _get_genai_client, _get_api_key
    api_key = _get_api_key()
    return _get_genai_client(api_key)


def image2video(
    *,
    prompt: str,
    image_url: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime_type: str = "image/jpeg",
    aspect_ratio: str = "9:16",
    model: str = None,
) -> bytes:
    """
    Generate video from image using Veo.
    Returns video bytes (mp4).
    """
    model = model or VEO_MODEL

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise VeoNotConfigured(
            "google-genai SDK is required for Veo. Install: pip install google-genai"
        ) from e

    if not os.environ.get("GOOGLE_API_KEY") and not config.GOOGLE_API_KEY:
        raise VeoNotConfigured("GOOGLE_API_KEY is not set.")

    client = _get_client()
    if client is None:
        raise VeoNotConfigured("Failed to create GenAI client.")

    # Resolve image bytes
    img_bytes = image_bytes
    if img_bytes is None and image_url:
        import requests
        resp = requests.get(image_url, timeout=60)
        resp.raise_for_status()
        img_bytes = resp.content

    if not img_bytes:
        raise ValueError("Either image_url or image_bytes is required.")

    image_input = types.Image(image_bytes=img_bytes, mime_type=image_mime_type)

    ar_map = {"9:16": "9:16", "16:9": "16:9", "1:1": "1:1"}
    ar = ar_map.get(aspect_ratio, aspect_ratio)
    veo_config = types.GenerateVideosConfig(aspect_ratio=ar)

    operation = client.models.generate_videos(
        model=model,
        prompt=(prompt or "Smooth motion, natural movement").strip(),
        image=image_input,
        config=veo_config,
    )

    max_wait = config.VIDEO_AI_MAX_WAIT
    poll_interval = config.VIDEO_AI_POLL_INTERVAL
    elapsed = 0
    while not operation.done and elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        operation = client.operations.get(operation)

    if not operation.done:
        raise VeoVideoGenerationFailed("Veo video generation timed out")

    resp = getattr(operation, "response", None) or getattr(operation, "result", None)
    if not resp:
        raise VeoVideoGenerationFailed("Veo returned no response")

    gen_videos = getattr(resp, "generated_videos", None) or []
    if not gen_videos:
        raise VeoVideoGenerationFailed("Veo returned no generated videos")

    video_obj = gen_videos[0]
    video_file = getattr(video_obj, "video", None) or video_obj
    if not video_file:
        raise VeoVideoGenerationFailed("Veo generated video has no video file")

    data = getattr(video_file, "video_bytes", None)
    if isinstance(data, bytes) and len(data) > 0:
        return data

    uri = getattr(video_file, "uri", None)
    if uri and str(uri).startswith("gs://"):
        raise VeoVideoGenerationFailed(
            "Veo output is in GCS. Ensure video_bytes is returned for Gemini API."
        )

    try:
        downloaded = client.files.download(file=video_file)
        if isinstance(downloaded, bytes):
            return downloaded
        if hasattr(downloaded, "read"):
            return downloaded.read()
    except Exception as e:
        logger.warning(f"Veo files.download: {e}")

    raise VeoVideoGenerationFailed("Could not retrieve video bytes from Veo response")

"""
Assemble the video generation prompt with preservation rules.
Order: reference → preservation → style → costume → background → pose → motion → camera → avoid → runtime rules.
Ported from sibling production project.
"""
from __future__ import annotations
from typing import Optional

from .prompt_constants import (
    VIDEO_REFERENCE_RULES,
    VIDEO_PRESERVATION_RULES,
    DEFAULT_VIDEO_NEGATIVE,
    VIDEO_RUNTIME_RULES,
)


def _get_costume_lock(style_profile: Optional[dict]) -> str:
    if not isinstance(style_profile, dict):
        return ""
    dna = style_profile.get("character_dna")
    if isinstance(dna, dict):
        return (dna.get("outfit") or "").strip()
    return ""


def _get_style_lock(style_profile: Optional[dict]) -> str:
    if not isinstance(style_profile, dict):
        return ""
    value = style_profile.get("style_lock") or ""
    if isinstance(value, list):
        value = ", ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _get_negative_prompt(style_profile: Optional[dict]) -> str:
    if not isinstance(style_profile, dict):
        return ""
    value = style_profile.get("negative_prompt") or ""
    if isinstance(value, list):
        value = ", ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def build_video_prompt(
    *,
    scene_action: str,
    scene_background: str,
    motion_prompt: str,
    style_profile: Optional[dict] = None,
    camera_prompt: str = "",
) -> str:
    """Build a complete video generation prompt with character preservation."""
    parts: list = []

    parts.append(f"[REFERENCE]\n{VIDEO_REFERENCE_RULES}")
    parts.append(f"[CHARACTER PRESERVATION]\n{VIDEO_PRESERVATION_RULES}")

    style_lock = _get_style_lock(style_profile)
    if style_lock:
        parts.append(f"[STYLE]\n{style_lock}")

    costume = _get_costume_lock(style_profile)
    if costume:
        parts.append(f"[COSTUME]\nPreserve exactly: {costume}")

    if scene_background.strip():
        parts.append(f"[BACKGROUND]\n{scene_background.strip()}")

    if scene_action.strip():
        parts.append(f"[POSE / STATE]\n{scene_action.strip()}")

    if motion_prompt.strip():
        parts.append(f"[MOTION]\n{motion_prompt.strip()}")

    if camera_prompt.strip():
        parts.append(f"[CAMERA]\n{camera_prompt.strip()}")

    negative_items = [DEFAULT_VIDEO_NEGATIVE]
    negative = _get_negative_prompt(style_profile)
    if negative:
        negative_items.append(negative)
    parts.append(f"[AVOID]\n{', '.join(negative_items)}")

    parts.append(VIDEO_RUNTIME_RULES)

    return "\n\n".join(parts)

"""Shared utilities for prompt compiler modules."""


def aspect_ratio_desc(ar: str) -> str:
    """Convert aspect ratio to prompt-friendly description (e.g. 'vertical 9:16')."""
    ar = (ar or "9:16").strip()
    if ar in ("9:16", "3:4"):
        return f"vertical {ar}"
    if ar in ("16:9", "4:3"):
        return f"horizontal {ar}"
    if ar == "1:1":
        return "square 1:1"
    return ar

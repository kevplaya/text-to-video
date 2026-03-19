"""
ElevenLabs TTS client with character-level timing support.
Ported from sibling production project.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, List, Dict, Any

try:
    import requests
except ImportError:
    requests = None

from .exceptions import ElevenLabsNotConfigured, ElevenLabsSpeechGenerationFailed
import config


def get_voice_preview_url(
    *,
    voice_id: str,
    api_key: Optional[str] = None,
    timeout_seconds: int = 10,
) -> Optional[str]:
    """Get the official ElevenLabs preview URL for a voice."""
    vid = (voice_id or "").strip()
    if not vid:
        return None
    key = (api_key or os.environ.get("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY or "").strip()
    if not key:
        return None
    if requests is None:
        return None

    url = f"https://api.elevenlabs.io/v1/voices/{vid}"
    headers = {"xi-api-key": key, "accept": "application/json"}
    resp = requests.get(url, headers=headers, timeout=timeout_seconds)
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
        return (data.get("preview_url") or "").strip() or None
    except Exception:
        return None


def _parse_timed_subtitles_from_alignment(
    text: str,
    alignment: Dict[str, Any],
    min_duration: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Parse ElevenLabs character-level alignment into sentence/phrase-level timed subtitles.

    Returns: [{"text": "...", "start": 0.0}, ...]
    """
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []

    if not chars or len(chars) != len(starts):
        return [{"text": text.strip(), "start": 0.0}] if text.strip() else []

    SPLIT_CHARS = set(".\n?!,")
    segments: List[Dict[str, Any]] = []
    seg_chars: List[str] = []
    seg_start: Optional[float] = None

    for i, ch in enumerate(chars):
        t_start = float(starts[i]) if i < len(starts) else 0.0

        if seg_start is None:
            seg_start = t_start

        seg_chars.append(ch)

        is_last = (i == len(chars) - 1)
        is_split = ch in SPLIT_CHARS

        if is_split or is_last:
            seg_text = "".join(seg_chars).strip()
            if seg_text:
                seg_end = float(ends[i]) if i < len(ends) else t_start
                duration = seg_end - seg_start
                if duration >= min_duration or is_last:
                    segments.append({"text": seg_text, "start": round(seg_start, 2)})
                    seg_chars = []
                    seg_start = None
                    continue
            if not seg_text:
                seg_chars = []
                seg_start = None

    if seg_chars:
        seg_text = "".join(seg_chars).strip()
        if seg_text and seg_start is not None:
            segments.append({"text": seg_text, "start": round(seg_start, 2)})

    return segments if segments else [{"text": text.strip(), "start": 0.0}]


def generate_elevenlabs_speech_with_timing(
    *,
    text: str,
    voice_id: Optional[str] = None,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    stability: float = 0.4,
    similarity_boost: float = 0.8,
    timeout_seconds: int = 60,
) -> Tuple[bytes, str, List[Dict[str, Any]]]:
    """
    Generate TTS audio with timing via ElevenLabs with-timestamps endpoint.

    Returns: (audio_bytes, mime_type, timed_subtitles)
    """
    import base64 as _b64

    prompt = (text or "").strip()
    if not prompt:
        raise ElevenLabsSpeechGenerationFailed("ElevenLabs TTS input text is empty.")

    key = (api_key or os.environ.get("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY or "").strip()
    if not key:
        raise ElevenLabsNotConfigured("ELEVENLABS_API_KEY is not configured.")

    vid = (voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or config.ELEVENLABS_DEFAULT_VOICE_ID or "").strip()
    if not vid:
        vid = "iWLjl1zCuqXRkW6494ve"

    mid = (model_id or os.environ.get("ELEVENLABS_MODEL_ID") or config.ELEVENLABS_MODEL_ID or "").strip() or "eleven_v3"

    if requests is None:
        raise RuntimeError("requests is required for ElevenLabs TTS.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/with-timestamps"
    headers = {
        "xi-api-key": key,
        "accept": "application/json",
        "content-type": "application/json",
    }
    payload = {
        "text": prompt,
        "model_id": mid,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
        "output_format": "mp3_44100_128",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)

    # Fallback to eleven_multilingual_v2 if eleven_v3 is not supported
    if resp.status_code >= 400 and mid == "eleven_v3":
        payload["model_id"] = "eleven_multilingual_v2"
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)

    if resp.status_code < 200 or resp.status_code >= 300:
        preview = ""
        try:
            preview = str(resp.json())[:600]
        except Exception:
            preview = (resp.text or "")[:600]
        raise ElevenLabsSpeechGenerationFailed(
            f"ElevenLabs TTS with-timestamps failed: HTTP {resp.status_code}",
            raw_text_preview=preview,
        )

    data = resp.json()
    audio_b64 = data.get("audio_base64") or ""
    if not audio_b64:
        raise ElevenLabsSpeechGenerationFailed("ElevenLabs with-timestamps returned no audio_base64.")

    audio_bytes = _b64.b64decode(audio_b64)
    alignment = data.get("alignment") or {}
    timed_subtitles = _parse_timed_subtitles_from_alignment(prompt, alignment)

    return (audio_bytes, "audio/mpeg", timed_subtitles)


def generate_elevenlabs_speech_bytes(
    *,
    text: str,
    voice_id: Optional[str] = None,
    api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    stability: float = 0.4,
    similarity_boost: float = 0.8,
    timeout_seconds: int = 60,
) -> Tuple[bytes, str]:
    """
    Generate TTS audio bytes using ElevenLabs.
    Returns: (audio_bytes, mime_type)
    """
    prompt = (text or "").strip()
    if not prompt:
        raise ElevenLabsSpeechGenerationFailed("ElevenLabs TTS input text is empty.")

    key = (api_key or os.environ.get("ELEVENLABS_API_KEY") or config.ELEVENLABS_API_KEY or "").strip()
    if not key:
        raise ElevenLabsNotConfigured("ELEVENLABS_API_KEY is not configured.")

    vid = (voice_id or os.environ.get("ELEVENLABS_VOICE_ID") or config.ELEVENLABS_DEFAULT_VOICE_ID or "").strip()
    if not vid:
        vid = "iWLjl1zCuqXRkW6494ve"

    mid = (model_id or os.environ.get("ELEVENLABS_MODEL_ID") or config.ELEVENLABS_MODEL_ID or "").strip() or "eleven_v3"

    if requests is None:
        raise RuntimeError("requests is required for ElevenLabs TTS.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {
        "xi-api-key": key,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": prompt,
        "model_id": mid,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }
    params = {"output_format": "mp3_44100_128"}

    resp = requests.post(url, headers=headers, json=payload, params=params, timeout=timeout_seconds)
    if 200 <= resp.status_code < 300:
        audio = resp.content or b""
        if not audio:
            raise ElevenLabsSpeechGenerationFailed("ElevenLabs returned empty audio bytes.")
        return (audio, "audio/mpeg")

    # Fallback
    if mid == "eleven_v3" and resp.status_code >= 400:
        payload["model_id"] = "eleven_multilingual_v2"
        resp2 = requests.post(url, headers=headers, json=payload, params=params, timeout=timeout_seconds)
        if 200 <= resp2.status_code < 300:
            audio = resp2.content or b""
            if audio:
                return (audio, "audio/mpeg")

    preview = ""
    try:
        preview = str(resp.json())[:600]
    except Exception:
        preview = (resp.text or "")[:600]
    raise ElevenLabsSpeechGenerationFailed(
        f"ElevenLabs TTS failed: HTTP {resp.status_code}",
        raw_text_preview=preview,
    )

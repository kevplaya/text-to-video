"""
Thread-safe Gemini API client with caching, fallback chains, and defensive response parsing.
Ported from sibling production project with Django references removed.
"""
import base64
import io
import json
import logging
import os
import re
import threading
import wave
from typing import Any, Dict, Optional

from .exceptions import (
    GeminiNotConfigured,
    GeminiEmptyResponse,
    GeminiInvalidJSON,
    GeminiImageGenerationFailed,
    GeminiSpeechGenerationFailed,
)

logger = logging.getLogger(__name__)

_RATE_RE = re.compile(r"(?:rate|sample_rate)=(\d+)", re.IGNORECASE)


def _looks_like_wav(b: bytes) -> bool:
    return len(b) > 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE"


def _parse_sample_rate(mime_type: str) -> int:
    m = _RATE_RE.search(mime_type or "")
    if not m:
        return 24000
    try:
        return int(m.group(1))
    except Exception:
        return 24000


def _pcm_to_wav(pcm_bytes: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return bio.getvalue()


# ---------------------------------------------------------------------------
# Module-level genai.Client cache (lazy singleton per api_key, thread-safe).
# ---------------------------------------------------------------------------
_GENAI_CLIENT_CACHE: Dict[str, Any] = {}
_GENAI_CLIENT_LOCK = threading.Lock()


def _get_genai_client(api_key: str):
    cached = _GENAI_CLIENT_CACHE.get(api_key)
    if cached is not None:
        return cached

    with _GENAI_CLIENT_LOCK:
        cached = _GENAI_CLIENT_CACHE.get(api_key)
        if cached is not None:
            return cached

        try:
            from google import genai
        except Exception:
            genai = None

        if genai is None:
            try:
                import google.genai as genai
            except Exception as e:
                raise RuntimeError(
                    "google-genai SDK is not installed or importable in this environment."
                ) from e

        client = genai.Client(api_key=api_key)
        _GENAI_CLIENT_CACHE[api_key] = client
        return client


def _get_api_key(override_api_key: Optional[str] = None) -> str:
    api_key = override_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiNotConfigured(
            "GOOGLE_API_KEY is not configured (set GOOGLE_API_KEY in environment)."
        )
    return api_key


def _extract_text_from_response(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    for attr in ("output_text", "generated_text"):
        t = getattr(resp, attr, None)
        if isinstance(t, str) and t.strip():
            return t

    candidates = getattr(resp, "candidates", None)
    if isinstance(candidates, (list, tuple)) and len(candidates) > 0:
        c0 = candidates[0]
        content = getattr(c0, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if isinstance(parts, (list, tuple)):
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t.strip():
                    return t

    return ""


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_json_text(text: str) -> str:
    t = text.strip()
    if not t:
        return ""

    m = _CODE_FENCE_RE.search(t)
    if m:
        fenced = m.group(1).strip()
        if fenced:
            return fenced

    if "{" in t and "}" in t:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            return t[start : end + 1].strip()

    if "[" in t and "]" in t:
        start = t.find("[")
        end = t.rfind("]")
        if start >= 0 and end > start:
            return t[start : end + 1].strip()

    return t


def _iter_parts(resp: Any):
    parts = getattr(resp, "parts", None)
    if isinstance(parts, (list, tuple)):
        for p in parts:
            yield p
        return

    candidates = getattr(resp, "candidates", None)
    if isinstance(candidates, (list, tuple)) and len(candidates) > 0:
        for c in candidates:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if isinstance(parts, (list, tuple)):
                for p in parts:
                    yield p


def _resp_to_dict(resp: Any) -> Any:
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        return resp.__dict__
    except Exception:
        return None


def _find_inline_image_bytes(obj: Any) -> Optional[bytes]:
    if obj is None:
        return None

    if isinstance(obj, (bytes, bytearray)) and len(obj) > 0:
        return bytes(obj)

    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 128:
            try:
                return base64.b64decode(s)
            except Exception:
                return None
        return None

    if isinstance(obj, list):
        for it in obj:
            found = _find_inline_image_bytes(it)
            if found:
                return found
        return None

    if isinstance(obj, dict):
        for key in ("inline_data", "inlineData"):
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                data = inner.get("data") or inner.get("bytesBase64Encoded") or inner.get("content")
                found = _find_inline_image_bytes(data)
                if found:
                    return found

        if "data" in obj:
            found = _find_inline_image_bytes(obj.get("data"))
            if found:
                return found

        for v in obj.values():
            found = _find_inline_image_bytes(v)
            if found:
                return found

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_image_bytes(
    *,
    prompt: str,
    model: str,
    reference_image_bytes: Optional[bytes] = None,
    reference_image_mime_type: str = "image/jpeg",
    reference_images: Optional[list] = None,
    aspect_ratio: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    """Generate an image using Gemini image-capable models. Returns raw image bytes."""
    key = _get_api_key(api_key)
    client = _get_genai_client(key)

    normalized = model.strip()
    if normalized == "nano-banana":
        normalized = "gemini-2.5-flash-image"

    fallback_models = [
        normalized,
        os.environ.get("GEMINI_IMAGE_MODEL_FALLBACK_1", "").strip(),
        os.environ.get("GEMINI_IMAGE_MODEL_FALLBACK_2", "").strip(),
        "gemini-2.5-flash-image",
    ]
    fallback_models = [m for m in fallback_models if m]

    last_exc: Optional[Exception] = None
    resp = None
    for m in fallback_models:
        try:
            configs = [
                {"response_modalities": ["IMAGE"]},
                {"response_modalities": ["IMAGE", "TEXT"]},
                {"response_mime_type": "image/png"},
                None,
            ]
            if (aspect_ratio or "").strip():
                ar = str(aspect_ratio).strip()
                configs = [
                    {"response_modalities": ["IMAGE"], "image_config": {"aspect_ratio": ar}},
                    {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": ar}},
                    *configs,
                ]

            contents: Any = [prompt]
            refs = reference_images or []
            if not refs and reference_image_bytes:
                refs = [(reference_image_bytes, reference_image_mime_type or "image/jpeg")]
            if refs:
                parts: list = [{"text": prompt}]
                for img_bytes, mime in refs[:3]:
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                contents = [{"role": "user", "parts": parts}]

            for cfg in configs:
                try:
                    if cfg is None:
                        resp = client.models.generate_content(model=m, contents=contents)
                    else:
                        resp = client.models.generate_content(model=m, contents=contents, config=cfg)
                except TypeError:
                    resp = client.models.generate_content(model=m, contents=contents)
                except Exception:
                    if refs:
                        if cfg is None:
                            resp = client.models.generate_content(model=m, contents=[prompt])
                        else:
                            resp = client.models.generate_content(model=m, contents=[prompt], config=cfg)
                    else:
                        raise

                for part in _iter_parts(resp):
                    inline = getattr(part, "inline_data", None)
                    if inline is None:
                        continue
                    data = getattr(inline, "data", None)
                    if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                        return bytes(data)
                    if isinstance(data, str) and data.strip():
                        try:
                            return base64.b64decode(data)
                        except Exception:
                            pass

                    as_image = getattr(part, "as_image", None)
                    if callable(as_image):
                        try:
                            img = as_image()
                            bio = io.BytesIO()
                            img.save(bio, format="PNG")
                            return bio.getvalue()
                        except Exception:
                            pass

                found = _find_inline_image_bytes(_resp_to_dict(resp))
                if found:
                    return found

            continue
        except Exception as e:
            last_exc = e
            continue

    if resp is None:
        raise GeminiImageGenerationFailed(
            f"Gemini image generation call failed for all candidate models (first={normalized}).",
            raw_text_preview=str(last_exc)[:500] if last_exc else "",
        )

    preview = _extract_text_from_response(resp).strip().replace("\n", "\\n")[:500]
    raise GeminiImageGenerationFailed(
        "Gemini image generation returned no image bytes.",
        raw_text_preview=preview,
    )


def regenerate_image_prompt(
    *,
    original_prompt: str,
    error_message: str,
    scene_text: str = "",
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """Generate a safer/simpler image prompt when image generation fails."""
    key = _get_api_key(api_key)
    orig = (original_prompt or "").strip()
    err = (error_message or "").strip()
    seg = (scene_text or "").strip()
    if not orig:
        raise GeminiImageGenerationFailed("regenerate_image_prompt: original_prompt is empty.")

    system_prompt = """당신은 AI 이미지 생성 전문가입니다. 이미지 생성에 실패한 프롬프트를 분석하고, 성공 가능성이 높은 새로운 프롬프트를 생성해야 합니다.

## 실패 원인 분석 가이드
- "No image data in response": 프롬프트가 너무 복잡하거나, 생성하기 어려운 내용 포함
- "timeout": 너무 복잡한 장면

## 프롬프트 수정 가이드라인
1. 장면을 단순화하되 핵심 분위기/감정은 유지
2. 사람의 얼굴이나 신체를 직접 묘사하지 말고, 실루엣/뒷모습/손만 등으로 표현
3. 텍스트나 글자가 포함된 요소는 제거
4. 추상적인 개념은 구체적인 시각적 메타포로 변환
5. 민감하거나 폭력적인 내용은 상징적/예술적으로 표현
6. 실사 스타일보다는 일러스트/아트 스타일 권장

## 출력 형식
새로운 프롬프트만 출력하세요. 설명이나 부연 없이 영어로 된 이미지 프롬프트만 출력합니다."""

    user_prompt = f"""## 실패한 프롬프트
{orig}

## 에러 메시지
{err}

## 이 장면의 내용 (참고용)
{seg}

위 정보를 바탕으로 동일한 장면을 묘사하되, 이미지 생성에 더 적합한 새로운 프롬프트를 생성해주세요."""

    client = _get_genai_client(key)

    resp = None
    last_exc: Optional[Exception] = None
    configs = [
        {"system_instruction": system_prompt, "temperature": 0.7},
        {"systemInstruction": system_prompt, "temperature": 0.7},
        None,
    ]

    for cfg in configs:
        try:
            if cfg is None:
                resp = client.models.generate_content(
                    model=model, contents=[system_prompt, user_prompt]
                )
            else:
                resp = client.models.generate_content(
                    model=model, contents=[user_prompt], config=cfg
                )
            break
        except Exception as e:
            last_exc = e
            resp = None
            continue

    if resp is None:
        raise GeminiImageGenerationFailed(
            f"Failed to regenerate image prompt: {last_exc.__class__.__name__ if last_exc else 'Unknown'}",
            raw_text_preview=str(last_exc)[:500] if last_exc else "",
        )

    text = _extract_text_from_response(resp).strip()
    if not text:
        raise GeminiImageGenerationFailed("Failed to regenerate image prompt: empty response.")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else text


def generate_speech_bytes(
    *,
    text: str,
    language: str = "ko",
    model: Optional[str] = None,
    voice_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple:
    """
    Generate TTS audio bytes using Gemini TTS-capable models.
    Returns: (audio_bytes, mime_type)
    """
    key = _get_api_key(api_key)
    prompt = (text or "").strip()
    if not prompt:
        raise GeminiSpeechGenerationFailed("TTS input text is empty.")

    client = _get_genai_client(key)

    tts_model = (
        model
        or os.environ.get("GEMINI_TTS_MODEL")
        or "gemini-2.5-flash-preview-tts"
    )
    vname = (
        voice_name
        or os.environ.get("GEMINI_TTS_VOICE")
        or ("Kore" if language.startswith("ko") else "Kore")
    )

    resp = None
    last_exc: Optional[Exception] = None

    # Try typed config first (newer SDK)
    try:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=vname)
                )
            ),
        )
        resp = client.models.generate_content(model=tts_model, contents=prompt, config=cfg)
    except Exception as e:
        last_exc = e

    if resp is None:
        try:
            resp = client.models.generate_content(
                model=tts_model,
                contents=prompt,
                config={
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": vname}
                        }
                    },
                },
            )
        except Exception as e:
            raise GeminiSpeechGenerationFailed(
                f"Gemini TTS call failed: {e.__class__.__name__}",
                raw_text_preview=str(last_exc or e)[:500],
            ) from e

    for part in _iter_parts(resp):
        inline = getattr(part, "inline_data", None)
        if inline is None:
            continue
        mime = getattr(inline, "mime_type", None) or getattr(inline, "mimeType", None) or "audio/wav"
        data = getattr(inline, "data", None)
        if isinstance(data, (bytes, bytearray)) and len(data) > 0:
            audio = bytes(data)
            mime_s = str(mime)
            if (("L16" in mime_s) or ("pcm" in mime_s.lower()) or ("audio/raw" in mime_s.lower())) and not _looks_like_wav(audio):
                audio = _pcm_to_wav(audio, sample_rate=_parse_sample_rate(mime_s))
                mime_s = "audio/wav"
            elif mime_s.lower().startswith("audio/wav") and not _looks_like_wav(audio):
                audio = _pcm_to_wav(audio, sample_rate=_parse_sample_rate(mime_s))
                mime_s = "audio/wav"
            return (audio, mime_s)
        if isinstance(data, str) and data.strip():
            try:
                audio = base64.b64decode(data)
                mime_s = str(mime)
                if (("L16" in mime_s) or ("pcm" in mime_s.lower()) or ("audio/raw" in mime_s.lower())) and not _looks_like_wav(audio):
                    audio = _pcm_to_wav(audio, sample_rate=_parse_sample_rate(mime_s))
                    mime_s = "audio/wav"
                elif mime_s.lower().startswith("audio/wav") and not _looks_like_wav(audio):
                    audio = _pcm_to_wav(audio, sample_rate=_parse_sample_rate(mime_s))
                    mime_s = "audio/wav"
                return (audio, mime_s)
            except Exception:
                pass

    found = _find_inline_image_bytes(_resp_to_dict(resp))
    if found:
        audio = found
        if not _looks_like_wav(audio):
            audio = _pcm_to_wav(audio, sample_rate=24000)
        return (audio, "audio/wav")

    preview = _extract_text_from_response(resp).strip().replace("\n", "\\n")[:500]
    raise GeminiSpeechGenerationFailed(
        "Gemini TTS returned no audio bytes.",
        raw_text_preview=preview,
    )


def generate_json(
    *,
    system_prompt: str,
    user_prompt: str,
    image_bytes: Optional[bytes] = None,
    image_mime_type: str = "image/jpeg",
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> Any:
    """Generate a JSON response from Gemini."""
    key = _get_api_key(api_key)
    client = _get_genai_client(key)

    try:
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            resp = client.models.generate_content(
                model=model,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": system_prompt},
                            {"text": user_prompt},
                            {"inline_data": {"mime_type": image_mime_type, "data": b64}},
                        ],
                    }
                ],
                config={"response_mime_type": "application/json"},
            )
        else:
            resp = client.models.generate_content(
                model=model,
                contents=[system_prompt, user_prompt],
                config={"response_mime_type": "application/json"},
            )
    except TypeError:
        resp = client.models.generate_content(model=model, contents=[system_prompt, user_prompt])

    text = _extract_text_from_response(resp)
    json_text = _extract_json_text(text)
    if not json_text.strip():
        raise GeminiEmptyResponse("Gemini returned an empty response.")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        preview = (text or "").strip().replace("\n", "\\n")[:500]
        logger.warning(f"Gemini returned non-JSON text preview: {preview}")
        raise GeminiInvalidJSON(
            "Gemini returned invalid JSON.",
            raw_text_preview=preview,
        ) from e

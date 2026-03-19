"""
GPT-4o Vision client for reference image analysis.
Extracts character style profiles (DNA) for consistent image generation.
Ported from sibling production project.
"""
import base64
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from .exceptions import GPTNotConfigured, GPTEmptyResponse, GPTInvalidJSON

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_STYLE_PROFILE_SYSTEM_PROMPT = """
You are a Visual AI Engineer specializing in reference image analysis for AI image generation (Grok / Gemini).

Your output will be reused across multiple scenes.
The subject's visual identity must remain identical in every generated image.
Only scene, pose, action, camera, and background may change later.

### TASK
The exact same character from image_0.png.
Do not change the character design.

1. IMAGE TYPE DETECTION
Classify the uploaded image into exactly one of these values:

- "illustration"   → cartoon, anime, 2D art, chibi, webtoon, drawing
- "plush_toy"      → stuffed animal, plushie
- "real_animal"    → live or photographed animal
- "real_object"    → product, object, scenery
- "human_photo"    → real human photograph
- "other_real"     → any other real-world photograph
Return only the label.

---

2. CHARACTER DNA
Extract the subject's immutable visual identity.
Only include traits that must stay consistent across all scenes.

Allowed attributes:
- subject_type
- body_structure
- face
- hair
- materials
- colors
- outfit
- distinctive_features

Rules:
- DO NOT describe background or environment.
- DO NOT describe pose or action.
- DO NOT write prompt-style instructions.
- DO NOT describe camera angle or lighting.
- Keep each field concise and factual.
- If a field is not applicable, return an empty string.

For real-world subjects:
- Never convert them into illustration or cartoon.
- Preserve real-world appearance, material, and proportions.

For illustration subjects:
- Preserve original drawing identity, proportions, line style, and costume design.

---

3. STYLE LOCK

Extract concise rendering style keywords that must stay consistent.

Rules:
- Use short comma-separated keyword phrases.
- Do NOT write sentences.
- Focus only on rendering style, not character identity.
- 3 to 12 keywords maximum.

Examples:

illustration:
"flat 2D anime illustration, clean line art, cel shading, bold outlines, vibrant colors"

real-world:
"photorealistic, natural lighting, real fabric texture, realistic fur detail"

---

4. NEGATIVE PROMPT

List visual qualities that must NOT appear.

Rules:
- Return as comma-separated keywords only.
- Focus on style drift, rendering errors, and identity corruption.

Examples:

illustration:
"3D render, photorealistic, realistic skin, painterly style, different art style"

real-world:
"cartoon, anime, illustration, stylized, painted"

---

### OUTPUT FORMAT

Return ONLY valid JSON with exactly these 4 fields:

{
  "image_type": "",
  "character_dna": {
    "subject_type": "",
    "body_structure": "",
    "face": "",
    "hair": "",
    "materials": "",
    "colors": "",
    "outfit": "",
    "distinctive_features": ""
  },
  "style_lock": "",
  "negative_prompt": ""
}
"""


def _get_api_key(override_api_key: Optional[str] = None) -> str:
    api_key = override_api_key or os.environ.get("GPT_API_KEY")
    if not api_key:
        raise GPTNotConfigured(
            "GPT_API_KEY is not configured (set GPT_API_KEY in environment)."
        )
    return api_key


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


def extract_character_style_profile(
    *,
    image_bytes: bytes,
    image_mime_type: str = "image/jpeg",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze reference image via GPT-4o Vision to extract character style profile.

    Returns dict with keys: image_type, character_dna, style_lock, negative_prompt
    """
    result = generate_json(
        system_prompt=_STYLE_PROFILE_SYSTEM_PROMPT,
        user_prompt="Analyze the reference image and return the JSON.",
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        model=model,
        api_key=api_key,
    )
    required_keys = {"image_type", "character_dna", "style_lock", "negative_prompt"}
    if not isinstance(result, dict):
        logger.warning("GPT style_profile response is not a dict: %s", result)
        return {}
    missing = required_keys - result.keys()
    if missing:
        logger.warning("GPT style_profile response missing keys %s: %s", missing, result)

    # Normalize string fields
    for key in ("image_type", "style_lock", "negative_prompt"):
        val = result.get(key, "")
        if isinstance(val, list):
            result[key] = ", ".join(str(x) for x in val)
        elif not isinstance(val, str):
            result[key] = str(val) if val else ""

    # character_dna must stay as a dict
    dna = result.get("character_dna")
    if isinstance(dna, str):
        try:
            parsed = json.loads(dna)
            if isinstance(parsed, dict):
                result["character_dna"] = parsed
        except Exception:
            logger.warning("character_dna is a plain string: %s", dna[:200])
    elif not isinstance(dna, dict):
        result["character_dna"] = {}

    # Backward-compat alias
    if not result.get("character_dna") and result.get("gemini_prompt"):
        result["character_dna"] = result["gemini_prompt"]

    return result


def generate_json(
    *,
    system_prompt: str,
    user_prompt: str,
    image_bytes: Optional[bytes] = None,
    image_mime_type: str = "image/jpeg",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
) -> Any:
    """Generate a JSON response from OpenAI GPT."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise GPTNotConfigured(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    key = _get_api_key(api_key)
    client = OpenAI(api_key=key)

    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{image_mime_type};base64,{encoded}"
        user_content: Any = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            },
        ]
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    choice = response.choices[0] if response.choices else None
    if not choice:
        raise GPTEmptyResponse("GPT returned no choices")

    raw_text = (choice.message.content or "").strip()
    if not raw_text:
        raise GPTEmptyResponse("GPT returned an empty response")

    json_text = _extract_json_text(raw_text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        preview = json_text[:300]
        raise GPTInvalidJSON(
            f"GPT response is not valid JSON: {exc}",
            raw_text_preview=preview,
        ) from exc

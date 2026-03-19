"""
Story Generator Module — Enhanced with 6-beat narrative arc.
Ported from sibling production project with CLI adaptations.

Generates structured stories with:
- scene_action / scene_background separation
- motion_prompt per scene
- Model fallback chains
- Character DNA style hint injection
- Location lock support
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback model order
# ---------------------------------------------------------------------------

STORY_MODEL_FALLBACK_ORDER = [
    config.STORY_MODEL,
    config.STORY_MODEL_FALLBACK,
    "gemini-2.0-flash",
]

_RETRYABLE_ERROR_PATTERNS = (
    "503",
    "resource exhausted",
    "overloaded",
    "quota exceeded",
    "rate limit",
    "service unavailable",
)


def _is_retryable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Structure description builder
# ---------------------------------------------------------------------------

_FORMAT_PRESET_IDS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
_DEFAULT_FORMAT_PRESET_ID = "P01"


def _build_structure_desc(num_scenes: int) -> str:
    """Build narrative arc distribution guide for N scenes."""
    if num_scenes == 1:
        return "장면 1개: Hook + Punchline을 한 장면에 압축하세요."
    if num_scenes <= 3:
        beats = ["Hook", "Problem/Reaction", "Punchline"]
        return "\n".join(f"장면 {i+1}: {b}" for i, b in enumerate(beats[:num_scenes]))
    if num_scenes <= 6:
        base = ["Hook", "Situation", "Problem", "Reaction", "Escalation", "Punchline"]
        selected = base[:num_scenes]
        return "\n".join(f"장면 {i+1}: {b}" for i, b in enumerate(selected))
    # num_scenes > 6: distribute middle stages
    lines = ["장면 1: Hook"]
    middle_count = num_scenes - 2
    stages = ["Situation", "Problem", "Reaction", "Escalation"]
    per_stage = max(1, middle_count // len(stages))
    remainder = middle_count - per_stage * len(stages)
    scene_idx = 2
    for i, stage in enumerate(stages):
        count = per_stage + (1 if i < remainder else 0)
        for j in range(count):
            suffix = f" (심화 {j+1})" if count > 1 else ""
            lines.append(f"장면 {scene_idx}: {stage}{suffix}")
            scene_idx += 1
    lines.append(f"장면 {num_scenes}: Punchline")
    return "\n".join(lines)


def _style_hint_block(image_type: Optional[str], style_lock: Optional[str]) -> str:
    """Build the STYLE rule injected into image_prompt instructions."""
    if not image_type and not style_lock:
        return ""

    is_real = image_type in ("plush_toy", "real_animal", "real_object", "human_photo", "other_real")
    sl = (style_lock or "").strip()

    if is_real:
        style_desc = "photorealistic rendering. Reference image is a real-world subject — preserve as-is. DO NOT convert to illustration or cartoon."
        if sl:
            style_desc += f" Style keywords: {sl}"
    else:
        style_desc = "illustration/2D art style matching the reference image exactly."
        if sl:
            style_desc += f" Style keywords: {sl}"
        style_desc += " DO NOT convert to 3D or photorealistic."

    return f"""
[CHARACTER DNA STYLE RULE — from GPT-4o analysis]
  2. STYLE: {style_desc}
  This overrides any default style guess. Always follow this style rule for every scene.
"""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _system_prompt(
    num_scenes: int,
    language: str,
    location_lock: Optional[str] = None,
    image_type: Optional[str] = None,
    style_lock: Optional[str] = None,
) -> str:
    format_ids = ", ".join(_FORMAT_PRESET_IDS)
    location_rule = ""
    if location_lock and location_lock.strip():
        loc = location_lock.strip()
        location_rule = f"""

[필수 - 장소 일관성] 인터뷰, 상황연출 등 고정 장소 시나리오:
- 고정 장소: "{loc}"
- 모든 장면의 image_prompt에는 반드시 동일한 장소 "{loc}"를 포함하세요.
- 장면마다 배경/장소를 바꾸지 마세요. 씬별 변주는 캐릭터 표정, 제스처, 대사 톤만 허용합니다.
- 예: 1번 씬이 "{loc}"이면, 2~{num_scenes}번 씬도 전부 "{loc}"입니다.
"""
    structure_desc = _build_structure_desc(num_scenes)
    style_hint = _style_hint_block(image_type, style_lock)

    return f"""당신은 유튜브 쇼츠(Shorts) 숏폼 콘텐츠 전문가입니다.
주어진 주제에 대해 정확히 {num_scenes}개의 장면으로 구성된 짧고 임팩트 있는 스크립트를 작성하세요.

목표:
시청 지속 시간(Retention), 재시청률(Replay Rate), 댓글 참여(Comment Engagement)를 극대화하는 것입니다.

영상 구조 ({num_scenes}개 장면 기준):
아래 6단계 내러티브 아크를 {num_scenes}개 장면에 비례 배분하세요.
- Hook: 강한 호기심/공감 유발 (최대 20단어)
- Situation: 상황 설명 (짧고 명확)
- Problem: 갈등 또는 이상한 상황 발생
- Reaction: 캐릭터의 감정 반응
- Escalation: 긴장 상승 또는 심화
- Punchline: 웃음/반전/여운 결론

{structure_desc}

규칙:
- 반드시 정확히 {num_scenes}개의 장면을 생성하세요
- 각 장면은 4~6초 길이
- 각 narration은 20단어 이하
- 짧고 대화체로 작성
- 인사말 금지
- 설명문 스타일 금지

각 scene에는 다음을 포함하세요:
- narration
- duration
- scene_action
- scene_background
- motion_prompt

scene_action 작성 규칙:
- 영어로 작성. 한 문장 이상.
- 캐릭터의 포즈, 작은 동작, 간단한 표정 상태만 묘사하세요.
- 장면의 "정적인 상태" 설명 위주로 작성하세요.

허용:
- small gesture, slight posture change, shy smile, subtle reaction

금지:
- 캐릭터 외형 묘사 (헤어, 의상, 색상, 재질, 종족)
- 카메라 연출, 시네마틱 연기, 강한 감정 표현

절대 금지 단어/표현:
deeply blushing, furiously, intense eye contact, dramatic reaction,
screaming, crying, dramatic tension, cinematic moment

예:
"The character pauses with a shy smile."
"The character raises one hand slightly in greeting."

scene_background 작성 규칙:
- 영어로 작성. 한 문장 이상.
- 배경 환경, 장소, 조명만 묘사하세요.
- 캐릭터 묘사 금지.

금지:
- 카메라 앵글, 렌즈 표현, 구도 설명, 시네마틱 프레이밍

절대 금지 표현:
over-the-shoulder, wide-angle shot, close-up,
low angle, high angle, dramatic framing

예:
"busy traditional market stalls, warm afternoon sunlight"
"modern office lobby interior, cool fluorescent lighting"

motion_prompt 작성 규칙 (image-to-video용):
- 영어로 작성.
- motion은 항상 subtle, minimal, stable 해야 합니다.
- 2~4문장.

우선순위:
1. 작은 ACTION (small hand wave, slight nod)
2. 미세한 BODY MOVEMENT (slight posture shift, blinking)
3. 약한 FABRIC / HAIR MOVEMENT (scarf moves slightly)
4. 약한 ENVIRONMENT 분위기 (soft light flicker)

CAMERA MOTION 규칙:
- 카메라 움직임을 작성하지 마세요. 카메라는 fixed framing입니다.

절대 금지 표현:
slow push in, pan, tilt, dolly, zoom, tracking shot

title:
- 영상 상단에 사용할 15자 미만의 짧은 제목
- 충격, 호기심, 공감 중 최소 하나 포함

format_preset_id:
다음 중 하나 선택: {format_ids}

출력은 반드시 JSON만 생성하세요.
{style_hint}

출력 JSON 스키마:
{{
  "title": string,
  "theme": string,
  "format_preset_id": string,
  "scenes": [
    {{
      "scene_number": number,
      "narration": string,
      "duration": number,
      "scene_action": string,
      "scene_background": string,
      "motion_prompt": string
    }}
  ]
}}

추가:
- narration은 {language} 언어로 자연스럽게 작성하세요.
- 참고 이미지가 제공된 경우, 이미지의 분위기와 스타일을 참고해 스토리를 작성하세요. 단, scene_action/scene_background에 캐릭터의 외형·종류·재질·색상을 묘사하지 마세요.
{location_rule}"""


# ---------------------------------------------------------------------------
# Gemini JSON generation (delegated to gemini_client)
# ---------------------------------------------------------------------------

def _generate_json(
    *,
    system_prompt: str,
    user_prompt: str,
    image_bytes: Optional[bytes],
    image_mime_type: str,
    model: str,
) -> Any:
    from .gemini_client import generate_json as gemini_generate_json
    return gemini_generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_bytes=image_bytes,
        image_mime_type=image_mime_type,
        model=model,
    )


# ---------------------------------------------------------------------------
# StoryGenerator class
# ---------------------------------------------------------------------------

class StoryGenerator:
    """Generates structured stories using Gemini API with 6-beat narrative arc."""

    DEFAULT_MODEL = config.STORY_MODEL

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key
        logger.info("StoryGenerator initialized: model=%s", self.model)

    def _fallback_models(self) -> List[str]:
        all_models = list(dict.fromkeys(STORY_MODEL_FALLBACK_ORDER))
        return [m for m in all_models if m != self.model]

    def generate_story(
        self,
        prompt: str = "",
        num_scenes: int = 6,
        language: str = "ko",
        *,
        reference_image_bytes: Optional[bytes] = None,
        reference_image_mime_type: str = "image/jpeg",
        location_lock: Optional[str] = None,
        character_dna_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured story.

        Args:
            prompt: User's story topic/prompt
            num_scenes: Number of scenes to generate
            language: Language code for narration
            reference_image_bytes: Optional reference image for style hints
            reference_image_mime_type: MIME type of reference image
            location_lock: Fixed location for all scenes
            character_dna_profile: GPT-4o extracted character DNA profile

        Returns:
            Story dict with title, theme, scenes (scene_action, scene_background, motion_prompt, etc.)
        """
        # Extract style context from character DNA
        image_type: Optional[str] = None
        style_lock: Optional[str] = None
        if character_dna_profile and isinstance(character_dna_profile, dict):
            image_type = (character_dna_profile.get("image_type") or "").strip() or None
            _sl_raw = character_dna_profile.get("style_lock") or ""
            if isinstance(_sl_raw, list):
                _sl_raw = ", ".join(str(x) for x in _sl_raw)
            style_lock = str(_sl_raw).strip() or None
            if image_type or style_lock:
                logger.info(
                    "StoryGenerator: injecting character DNA — image_type=%s, style_lock=%s",
                    image_type, (style_lock or "")[:80],
                )

        system_prompt = _system_prompt(
            num_scenes=num_scenes,
            language=language,
            location_lock=location_lock,
            image_type=image_type,
            style_lock=style_lock,
        )
        user_prompt = f"주제: {prompt}"
        models_to_try = [self.model] + self._fallback_models()
        data = None
        last_exc: Optional[Exception] = None

        for model in models_to_try:
            try:
                data = _generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_bytes=reference_image_bytes,
                    image_mime_type=reference_image_mime_type or "image/jpeg",
                    model=model,
                )
                if model != self.model:
                    logger.info(
                        "Story generated with fallback model: %s (primary=%s failed)",
                        model, self.model,
                    )
                break
            except Exception as e:
                last_exc = e
                if _is_retryable_error(e) and model != models_to_try[-1]:
                    logger.warning(
                        "Story generation failed with %s (retryable): %s, trying fallback",
                        model, str(e)[:200],
                    )
                    continue
                raise

        if data is None and last_exc is not None:
            raise last_exc

        # Basic validation + normalization
        raw_title = str(data.get("title") or "").strip()
        if raw_title and not raw_title.startswith("Create") and not raw_title.startswith("주제:"):
            title = raw_title[:14]
        else:
            title = (raw_title or str(prompt))[:14]
        theme = str(data.get("theme") or "")
        format_preset_id = str(data.get("format_preset_id") or "").strip() or _DEFAULT_FORMAT_PRESET_ID
        if format_preset_id not in _FORMAT_PRESET_IDS:
            format_preset_id = _DEFAULT_FORMAT_PRESET_ID

        scenes_raw: List[Dict[str, Any]] = data.get("scenes") or []
        if not isinstance(scenes_raw, list) or len(scenes_raw) == 0:
            raise ValueError("Gemini returned no scenes")

        scenes: List[Dict[str, Any]] = []
        for i, sc in enumerate(scenes_raw, start=1):
            scene_number = int(sc.get("scene_number") or i)
            narration = str(sc.get("narration") or "").strip()
            scene_action = str(sc.get("scene_action") or "").strip()
            scene_background = str(sc.get("scene_background") or "").strip()
            legacy_image_prompt = str(sc.get("image_prompt") or "").strip()
            motion_prompt = str(sc.get("motion_prompt") or "").strip()
            duration = float(sc.get("duration") or 5.0)

            if not narration:
                raise ValueError(f"Scene {scene_number} missing narration")
            if not scene_action and not scene_background and not legacy_image_prompt:
                raise ValueError(f"Scene {scene_number} has no scene content")

            # Combined image_prompt for backward compat
            if scene_action or scene_background:
                combined_prompt = " in ".join(p for p in [scene_action, scene_background] if p)
            else:
                combined_prompt = legacy_image_prompt

            scenes.append({
                "scene_number": scene_number,
                "title_text": title,
                "narration": narration,
                "duration": duration,
                "scene_action": scene_action,
                "scene_background": scene_background,
                "image_prompt": combined_prompt,
                "motion_prompt": motion_prompt,
            })

        scenes.sort(key=lambda x: x["scene_number"])

        return {
            "title": title,
            "theme": theme,
            "format_preset_id": format_preset_id,
            "scenes": scenes,
            "prompt": prompt,
            "created_at": datetime.now().isoformat(),
            "num_scenes": len(scenes),
        }

    def save_story(self, story_data: Dict, filename: Optional[str] = None) -> Path:
        if filename is None:
            title_slug = story_data.get("title", "story").lower()
            title_slug = "".join(c if c.isalnum() else "_" for c in title_slug)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{title_slug}_{timestamp}.json"

        filepath = config.STORIES_DIR / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Story saved to: {filepath}")
        return filepath

    def load_story(self, filepath: Path) -> Dict:
        with open(filepath, "r", encoding="utf-8") as f:
            story_data = json.load(f)
        return story_data

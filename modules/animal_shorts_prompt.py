"""
#동물쇼츠 - Animal Interview Video Prompt Compiler

Produces ONE final generation prompt from ROLE_SITUATION, LOCATION, TONE, ASPECT_RATIO, NUM_SCENES.
Uses generate_json (prompt field) for consistency with existing flow.
"""
from typing import Optional

from .gemini_client import generate_json
from .prompt_utils import aspect_ratio_desc


ANIMAL_SHORTS_SYSTEM_PROMPT = """You are a prompt compiler for a short-form AI video generator. Your job is to produce ONE final generation prompt that strictly follows the rules below while customizing only the allowed fields from user inputs.

INPUTS (from user UI):
- ROLE_SITUATION: {ROLE_SITUATION}  (e.g., "전통시장 상인", "신입 직장인", "연애상담 중", "알바 첫날")
- LOCATION: {LOCATION}  (e.g., "전통시장 채소가게 앞", "회사 로비", "카페 테라스", "동네 골목")
- TONE: {TONE}  (e.g., "잔잔", "현실적", "살짝 장난", "덤덤", "약간 찔림", "할머니 말투", "MZ 말투")
- NUM_SCENES: {NUM_SCENES}
- USER_IMAGE: an uploaded image of the animal character (reference)

NON-NEGOTIABLE CONSISTENCY RULES:
1) Character identity lock: Use USER_IMAGE as the reference. Do NOT alter species, face, eye/nose/mouth design, fur texture, proportions, or recognizable identity. No redesign, no deformation, no style change of the character.
   - CRITICAL: If USER_IMAGE has a background, IGNORE the background entirely. Use ONLY the character (animal figure) from the reference. The background in the output must come from {{LOCATION}}, never from the user's photo.
2) Outfit lock: Outfit must remain the same across all {NUM_SCENES} scenes. Outfit should match ROLE_SITUATION while respecting the character's identity.
3) Location lock: LOCATION is fixed across all {NUM_SCENES} scenes (same layout/composition). Only minor background motion allowed (light passerby blur).
4) Reporter prop lock: Reporter stays off-camera. The same reporter hand and the same microphone design must be visible in ALL {NUM_SCENES} scenes consistently.
5) Camera continuity: Handheld street interview vibe. Keep framing consistent with small natural camera variation only (no major angle jumps).
6) Safety/cleanliness: No logos, no watermarks, no brand marks. Avoid copyrighted characters/brands unless the provided USER_IMAGE is the user's own.

VISUAL STYLE (fixed defaults unless contradictory to user inputs):
- Handheld street interview feel, natural lighting
- Warm Instagram-style golden filter (cozy, slightly nostalgic)
- Bold Korean subtitles in lower third (short lines, meme-friendly)
- Realistic-ish background with webtoon/soft cinematic clarity (prioritize character consistency over style experimentation)

DIALOGUE & SUBTITLES RULES:
- Interviewer voice is off-camera; only subtitles show the interviewer questions and character answers.
- Korean dialogue only.
- Short, punchy, relatable lines.
- Humor comes from contrast, timing, and tone (TONE).
- Final scene must end upbeat with visible bright laugh/smile.
- No profanity.

SCENE BLUEPRINT (must always be {NUM_SCENES} scenes, fixed beat order):
The blueprint below uses 6 beats as a narrative reference arc. If NUM_SCENES differs from 6, proportionally adjust the number of scenes while maintaining the arc: Opening → Premise → Tension → Confidence → Humor → Wrap-up.
Scene 1: Opening question about ROLE_SITUATION performance at LOCATION.
Scene 2: Character reply (sets comedic premise).
Scene 3: Follow-up question that raises tension (competition/pressure/problem).
Scene 4: Character responds with confidence/pride; gesture toward relevant surroundings/props.
Scene 5: Character adds a playful aside/shrug/sigh (comedic timing).
Scene 6: Positive wrap-up line + bright laugh/smile ending.

PROMPT ASSEMBLY INSTRUCTIONS:
- Replace placeholders {{ROLE_SITUATION}}, {{LOCATION}}, {{TONE}}, {{ASPECT_RATIO_DESC}}, {{NUM_SCENES}} into the final prompt.
- Ensure outfit description matches {{ROLE_SITUATION}}. If {{ROLE_SITUATION}} implies a specific uniform, adapt accordingly (e.g., officewear, apron, delivery jacket). If not specified, use a sensible outfit for the role.
- Ensure background props match {{LOCATION}} (e.g., market baskets, office lobby signage, café tables).
- CRITICAL: Every scene's visual/image description MUST explicitly include "{{LOCATION}}" so that all {NUM_SCENES} scenes show the SAME place. No scene may have a different background.
- Keep all other constraints unchanged.

OUTPUT:
Return JSON with a single "prompt" field containing the compiled prompt. Example: {{"prompt": "Create a 9:16 vertical short-form street interview video..."}}

FINAL GENERATION PROMPT TO OUTPUT (put inside the "prompt" field):

Create a {{ASPECT_RATIO_DESC}} short-form street interview video with {NUM_SCENES} sequential scenes.

IMPORTANT CONSISTENCY:
- Use the provided reference image (USER_IMAGE) to keep the animal character EXACTLY the same in all {NUM_SCENES} scenes.
- CRITICAL: If USER_IMAGE has a background, IGNORE it. Use ONLY the character from the reference. Background must come from {{LOCATION}}, never from the user's photo.
- Do NOT change species, face, eye/nose/mouth design, fur texture, proportions, or identity.
- Outfit must remain identical across all scenes and must fit: {{ROLE_SITUATION}}.
- The reporter remains off-camera, but the SAME reporter hand and the SAME microphone design must be visible in EVERY scene.
- Location is fixed and must remain the SAME across all {NUM_SCENES} scenes: {{LOCATION}}.
- No logos, no watermarks.

CHARACTER:
Anthropomorphized animal character from USER_IMAGE, standing upright, same exact appearance.
Outfit (fixed across all scenes): outfit appropriate to {{ROLE_SITUATION}} (keep consistent).

SETTING (fixed):
{{LOCATION}}. Keep identical layout and background across all scenes. Warm daylight.
Apply a warm Instagram-style golden filter (cozy, nostalgic).

STYLE:
Handheld street interview vibe, natural lighting, slight realistic depth of field.
Bold Korean subtitles in the lower third.
Tone of dialogue and timing: {{TONE}}.
Keep camera framing consistent with only subtle handheld variation.

RULES:
Location fixed. Outfit fixed. Character fixed. Reporter hand + microphone fixed.
Only small gestures, facial expressions, and posture change per scene.

{NUM_SCENES} SCENES (dialogue must be short and punchy, Korean only):
Distribute {NUM_SCENES} scenes following the arc: Opening question → Character premise → Tension/pressure → Confidence/pride → Playful aside → Positive wrap-up (expand or compress intermediate scenes proportionally).
1) Reporter (subtitle): "{{ROLE_SITUATION}} 요즘 어때요? 잘 돼요?"  Character reacts (calm/ready).
2) Character answers in {{TONE}}, setting the premise about how it's going.
3) Reporter follow-up about competition/pressure relevant to {{ROLE_SITUATION}}.
4) Character responds with confidence/pride and gestures toward the surroundings/props that fit {{LOCATION}}.
5) Character adds a playful aside (shrug/sigh) with comedic timing in {{TONE}}.
6) Character closes with an upbeat line and ends with a bright laugh and warm smile. Positive ending energy.

Ensure the character and reporter hand/microphone remain visually identical across all {NUM_SCENES} scenes. Emphasize relatable humor and crisp subtitle readability."""


def compile_animal_shorts_prompt(
    role_situation: str,
    location: str,
    tone: str,
    aspect_ratio: str = "9:16",
    num_scenes: int = 6,
    additional_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """Compile final generation prompt from user inputs. Returns prompt string."""
    extra = f"\n- ADDITIONAL (user's extra wishes): {additional_prompt}" if (additional_prompt or "").strip() else ""
    ar_desc = aspect_ratio_desc(aspect_ratio)
    user_prompt = f"""INPUTS:
- ROLE_SITUATION: {role_situation}
- LOCATION: {location}
- TONE: {tone}
- NUM_SCENES: {num_scenes}
- ASPECT_RATIO_DESC: {ar_desc}{extra}

Produce the final generation prompt. Replace {{ROLE_SITUATION}}, {{LOCATION}}, {{TONE}}, {{NUM_SCENES}}, {{ASPECT_RATIO_DESC}} with the values above. If ADDITIONAL is provided, incorporate those wishes into the prompt where appropriate. Output JSON with a "prompt" field containing the compiled prompt."""

    data = generate_json(
        system_prompt=ANIMAL_SHORTS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
    )
    return (data.get("prompt") or "").strip()

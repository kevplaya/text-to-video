"""
#상황극 - Situational Drama Video Prompt Compiler

Produces ONE final generation prompt from SITUATION, LOCATION, ROLE_TYPE, MOOD, ASPECT_RATIO, NUM_SCENES.
Uses generate_json (prompt field) for consistency with existing flow.
"""
from typing import Optional

from .gemini_client import generate_json
from .prompt_utils import aspect_ratio_desc


SITUATION_DRAMA_SYSTEM_PROMPT = """You are a prompt compiler for a short-form AI video generator. Your job is to produce ONE final generation prompt. You will receive SITUATION (상황), LOCATION (장소), ROLE_TYPE (주인공 포지션), TONE (분위기), NUM_SCENES, and ASPECT_RATIO_DESC. Output the prompt below with those placeholders replaced by the actual values. Put the result in a JSON "prompt" field.

---

SYSTEM PROMPT — {NUM_SCENES}-Scene Realistic Two-Character Short Drama

Create a {ASPECT_RATIO_DESC} realistic short-form situational drama with {NUM_SCENES} sequential scenes.

INPUTS:
- USER_IMAGE (main character reference)
- SCENARIO: {SCENARIO}
- LOCATION: {LOCATION}
- ROLE_TYPE: {ROLE_TYPE}
- TONE: {TONE}
- NUM_SCENES: {NUM_SCENES}

GLOBAL GOAL:
Create a realistic, subtle short drama based on {SCENARIO}.
The same main character from USER_IMAGE must remain perfectly consistent across all {NUM_SCENES} scenes.
- CRITICAL: If USER_IMAGE has a background, IGNORE it. Use ONLY the character from the reference. Background must come from {LOCATION}, never from the user's photo.

MAIN CHARACTER LOCK:
- Do NOT change species, face, proportions, fur/skin texture, or identity.
- Outfit remains consistent.
- Natural, restrained expressions only (no exaggerated cartoon emotion).

SECOND CHARACTER RULES:
- Korean human.
- Face must NEVER be shown.
- Only body, shoulder, back, hands, silhouette allowed.
- Use realistic framing (over-the-shoulder, cropped angles).
- No dramatic cinematic reveals.

CONSISTENCY:
- Location lock: {LOCATION} is fixed across all {NUM_SCENES} scenes (same layout/composition). Only minor background motion allowed.
- CRITICAL: Every scene's visual/image description MUST explicitly include "{LOCATION}" so that all {NUM_SCENES} scenes show the SAME place. No scene may have a different background.
- Lighting natural and consistent.
- Same continuous filming session feel.
- No logos, no watermarks.

REALISTIC STYLE:
- Minimal background music implied.
- Subtle handheld camera.
- Natural pauses between lines.
- Korean subtitles, short and understated.
- Avoid exaggerated punchlines.
- Dialogue should feel like something someone actually says.

EMOTIONAL DIRECTION:
ROLE_TYPE influences posture and response style.
TONE influences pacing and intensity.
Keep reactions small but meaningful.

CINEMATIC DETAIL:
Use physical cues instead of dramatic acting:
- Fingers tightening around a cup
- Papers sliding across table
- Shoulder brushing lightly
- Small sigh before speaking
- Eye movement (main character only)

SCENE STRUCTURE:
The blueprint below uses 6 beats as a narrative reference arc. If NUM_SCENES differs from 6, proportionally adjust the number of scenes while maintaining the arc: Setup → Small tension → Pause → Honest line → Quiet escalation → Soft ending.

Scene 1 – Setup
Establish {SCENARIO} at {LOCATION}. Setting: {LOCATION}.
Second character present (face hidden).
Normal tone.

Scene 2 – Small tension
Short line from second character (subtitle only).
Subtle body movement.

Scene 3 – Pause moment
Main character hesitates before responding.

Scene 4 – Honest line
Main character gives realistic, slightly vulnerable line.

Scene 5 – Quiet escalation
Physical detail shot (cup, paper, distance shift).
Minimal dialogue.

Scene 6 – Soft ending
No dramatic twist.
Either quiet smile, small nod, or slow walk away.
Emotion lingers.

Ensure:
- Main character remains identical.
- Second character face never visible.
- Emotions conveyed through restraint.
- Avoid melodrama.
- Location fixed: {LOCATION}. Keep identical layout and background across all {NUM_SCENES} scenes.

---

Replace {SCENARIO}, {LOCATION}, {ROLE_TYPE}, {TONE}, {NUM_SCENES} with the values above. Replace {ASPECT_RATIO_DESC} with the ASPECT_RATIO_DESC value (e.g. "vertical 9:16", "horizontal 16:9", "square 1:1"). If ADDITIONAL is provided, weave it into the prompt where appropriate. Output JSON: {"prompt": "..."}."""


def compile_situation_drama_prompt(
    situation: str,
    location: str,
    character_position: str,
    mood: str,
    aspect_ratio: str = "9:16",
    num_scenes: int = 6,
    additional_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """Compile final generation prompt from situation drama inputs. Returns prompt string."""
    extra = f"\n- ADDITIONAL: {additional_prompt}" if (additional_prompt or "").strip() else ""
    ar_desc = aspect_ratio_desc(aspect_ratio)
    user_prompt = f"""INPUTS:
- SITUATION (→ SCENARIO): {situation}
- LOCATION (장소): {location}
- ROLE_TYPE (주인공 포지션): {character_position}
- TONE (분위기): {mood}
- NUM_SCENES: {num_scenes}
- ASPECT_RATIO_DESC: {ar_desc}{extra}

Replace {{SCENARIO}}, {{LOCATION}}, {{ROLE_TYPE}}, {{TONE}}, {{NUM_SCENES}}, {{ASPECT_RATIO_DESC}} in the template with these values. Output JSON with a "prompt" field containing the compiled prompt."""

    data = generate_json(
        system_prompt=SITUATION_DRAMA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
    )
    return (data.get("prompt") or "").strip()

"""
#요리 - Cooking Short-Form Video Prompt Compiler

Produces ONE final generation prompt from MENU, COOKING_MOOD, LOCATION, ASPECT_RATIO, NUM_SCENES.
Uses generate_json (prompt field) for consistency with existing flow.
"""
from typing import Optional

from .gemini_client import generate_json
from .prompt_utils import aspect_ratio_desc


COOKING_SYSTEM_PROMPT = """You are a prompt compiler for a short-form AI video generator. Your job is to produce ONE final generation prompt. You will receive MENU (요리 이름), COOKING_MOOD (요리 분위기), LOCATION (장소), NUM_SCENES, and ASPECT_RATIO_DESC. Output the prompt below with those placeholders replaced by the actual values. Put the result in a JSON "prompt" field.

---

SYSTEM PROMPT — {NUM_SCENES}-Scene Cooking Short-Form Video

You are generating a {ASPECT_RATIO_DESC} short-form cooking video with {NUM_SCENES} sequential scenes.

INPUTS:
- USER_IMAGE (reference character image)
- MENU: {MENU}
- MOOD: {MOOD}
- LOCATION: {LOCATION}
- NUM_SCENES: {NUM_SCENES}

GLOBAL GOAL:
Create a {NUM_SCENES}-scene cooking video where the same exact anthropomorphic character from USER_IMAGE cooks {MENU}.
Character identity must remain perfectly consistent across all scenes.

NON-NEGOTIABLE CONSISTENCY RULES:
1) Use USER_IMAGE as reference.
   Do NOT change species, face, fur/skin texture, proportions, or identity.
   - CRITICAL: If USER_IMAGE has a background, IGNORE it. Use ONLY the character from the reference. Background must come from {LOCATION}, never from the user's photo.
2) Outfit must remain identical across all {NUM_SCENES} scenes.
   Outfit should logically match cooking (e.g., apron added if appropriate) but remain consistent.
3) Location {LOCATION} must remain fixed across all scenes.
   - CRITICAL: Every scene's visual/image description MUST explicitly include "{LOCATION}" so that all {NUM_SCENES} scenes show the SAME place.
4) No logos, no watermarks.

STYLE:
{ASPECT_RATIO_DESC} format.
Warm, cozy Instagram-style filter.
Soft natural lighting.
Cinematic but casual cooking vlog feel.
Clear food visuals.
Korean subtitle captions (short, minimal).

MOOD CONTROL:
Video pacing, facial expression, subtitle tone, and gesture style must reflect {MOOD}.

SCENE STRUCTURE ({NUM_SCENES} beats):
The blueprint below uses 6 beats as a narrative reference arc. If NUM_SCENES differs from 6, proportionally adjust the number of scenes while maintaining the arc: Intro → Prep → Cooking → Reaction → Plating → Final bite/ending.

Scene 1 - Intro
Character shows ingredients for {MENU}.
Medium shot. Friendly or mood-matching expression.

Scene 2 - Preparation
Cutting or prepping ingredients.
Close-up on hands + food texture.

Scene 3 - Cooking action
Stirring / frying / boiling.
Steam or sizzling visible.

Scene 4 - Reaction moment
Character reacts to smell or taste.
Expression matches {MOOD}.

Scene 5 - Plating
Food neatly plated.
Slight proud gesture.

Scene 6 - Final bite & ending
Character takes a bite.
Bright smile or satisfying expression.
Positive closing vibe.

DIALOGUE RULES:
Short Korean subtitles.
No long narration.
Natural, relatable tone.
End on warm, satisfying feeling.

Ensure character consistency across all {NUM_SCENES} scenes.
Only cooking actions and hand movements vary.
Location and outfit remain fixed.

---

Replace {MENU}, {MOOD}, {LOCATION}, {NUM_SCENES}, {ASPECT_RATIO_DESC} with the values above. If ADDITIONAL is provided, weave it into the prompt where appropriate. Output JSON: {"prompt": "..."}."""


def compile_cooking_prompt(
    menu: str,
    cooking_mood: str,
    location: str,
    aspect_ratio: str = "9:16",
    num_scenes: int = 6,
    additional_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """Compile final generation prompt from cooking inputs. Returns prompt string."""
    extra = f"\n- ADDITIONAL: {additional_prompt}" if (additional_prompt or "").strip() else ""
    ar_desc = aspect_ratio_desc(aspect_ratio)
    user_prompt = f"""INPUTS:
- MENU (요리 이름): {menu}
- COOKING_MOOD (요리 분위기): {cooking_mood}
- LOCATION (장소): {location}
- NUM_SCENES: {num_scenes}
- ASPECT_RATIO_DESC: {ar_desc}{extra}

Replace {{MENU}}, {{MOOD}}, {{LOCATION}}, {{NUM_SCENES}}, {{ASPECT_RATIO_DESC}} in the template with these values. Output JSON with a "prompt" field containing the compiled prompt."""

    data = generate_json(
        system_prompt=COOKING_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
    )
    return (data.get("prompt") or "").strip()

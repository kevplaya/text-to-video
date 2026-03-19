"""
#룩북 - Fashion Lookbook Short-Form Video Prompt Compiler

Produces ONE final generation prompt from STYLE_CONCEPT, MOOD, LOCATION, ASPECT_RATIO, NUM_SCENES.
Uses generate_json (prompt field) for consistency with existing flow.
"""
from typing import Optional

from .gemini_client import generate_json
from .prompt_utils import aspect_ratio_desc


LOOKBOOK_SYSTEM_PROMPT = """You are a prompt compiler for a short-form AI video generator. Your job is to produce ONE final generation prompt. You will receive STYLE_CONCEPT (컨셉), MOOD (분위기), LOCATION (촬영 장소), NUM_SCENES, and ASPECT_RATIO_DESC. Output the prompt below with those placeholders replaced by the actual values. Put the result in a JSON "prompt" field.

---

SYSTEM PROMPT — {NUM_SCENES}-Scene Fashion Lookbook Short-Form Video

You are generating a {ASPECT_RATIO_DESC} short-form fashion lookbook video with {NUM_SCENES} sequential scenes.

INPUTS:
- USER_IMAGE (reference character image)
- STYLE_CONCEPT: {STYLE_CONCEPT}
- MOOD: {MOOD}
- LOCATION: {LOCATION}
- NUM_SCENES: {NUM_SCENES}

GLOBAL GOAL:
Create a {NUM_SCENES}-scene fashion lookbook video where the same exact character from USER_IMAGE models outfits based on {STYLE_CONCEPT}.
Character identity must remain perfectly consistent.

NON-NEGOTIABLE CONSISTENCY RULES:
1) Use USER_IMAGE as reference.
   Do NOT change species, face, body proportions, or identity.
   - CRITICAL: If USER_IMAGE has a background, IGNORE it. Use ONLY the character from the reference. Background must come from {LOCATION}, never from the user's photo.
2) Location {LOCATION} must remain fixed across all {NUM_SCENES} scenes.
   - CRITICAL: Every scene's visual/image description MUST explicitly include "{LOCATION}" so that all {NUM_SCENES} scenes show the SAME place.
3) Lighting and color grading remain consistent.
4) No logos, no watermarks.

OUTFIT RULE:
Outfits change per scene but must align with {STYLE_CONCEPT}.
Each scene shows a different variation within the same concept.
Example: layering change, color variation, accessory change.

STYLE:
{ASPECT_RATIO_DESC}.
Cinematic fashion reel vibe.
Smooth camera motion.
Warm or stylized color grading matching {MOOD}.
Minimal Korean subtitle captions (short, aesthetic).

SCENE STRUCTURE ({NUM_SCENES} beats):
The blueprint below uses 6 beats as a narrative reference arc. If NUM_SCENES differs from 6, proportionally adjust the number of scenes while maintaining the arc: Intro walk → Profile pose → Detail shot → Confident pose → Natural movement → Final hero shot.

Scene 1 – Intro Walk
Character walking toward camera.
Outfit look #1.

Scene 2 – Side Profile Pose
Subtle turn and glance.
Outfit look #2.

Scene 3 – Detail Shot
Close-up of fabric, accessory, or shoes.
Outfit look #3.

Scene 4 – Confident Pose
Mid shot, strong stance.
Outfit look #4.

Scene 5 – Natural Movement
Hair movement / light spin / relaxed pose.
Outfit look #5.

Scene 6 – Final Hero Shot
Strong finishing pose.
Confident expression matching {MOOD}.
Stylish ending.

MOOD CONTROL:
Facial expression, walk speed, camera rhythm must reflect {MOOD}.

Ensure character facial identity remains identical across all {NUM_SCENES} scenes.
Only outfits and poses change.
Location remains fixed.

---

Replace {STYLE_CONCEPT}, {MOOD}, {LOCATION}, {NUM_SCENES}, {ASPECT_RATIO_DESC} with the values above. If ADDITIONAL is provided, weave it into the prompt where appropriate. Output JSON: {"prompt": "..."}."""


def compile_lookbook_prompt(
    style_concept: str,
    mood: str,
    location: str,
    aspect_ratio: str = "9:16",
    num_scenes: int = 6,
    additional_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """Compile final generation prompt from lookbook inputs. Returns prompt string."""
    extra = f"\n- ADDITIONAL: {additional_prompt}" if (additional_prompt or "").strip() else ""
    ar_desc = aspect_ratio_desc(aspect_ratio)
    user_prompt = f"""INPUTS:
- STYLE_CONCEPT (컨셉): {style_concept}
- MOOD (분위기): {mood}
- LOCATION (촬영 장소): {location}
- NUM_SCENES: {num_scenes}
- ASPECT_RATIO_DESC: {ar_desc}{extra}

Replace {{STYLE_CONCEPT}}, {{MOOD}}, {{LOCATION}}, {{NUM_SCENES}}, {{ASPECT_RATIO_DESC}} in the template with these values. Output JSON with a "prompt" field containing the compiled prompt."""

    data = generate_json(
        system_prompt=LOOKBOOK_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        api_key=api_key,
    )
    return (data.get("prompt") or "").strip()

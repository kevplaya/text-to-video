"""
All prompt constants centralized in one place.
Ported from sibling production project.
"""

# Character preservation rules when reference images are used
CHARACTER_PRESERVATION_RULES = """CRITICAL CHARACTER PRESERVATION RULES:
The input image is the exact reference of the character.
You MUST:
- Preserve 100% of the original facial structure.
- Do NOT alter eye shape, eye size, eye position.
- Do NOT alter nose shape or position.
- Do NOT alter mouth shape, expression geometry, or proportions.
- Do NOT redesign or stylize the character.
- Do NOT reinterpret the character in a different art style.
- Do NOT add new facial details.
- Do NOT beautify or exaggerate proportions.
The character must remain visually identical to the uploaded image.
Only change movement, camera angle, lighting, or background as directed by the scene prompt.
If uncertain, prioritize preservation over creativity.
Character consistency is more important than cinematic effect."""

# Image quality suffix
IMAGE_ENHANCEMENT_SUFFIX = "high quality, professional photography, 8k resolution"

# DNA field labels for rendering character_dna dict into prompt
DNA_FIELD_LABELS = {
    "subject_type": "Subject type",
    "body_structure": "Body structure",
    "face": "Face",
    "hair": "Hair",
    "materials": "Materials",
    "colors": "Colors",
    "outfit": "Outfit",
    "distinctive_features": "Distinctive features",
}

# Video prompt constants
VIDEO_REFERENCE_RULES = "An image of the characters from image_0.png."

VIDEO_PRESERVATION_RULES = (
    "Preserve the exact same face, eyes, hair, body proportions, outfit, colors, and 2D character design from the input image. "
    "Do not redesign, reinterpret, replace, or restyle the character in any way."
)

DEFAULT_VIDEO_NEGATIVE = (
    "different character, different face, different hairstyle, different outfit, "
    "identity drift, face drift, style inconsistency, dramatic reframing, camera jump"
)

VIDEO_RUNTIME_RULES = (
    "AUDIO: Output only diegetic sound unless narration voice is provided.\n"
    "VISUAL: Keep the exact original character identity and 2D appearance stable across the whole clip.\n"
    "CAMERA: Keep framing fixed or nearly fixed. Avoid reframing, push-in, angle changes, or dramatic camera motion unless explicitly required."
)

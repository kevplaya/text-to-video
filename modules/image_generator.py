"""
Image Generator Module - Multi-backend support with character preservation.
Supports: Local SD, HuggingFace API, FLUX.1-dev, Gemini Imagen/Nano Banana, Grok

Enhanced with:
- Structured prompt assembly (reference → preservation → DNA → style → scene → quality → avoid)
- scene_action + scene_background separation
- Prompt regeneration on failure
- Character DNA support
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, List
import io
import time
import requests
from PIL import Image

from .prompt_constants import (
    CHARACTER_PRESERVATION_RULES,
    IMAGE_ENHANCEMENT_SUFFIX,
    DNA_FIELD_LABELS,
)
from .exceptions import GeminiImageGenerationFailed

try:
    import google.genai as genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import torch
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    torch = None
    FluxPipeline = None

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resize helpers (unchanged from original)
# ---------------------------------------------------------------------------

def resize_with_padding(image: Image.Image, target_width: int, target_height: int, bg_color=(0, 0, 0)) -> Image.Image:
    img_ratio = image.width / image.height
    target_ratio = target_width / target_height
    if img_ratio > target_ratio:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * img_ratio)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (target_width, target_height), bg_color)
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def resize_with_crop(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    img_ratio = image.width / image.height
    target_ratio = target_width / target_height
    if img_ratio > target_ratio:
        new_height = image.height
        new_width = int(new_height * target_ratio)
        left = (image.width - new_width) // 2
        image = image.crop((left, 0, left + new_width, new_height))
    else:
        new_width = image.width
        new_height = int(new_width / target_ratio)
        top = (image.height - new_height) // 2
        image = image.crop((0, top, new_width, top + new_height))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def smart_resize(image: Image.Image, target_width: int, target_height: int, mode: str = "padding") -> Image.Image:
    if mode == "padding":
        return resize_with_padding(image, target_width, target_height)
    elif mode == "crop":
        return resize_with_crop(image, target_width, target_height)
    elif mode == "stretch":
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    else:
        raise ValueError(f"Unknown resize mode: {mode}")


# ---------------------------------------------------------------------------
# Prompt assembly helpers (ported from sibling)
# ---------------------------------------------------------------------------

def _get_style_lock(style_profile: Optional[dict]) -> str:
    if not style_profile or not isinstance(style_profile, dict):
        return ""
    _sl_raw = style_profile.get("style_lock") or ""
    if isinstance(_sl_raw, list):
        _sl_raw = ", ".join(str(x) for x in _sl_raw)
    return str(_sl_raw).strip()


def _get_negative_prompt(style_profile: Optional[dict]) -> str:
    if not style_profile or not isinstance(style_profile, dict):
        return ""
    _raw = style_profile.get("negative_prompt") or ""
    if isinstance(_raw, list):
        _raw = ", ".join(str(x) for x in _raw)
    return str(_raw).strip()


def _resolve_scene_prompt(scene: dict) -> str:
    """Extract scene description. Prefers scene_action + scene_background, falls back to image_prompt."""
    action = (scene.get("scene_action") or "").strip()
    background = (scene.get("scene_background") or "").strip()
    if action or background:
        return " in ".join(p for p in [action, background] if p)
    legacy = (scene.get("image_prompt") or "").strip()
    legacy = re.sub(
        r"^(?:An image of the character from image_0\.png[.,]?\s*)",
        "", legacy, flags=re.IGNORECASE,
    ).strip()
    return legacy


def _format_character_dna(dna: Any) -> str:
    """Render character_dna into a prompt block."""
    if isinstance(dna, dict):
        lines = []
        for field, label in DNA_FIELD_LABELS.items():
            val = (dna.get(field) or "").strip()
            if val:
                lines.append(f"  {label}: {val}")
        for k, v in dna.items():
            if k not in DNA_FIELD_LABELS and v:
                lines.append(f"  {k.replace('_', ' ').capitalize()}: {v}")
        return "\n".join(lines)
    if isinstance(dna, str):
        return dna.strip()
    return ""


def _build_grok_prompt(
    *,
    scene: dict,
    style_guide: Optional[str],
    style_profile: Optional[dict],
    has_ref: bool,
    ignore_reference_background: bool,
    aspect_ratio: str,
    scene_number: int,
) -> str:
    """Assemble Grok image-gen prompt in strict priority order."""
    parts: list = []

    if has_ref:
        bg_note = (
            " Use only the character; ignore the background in the reference image."
            if ignore_reference_background else ""
        )
        parts.append(f"The same character as in image_0.png.{bg_note}")
        parts.append(CHARACTER_PRESERVATION_RULES)

        if style_profile and isinstance(style_profile, dict):
            dna_raw = style_profile.get("character_dna") or style_profile.get("gemini_prompt") or ""
            sp_dna = _format_character_dna(dna_raw)
            if sp_dna:
                parts.append(f"Character identity:\n{sp_dna}")

        parts.append(
            "Preserve the original color palette and outfit details exactly.\n"
            "Do not change hair color, outfit colors, or accessory colors."
        )

        sl = _get_style_lock(style_profile)
        if sl:
            parts.append(f"Rendering style: {sl}")

    scene_desc = _resolve_scene_prompt(scene)
    scene_sections = [scene_desc]
    if style_guide:
        scene_sections.append(f"Style guide:\n{style_guide}")
    scene_block = "\n\n".join(s for s in scene_sections if s)
    if scene_block:
        parts.append(f"Scene:\n{scene_block}")

    parts.append(IMAGE_ENHANCEMENT_SUFFIX)

    neg = _get_negative_prompt(style_profile)
    if neg:
        parts.append(f"Avoid: {neg}")

    body = "\n\n".join(p for p in parts if p)
    return (
        f"Aspect ratio: {aspect_ratio}, high resolution. Scene {scene_number}.\n\n"
        f"{body}\n\n"
        "Do not include text, watermarks, or logos."
    )


def _build_nano_banana_prompt(
    *,
    scene: dict,
    style_guide: Optional[str],
    style_profile: Optional[dict],
    has_ref: bool,
    ignore_reference_background: bool,
    aspect_ratio: str,
    scene_number: int,
    target_w: int,
    target_h: int,
) -> str:
    """Assemble Gemini image-gen (nano-banana) prompt in strict priority order."""
    parts: list = []

    if has_ref:
        bg_note = (
            " 배경은 무시하고 캐릭터만 참조하세요."
            if ignore_reference_background else ""
        )
        parts.append(f"image_0.png와 동일한 캐릭터를 그리세요.{bg_note}")
        parts.append(CHARACTER_PRESERVATION_RULES)

        if style_profile and isinstance(style_profile, dict):
            dna_raw = style_profile.get("character_dna") or style_profile.get("gemini_prompt") or ""
            sp_dna = _format_character_dna(dna_raw)
            if sp_dna:
                parts.append(f"캐릭터 정체성:\n{sp_dna}")

        parts.append(
            "원본 컬러 팔레트와 의상 디테일을 정확히 유지하세요.\n"
            "헤어 색상, 의상 색상, 액세서리 색상을 변경하지 마세요."
        )

        sl = _get_style_lock(style_profile)
        if sl:
            parts.append(f"렌더링 스타일: {sl}")

    scene_desc = _resolve_scene_prompt(scene)
    scene_sections = [scene_desc]
    if style_guide:
        scene_sections.append(f"스타일 가이드:\n{style_guide}")
    scene_block = "\n\n".join(s for s in scene_sections if s)
    if scene_block:
        parts.append(f"장면:\n{scene_block}")

    parts.append(f"Quality: {IMAGE_ENHANCEMENT_SUFFIX}")

    neg = _get_negative_prompt(style_profile)
    if neg:
        parts.append(f"금지: {neg}")

    body = "\n\n".join(p for p in parts if p)
    return (
        f"이미지 비율 {aspect_ratio}, {target_w}x{target_h}, 고해상도. 장면 {scene_number}.\n\n"
        f"{body}\n\n"
        "텍스트/워터마크/로고는 넣지 마세요."
    )


# ---------------------------------------------------------------------------
# Aspect ratio helpers
# ---------------------------------------------------------------------------

def _target_size_for_aspect_ratio(aspect_ratio: str) -> tuple:
    ar = (aspect_ratio or "").strip()
    if ar == "1:1":
        return (1080, 1080)
    if ar == "3:4":
        return (1080, 1440)
    if ar == "4:3":
        return (1080, 810)
    if ar == "16:9":
        return (1080, 608)
    # default 9:16
    return (1080, 1920)


# ---------------------------------------------------------------------------
# ImageGenerator class
# ---------------------------------------------------------------------------

class ImageGenerator:
    """Multi-backend image generator with character preservation support."""

    def __init__(self, backend: str = "local", model_id: Optional[str] = None, resize_mode: str = "padding"):
        self.backend = backend
        self.resize_mode = resize_mode

        if backend == "local":
            self._init_local_sd(model_id)
        elif backend == "huggingface":
            self._init_huggingface(model_id)
        elif backend == "flux":
            self._init_flux_local(model_id)
        elif backend in ["imagen", "imagen-fast", "imagen-ultra"]:
            self._init_gemini_imagen(backend, model_id)
        elif backend == "nano-banana":
            self._init_nano_banana(model_id)
        elif backend in ("grok", "grok-imagine-image"):
            self._init_grok(model_id)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_local_sd(self, model_id: Optional[str]):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers and torch are required for local SD.")
        self.model_id = model_id or "runwayml/stable-diffusion-v1-5"
        self.hf_token = config.HF_TOKEN
        logger.info(f"Loading local Stable Diffusion: {self.model_id}")
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32
            else:
                device = "cpu"
                dtype = torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype=dtype, safety_checker=None,
                requires_safety_checker=False, token=self.hf_token
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(device)
            if device == "cuda":
                self.pipe.enable_attention_slicing()
        except Exception as e:
            logger.error(f"Failed to load local SD: {e}")
            raise

    def _init_huggingface(self, model_id: Optional[str]):
        self.hf_token = config.HF_TOKEN
        self.model_id = model_id or config.HF_SD_MODEL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

    def _init_gemini_imagen(self, backend: str, model_id: Optional[str]):
        if not GENAI_AVAILABLE:
            raise ImportError("google.genai not installed")
        self.api_key = config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for Gemini")
        model_map = {
            'imagen': 'imagen-4.0-fast-generate-001',
            'imagen-fast': 'imagen-4.0-fast-generate-001',
            'imagen-ultra': 'imagen-4.0-generate-001',
        }
        self.model_id = model_id or model_map.get(backend, 'imagen-4.0-fast-generate-001')
        self.client = genai.Client(api_key=self.api_key)

    def _init_nano_banana(self, model_id: Optional[str]):
        if not GENAI_AVAILABLE:
            raise ImportError("google.genai not installed")
        self.api_key = config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for Nano Banana")
        self.model_id = model_id or config.GEMINI_IMAGE_MODEL
        self.client = genai.Client(api_key=self.api_key)

    def _init_flux_local(self, model_id: Optional[str]):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers and torch are required for FLUX.1-dev.")
        self.model_id = model_id or config.FLUX_DEV_MODEL
        self.hf_token = config.HF_TOKEN
        logger.info(f"Loading FLUX.1-dev model: {self.model_id}")
        try:
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, token=self.hf_token
            )
            self.pipe.enable_sequential_cpu_offload()
            if hasattr(self.pipe, 'vae'):
                self.pipe.vae.enable_tiling()
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing(1)
        except Exception as e:
            logger.error(f"Failed to load FLUX.1-dev: {e}")
            raise

    def _init_grok(self, model_id: Optional[str]):
        """Initialize Grok backend (requires xai-sdk)."""
        self.model_id = model_id or config.GROK_IMAGE_MODEL
        logger.info(f"Initialized Grok image backend: {self.model_id}")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, ugly",
        width: int = None,
        height: int = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        width = width or config.IMAGE_WIDTH
        height = height or config.IMAGE_HEIGHT

        if self.backend == "local":
            return self._generate_local_sd(prompt, negative_prompt, width, height, seed)
        elif self.backend == "huggingface":
            return self._generate_huggingface(prompt, negative_prompt, width, height, seed)
        elif self.backend == "flux":
            return self._generate_flux_local(prompt, negative_prompt, width, height, seed)
        elif self.backend in ["imagen", "imagen-fast", "imagen-ultra"]:
            return self._generate_gemini_imagen(prompt, width, height, seed)
        elif self.backend == "nano-banana":
            return self._generate_nano_banana(prompt, width, height, seed)
        elif self.backend in ("grok", "grok-imagine-image"):
            raise ValueError("Use generate_image_from_scene() for Grok backend")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_huggingface(self, prompt, negative_prompt, width, height, seed):
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        payload = {"inputs": prompt, "parameters": {"negative_prompt": negative_prompt, "width": width, "height": height}}
        if seed is not None:
            payload["parameters"]["seed"] = seed
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 503:
                    logger.info("Model is loading... waiting 20 seconds")
                    time.sleep(20)
                    continue
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                return image
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(10)
                else:
                    raise

    def _generate_gemini_imagen(self, prompt, width, height, seed):
        if height > width:
            aspect_ratio = "9:16"
        elif width > height:
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"
        try:
            generation_config = types.GenerateContentConfig(response_modalities=["IMAGE"])
            image_config = {"aspectRatio": aspect_ratio}
            if seed is not None:
                image_config["seed"] = seed
            if hasattr(generation_config, 'image_config'):
                generation_config.image_config = image_config
            response = self.client.models.generate_content(
                model=self.model_id, contents=prompt, config=generation_config
            )
            for part in response.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    return image
            raise ValueError("No image in response")
        except Exception as e:
            logger.error(f"Imagen generation failed: {e}")
            raise

    def _generate_nano_banana(self, prompt, width, height, seed):
        if height > width:
            aspect_ratio = "9:16"
        elif width > height:
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"
        try:
            generation_config = types.GenerateContentConfig(temperature=0.9, response_modalities=["IMAGE"])
            image_config = {"aspect_ratio": aspect_ratio}
            if seed is not None:
                image_config["seed"] = seed
            if hasattr(generation_config, 'image_generation_config'):
                generation_config.image_generation_config = image_config
            response = self.client.models.generate_content(
                model=self.model_id, contents=prompt, config=generation_config
            )
            for part in response.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    if image.size != (width, height):
                        image = smart_resize(image, width, height, self.resize_mode)
                    return image
            raise ValueError("No image in response")
        except Exception as e:
            logger.error(f"Nano Banana generation failed: {e}")
            raise

    def _generate_flux_local(self, prompt, negative_prompt, width, height, seed):
        try:
            generator = None
            if seed is not None:
                if torch.cuda.is_available():
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                else:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
            adjusted_width = (width // 16) * 16
            adjusted_height = (height // 16) * 16
            result = self.pipe(
                prompt, height=adjusted_height, width=adjusted_width,
                guidance_scale=3.5, num_inference_steps=28,
                max_sequence_length=256, generator=generator
            )
            image = result.images[0]
            if image.size != (width, height):
                image = smart_resize(image, width, height, self.resize_mode)
            return image
        except Exception as e:
            logger.error(f"FLUX.1-dev generation failed: {e}")
            raise

    def _generate_local_sd(self, prompt, negative_prompt, width, height, seed):
        try:
            generator = None
            if seed is not None:
                device = self.pipe.device
                generator = torch.Generator(device=device).manual_seed(seed)
            result = self.pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_inference_steps=20,
                guidance_scale=7.5, generator=generator
            )
            image = result.images[0]
            if image.size != (width, height):
                image = smart_resize(image, width, height, self.resize_mode)
            return image
        except Exception as e:
            logger.error(f"Local SD generation failed: {e}")
            raise

    def generate_image_from_scene(
        self,
        scene: dict,
        output_path: Path,
        seed: Optional[int] = None,
        style_guide: Optional[str] = None,
        style_profile: Optional[dict] = None,
        reference_image_bytes: Optional[bytes] = None,
        reference_image_mime_type: str = "image/jpeg",
        reference_images: Optional[list] = None,
        aspect_ratio: str = "9:16",
        ignore_reference_background: bool = False,
    ) -> Path:
        """Generate a single image from a scene dict with full character preservation support."""
        # Resolve reference images
        refs: list = []
        if reference_images and len(reference_images) > 0:
            refs = list(reference_images)[:3]
        elif reference_image_bytes:
            refs = [(reference_image_bytes, reference_image_mime_type or "image/jpeg")]

        scene_number = int(scene.get("scene_number") or 0)

        # --- Grok backend ---
        if self.backend in ("grok", "grok-imagine-image"):
            from .grok_client import generate_image_bytes as grok_generate_image_bytes
            from .gemini_client import regenerate_image_prompt

            model = os.environ.get("GROK_IMAGE_MODEL", config.GROK_IMAGE_MODEL)
            full_prompt = _build_grok_prompt(
                scene=scene, style_guide=style_guide, style_profile=style_profile,
                has_ref=bool(refs), ignore_reference_background=ignore_reference_background,
                aspect_ratio=aspect_ratio, scene_number=scene_number,
            )
            try:
                img_bytes = grok_generate_image_bytes(
                    prompt=full_prompt, model=model,
                    reference_images=refs, aspect_ratio=aspect_ratio,
                )
            except Exception as e:
                # Prompt regeneration on failure
                err_msg = f"{e.__class__.__name__}: {str(e)[:400]}"
                scene_text = str(scene.get("narration") or "").strip()
                try:
                    new_prompt = regenerate_image_prompt(
                        original_prompt=_resolve_scene_prompt(scene),
                        error_message=err_msg, scene_text=scene_text,
                        model=os.environ.get("GEMINI_PROMPT_REWRITE_MODEL", config.GEMINI_PROMPT_REWRITE_MODEL),
                    )
                    retry_scene = dict(scene)
                    retry_scene["scene_action"] = new_prompt
                    retry_scene["scene_background"] = ""
                    retry_full = _build_grok_prompt(
                        scene=retry_scene, style_guide=style_guide, style_profile=style_profile,
                        has_ref=bool(refs), ignore_reference_background=ignore_reference_background,
                        aspect_ratio=aspect_ratio, scene_number=scene_number,
                    )
                    img_bytes = grok_generate_image_bytes(
                        prompt=retry_full, model=model,
                        reference_images=refs, aspect_ratio=aspect_ratio,
                    )
                except Exception as e2:
                    from .exceptions import GrokImageGenerationFailed
                    raise GrokImageGenerationFailed(
                        "Grok image generation failed after prompt rewrite retry.",
                        raw_text_preview=f"{err_msg} | retry={e2.__class__.__name__}: {str(e2)[:300]}",
                    ) from e2

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(img_bytes)
            return output_path

        # --- Nano Banana backend (structured prompt) ---
        if self.backend == "nano-banana" and (refs or style_profile):
            from .gemini_client import (
                generate_image_bytes as gemini_generate_image_bytes,
                regenerate_image_prompt,
            )

            model = os.environ.get("GEMINI_IMAGE_MODEL", config.GEMINI_IMAGE_MODEL)
            target_w, target_h = _target_size_for_aspect_ratio(aspect_ratio)
            full_prompt = _build_nano_banana_prompt(
                scene=scene, style_guide=style_guide, style_profile=style_profile,
                has_ref=bool(refs), ignore_reference_background=ignore_reference_background,
                aspect_ratio=aspect_ratio, scene_number=scene_number,
                target_w=target_w, target_h=target_h,
            )
            try:
                img_bytes = gemini_generate_image_bytes(
                    prompt=full_prompt, model=model,
                    reference_images=refs, aspect_ratio=aspect_ratio,
                )
            except Exception as e:
                err_msg = f"{e.__class__.__name__}: {str(e)[:400]}"
                scene_text = str(scene.get("narration") or "").strip()
                try:
                    new_prompt = regenerate_image_prompt(
                        original_prompt=_resolve_scene_prompt(scene),
                        error_message=err_msg, scene_text=scene_text,
                        model=os.environ.get("GEMINI_PROMPT_REWRITE_MODEL", config.GEMINI_PROMPT_REWRITE_MODEL),
                    )
                    retry_scene = dict(scene)
                    retry_scene["scene_action"] = new_prompt
                    retry_scene["scene_background"] = ""
                    retry_full = _build_nano_banana_prompt(
                        scene=retry_scene, style_guide=style_guide, style_profile=style_profile,
                        has_ref=bool(refs), ignore_reference_background=ignore_reference_background,
                        aspect_ratio=aspect_ratio, scene_number=scene_number,
                        target_w=target_w, target_h=target_h,
                    )
                    img_bytes = gemini_generate_image_bytes(
                        prompt=retry_full, model=model,
                        reference_images=refs, aspect_ratio=aspect_ratio,
                    )
                except Exception as e2:
                    raise GeminiImageGenerationFailed(
                        "Image generation failed after prompt rewrite retry.",
                        raw_text_preview=f"{err_msg} | retry={e2.__class__.__name__}: {str(e2)[:300]}",
                    ) from e2

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(img_bytes)
            return output_path

        # --- Legacy backends (local, huggingface, flux, imagen, nano-banana without refs) ---
        prompt = _resolve_scene_prompt(scene)
        if not prompt:
            prompt = scene.get("image_prompt", "")
        if not prompt:
            raise ValueError("Scene must have scene content for image generation")

        if style_guide:
            enhanced_prompt = f"{prompt}, {style_guide}"
        else:
            enhanced_prompt = f"{prompt}, {config.DEFAULT_IMAGE_STYLE}"

        image = self.generate_image(prompt=enhanced_prompt, seed=seed)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logger.info(f"Saved image to: {output_path}")
        return output_path

    def generate_scene_images(
        self,
        scenes: List[dict],
        output_dir: Optional[Path] = None,
        base_filename: str = "scene",
        seed: Optional[int] = None
    ) -> List[Path]:
        output_dir = output_dir or config.IMAGES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for i, scene in enumerate(scenes):
            scene_number = scene.get("scene_number", i + 1)
            prompt = scene.get("image_prompt", "")
            if not prompt:
                continue
            enhanced_prompt = f"{prompt}, cinematic, highly detailed, professional quality"
            current_seed = seed + i if seed is not None else None
            image = self.generate_image(prompt=enhanced_prompt, seed=current_seed)
            filename = f"{base_filename}_{scene_number:02d}.png"
            filepath = output_dir / filename
            image.save(filepath)
            image_paths.append(filepath)
        return image_paths

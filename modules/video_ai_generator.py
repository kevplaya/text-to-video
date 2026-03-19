"""
Video AI Generator — Orchestrates per-scene video generation using Grok, Veo, or Kling.
Uses video_prompt_builder for prompt assembly.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from .video_prompt_builder import build_video_prompt

logger = logging.getLogger(__name__)


class VideoAIGenerator:
    """Orchestrate AI video generation for individual scenes."""

    def __init__(self, backend: str = "grok"):
        """
        Args:
            backend: Video AI backend — "grok", "veo", or "kling"
        """
        self.backend = backend
        logger.info(f"VideoAIGenerator initialized: backend={backend}")

    def generate_scene_video(
        self,
        scene: dict,
        image_path: Path,
        output_path: Path,
        style_profile: Optional[dict] = None,
        aspect_ratio: str = "9:16",
        duration: int = 6,
    ) -> Optional[Path]:
        """
        Generate a video clip for a single scene.

        Args:
            scene: Scene dict with scene_action, scene_background, motion_prompt
            image_path: Path to the scene's source image
            output_path: Where to save the video clip
            style_profile: Character DNA style profile
            aspect_ratio: Output aspect ratio
            duration: Video duration in seconds

        Returns:
            Path to generated video, or None if generation fails gracefully
        """
        scene_action = (scene.get("scene_action") or "").strip()
        scene_background = (scene.get("scene_background") or "").strip()
        motion_prompt = (scene.get("motion_prompt") or "").strip()
        scene_number = scene.get("scene_number", 0)

        video_prompt = build_video_prompt(
            scene_action=scene_action,
            scene_background=scene_background,
            motion_prompt=motion_prompt,
            style_profile=style_profile,
        )

        logger.info(f"Generating video for scene {scene_number} via {self.backend}")

        try:
            if self.backend == "grok":
                video_bytes = self._generate_grok(
                    video_prompt, image_path, aspect_ratio, duration
                )
            elif self.backend == "veo":
                video_bytes = self._generate_veo(
                    video_prompt, image_path, aspect_ratio
                )
            elif self.backend == "kling":
                video_bytes = self._generate_kling(
                    video_prompt, image_path, aspect_ratio, duration
                )
            else:
                logger.error(f"Unknown video AI backend: {self.backend}")
                return None

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(video_bytes)
            logger.info(f"Video saved: {output_path} ({len(video_bytes)} bytes)")
            return output_path

        except Exception as e:
            logger.error(f"Video AI generation failed for scene {scene_number}: {e}")
            return None

    def _generate_grok(self, prompt: str, image_path: Path, aspect_ratio: str, duration: int) -> bytes:
        from .grok_client import image2video

        image_bytes = image_path.read_bytes()
        import mimetypes
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        return image2video(
            prompt=prompt,
            image_base64_data_uri=data_uri,
            duration=duration,
            aspect_ratio=aspect_ratio,
        )

    def _generate_veo(self, prompt: str, image_path: Path, aspect_ratio: str) -> bytes:
        from .veo_client import image2video

        image_bytes = image_path.read_bytes()
        import mimetypes
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"

        return image2video(
            prompt=prompt,
            image_bytes=image_bytes,
            image_mime_type=mime,
            aspect_ratio=aspect_ratio,
        )

    def _generate_kling(self, prompt: str, image_path: Path, aspect_ratio: str, duration: int) -> bytes:
        from .kling_client import image2video_sync

        image_bytes = image_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode("ascii")

        return image2video_sync(
            prompt=prompt,
            image_base64=b64,
            duration=duration,
            aspect_ratio=aspect_ratio,
        )

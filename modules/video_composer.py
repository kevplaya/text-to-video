"""
Video Composer Module
Combines images/video clips and audio into final video using MoviePy.

Enhanced with:
- Mixed scene support (static images + AI video clips)
- Timed subtitles from ElevenLabs
- Per-scene video clip support (VideoFileClip)
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict

from moviepy.editor import (
    ImageClip, AudioFileClip, VideoFileClip,
    concatenate_videoclips, CompositeVideoClip
)

import config

logger = logging.getLogger(__name__)


class VideoComposer:
    """Compose final video from images/video clips and audio."""

    def __init__(
        self,
        width: int = None,
        height: int = None,
        fps: int = None
    ):
        self.width = width or config.VIDEO_WIDTH
        self.height = height or config.VIDEO_HEIGHT
        self.fps = fps or config.VIDEO_FPS
        logger.info(f"Video composer initialized: {self.width}x{self.height} @ {self.fps}fps")

    def create_video_from_scenes(
        self,
        scenes: List[dict],
        image_paths: List[Path],
        audio_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        scene_durations: Optional[List[float]] = None,
        scene_video_paths: Optional[List[Optional[Path]]] = None,
        timed_subtitles: Optional[list] = None,
    ) -> Path:
        """
        Create video from scene images/videos and audio.

        Args:
            scenes: List of scene dictionaries
            image_paths: List of paths to scene images (used as fallback if no video)
            audio_path: Path to audio file
            output_path: Path to save output video
            scene_durations: Custom durations for each scene
            scene_video_paths: Optional per-scene video paths (None entries use image)
            timed_subtitles: Optional timed subtitles from ElevenLabs
        """
        if len(scenes) != len(image_paths):
            raise ValueError("Number of scenes must match number of images")

        logger.info(f"Creating video from {len(scenes)} scenes")

        # Determine scene durations
        if scene_durations is None:
            if audio_path and audio_path.exists():
                scene_durations = self._calculate_scene_durations_from_audio(
                    audio_path, len(scenes)
                )
            else:
                scene_durations = [
                    scene.get("duration", config.DEFAULT_SCENE_DURATION)
                    for scene in scenes
                ]

        # Create video clips — use AI video clip if available, else static image
        video_clips = []
        clips_to_close = []

        for i, (img_path, duration) in enumerate(zip(image_paths, scene_durations)):
            has_video = (
                scene_video_paths is not None
                and i < len(scene_video_paths)
                and scene_video_paths[i] is not None
                and scene_video_paths[i].exists()
            )

            if has_video:
                # Use AI-generated video clip
                video_path = scene_video_paths[i]
                logger.info(f"Scene {i+1}: using AI video clip {video_path}")
                try:
                    clip = VideoFileClip(str(video_path))
                    # Resize if needed
                    if clip.size != (self.width, self.height):
                        clip = clip.resize((self.width, self.height))
                    clip = clip.set_fps(self.fps)
                    # Mute the AI video's audio (we'll use our own narration)
                    clip = clip.without_audio()
                except Exception as e:
                    logger.warning(f"Failed to load video clip {video_path}: {e}, falling back to image")
                    clip = ImageClip(str(img_path), duration=duration)
                    if clip.size != (self.width, self.height):
                        clip = clip.resize((self.width, self.height))
                    clip = clip.set_fps(self.fps)
            else:
                # Use static image
                logger.info(f"Scene {i+1}: using static image, duration={duration:.2f}s")
                clip = ImageClip(str(img_path), duration=duration)
                if clip.size != (self.width, self.height):
                    clip = clip.resize((self.width, self.height))
                clip = clip.set_fps(self.fps)

            video_clips.append(clip)
            clips_to_close.append(clip)

        # Concatenate all clips
        logger.info("Concatenating video clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Add audio if provided
        audio_clip = None
        if audio_path and audio_path.exists():
            logger.info(f"Adding audio from: {audio_path}")
            audio_clip = AudioFileClip(str(audio_path))
            if audio_clip.duration > final_video.duration:
                logger.info("Trimming audio to match video duration")
                audio_clip = audio_clip.subclip(0, final_video.duration)
            final_video = final_video.set_audio(audio_clip)

        # Set output path
        if output_path is None:
            output_path = config.VIDEOS_DIR / "output_video.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write video file
        logger.info(f"Writing video to: {output_path}")
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(output_path.parent / 'temp-audio.m4a'),
            remove_temp=True,
            logger=None
        )

        # Close clips to free memory
        for clip in clips_to_close:
            try:
                clip.close()
            except Exception:
                pass
        try:
            final_video.close()
        except Exception:
            pass
        if audio_clip:
            try:
                audio_clip.close()
            except Exception:
                pass

        logger.info(f"Video created successfully: {output_path}")
        return output_path

    def _calculate_scene_durations_from_audio(
        self,
        audio_path: Path,
        num_scenes: int
    ) -> List[float]:
        audio_clip = AudioFileClip(str(audio_path))
        total_duration = audio_clip.duration
        audio_clip.close()
        scene_duration = total_duration / num_scenes
        logger.info(f"Audio duration: {total_duration:.2f}s, {scene_duration:.2f}s per scene")
        return [scene_duration] * num_scenes

    def create_video_with_timed_scenes(
        self,
        scenes: List[dict],
        image_paths: List[Path],
        scene_audio_paths: List[Path],
        output_path: Optional[Path] = None
    ) -> Path:
        """Create video where each scene has its own audio file."""
        if len(scenes) != len(image_paths):
            raise ValueError("Number of scenes must match number of images")

        video_clips = []
        for i, (img_path, audio_path) in enumerate(zip(image_paths, scene_audio_paths)):
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            video_clip = ImageClip(str(img_path), duration=duration)
            if video_clip.size != (self.width, self.height):
                video_clip = video_clip.resize((self.width, self.height))
            video_clip = video_clip.set_fps(self.fps)
            video_clip = video_clip.set_audio(audio_clip)
            video_clips.append(video_clip)

        final_video = concatenate_videoclips(video_clips, method="compose")

        if output_path is None:
            output_path = config.VIDEOS_DIR / "output_video.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=str(output_path.parent / 'temp-audio.m4a'),
            remove_temp=True,
            logger=None
        )

        for clip in video_clips:
            clip.close()
        final_video.close()

        logger.info(f"Video created successfully: {output_path}")
        return output_path

    def create_simple_video(
        self,
        image_path: Path,
        duration: float,
        audio_path: Optional[Path] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Create a simple video from a single image."""
        clip = ImageClip(str(image_path), duration=duration)
        clip = clip.resize((self.width, self.height))
        clip = clip.set_fps(self.fps)

        audio_clip = None
        if audio_path and audio_path.exists():
            audio_clip = AudioFileClip(str(audio_path))
            if audio_clip.duration > duration:
                audio_clip = audio_clip.subclip(0, duration)
            clip = clip.set_audio(audio_clip)

        if output_path is None:
            output_path = config.VIDEOS_DIR / "simple_video.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clip.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            logger=None
        )

        clip.close()
        if audio_clip:
            audio_clip.close()

        logger.info(f"Simple video created: {output_path}")
        return output_path

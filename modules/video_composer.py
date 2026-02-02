"""
Video Composer Module
Combines images and audio into final video using MoviePy
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, CompositeAudioClip
)

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoComposer:
    """Compose final video from images and audio"""
    
    def __init__(
        self,
        width: int = None,
        height: int = None,
        fps: int = None
    ):
        """
        Initialize video composer
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
        """
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
        scene_durations: Optional[List[float]] = None
    ) -> Path:
        """
        Create video from scene images and audio
        
        Args:
            scenes: List of scene dictionaries
            image_paths: List of paths to scene images
            audio_path: Path to audio file (optional)
            output_path: Path to save output video
            scene_durations: Custom durations for each scene (in seconds)
            
        Returns:
            Path to created video
        """
        if len(scenes) != len(image_paths):
            raise ValueError("Number of scenes must match number of images")
        
        logger.info(f"Creating video from {len(scenes)} scenes")
        
        # Determine scene durations
        if scene_durations is None:
            if audio_path and audio_path.exists():
                # Calculate durations based on audio
                scene_durations = self._calculate_scene_durations_from_audio(
                    audio_path,
                    len(scenes)
                )
            else:
                # Use durations from scene data or default
                scene_durations = [
                    scene.get("duration", config.DEFAULT_SCENE_DURATION)
                    for scene in scenes
                ]
        
        # Create video clips from images
        video_clips = []
        for i, (img_path, duration) in enumerate(zip(image_paths, scene_durations)):
            logger.info(f"Creating clip {i+1}/{len(image_paths)}: duration={duration:.2f}s")
            
            clip = ImageClip(str(img_path), duration=duration)
            
            # Resize if needed
            if clip.size != (self.width, self.height):
                clip = clip.resize((self.width, self.height))
            
            clip = clip.set_fps(self.fps)
            video_clips.append(clip)
        
        # Concatenate all clips
        logger.info("Concatenating video clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        # Add audio if provided
        if audio_path and audio_path.exists():
            logger.info(f"Adding audio from: {audio_path}")
            audio_clip = AudioFileClip(str(audio_path))
            
            # Trim or loop audio to match video duration
            if audio_clip.duration < final_video.duration:
                logger.warning(f"Audio ({audio_clip.duration:.2f}s) shorter than video ({final_video.duration:.2f}s)")
            elif audio_clip.duration > final_video.duration:
                logger.info(f"Trimming audio to match video duration")
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
            logger=None  # Suppress moviepy's verbose output
        )
        
        # Close clips to free memory
        for clip in video_clips:
            clip.close()
        final_video.close()
        if audio_path and audio_path.exists():
            audio_clip.close()
        
        logger.info(f"Video created successfully: {output_path}")
        return output_path
    
    def _calculate_scene_durations_from_audio(
        self,
        audio_path: Path,
        num_scenes: int
    ) -> List[float]:
        """
        Calculate scene durations based on total audio duration
        
        Args:
            audio_path: Path to audio file
            num_scenes: Number of scenes
            
        Returns:
            List of durations for each scene
        """
        audio_clip = AudioFileClip(str(audio_path))
        total_duration = audio_clip.duration
        audio_clip.close()
        
        # Divide equally among scenes
        scene_duration = total_duration / num_scenes
        
        logger.info(f"Audio duration: {total_duration:.2f}s, {num_scenes} scenes, {scene_duration:.2f}s per scene")
        
        return [scene_duration] * num_scenes
    
    def create_video_with_timed_scenes(
        self,
        scenes: List[dict],
        image_paths: List[Path],
        scene_audio_paths: List[Path],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create video where each scene has its own audio file
        (Each scene duration is determined by its audio length)
        
        Args:
            scenes: List of scene dictionaries
            image_paths: List of paths to scene images
            scene_audio_paths: List of paths to scene audio files
            output_path: Path to save output video
            
        Returns:
            Path to created video
        """
        if len(scenes) != len(image_paths) != len(scene_audio_paths):
            raise ValueError("Number of scenes, images, and audio files must match")
        
        logger.info(f"Creating video with individual scene audio timing")
        
        # Create clips with audio-based timing
        video_clips = []
        for i, (img_path, audio_path) in enumerate(zip(image_paths, scene_audio_paths)):
            # Get audio duration
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            
            logger.info(f"Scene {i+1}: duration={duration:.2f}s")
            
            # Create video clip
            video_clip = ImageClip(str(img_path), duration=duration)
            
            # Resize if needed
            if video_clip.size != (self.width, self.height):
                video_clip = video_clip.resize((self.width, self.height))
            
            video_clip = video_clip.set_fps(self.fps)
            video_clip = video_clip.set_audio(audio_clip)
            
            video_clips.append(video_clip)
        
        # Concatenate all clips
        logger.info("Concatenating video clips...")
        final_video = concatenate_videoclips(video_clips, method="compose")
        
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
        
        # Close clips
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
        """
        Create a simple video from a single image
        
        Args:
            image_path: Path to image
            duration: Video duration in seconds
            audio_path: Optional audio file
            output_path: Path to save video
            
        Returns:
            Path to created video
        """
        logger.info(f"Creating simple video from {image_path}")
        
        # Create clip
        clip = ImageClip(str(image_path), duration=duration)
        clip = clip.resize((self.width, self.height))
        clip = clip.set_fps(self.fps)
        
        # Add audio if provided
        if audio_path and audio_path.exists():
            audio_clip = AudioFileClip(str(audio_path))
            if audio_clip.duration > duration:
                audio_clip = audio_clip.subclip(0, duration)
            clip = clip.set_audio(audio_clip)
        
        # Set output path
        if output_path is None:
            output_path = config.VIDEOS_DIR / "simple_video.mp4"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write video
        clip.write_videofile(
            str(output_path),
            fps=self.fps,
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        
        clip.close()
        if audio_path and audio_path.exists():
            audio_clip.close()
        
        logger.info(f"Simple video created: {output_path}")
        return output_path


def main():
    """Test the video composer"""
    from PIL import Image
    
    # Create test images
    test_dir = config.IMAGES_DIR / "test"
    test_dir.mkdir(exist_ok=True)
    
    for i in range(3):
        img = Image.new('RGB', (1920, 1080), color=(50 + i*50, 100, 150 - i*30))
        img.save(test_dir / f"test_scene_{i+1}.png")
    
    # Test scenes
    scenes = [
        {"scene_number": 1, "duration": 3},
        {"scene_number": 2, "duration": 3},
        {"scene_number": 3, "duration": 3},
    ]
    
    image_paths = [test_dir / f"test_scene_{i+1}.png" for i in range(3)]
    
    # Create video
    composer = VideoComposer()
    output = composer.create_video_from_scenes(
        scenes,
        image_paths,
        output_path=config.VIDEOS_DIR / "test_video.mp4"
    )
    
    print(f"Test video created: {output}")


if __name__ == "__main__":
    main()

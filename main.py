#!/usr/bin/env python3
"""
Text-to-Video Generator - Main CLI Interface
Generates videos from text prompts using AI
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import config
from modules.story_generator import StoryGenerator
from modules.image_generator import ImageGenerator
from modules.tts_generator import TTSGenerator
from modules.subtitle_overlay import SubtitleOverlay
from modules.video_composer import VideoComposer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextToVideoGenerator:
    """Main orchestrator for text-to-video generation"""
    
    def __init__(
        self,
        tts_provider: str = "gemini",
        image_model: str = "imagen",
        num_scenes: int = 6,
        add_subtitles: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the text-to-video generator
        
        Args:
            tts_provider: TTS provider ('gemini' or 'elevenlabs')
            image_model: Image generation model ('imagen', 'imagen-ultra', 'nano-banana')
            num_scenes: Number of scenes to generate
            add_subtitles: Whether to add subtitles to images
            seed: Random seed for reproducible generation
        """
        self.tts_provider = tts_provider
        self.image_model = image_model
        self.num_scenes = num_scenes
        self.add_subtitles = add_subtitles
        self.seed = seed
        
        logger.info("Initializing Text-to-Video Generator...")
        logger.info(f"TTS Provider: {tts_provider}")
        logger.info(f"Image Model: {image_model}")
        logger.info(f"Number of scenes: {num_scenes}")
        logger.info(f"Add subtitles: {add_subtitles}")
        
        # Initialize modules
        try:
            self.story_gen = StoryGenerator()
            self.image_gen = ImageGenerator(backend=image_model)
            self.tts_gen = TTSGenerator(provider=tts_provider)
            self.subtitle_overlay = SubtitleOverlay()
            self.video_composer = VideoComposer()
            logger.info("All modules initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise
    
    def generate(self, prompt: str, output_name: Optional[str] = None) -> Path:
        """
        Generate video from text prompt
        
        Args:
            prompt: User's text prompt
            output_name: Optional custom name for output files
            
        Returns:
            Path to generated video
        """
        logger.info("="*80)
        logger.info("Starting Text-to-Video Generation")
        logger.info(f"Prompt: {prompt}")
        logger.info("="*80)
        
        # Generate output name if not provided
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"video_{timestamp}"
        
        try:
            # Step 1: Generate story
            print("\n📖 Step 1/6: Generating story...")
            story = self._generate_story(prompt)
            
            # Step 2: Generate images
            print("\n🎨 Step 2/6: Generating images...")
            image_paths = self._generate_images(story, output_name)
            
            # Step 3: Add subtitles (optional)
            if self.add_subtitles:
                print("\n📝 Step 3/6: Adding subtitles to images...")
                image_paths = self._add_subtitles(image_paths, story, output_name)
            else:
                print("\n📝 Step 3/6: Skipping subtitles...")
            
            # Step 4: Generate audio
            print("\n🎤 Step 4/6: Generating narration audio...")
            audio_path = self._generate_audio(story, output_name)
            
            # Step 5: Compose video
            print("\n🎬 Step 5/6: Composing final video...")
            video_path = self._compose_video(story, image_paths, audio_path, output_name)
            
            # Step 6: Done!
            print("\n✅ Step 6/6: Video generation complete!")
            print(f"\n{'='*80}")
            print(f"🎉 Video successfully created!")
            print(f"📁 Output location: {video_path}")
            print(f"{'='*80}\n")
            
            return video_path
            
        except Exception as e:
            logger.error(f"Error during video generation: {e}", exc_info=True)
            raise
    
    def _generate_story(self, prompt: str) -> dict:
        """Generate story from prompt"""
        story = self.story_gen.generate_story(prompt, num_scenes=self.num_scenes)
        
        # Save story
        story_path = self.story_gen.save_story(story)
        print(f"   ✓ Story saved to: {story_path}")
        print(f"   ✓ Title: {story.get('title', 'Untitled')}")
        print(f"   ✓ Scenes: {len(story['scenes'])}")
        
        return story
    
    def _generate_images(self, story: dict, output_name: str) -> list:
        """Generate images for each scene"""
        scenes = story["scenes"]
        image_paths = []
        
        with tqdm(total=len(scenes), desc="   Generating images") as pbar:
            for i, scene in enumerate(scenes):
                current_seed = self.seed + i if self.seed is not None else None
                
                # Generate image
                img_filename = f"{output_name}_scene_{scene['scene_number']:02d}.png"
                img_path = config.IMAGES_DIR / img_filename
                
                self.image_gen.generate_image_from_scene(
                    scene,
                    img_path,
                    seed=current_seed
                )
                
                image_paths.append(img_path)
                pbar.update(1)
        
        print(f"   ✓ Generated {len(image_paths)} images")
        return image_paths
    
    def _add_subtitles(self, image_paths: list, story: dict, output_name: str) -> list:
        """Add subtitles to images"""
        scenes = story["scenes"]
        subtitled_paths = []
        
        with tqdm(total=len(image_paths), desc="   Adding subtitles") as pbar:
            for img_path, scene in zip(image_paths, scenes):
                # Create output path for subtitled image
                subtitled_filename = f"{output_name}_subtitled_{scene['scene_number']:02d}.png"
                subtitled_path = config.IMAGES_DIR / subtitled_filename
                
                # Add subtitle
                self.subtitle_overlay.process_image_with_subtitle(
                    img_path,
                    subtitled_path,
                    scene.get('narration', '')
                )
                
                subtitled_paths.append(subtitled_path)
                pbar.update(1)
        
        print(f"   ✓ Added subtitles to {len(subtitled_paths)} images")
        return subtitled_paths
    
    def _generate_audio(self, story: dict, output_name: str) -> Path:
        """Generate narration audio"""
        # Combine all narrations
        full_narration = " ".join([scene.get('narration', '') for scene in story['scenes']])
        
        # Generate audio
        audio_filename = f"{output_name}_narration.mp3"
        audio_path = config.AUDIO_DIR / audio_filename
        
        print(f"   Generating speech ({len(full_narration)} characters)...")
        self.tts_gen.generate_speech(full_narration, audio_path)
        
        # Get duration
        duration = self.tts_gen.get_audio_duration(audio_path)
        print(f"   ✓ Audio generated: {duration:.2f} seconds")
        
        return audio_path
    
    def _compose_video(
        self,
        story: dict,
        image_paths: list,
        audio_path: Path,
        output_name: str
    ) -> Path:
        """Compose final video"""
        scenes = story["scenes"]
        
        # Create output path
        video_filename = f"{output_name}_final.mp4"
        video_path = config.VIDEOS_DIR / video_filename
        
        print(f"   Composing video with {len(image_paths)} scenes...")
        self.video_composer.create_video_from_scenes(
            scenes,
            image_paths,
            audio_path,
            video_path
        )
        
        print(f"   ✓ Video composed successfully")
        return video_path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Generate videos from text prompts using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "A brave astronaut explores a mysterious alien planet"
  python main.py "우주 탐험가의 모험" --scenes 8 --image-model huggingface
  python main.py "호랑이형님의 모험" --image-model huggingface --scenes 3
  python main.py "A day in the life of a robot" --no-subtitles --seed 42
        """
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the video content"
    )
    
    parser.add_argument(
        "--scenes",
        type=int,
        default=6,
        help="Number of scenes to generate (default: 6)"
    )
    
    parser.add_argument(
        "--tts",
        type=str,
        choices=["gemini", "elevenlabs"],
        default="gemini",
        help="TTS provider to use (default: gemini)"
    )
    
    parser.add_argument(
        "--image-model",
        type=str,
        choices=["huggingface", "flux", "imagen", "imagen-ultra", "nano-banana"],
        default="huggingface",
        help="Image backend (default: huggingface). flux=FLUX.1-dev (local, most accurate), huggingface=Free & Fast, imagen/nano-banana=Gemini API (paid)"
    )
    
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Disable subtitle overlay on images"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name for output files (default: auto-generated)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check API keys
    if not config.GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please set your API key in the .env file")
        print("See .env.example for reference")
        sys.exit(1)
    
    if args.tts == "elevenlabs" and not config.ELEVENLABS_API_KEY:
        print("❌ Error: ELEVENLABS_API_KEY not found in environment variables")
        print("Please set your API key in the .env file or use --tts gemini")
        sys.exit(1)
    
    try:
        # Create generator
        generator = TextToVideoGenerator(
            tts_provider=args.tts,
            image_model=args.image_model,
            num_scenes=args.scenes,
            add_subtitles=not args.no_subtitles,
            seed=args.seed
        )
        
        # Generate video
        video_path = generator.generate(args.prompt, args.output_name)
        
        print(f"\n✅ Success! Your video is ready at:")
        print(f"   {video_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

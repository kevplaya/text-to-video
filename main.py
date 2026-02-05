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
from modules.interactive_ui import interactive_setup, confirm_step, display_story, display_images_info
from modules.image_editor import ImageEditor

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
        tts_language: str = "ko",
        voice_id: Optional[str] = None,
        image_model: str = "imagen",
        image_style: Optional[str] = None,
        resize_mode: str = "padding",
        num_scenes: int = 6,
        add_subtitles: bool = True,
        seed: Optional[int] = None,
        interactive: bool = False,
        project_name: Optional[str] = None
    ):
        """
        Initialize the text-to-video generator
        
        Args:
            tts_provider: TTS provider ('gemini' or 'elevenlabs')
            tts_language: Language code for TTS and story narration (ko, en, ja, zh-CN, etc.)
            voice_id: Voice ID for ElevenLabs TTS
            image_model: Image generation model ('imagen', 'imagen-ultra', 'nano-banana')
            image_style: Custom style guide for image generation
            resize_mode: Image resize mode ('padding', 'crop', 'stretch')
            num_scenes: Number of scenes to generate
            add_subtitles: Whether to add subtitles to images
            seed: Random seed for reproducible generation
            interactive: Enable interactive mode with step confirmations
            project_name: Custom project folder name
        """
        self.tts_provider = tts_provider
        self.tts_language = tts_language
        self.voice_id = voice_id
        self.image_model = image_model
        self.image_style = image_style
        self.resize_mode = resize_mode
        self.num_scenes = num_scenes
        self.add_subtitles = add_subtitles
        self.seed = seed
        self.interactive = interactive
        
        # Setup project folder
        if project_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"video_{timestamp}"
        
        if config.USE_PROJECT_FOLDERS:
            self.project_paths = config.create_project_folder(project_name)
            logger.info(f"Project folder: {self.project_paths['project']}")
        else:
            self.project_paths = None
        
        logger.info("Initializing Text-to-Video Generator...")
        logger.info(f"TTS Provider: {tts_provider}")
        logger.info(f"TTS Language: {tts_language}")
        if voice_id:
            logger.info(f"Voice ID: {voice_id}")
        logger.info(f"Image Model: {image_model}")
        logger.info(f"Number of scenes: {num_scenes}")
        logger.info(f"Add subtitles: {add_subtitles}")
        
        # Initialize modules
        try:
            self.story_gen = StoryGenerator()
            self.image_gen = ImageGenerator(backend=image_model, resize_mode=resize_mode)
            self.tts_gen = TTSGenerator(
                provider=tts_provider,
                language=tts_language,
                voice_id=voice_id
            )
            self.subtitle_overlay = SubtitleOverlay()
            self.video_composer = VideoComposer()
            self.image_editor = ImageEditor(self.image_gen)
            logger.info("All modules initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise
    
    def generate(self, prompt: str, output_name: Optional[str] = None) -> Path:
        """
        Generate video from text prompt
        
        Args:
            prompt: User's text prompt
            output_name: Optional custom name for output files (ignored if using project folders)
            
        Returns:
            Path to generated video
        """
        logger.info("="*80)
        logger.info("Starting Text-to-Video Generation")
        logger.info(f"Prompt: {prompt}")
        logger.info("="*80)
        
        try:
            # Step 1: Generate story
            print("\n📖 Step 1/6: Generating story...")
            story = self._generate_story_interactive(prompt)
            
            # Step 2: Generate images
            print("\n🎨 Step 2/6: Generating images...")
            image_paths = self._generate_images_interactive(story)
            
            # Step 3: Add subtitles (optional)
            if self.add_subtitles:
                print("\n📝 Step 3/6: Adding subtitles to images...")
                image_paths = self._add_subtitles_interactive(image_paths, story)
            else:
                print("\n📝 Step 3/6: Skipping subtitles...")
            
            # Step 4: Generate audio
            print("\n🎤 Step 4/6: Generating narration audio...")
            audio_path = self._generate_audio_interactive(story)
            
            # Step 5: Compose video
            print("\n🎬 Step 5/6: Composing final video...")
            video_path = self._compose_video_interactive(story, image_paths, audio_path)
            
            # Step 6: Done!
            print("\n✅ Step 6/6: Video generation complete!")
            print(f"\n{'='*80}")
            print(f"🎉 Video successfully created!")
            print(f"📁 Output location: {video_path}")
            if self.project_paths:
                print(f"📁 Project folder: {self.project_paths['project']}")
            print(f"{'='*80}\n")
            
            return video_path
            
        except Exception as e:
            logger.error(f"Error during video generation: {e}", exc_info=True)
            raise
    
    def _generate_story_interactive(self, prompt: str) -> dict:
        """Generate story with optional interactive confirmation"""
        while True:
            story = self.story_gen.generate_story(
                prompt,
                num_scenes=self.num_scenes,
                language=self.tts_language
            )
            
            # Save story
            if self.project_paths:
                story_path = self.project_paths['story']
                import json
                with open(story_path, 'w', encoding='utf-8') as f:
                    json.dump(story, f, indent=2, ensure_ascii=False)
            else:
                story_path = self.story_gen.save_story(story)
            
            print(f"   ✓ Story saved to: {story_path}")
            print(f"   ✓ Title: {story.get('title', 'Untitled')}")
            print(f"   ✓ Scenes: {len(story['scenes'])}")
            
            # Interactive mode: show story and confirm
            if self.interactive:
                display_story(story)
                action = confirm_step("스토리 생성", f"제목: {story.get('title')}")
                
                if action == 'continue':
                    return story
                elif action == 'retry':
                    print("\n🔄 스토리 재생성 중...")
                    continue
                elif action == 'quit':
                    print("\n❌ 프로그램을 종료합니다.")
                    sys.exit(0)
            else:
                return story
    
    def _generate_images_interactive(self, story: dict) -> list:
        """Generate images with optional interactive review"""
        while True:
            scenes = story["scenes"]
            image_paths = []
            
            # Determine output directory
            if self.project_paths:
                output_dir = self.project_paths['images']
            else:
                output_dir = config.IMAGES_DIR
            
            with tqdm(total=len(scenes), desc="   Generating images") as pbar:
                for i, scene in enumerate(scenes):
                    current_seed = self.seed + i if self.seed is not None else None
                    
                    # Generate image
                    img_filename = f"scene_{scene['scene_number']:02d}.png"
                    img_path = output_dir / img_filename
                    
                    self.image_gen.generate_image_from_scene(
                        scene,
                        img_path,
                        seed=current_seed,
                        style_guide=self.image_style
                    )
                    
                    image_paths.append(img_path)
                    pbar.update(1)
            
            print(f"   ✓ Generated {len(image_paths)} images")
            
            # Interactive mode: review and modify images
            if self.interactive:
                display_images_info(image_paths)
                image_paths = self.image_editor.interactive_image_review(
                    scenes,
                    image_paths,
                    style_guide=self.image_style,
                    seed=self.seed
                )
                
                action = confirm_step("이미지 생성", f"{len(image_paths)}개 이미지 준비 완료")
                
                if action == 'continue':
                    return image_paths
                elif action == 'retry':
                    print("\n🔄 모든 이미지 재생성 중...")
                    continue
                elif action == 'quit':
                    print("\n❌ 프로그램을 종료합니다.")
                    sys.exit(0)
            else:
                return image_paths
    
    def _add_subtitles_interactive(self, image_paths: list, story: dict) -> list:
        """Add subtitles with optional interactive confirmation"""
        while True:
            scenes = story["scenes"]
            subtitled_paths = []
            
            # Determine output directory
            if self.project_paths:
                output_dir = self.project_paths['images_subtitled']
            else:
                output_dir = config.IMAGES_DIR
            
            with tqdm(total=len(image_paths), desc="   Adding subtitles") as pbar:
                for img_path, scene in zip(image_paths, scenes):
                    # Create output path for subtitled image
                    subtitled_filename = f"scene_{scene['scene_number']:02d}.png"
                    subtitled_path = output_dir / subtitled_filename
                    
                    # Add subtitle
                    self.subtitle_overlay.process_image_with_subtitle(
                        img_path,
                        subtitled_path,
                        scene.get('narration', '')
                    )
                    
                    subtitled_paths.append(subtitled_path)
                    pbar.update(1)
            
            print(f"   ✓ Added subtitles to {len(subtitled_paths)} images")
            
            # Interactive mode: confirm
            if self.interactive:
                action = confirm_step("자막 추가", f"{len(subtitled_paths)}개 이미지에 자막 추가 완료")
                
                if action == 'continue':
                    return subtitled_paths
                elif action == 'retry':
                    print("\n🔄 자막 재생성 중...")
                    continue
                elif action == 'quit':
                    print("\n❌ 프로그램을 종료합니다.")
                    sys.exit(0)
            else:
                return subtitled_paths
    
    def _generate_audio_interactive(self, story: dict) -> Path:
        """Generate narration audio with optional interactive confirmation"""
        while True:
            # Combine all narrations
            full_narration = " ".join([scene.get('narration', '') for scene in story['scenes']])
            
            # Determine output path
            if self.project_paths:
                audio_path = self.project_paths['audio'] / 'narration.mp3'
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = config.AUDIO_DIR / f"video_{timestamp}_narration.mp3"
            
            print(f"   Generating speech ({len(full_narration)} characters)...")
            self.tts_gen.generate_speech(full_narration, audio_path)
            
            # Get duration
            duration = self.tts_gen.get_audio_duration(audio_path)
            print(f"   ✓ Audio generated: {duration:.2f} seconds")
            print(f"   ✓ Location: {audio_path}")
            
            # Interactive mode: confirm
            if self.interactive:
                print(f"\n🎧 오디오 파일을 확인하세요: {audio_path}")
                action = confirm_step("음성 생성", f"{duration:.2f}초 오디오 생성 완료")
                
                if action == 'continue':
                    return audio_path
                elif action == 'retry':
                    print("\n🔄 음성 재생성 중...")
                    continue
                elif action == 'quit':
                    print("\n❌ 프로그램을 종료합니다.")
                    sys.exit(0)
            else:
                return audio_path
    
    def _compose_video_interactive(
        self,
        story: dict,
        image_paths: list,
        audio_path: Path
    ) -> Path:
        """Compose final video with optional interactive confirmation"""
        scenes = story["scenes"]
        
        # Interactive mode: final confirmation before composing
        if self.interactive:
            print(f"\n{'='*80}")
            print("🎬 최종 비디오 합성 준비")
            print(f"{'='*80}")
            print(f"씬 개수: {len(image_paths)}")
            print(f"오디오: {audio_path}")
            print(f"비디오 크기: {config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}")
            
            if not input("\n비디오를 합성하시겠습니까? (Y/n): ").strip().lower() in ['n', 'no']:
                pass
            else:
                print("\n❌ 비디오 합성을 취소했습니다.")
                sys.exit(0)
        
        # Determine output path
        if self.project_paths:
            video_path = self.project_paths['video']
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = config.VIDEOS_DIR / f"video_{timestamp}_final.mp4"
        
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
  # Interactive mode
  python main.py --interactive
  python main.py -i
  
  # CLI mode
  python main.py "A brave astronaut explores a mysterious alien planet" --tts-lang en
  python main.py "우주 탐험가의 모험" --scenes 8 --tts-lang ko
  python main.py "호랑이형님의 모험" --scenes 3 --image-style "traditional Korean art, vibrant colors"
  python main.py "Space Story" --tts elevenlabs --voice-id YOUR_VOICE_ID --tts-lang en --image-style "sci-fi, neon lights, cyberpunk"
        """
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        nargs='?',
        default=None,
        help="Text prompt describing the video content (not needed in interactive mode)"
    )
    
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode with step-by-step confirmations"
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
        "--tts-lang",
        type=str,
        default="ko",
        help="TTS language code (default: ko). Examples: ko, en, ja, zh-CN, es, fr"
    )
    
    parser.add_argument(
        "--voice-id",
        type=str,
        default=None,
        help="Voice ID for ElevenLabs TTS (optional). Example: 21m00Tcm4TlvDq8ikWAM"
    )
    
    parser.add_argument(
        "--image-model",
        type=str,
        choices=["local", "huggingface", "flux", "imagen", "nano-banana"],
        default="local",
        help="Image backend (default: local). local=Local Stable Diffusion, flux=FLUX.1-dev (most accurate), huggingface=HF API (deprecated), imagen/nano-banana=Gemini API (paid)"
    )
    
    parser.add_argument(
        "--image-style",
        type=str,
        default=None,
        help='Image style guide (default: "cinematic, highly detailed, professional quality"). Example: "anime style, vibrant colors, Studio Ghibli"'
    )
    
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=["padding", "crop", "stretch"],
        default="padding",
        help="Image resize mode (default: padding). padding=비율유지+여백, crop=비율유지+자르기, stretch=비율무시"
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
        help="Custom name for project folder (default: video_TIMESTAMP)"
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
    
    # Interactive mode
    if args.interactive:
        # Get all options interactively
        options = interactive_setup()
        
        # Override args with interactive choices
        args.prompt = options['prompt']
        args.scenes = options['num_scenes']
        args.image_model = options['image_model']
        args.tts_lang = options['tts_lang']
        args.image_style = options.get('image_style')
        args.resize_mode = options['resize_mode']
        args.no_subtitles = not options['add_subtitles']
        args.seed = options.get('seed')
    
    # Check that we have a prompt
    if not args.prompt:
        print("❌ Error: Prompt is required")
        print("Use --interactive mode or provide a prompt")
        parser.print_help()
        sys.exit(1)
    
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
            tts_language=args.tts_lang,
            voice_id=args.voice_id,
            image_model=args.image_model,
            image_style=args.image_style,
            resize_mode=args.resize_mode,
            num_scenes=args.scenes,
            add_subtitles=not args.no_subtitles,
            seed=args.seed,
            interactive=args.interactive,
            project_name=args.output_name
        )
        
        # Generate video
        video_path = generator.generate(args.prompt)
        
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

#!/usr/bin/env python3
"""
Text-to-Video Generator - Main CLI Interface
Generates videos from text prompts using AI

Enhanced 8-step pipeline:
1. Character DNA extraction (optional, GPT-4o)
2. Domain prompt compilation (optional)
3. Story generation (6-beat narrative arc, DNA, location lock)
4. Image generation (structured prompts, character preservation)
5. Video AI generation (optional — Grok/Veo/Kling)
6. Subtitle overlay
7. Audio generation (Gemini TTS, ElevenLabs timing, gTTS fallback)
8. Video composition (mixed scenes, timed subtitles)
"""
import argparse
import json
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
    """Main orchestrator for text-to-video generation — 8-step pipeline."""

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
        project_name: Optional[str] = None,
        reference_image: Optional[str] = None,
        video_ai: str = "none",
        domain: str = "general",
        location_lock: Optional[str] = None,
        aspect_ratio: str = "9:16",
        # Domain-specific params
        role_situation: Optional[str] = None,
        location: Optional[str] = None,
        tone: Optional[str] = None,
        menu: Optional[str] = None,
        cooking_mood: Optional[str] = None,
        style_concept: Optional[str] = None,
        character_position: Optional[str] = None,
        mood: Optional[str] = None,
    ):
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
        self.reference_image = reference_image
        self.video_ai = video_ai
        self.domain = domain
        self.location_lock = location_lock
        self.aspect_ratio = aspect_ratio

        # Domain params
        self.role_situation = role_situation
        self.location = location
        self.tone = tone
        self.menu = menu
        self.cooking_mood = cooking_mood
        self.style_concept = style_concept
        self.character_position = character_position
        self.mood = mood

        # State
        self.style_profile = None
        self.reference_image_bytes = None
        self.timed_subtitles = None

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
        logger.info(f"Image Model: {image_model}, TTS: {tts_provider}, Video AI: {video_ai}")

        # Initialize modules
        try:
            self.story_gen = StoryGenerator()
            self.image_gen = ImageGenerator(backend=image_model, resize_mode=resize_mode)
            self.tts_gen = TTSGenerator(
                provider=tts_provider, language=tts_language, voice_id=voice_id
            )
            self.subtitle_overlay = SubtitleOverlay()
            self.video_composer = VideoComposer()
            self.image_editor = ImageEditor(self.image_gen)

            # Video AI generator (optional)
            self.video_ai_gen = None
            if video_ai and video_ai != "none":
                from modules.video_ai_generator import VideoAIGenerator
                self.video_ai_gen = VideoAIGenerator(backend=video_ai)

            logger.info("All modules initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise

    def generate(self, prompt: str, output_name: Optional[str] = None) -> Path:
        """Generate video from text prompt — 8-step pipeline."""
        logger.info("="*80)
        logger.info(f"Starting Text-to-Video Generation: {prompt}")
        logger.info("="*80)

        try:
            # Step 1: Character DNA extraction (optional)
            if self.reference_image:
                print("\n🧬 Step 1/8: Extracting character DNA...")
                self._extract_character_dna()
            else:
                print("\n🧬 Step 1/8: No reference image, skipping DNA extraction")

            # Step 2: Domain prompt compilation (optional)
            if self.domain != "general":
                print(f"\n📋 Step 2/8: Compiling {self.domain} domain prompt...")
                prompt = self._compile_domain_prompt(prompt)
                print(f"   ✓ Domain prompt compiled")
            else:
                print("\n📋 Step 2/8: General mode, skipping domain compilation")

            # Step 3: Generate story
            print("\n📖 Step 3/8: Generating story...")
            story = self._generate_story_interactive(prompt)

            # Step 4: Generate images
            print("\n🎨 Step 4/8: Generating images...")
            image_paths = self._generate_images_interactive(story)

            # Step 5: Video AI generation (optional)
            scene_video_paths = None
            if self.video_ai_gen:
                print(f"\n🎬 Step 5/8: Generating AI videos ({self.video_ai})...")
                scene_video_paths = self._generate_videos_ai(story, image_paths)
            else:
                print("\n🎬 Step 5/8: No Video AI, skipping")

            # Step 6: Add subtitles (optional)
            if self.add_subtitles:
                print("\n📝 Step 6/8: Adding subtitles to images...")
                image_paths = self._add_subtitles_interactive(image_paths, story)
            else:
                print("\n📝 Step 6/8: Skipping subtitles...")

            # Step 7: Generate audio
            print("\n🎤 Step 7/8: Generating narration audio...")
            audio_path = self._generate_audio_interactive(story)

            # Step 8: Compose video
            print("\n🎬 Step 8/8: Composing final video...")
            video_path = self._compose_video_interactive(
                story, image_paths, audio_path, scene_video_paths
            )

            print(f"\n{'='*80}")
            print(f"✅ Video successfully created!")
            print(f"📁 Output: {video_path}")
            if self.project_paths:
                print(f"📁 Project: {self.project_paths['project']}")
            print(f"{'='*80}\n")

            return video_path

        except Exception as e:
            logger.error(f"Error during video generation: {e}", exc_info=True)
            raise

    def _extract_character_dna(self):
        """Extract character DNA from reference image using GPT-4o."""
        ref_path = Path(self.reference_image)
        if not ref_path.exists():
            logger.warning(f"Reference image not found: {ref_path}")
            return

        self.reference_image_bytes = ref_path.read_bytes()

        # Determine MIME type
        suffix = ref_path.suffix.lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime_type = mime_map.get(suffix, 'image/jpeg')

        # Try GPT-4o DNA extraction
        try:
            from modules.gpt_client import extract_character_style_profile
            self.style_profile = extract_character_style_profile(
                image_bytes=self.reference_image_bytes,
                image_mime_type=mime_type,
            )
            logger.info(f"Character DNA extracted: image_type={self.style_profile.get('image_type')}")
            print(f"   ✓ DNA extracted: {self.style_profile.get('image_type', 'unknown')} type")

            # Save DNA profile
            if self.project_paths:
                dna_path = self.project_paths['character_dna'] / 'profile.json'
                with open(dna_path, 'w', encoding='utf-8') as f:
                    json.dump(self.style_profile, f, indent=2, ensure_ascii=False)
                print(f"   ✓ DNA saved to: {dna_path}")

        except Exception as e:
            logger.warning(f"GPT-4o DNA extraction failed ({e}), continuing without DNA")
            print(f"   ⚠ DNA extraction failed: {e}")
            print(f"   → Continuing without character DNA")

    def _compile_domain_prompt(self, original_prompt: str) -> str:
        """Compile domain-specific prompt."""
        try:
            if self.domain == "animal-shorts":
                from modules.animal_shorts_prompt import compile_animal_shorts_prompt
                return compile_animal_shorts_prompt(
                    role_situation=self.role_situation or original_prompt,
                    location=self.location or "거리",
                    tone=self.tone or "현실적",
                    aspect_ratio=self.aspect_ratio,
                    num_scenes=self.num_scenes,
                )
            elif self.domain == "cooking":
                from modules.cooking_prompt import compile_cooking_prompt
                return compile_cooking_prompt(
                    menu=self.menu or original_prompt,
                    cooking_mood=self.cooking_mood or "따뜻한",
                    location=self.location or "주방",
                    aspect_ratio=self.aspect_ratio,
                    num_scenes=self.num_scenes,
                )
            elif self.domain == "lookbook":
                from modules.lookbook_prompt import compile_lookbook_prompt
                return compile_lookbook_prompt(
                    style_concept=self.style_concept or original_prompt,
                    mood=self.mood or "세련된",
                    location=self.location or "거리",
                    aspect_ratio=self.aspect_ratio,
                    num_scenes=self.num_scenes,
                )
            elif self.domain == "situation-drama":
                from modules.situation_drama_prompt import compile_situation_drama_prompt
                return compile_situation_drama_prompt(
                    situation=original_prompt,
                    location=self.location or "카페",
                    character_position=self.character_position or "주인공",
                    mood=self.mood or "잔잔한",
                    aspect_ratio=self.aspect_ratio,
                    num_scenes=self.num_scenes,
                )
        except Exception as e:
            logger.warning(f"Domain prompt compilation failed ({e}), using original prompt")
            print(f"   ⚠ Domain compilation failed: {e}")

        return original_prompt

    def _generate_story_interactive(self, prompt: str) -> dict:
        while True:
            story = self.story_gen.generate_story(
                prompt,
                num_scenes=self.num_scenes,
                language=self.tts_language,
                reference_image_bytes=self.reference_image_bytes,
                location_lock=self.location_lock,
                character_dna_profile=self.style_profile,
            )

            if self.project_paths:
                story_path = self.project_paths['story']
                with open(story_path, 'w', encoding='utf-8') as f:
                    json.dump(story, f, indent=2, ensure_ascii=False)
            else:
                story_path = self.story_gen.save_story(story)

            print(f"   ✓ Story saved to: {story_path}")
            print(f"   ✓ Title: {story.get('title', 'Untitled')}")
            print(f"   ✓ Scenes: {len(story['scenes'])}")

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
        while True:
            scenes = story["scenes"]
            image_paths = []

            if self.project_paths:
                output_dir = self.project_paths['images']
            else:
                output_dir = config.IMAGES_DIR

            with tqdm(total=len(scenes), desc="   Generating images") as pbar:
                for i, scene in enumerate(scenes):
                    current_seed = self.seed + i if self.seed is not None else None
                    img_filename = f"scene_{scene['scene_number']:02d}.png"
                    img_path = output_dir / img_filename

                    self.image_gen.generate_image_from_scene(
                        scene,
                        img_path,
                        seed=current_seed,
                        style_guide=self.image_style,
                        style_profile=self.style_profile,
                        reference_image_bytes=self.reference_image_bytes,
                        aspect_ratio=self.aspect_ratio,
                    )

                    image_paths.append(img_path)
                    pbar.update(1)

            print(f"   ✓ Generated {len(image_paths)} images")

            if self.interactive:
                display_images_info(image_paths)
                image_paths = self.image_editor.interactive_image_review(
                    scenes, image_paths, style_guide=self.image_style, seed=self.seed
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

    def _generate_videos_ai(self, story: dict, image_paths: list) -> list:
        """Generate AI video clips for each scene."""
        scenes = story["scenes"]
        video_paths = []

        if self.project_paths:
            output_dir = self.project_paths['videos_ai']
        else:
            output_dir = config.OUTPUT_DIR / "videos_ai"
            output_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=len(scenes), desc=f"   Generating {self.video_ai} videos") as pbar:
            for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
                video_filename = f"scene_{scene['scene_number']:02d}.mp4"
                video_path = output_dir / video_filename

                result = self.video_ai_gen.generate_scene_video(
                    scene=scene,
                    image_path=img_path,
                    output_path=video_path,
                    style_profile=self.style_profile,
                    aspect_ratio=self.aspect_ratio,
                )
                video_paths.append(result)  # None if failed
                pbar.update(1)

        success_count = sum(1 for p in video_paths if p is not None)
        print(f"   ✓ Generated {success_count}/{len(scenes)} video clips")
        return video_paths

    def _add_subtitles_interactive(self, image_paths: list, story: dict) -> list:
        while True:
            scenes = story["scenes"]
            subtitled_paths = []

            if self.project_paths:
                output_dir = self.project_paths['images_subtitled']
            else:
                output_dir = config.IMAGES_DIR

            with tqdm(total=len(image_paths), desc="   Adding subtitles") as pbar:
                for img_path, scene in zip(image_paths, scenes):
                    subtitled_filename = f"scene_{scene['scene_number']:02d}.png"
                    subtitled_path = output_dir / subtitled_filename
                    self.subtitle_overlay.process_image_with_subtitle(
                        img_path, subtitled_path, scene.get('narration', '')
                    )
                    subtitled_paths.append(subtitled_path)
                    pbar.update(1)

            print(f"   ✓ Added subtitles to {len(subtitled_paths)} images")

            if self.interactive:
                action = confirm_step("자막 추가", f"{len(subtitled_paths)}개 이미지에 자막 추가 완료")
                if action == 'continue':
                    return subtitled_paths
                elif action == 'retry':
                    continue
                elif action == 'quit':
                    sys.exit(0)
            else:
                return subtitled_paths

    def _generate_audio_interactive(self, story: dict) -> Path:
        while True:
            full_narration = " ".join([scene.get('narration', '') for scene in story['scenes']])

            if self.project_paths:
                audio_path = self.project_paths['audio'] / 'narration.mp3'
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_path = config.AUDIO_DIR / f"video_{timestamp}_narration.mp3"

            print(f"   Generating speech ({len(full_narration)} characters)...")
            result = self.tts_gen.generate_speech(full_narration, audio_path)

            # Handle both old (Path) and new (Path, timed_subtitles) return types
            if isinstance(result, tuple):
                audio_path, self.timed_subtitles = result
            else:
                audio_path = result

            duration = self.tts_gen.get_audio_duration(audio_path)
            print(f"   ✓ Audio generated: {duration:.2f} seconds")
            if self.timed_subtitles:
                print(f"   ✓ Timed subtitles: {len(self.timed_subtitles)} segments")

            if self.interactive:
                print(f"\n🎧 오디오 파일: {audio_path}")
                action = confirm_step("음성 생성", f"{duration:.2f}초 오디오 생성 완료")
                if action == 'continue':
                    return audio_path
                elif action == 'retry':
                    continue
                elif action == 'quit':
                    sys.exit(0)
            else:
                return audio_path

    def _compose_video_interactive(
        self,
        story: dict,
        image_paths: list,
        audio_path: Path,
        scene_video_paths: Optional[list] = None,
    ) -> Path:
        scenes = story["scenes"]

        if self.interactive:
            print(f"\n{'='*80}")
            print("🎬 최종 비디오 합성 준비")
            print(f"{'='*80}")
            print(f"씬 개수: {len(image_paths)}")
            print(f"오디오: {audio_path}")
            if scene_video_paths:
                video_count = sum(1 for p in scene_video_paths if p is not None)
                print(f"AI 비디오 클립: {video_count}개")
            if not input("\n비디오를 합성하시겠습니까? (Y/n): ").strip().lower() in ['n', 'no']:
                pass
            else:
                sys.exit(0)

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
            video_path,
            scene_video_paths=scene_video_paths,
            timed_subtitles=self.timed_subtitles,
        )

        print(f"   ✓ Video composed successfully")
        return video_path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate videos from text prompts using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py "프롬프트" --scenes 6 --tts-lang ko
  python main.py --interactive

  # With character DNA (reference image)
  python main.py "테스트" --reference-image ./character.jpg --image-model grok

  # With Video AI
  python main.py "테스트" --video-ai grok --scenes 3

  # Domain modes
  python main.py "전통시장 상인" --domain animal-shorts --location "시장 앞" --tone "현실적"
  python main.py "김치찌개" --domain cooking --location "주방" --cooking-mood "따뜻한"
  python main.py "캐주얼 룩" --domain lookbook --location "거리" --mood "세련된"
  python main.py "퇴사 면담" --domain situation-drama --location "회의실" --mood "잔잔한"

  # Location lock (same background across all scenes)
  python main.py "인터뷰" --location-lock "전통시장 앞"
        """
    )

    parser.add_argument("prompt", type=str, nargs='?', default=None,
                        help="Text prompt (not needed in interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--scenes", type=int, default=6,
                        help="Number of scenes (default: 6)")
    parser.add_argument("--tts", type=str, choices=["gemini", "elevenlabs"], default="gemini",
                        help="TTS provider (default: gemini)")
    parser.add_argument("--tts-lang", type=str, default="ko",
                        help="TTS language code (default: ko)")
    parser.add_argument("--voice-id", type=str, default=None,
                        help="Voice ID for ElevenLabs TTS")
    parser.add_argument("--image-model", type=str,
                        choices=["local", "huggingface", "flux", "imagen", "nano-banana", "grok"],
                        default="local",
                        help="Image backend (default: local)")
    parser.add_argument("--image-style", type=str, default=None,
                        help="Image style guide")
    parser.add_argument("--resize-mode", type=str, choices=["padding", "crop", "stretch"],
                        default="padding", help="Image resize mode (default: padding)")
    parser.add_argument("--no-subtitles", action="store_true",
                        help="Disable subtitle overlay")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Custom project folder name")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    # New options
    parser.add_argument("--reference-image", type=str, default=None,
                        help="Path to reference image for character DNA extraction")
    parser.add_argument("--video-ai", type=str, choices=["none", "grok", "veo", "kling"],
                        default="none", help="Video AI backend (default: none)")
    parser.add_argument("--domain", type=str,
                        choices=["general", "animal-shorts", "cooking", "lookbook", "situation-drama"],
                        default="general", help="Domain mode (default: general)")
    parser.add_argument("--location-lock", type=str, default=None,
                        help="Fix location across all scenes")
    parser.add_argument("--aspect-ratio", type=str, choices=["9:16", "16:9", "1:1"],
                        default="9:16", help="Aspect ratio (default: 9:16)")

    # Domain-specific params
    parser.add_argument("--role-situation", type=str, default=None,
                        help="Role/situation for animal-shorts domain")
    parser.add_argument("--location", type=str, default=None,
                        help="Location for domain modes")
    parser.add_argument("--tone", type=str, default=None,
                        help="Tone for animal-shorts domain")
    parser.add_argument("--menu", type=str, default=None,
                        help="Menu name for cooking domain")
    parser.add_argument("--cooking-mood", type=str, default=None,
                        help="Cooking mood for cooking domain")
    parser.add_argument("--style-concept", type=str, default=None,
                        help="Style concept for lookbook domain")
    parser.add_argument("--character-position", type=str, default=None,
                        help="Character position for situation-drama domain")
    parser.add_argument("--mood", type=str, default=None,
                        help="Mood for lookbook/situation-drama domains")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Interactive mode
    if args.interactive:
        options = interactive_setup()
        args.prompt = options['prompt']
        args.scenes = options['num_scenes']
        args.image_model = options['image_model']
        args.tts_lang = options['tts_lang']
        args.image_style = options.get('image_style')
        args.resize_mode = options['resize_mode']
        args.no_subtitles = not options['add_subtitles']
        args.seed = options.get('seed')
        # New interactive options
        args.reference_image = options.get('reference_image')
        args.video_ai = options.get('video_ai', 'none')
        args.domain = options.get('domain', 'general')
        args.location_lock = options.get('location_lock')
        args.aspect_ratio = options.get('aspect_ratio', '9:16')
        args.role_situation = options.get('role_situation')
        args.location = options.get('location')
        args.tone = options.get('tone')
        args.menu = options.get('menu')
        args.cooking_mood = options.get('cooking_mood')
        args.style_concept = options.get('style_concept')
        args.character_position = options.get('character_position')
        args.mood = options.get('mood')

    if not args.prompt:
        print("❌ Error: Prompt is required")
        parser.print_help()
        sys.exit(1)

    if not config.GOOGLE_API_KEY:
        print("❌ Error: GOOGLE_API_KEY not found")
        print("Set your API key in .env file. See .env.example")
        sys.exit(1)

    if args.tts == "elevenlabs" and not config.ELEVENLABS_API_KEY:
        print("❌ Error: ELEVENLABS_API_KEY not found")
        sys.exit(1)

    try:
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
            project_name=args.output_name,
            reference_image=args.reference_image,
            video_ai=args.video_ai,
            domain=args.domain,
            location_lock=args.location_lock,
            aspect_ratio=args.aspect_ratio,
            role_situation=args.role_situation,
            location=args.location,
            tone=args.tone,
            menu=args.menu,
            cooking_mood=args.cooking_mood,
            style_concept=args.style_concept,
            character_position=args.character_position,
            mood=args.mood,
        )

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

"""
TTS (Text-to-Speech) Generator Module
Supports: Gemini TTS, ElevenLabs (with timing), gTTS fallback

Enhanced with:
- Real Gemini TTS (PCM→WAV conversion)
- ElevenLabs timed subtitles support
- gTTS as final fallback
- Per-scene audio generation
"""
import logging
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    AudioFileClip = None

import config

logger = logging.getLogger(__name__)


class TTSGenerator:
    """Text-to-Speech generator with multiple provider support."""

    def __init__(
        self,
        provider: str = None,
        language: str = 'ko',
        voice_id: Optional[str] = None,
        google_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None
    ):
        self.provider = provider or config.TTS_PROVIDER
        self.language = language
        self.voice_id = voice_id
        logger.info(f"TTSGenerator initialized: provider={self.provider}, language={self.language}")

    def generate_speech(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None
    ) -> Tuple[Path, Optional[list]]:
        """
        Generate speech from text.

        Returns:
            Tuple of (path_to_audio, timed_subtitles_or_None)
        """
        logger.info(f"Generating speech with {self.provider}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.provider == "gemini":
            return self._generate_with_gemini(text, output_path)
        elif self.provider == "elevenlabs":
            return self._generate_with_elevenlabs(text, output_path, voice_id)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_with_gemini(self, text: str, output_path: Path) -> Tuple[Path, None]:
        """Generate speech using Gemini TTS, falling back to gTTS."""
        # Try real Gemini TTS first
        try:
            from .gemini_client import generate_speech_bytes
            audio_bytes, mime_type = generate_speech_bytes(
                text=text,
                language=self.language,
            )
            # Determine extension based on mime type
            if "wav" in mime_type:
                output_path = output_path.with_suffix('.wav')
            else:
                output_path = output_path.with_suffix('.mp3')

            with open(output_path, 'wb') as f:
                f.write(audio_bytes)

            logger.info(f"Gemini TTS generated: {output_path}")
            return (output_path, None)

        except Exception as e:
            logger.warning(f"Gemini TTS failed ({e}), falling back to gTTS")

        # Fallback to gTTS
        return self._generate_with_gtts(text, output_path)

    def _generate_with_gtts(self, text: str, output_path: Path) -> Tuple[Path, None]:
        """Generate speech using gTTS as fallback."""
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError("gTTS package not installed. Run: pip install gtts")

        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            output_path = output_path.with_suffix('.mp3')
            tts.save(str(output_path))
            logger.info(f"gTTS generated in {self.language}: {output_path}")
            return (output_path, None)
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            raise

    def _generate_with_elevenlabs(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None
    ) -> Tuple[Path, Optional[list]]:
        """Generate speech using ElevenLabs with timed subtitles."""
        vid = voice_id or self.voice_id

        # Try with-timestamps endpoint for timed subtitles
        try:
            from .elevenlabs_client import generate_elevenlabs_speech_with_timing
            audio_bytes, mime_type, timed_subtitles = generate_elevenlabs_speech_with_timing(
                text=text,
                voice_id=vid,
            )
            output_path = output_path.with_suffix('.mp3')
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"ElevenLabs TTS with timing generated: {output_path}")
            return (output_path, timed_subtitles)

        except Exception as e:
            logger.warning(f"ElevenLabs with-timestamps failed ({e}), trying basic endpoint")

        # Fallback to basic ElevenLabs endpoint
        try:
            from .elevenlabs_client import generate_elevenlabs_speech_bytes
            audio_bytes, mime_type = generate_elevenlabs_speech_bytes(
                text=text,
                voice_id=vid,
            )
            output_path = output_path.with_suffix('.mp3')
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"ElevenLabs TTS generated: {output_path}")
            return (output_path, None)

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            raise

    def generate_scene_audio(
        self,
        scenes: List[dict],
        output_dir: Optional[Path] = None,
        combine: bool = True
    ) -> Tuple[Path, Optional[list]]:
        """
        Generate audio for multiple scenes.

        Returns:
            Tuple of (path, timed_subtitles_or_None)
        """
        output_dir = output_dir or config.AUDIO_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if combine:
            full_text = " ".join([scene.get("narration", "") for scene in scenes])
            output_path = output_dir / "narration_full.mp3"
            return self.generate_speech(full_text, output_path)
        else:
            all_timed = []
            audio_paths = []
            for i, scene in enumerate(scenes):
                scene_number = scene.get("scene_number", i + 1)
                narration = scene.get("narration", "")
                if not narration:
                    continue
                output_path = output_dir / f"narration_scene_{scene_number:02d}.mp3"
                path, timed = self.generate_speech(narration, output_path)
                audio_paths.append(path)
                if timed:
                    all_timed.extend(timed)
            return (output_dir, all_timed if all_timed else None)

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            if AudioFileClip is None:
                raise ImportError("MoviePy not installed")
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            audio_clip.close()
            return duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            raise

"""
TTS (Text-to-Speech) Generator Module
Supports Google Gemini TTS and ElevenLabs
"""
import logging
from pathlib import Path
from typing import Optional, List
import time

try:
    from elevenlabs import generate, set_api_key, voices
except ImportError:
    generate = None
    set_api_key = None
    voices = None

try:
    from moviepy.editor import AudioFileClip
except ImportError:
    AudioFileClip = None

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSGenerator:
    """Text-to-Speech generator with multiple provider support"""
    
    def __init__(
        self,
        provider: str = None,
        google_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None
    ):
        """
        Initialize TTS generator
        
        Args:
            provider: TTS provider ('gemini' or 'elevenlabs'). If None, uses config.TTS_PROVIDER
            google_api_key: Google API key for Gemini TTS
            elevenlabs_api_key: ElevenLabs API key
        """
        self.provider = provider or config.TTS_PROVIDER
        
        if self.provider == "gemini":
            # Using gTTS as fallback for Gemini TTS (no direct Gemini TTS API available)
            logger.info("Initialized Gemini TTS (using gTTS)")
            
        elif self.provider == "elevenlabs":
            self.elevenlabs_api_key = elevenlabs_api_key or config.ELEVENLABS_API_KEY
            if not self.elevenlabs_api_key:
                raise ValueError("ElevenLabs API key required for ElevenLabs TTS")
            if set_api_key is None:
                raise ImportError("elevenlabs package not installed")
            set_api_key(self.elevenlabs_api_key)
            logger.info("Initialized ElevenLabs TTS")
            
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")
    
    def generate_speech(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None
    ) -> Path:
        """
        Generate speech from text
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice_id: Voice identifier (provider-specific)
            
        Returns:
            Path to saved audio file
        """
        logger.info(f"Generating speech with {self.provider}")
        logger.info(f"Text length: {len(text)} characters")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.provider == "gemini":
            return self._generate_with_gemini(text, output_path)
        elif self.provider == "elevenlabs":
            return self._generate_with_elevenlabs(text, output_path, voice_id)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_with_gemini(self, text: str, output_path: Path) -> Path:
        """
        Generate speech using Google Gemini TTS
        Note: As of now, Gemini doesn't have a direct TTS API.
        This is a placeholder implementation using gTTS as fallback.
        """
        logger.warning("Gemini TTS is not directly available. Using gTTS as fallback.")
        
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError("gTTS package not installed. Run: pip install gtts")
        
        try:
            # Use gTTS as fallback
            tts = gTTS(text=text, lang='ko', slow=False)  # Change 'ko' to 'en' for English
            
            # Save directly as mp3
            output_path = output_path.with_suffix('.mp3')
            tts.save(str(output_path))
            
            logger.info(f"Speech generated and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech with gTTS: {e}")
            raise
    
    def _generate_with_elevenlabs(
        self,
        text: str,
        output_path: Path,
        voice_id: Optional[str] = None
    ) -> Path:
        """
        Generate speech using ElevenLabs TTS
        
        Args:
            text: Text to convert
            output_path: Output file path
            voice_id: ElevenLabs voice ID
            
        Returns:
            Path to saved audio file
        """
        voice_id = voice_id or config.ELEVENLABS_VOICE_ID
        
        try:
            # Generate audio
            audio = generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2"
            )
            
            # Save audio
            with open(output_path, "wb") as f:
                f.write(audio)
            
            logger.info(f"Speech generated and saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech with ElevenLabs: {e}")
            raise
    
    def generate_scene_audio(
        self,
        scenes: List[dict],
        output_dir: Optional[Path] = None,
        combine: bool = True
    ) -> Path:
        """
        Generate audio for multiple scenes
        
        Args:
            scenes: List of scene dictionaries with 'narration' key
            output_dir: Directory to save audio files
            combine: If True, combine all scenes into one audio file
            
        Returns:
            Path to audio file (or directory if not combined)
        """
        output_dir = output_dir or config.AUDIO_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if combine:
            # Combine all narrations
            full_text = " ".join([scene.get("narration", "") for scene in scenes])
            output_path = output_dir / "narration_full.mp3"
            return self.generate_speech(full_text, output_path)
        else:
            # Generate separate audio for each scene
            audio_paths = []
            for i, scene in enumerate(scenes):
                scene_number = scene.get("scene_number", i + 1)
                narration = scene.get("narration", "")
                
                if not narration:
                    logger.warning(f"Scene {scene_number} has no narration")
                    continue
                
                output_path = output_dir / f"narration_scene_{scene_number:02d}.mp3"
                self.generate_speech(narration, output_path)
                audio_paths.append(output_path)
            
            return output_dir
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """
        Get duration of audio file in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            if AudioFileClip is None:
                raise ImportError("MoviePy not installed")
            
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            audio_clip.close()
            logger.info(f"Audio duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            raise


def main():
    """Test the TTS generator"""
    generator = TTSGenerator(provider="gemini")
    
    # Test text
    test_text = "안녕하세요. 이것은 텍스트 투 스피치 테스트입니다."
    output_path = config.AUDIO_DIR / "test_speech.mp3"
    
    # Generate speech
    generator.generate_speech(test_text, output_path)
    print(f"Test audio saved to: {output_path}")
    
    # Get duration
    duration = generator.get_audio_duration(output_path)
    print(f"Audio duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()

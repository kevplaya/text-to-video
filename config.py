"""
Configuration management for text-to-video generator
"""
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

# Legacy directories (for backward compatibility)
STORIES_DIR = OUTPUT_DIR / "stories"
IMAGES_DIR = OUTPUT_DIR / "images"
AUDIO_DIR = OUTPUT_DIR / "audio"
VIDEOS_DIR = OUTPUT_DIR / "videos"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Project-based output structure
USE_PROJECT_FOLDERS = True  # Set to False to use legacy structure


def create_project_folder(project_name: str) -> Dict[str, Path]:
    """
    Create a project-specific folder structure
    
    Args:
        project_name: Name for the project folder (timestamp or custom name)
        
    Returns:
        Dictionary with paths for story, images, audio, video
    """
    project_dir = OUTPUT_DIR / project_name
    
    paths = {
        'project': project_dir,
        'story': project_dir / 'story.json',
        'images': project_dir / 'images',
        'images_subtitled': project_dir / 'images' / 'subtitled',
        'audio': project_dir / 'audio',
        'videos_ai': project_dir / 'videos_ai',
        'character_dna': project_dir / 'character_dna',
        'video': project_dir / 'final_video.mp4'
    }
    
    # Create directories
    for key, path in paths.items():
        if key != 'story' and key != 'video':  # Don't create files
            path.mkdir(parents=True, exist_ok=True)
    
    return paths

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")
KLING_API_KEY = os.getenv("KLING_API_KEY")
KLING_API_SECRET = os.getenv("KLING_API_SECRET")
GPT_API_KEY = os.getenv("GPT_API_KEY")

# Set HuggingFace token if available
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Model configurations
GEMINI_MODEL = "gemini-2.5-flash"  # Text generation model

# Story generation models (with fallback)
STORY_MODEL = os.getenv("STORY_MODEL", "gemini-2.5-flash")
STORY_MODEL_FALLBACK = os.getenv("STORY_MODEL_FALLBACK", "gemini-2.0-flash")

# Gemini image/prompt models
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GEMINI_PROMPT_REWRITE_MODEL = os.getenv("GEMINI_PROMPT_REWRITE_MODEL", "gemini-2.0-flash")

# Gemini TTS
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")

# Grok models
GROK_IMAGE_MODEL = os.getenv("GROK_IMAGE_MODEL", "grok-imagine-image")
GROK_VIDEO_MODEL = os.getenv("GROK_VIDEO_MODEL", "grok-imagine-video")

# Veo model
VEO_MODEL = os.getenv("VEO_MODEL", "veo-3.1-fast-generate-preview")

# Kling models
KLING_MODEL = os.getenv("KLING_MODEL", "kling-v3")
KLING_API_BASE = os.getenv("KLING_API_BASE", "https://api.klingai.com")

# ElevenLabs
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3")
ELEVENLABS_DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "iWLjl1zCuqXRkW6494ve")

# Video AI polling
VIDEO_AI_POLL_INTERVAL = int(os.getenv("VIDEO_AI_POLL_INTERVAL", "10"))
VIDEO_AI_MAX_WAIT = int(os.getenv("VIDEO_AI_MAX_WAIT", "600"))

# Image generation backends and models
# Backend options: 'huggingface', 'imagen', 'nano-banana'
DEFAULT_IMAGE_BACKEND = "huggingface"  # Most reliable and free

# Hugging Face models (used when backend='huggingface')
# Using actively maintained models on HF Inference API
HF_SD_MODEL = "runwayml/stable-diffusion-v1-5"  # SD 1.5 - reliable and fast

# FLUX.1-dev model (used when backend='flux' for local inference)
# Requires: pip install torch diffusers
# Model: https://huggingface.co/black-forest-labs/FLUX.1-dev
FLUX_DEV_MODEL = "black-forest-labs/FLUX.1-dev"  # FLUX.1-dev - most accurate, local only

# Gemini models (used when backend='imagen' or 'nano-banana')
NANO_BANANA_MODEL = "gemini-2.5-flash-image"
IMAGEN_MODEL = "imagen-4.0-fast-generate-001"
IMAGEN_ULTRA_MODEL = "imagen-4.0-generate-001"

# Video settings (YouTube Shorts - 9:16 vertical)
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
DEFAULT_SCENE_DURATION = 5  # seconds

# Image generation settings
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920
# 9:16 aspect ratio for YouTube Shorts (vertical video)

# Default image style enhancement
DEFAULT_IMAGE_STYLE = "cinematic, highly detailed, professional quality"

# Image resize mode options
DEFAULT_RESIZE_MODE = "padding"  # Options: "padding", "crop", "stretch"
IMAGE_PADDING_COLOR = (0, 0, 0)  # Black background for letterbox/pillarbox

# Subtitle settings
SUBTITLE_FONT_SIZE = 60
SUBTITLE_COLOR = (255, 255, 255)  # White
SUBTITLE_BG_COLOR = (0, 0, 0, 180)  # Semi-transparent black
SUBTITLE_POSITION = "bottom"  # top, bottom, center
SUBTITLE_PADDING = 20

# TTS settings
TTS_PROVIDER = "gemini"  # gemini or elevenlabs
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Default voice

# API settings (Nano Banana is API-based, no local GPU needed)

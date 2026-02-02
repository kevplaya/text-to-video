"""
Configuration management for text-to-video generator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
STORIES_DIR = OUTPUT_DIR / "stories"
IMAGES_DIR = OUTPUT_DIR / "images"
AUDIO_DIR = OUTPUT_DIR / "audio"
VIDEOS_DIR = OUTPUT_DIR / "videos"

# Create output directories if they don't exist
for directory in [STORIES_DIR, IMAGES_DIR, AUDIO_DIR, VIDEOS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Set HuggingFace token if available
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Model configurations  
GEMINI_MODEL = "gemini-2.5-flash"  # Text generation model

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

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 30
DEFAULT_SCENE_DURATION = 5  # seconds

# Image generation settings (Nano Banana)
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920
# Nano Banana uses aspect ratio instead of guidance/steps
# 9:16 aspect ratio for YouTube Shorts (vertical video)

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

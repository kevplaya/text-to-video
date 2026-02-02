"""
Image Generator Module - Multi-backend support
Supports: Hugging Face Inference API, Gemini Imagen/Nano Banana, FLUX.1-dev (local)
"""
import logging
from pathlib import Path
from typing import Optional, List
import io
import time
import requests
from PIL import Image

try:
    import google.genai as genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import torch
    from diffusers import FluxPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    torch = None
    FluxPipeline = None

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """Multi-backend image generator"""
    
    def __init__(self, backend: str = "huggingface", model_id: Optional[str] = None):
        """
        Initialize the image generator
        
        Args:
            backend: Backend to use ('huggingface', 'flux', 'imagen', 'nano-banana')
            model_id: Optional specific model ID to override defaults
        """
        self.backend = backend
        
        if backend == "huggingface":
            self._init_huggingface(model_id)
        elif backend == "flux":
            self._init_flux_local(model_id)
        elif backend in ["imagen", "imagen-fast", "imagen-ultra"]:
            self._init_gemini_imagen(backend, model_id)
        elif backend == "nano-banana":
            self._init_nano_banana(model_id)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _init_huggingface(self, model_id: Optional[str]):
        """Initialize Hugging Face Inference API"""
        self.hf_token = config.HF_TOKEN
        if not self.hf_token:
            logger.warning("HF_TOKEN not found. Using public Inference API (may be slower)")
        
        # Default to Stable Diffusion 1.5 (reliable on HF Inference API)
        self.model_id = model_id or config.HF_SD_MODEL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        
        logger.info(f"Initialized Hugging Face Inference API: {self.model_id}")
    
    def _init_gemini_imagen(self, backend: str, model_id: Optional[str]):
        """Initialize Gemini Imagen"""
        if not GENAI_AVAILABLE:
            raise ImportError("google.genai not installed")
        
        self.api_key = config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for Gemini")
        
        # Model mapping
        model_map = {
            'imagen': 'imagen-4.0-fast-generate-001',
            'imagen-fast': 'imagen-4.0-fast-generate-001',
            'imagen-ultra': 'imagen-4.0-generate-001',
        }
        
        self.model_id = model_id or model_map.get(backend, 'imagen-4.0-fast-generate-001')
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Gemini Imagen: {self.model_id}")
    
    def _init_nano_banana(self, model_id: Optional[str]):
        """Initialize Nano Banana"""
        if not GENAI_AVAILABLE:
            raise ImportError("google.genai not installed")
        
        self.api_key = config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for Nano Banana")
        
        self.model_id = model_id or "gemini-2.5-flash-image"
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Nano Banana: {self.model_id}")
    
    def _init_flux_local(self, model_id: Optional[str]):
        """Initialize FLUX.1-dev for local inference"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers and torch are required for FLUX.1-dev. "
                "Install with: pip install diffusers torch"
            )
        
        self.model_id = model_id or config.FLUX_DEV_MODEL
        self.hf_token = config.HF_TOKEN
        
        logger.info(f"Loading FLUX.1-dev model: {self.model_id}")
        logger.info("This may take a few minutes on first run...")
        
        try:
            # Load model with aggressive memory optimization
            logger.info("Loading with maximum memory optimization...")
            
            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                token=self.hf_token
            )
            
            # Use sequential CPU offloading for maximum memory savings
            # This is slower but uses much less GPU memory
            self.pipe.enable_sequential_cpu_offload()
            
            # Enable VAE tiling to reduce memory usage
            if hasattr(self.pipe, 'vae'):
                self.pipe.vae.enable_tiling()
            
            # Enable attention slicing for memory efficiency
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing(1)
            
            logger.info("FLUX.1-dev model loaded successfully with memory optimization")
        except Exception as e:
            logger.error(f"Failed to load FLUX.1-dev: {e}")
            raise
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, ugly",
        width: int = None,
        height: int = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate image using selected backend
        
        Args:
            prompt: Text description
            negative_prompt: Things to avoid (HuggingFace only)
            width: Image width
            height: Image height
            seed: Random seed
            
        Returns:
            Generated PIL Image
        """
        width = width or config.IMAGE_WIDTH
        height = height or config.IMAGE_HEIGHT
        
        if self.backend == "huggingface":
            return self._generate_huggingface(prompt, negative_prompt, width, height, seed)
        elif self.backend == "flux":
            return self._generate_flux_local(prompt, negative_prompt, width, height, seed)
        elif self.backend in ["imagen", "imagen-fast", "imagen-ultra"]:
            return self._generate_gemini_imagen(prompt, width, height, seed)
        elif self.backend == "nano-banana":
            return self._generate_nano_banana(prompt, width, height, seed)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _generate_huggingface(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: Optional[int]
    ) -> Image.Image:
        """Generate image using Hugging Face Inference API"""
        logger.info(f"Generating with Hugging Face: {prompt[:50]}...")
        
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
            }
        }
        
        if seed is not None:
            payload["parameters"]["seed"] = seed
        
        # Retry logic for model loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 503:
                    # Model is loading
                    logger.info("Model is loading... waiting 20 seconds")
                    time.sleep(20)
                    continue
                
                response.raise_for_status()
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(response.content))
                logger.info(f"Image generated successfully: {image.size}")
                return image
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(10)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise
    
    def _generate_gemini_imagen(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int]
    ) -> Image.Image:
        """Generate using Gemini Imagen (if available)"""
        logger.info(f"Generating with Imagen: {prompt[:50]}...")
        
        # Determine aspect ratio
        if height > width:
            aspect_ratio = "9:16"
        elif width > height:
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"
        
        try:
            generation_config = types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
            
            # Try imageConfig for Imagen
            image_config = {"aspectRatio": aspect_ratio}
            if seed is not None:
                image_config["seed"] = seed
            
            if hasattr(generation_config, 'image_config'):
                generation_config.image_config = image_config
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=generation_config
            )
            
            # Extract image
            for part in response.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    return image
            
            raise ValueError("No image in response")
            
        except Exception as e:
            logger.error(f"Imagen generation failed: {e}")
            raise
    
    def _generate_nano_banana(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: Optional[int]
    ) -> Image.Image:
        """Generate using Nano Banana"""
        logger.info(f"Generating with Nano Banana: {prompt[:50]}...")
        
        # Determine aspect ratio
        if height > width:
            aspect_ratio = "9:16"
        elif width > height:
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"
        
        try:
            generation_config = types.GenerateContentConfig(
                temperature=0.9,
                response_modalities=["IMAGE"]
            )
            
            # Nano Banana config
            image_config = {"aspect_ratio": aspect_ratio}
            if seed is not None:
                image_config["seed"] = seed
            
            if hasattr(generation_config, 'image_generation_config'):
                generation_config.image_generation_config = image_config
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=generation_config
            )
            
            # Extract image
            for part in response.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    if image.size != (width, height):
                        image = image.resize((width, height), Image.Resampling.LANCZOS)
                    return image
            
            raise ValueError("No image in response")
            
        except Exception as e:
            logger.error(f"Nano Banana generation failed: {e}")
            raise
    
    def _generate_flux_local(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: Optional[int]
    ) -> Image.Image:
        """Generate using FLUX.1-dev locally"""
        logger.info(f"Generating with FLUX.1-dev: {prompt[:50]}...")
        
        try:
            # Create generator with seed if provided
            generator = None
            if seed is not None:
                if torch.cuda.is_available():
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                else:
                    generator = torch.Generator(device="cpu").manual_seed(seed)
            
            # FLUX.1-dev needs dimensions divisible by 16
            # Adjust dimensions
            adjusted_width = (width // 16) * 16
            adjusted_height = (height // 16) * 16
            
            logger.info(f"Adjusted dimensions: {adjusted_width}x{adjusted_height}")
            
            # Generate image with reduced steps for memory
            # FLUX.1-dev supports up to 512 tokens
            result = self.pipe(
                prompt,
                height=adjusted_height,
                width=adjusted_width,
                guidance_scale=3.5,
                num_inference_steps=28,  # Reduced from 50 for faster generation
                max_sequence_length=256,  # Reduced from 512 to save memory
                generator=generator
            )
            
            image = result.images[0]
            
            # Resize if needed (FLUX may generate slightly different sizes)
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            
            logger.info(f"FLUX.1-dev image generated successfully: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"FLUX.1-dev generation failed: {e}")
            raise
    
    def generate_scene_images(
        self,
        scenes: List[dict],
        output_dir: Optional[Path] = None,
        base_filename: str = "scene",
        seed: Optional[int] = None
    ) -> List[Path]:
        """Generate images for multiple scenes"""
        output_dir = output_dir or config.IMAGES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        
        for i, scene in enumerate(scenes):
            scene_number = scene.get("scene_number", i + 1)
            prompt = scene.get("image_prompt", "")
            
            if not prompt:
                logger.warning(f"Scene {scene_number} has no image_prompt, skipping")
                continue
            
            # Add style enhancements (keep concise)
            enhanced_prompt = f"{prompt}, cinematic, highly detailed, professional quality"
            
            # Generate image
            current_seed = seed + i if seed is not None else None
            image = self.generate_image(
                prompt=enhanced_prompt,
                seed=current_seed
            )
            
            # Save image
            filename = f"{base_filename}_{scene_number:02d}.png"
            filepath = output_dir / filename
            image.save(filepath)
            logger.info(f"Saved image to: {filepath}")
            
            image_paths.append(filepath)
        
        return image_paths
    
    def generate_image_from_scene(
        self,
        scene: dict,
        output_path: Path,
        seed: Optional[int] = None
    ) -> Path:
        """Generate a single image from a scene dictionary"""
        prompt = scene.get("image_prompt", "")
        if not prompt:
            raise ValueError("Scene must have 'image_prompt' key")
        
        # Add style enhancements (keep concise)
        enhanced_prompt = f"{prompt}, cinematic, highly detailed, professional quality"
        
        # Generate image
        image = self.generate_image(prompt=enhanced_prompt, seed=seed)
        
        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logger.info(f"Saved image to: {output_path}")
        
        return output_path


def main():
    """Test the image generator"""
    generator = ImageGenerator(backend="huggingface")
    
    # Test with a simple prompt
    test_prompt = "A futuristic spaceship flying through a nebula, vibrant colors, space art"
    image = generator.generate_image(test_prompt, seed=42)
    
    # Save test image
    output_path = config.IMAGES_DIR / "test_image.png"
    image.save(output_path)
    print(f"Test image saved to: {output_path}")


if __name__ == "__main__":
    main()

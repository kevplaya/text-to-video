"""
Subtitle Overlay Module
Adds text subtitles to images using Pillow
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubtitleOverlay:
    """Overlay subtitles on images"""
    
    def __init__(
        self,
        font_size: int = None,
        font_color: Tuple[int, int, int] = None,
        bg_color: Tuple[int, int, int, int] = None,
        position: str = None,
        padding: int = None
    ):
        """
        Initialize subtitle overlay
        
        Args:
            font_size: Font size in pixels
            font_color: RGB color tuple for text
            bg_color: RGBA color tuple for background
            position: 'top', 'center', or 'bottom'
            padding: Padding around text in pixels
        """
        self.font_size = font_size or config.SUBTITLE_FONT_SIZE
        self.font_color = font_color or config.SUBTITLE_COLOR
        self.bg_color = bg_color or config.SUBTITLE_BG_COLOR
        self.position = position or config.SUBTITLE_POSITION
        self.padding = padding or config.SUBTITLE_PADDING
        
        # Try to load a nice font
        self.font = self._load_font()
        
    def _load_font(self) -> ImageFont.FreeTypeFont:
        """
        Load a TrueType font
        
        Returns:
            ImageFont object
        """
        # Try common font locations
        font_options = [
            # macOS fonts
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # Windows fonts (if running on Windows)
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\ariblk.ttf",
        ]
        
        # Try each font
        for font_path in font_options:
            try:
                if Path(font_path).exists():
                    font = ImageFont.truetype(font_path, self.font_size)
                    logger.info(f"Loaded font: {font_path}")
                    return font
            except Exception as e:
                continue
        
        # Fallback to default font
        logger.warning("Could not load TrueType font, using default font")
        return ImageFont.load_default()
    
    def _wrap_text(self, text: str, max_width: int, draw: ImageDraw.Draw) -> List[str]:
        """
        Wrap text to fit within max_width
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            draw: ImageDraw object
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Try adding this word to current line
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=self.font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word is too long, add it anyway
                    lines.append(word)
        
        # Add remaining words
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def add_subtitle(
        self,
        image: Image.Image,
        text: str,
        position: Optional[str] = None
    ) -> Image.Image:
        """
        Add subtitle to an image
        
        Args:
            image: PIL Image object
            text: Subtitle text
            position: Override default position ('top', 'center', 'bottom')
            
        Returns:
            Image with subtitle overlay
        """
        position = position or self.position
        
        # Create a copy to avoid modifying original
        img = image.copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Calculate max text width (90% of image width)
        max_text_width = int(img.width * 0.9)
        
        # Wrap text
        lines = self._wrap_text(text, max_text_width, draw)
        
        # Calculate text dimensions
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=self.font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        
        total_height = sum(line_heights) + (len(lines) - 1) * 10  # 10px spacing between lines
        max_line_width = max(line_widths) if line_widths else 0
        
        # Calculate position
        x = (img.width - max_line_width) // 2
        
        if position == 'top':
            y = self.padding
        elif position == 'center':
            y = (img.height - total_height) // 2
        else:  # bottom
            y = img.height - total_height - self.padding
        
        # Draw background box
        bg_box = [
            x - self.padding,
            y - self.padding,
            x + max_line_width + self.padding,
            y + total_height + self.padding
        ]
        draw.rectangle(bg_box, fill=self.bg_color)
        
        # Draw each line of text
        current_y = y
        for i, line in enumerate(lines):
            # Center each line
            line_width = line_widths[i]
            line_x = (img.width - line_width) // 2
            
            draw.text(
                (line_x, current_y),
                line,
                font=self.font,
                fill=self.font_color
            )
            
            current_y += line_heights[i] + 10  # Move to next line
        
        logger.info(f"Added subtitle: {text[:30]}...")
        return img
    
    def process_image_with_subtitle(
        self,
        input_path: Path,
        output_path: Path,
        subtitle_text: str,
        position: Optional[str] = None
    ) -> Path:
        """
        Load image, add subtitle, and save
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            subtitle_text: Text to overlay
            position: Subtitle position
            
        Returns:
            Path to output image
        """
        # Load image
        image = Image.open(input_path)
        
        # Add subtitle
        result = self.add_subtitle(image, subtitle_text, position)
        
        # Save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        
        logger.info(f"Saved image with subtitle to: {output_path}")
        return output_path
    
    def process_scenes(
        self,
        image_paths: List[Path],
        scenes: List[dict],
        output_dir: Optional[Path] = None,
        base_filename: str = "scene_subtitled"
    ) -> List[Path]:
        """
        Add subtitles to multiple scene images
        
        Args:
            image_paths: List of input image paths
            scenes: List of scene dictionaries with 'narration' key
            output_dir: Directory to save processed images
            base_filename: Base name for output files
            
        Returns:
            List of paths to processed images
        """
        output_dir = output_dir or config.IMAGES_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for i, (img_path, scene) in enumerate(zip(image_paths, scenes)):
            scene_number = scene.get("scene_number", i + 1)
            narration = scene.get("narration", "")
            
            if not narration:
                logger.warning(f"Scene {scene_number} has no narration, copying image as-is")
                # Just copy the image without subtitle
                output_path = output_dir / f"{base_filename}_{scene_number:02d}.png"
                image = Image.open(img_path)
                image.save(output_path)
                output_paths.append(output_path)
                continue
            
            # Process image with subtitle
            output_path = output_dir / f"{base_filename}_{scene_number:02d}.png"
            self.process_image_with_subtitle(
                img_path,
                output_path,
                narration
            )
            output_paths.append(output_path)
        
        return output_paths


def main():
    """Test the subtitle overlay"""
    overlay = SubtitleOverlay()
    
    # Create a test image
    test_image = Image.new('RGB', (1920, 1080), color=(50, 50, 100))
    
    # Add subtitle
    subtitle_text = "This is a test subtitle that demonstrates the subtitle overlay functionality."
    result = overlay.add_subtitle(test_image, subtitle_text)
    
    # Save result
    output_path = config.IMAGES_DIR / "test_subtitle.png"
    result.save(output_path)
    print(f"Test image with subtitle saved to: {output_path}")


if __name__ == "__main__":
    main()

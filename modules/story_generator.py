"""
Story Generator Module using Google Gemini API
Generates structured stories with scenes for video creation
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import google.genai as genai
    NEW_API = True
except ImportError:
    try:
        import google.generativeai as genai
        NEW_API = False
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StoryGenerator:
    """Generates structured stories using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the story generator
        
        Args:
            api_key: Google API key. If None, uses config.GOOGLE_API_KEY
        """
        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env file")
        
        if NEW_API:
            # New google.genai API
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = config.GEMINI_MODEL
        else:
            # Old google.generativeai API
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    def generate_story(self, prompt: str, num_scenes: int = 6, language: str = 'en') -> Dict:
        """
        Generate a structured story based on user prompt
        
        Args:
            prompt: User's story topic/prompt
            num_scenes: Number of scenes to generate (default: 6)
            language: Language code for narration (ko, en, ja, zh-CN, etc.)
            
        Returns:
            Dictionary containing story metadata and scenes
        """
        logger.info(f"Generating story for prompt: {prompt} (language: {language})")
        
        # Language name mapping
        language_names = {
            'ko': 'Korean (한국어)',
            'en': 'English',
            'ja': 'Japanese (日本語)',
            'zh-CN': 'Chinese (中文)',
            'zh': 'Chinese (中文)',
            'es': 'Spanish (Español)',
            'fr': 'French (Français)',
            'de': 'German (Deutsch)',
        }
        
        language_name = language_names.get(language, 'English')
        
        # Create detailed prompt for Gemini with language specification
        system_prompt = f"""Create a compelling short story based on the following topic: "{prompt}"

IMPORTANT LANGUAGE REQUIREMENT:
- Write ALL narration text in {language_name}
- The title should also be in {language_name}
- Image prompts can be in English (for better image generation)
- But narration MUST be in {language_name}

The story should be divided into exactly {num_scenes} scenes. For each scene, provide:
1. A detailed visual description suitable for image generation (describe characters, setting, mood, lighting, style) - This can be in English
2. A narration text that will be spoken in {language_name} (keep it concise, engaging, and suitable for a short video) - MUST be in {language_name}
3. Duration in seconds (distribute time naturally across scenes, total should be 30-60 seconds)

Output the story in the following JSON format:
{{
    "title": "Story Title (in {language_name})",
    "theme": "Brief theme description (in {language_name})",
    "scenes": [
        {{
            "scene_number": 1,
            "image_prompt": "Detailed visual description for image generation (can be in English)",
            "narration": "Spoken narration text in {language_name}",
            "duration": 5
        }},
        ...
    ]
}}

CRITICAL REMINDERS:
- Image prompts: Can be in English for better AI image generation
- Narration text: MUST be in {language_name} - this will be spoken aloud
- Title and theme: Should be in {language_name}

Make the visual descriptions cinematic and detailed. Use artistic styles like "cinematic lighting", "digital art", "highly detailed" in the image prompts.
Ensure the narration in {language_name} flows naturally from scene to scene.
"""
        
        try:
            # Generate content using Gemini
            if NEW_API:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=system_prompt
                )
                response_text = response.text
            else:
                response = self.model.generate_content(system_prompt)
                response_text = response.text
            
            # Extract JSON from response
            story_data = self._parse_response(response_text)
            
            # Add metadata
            story_data["prompt"] = prompt
            story_data["created_at"] = datetime.now().isoformat()
            story_data["num_scenes"] = len(story_data.get("scenes", []))
            
            # Validate story structure
            self._validate_story(story_data)
            
            logger.info(f"Successfully generated story with {story_data['num_scenes']} scenes")
            return story_data
            
        except Exception as e:
            logger.error(f"Error generating story: {e}")
            raise
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse the Gemini response to extract JSON
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed story dictionary
        """
        # Try to find JSON in the response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.error(f"Response text: {response_text}")
            raise ValueError("Could not parse valid JSON from Gemini response")
    
    def _validate_story(self, story_data: Dict) -> None:
        """
        Validate the story structure
        
        Args:
            story_data: Story dictionary to validate
            
        Raises:
            ValueError: If story structure is invalid
        """
        required_fields = ["title", "scenes"]
        for field in required_fields:
            if field not in story_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not story_data["scenes"]:
            raise ValueError("Story must have at least one scene")
        
        # Validate each scene
        for i, scene in enumerate(story_data["scenes"]):
            required_scene_fields = ["scene_number", "image_prompt", "narration", "duration"]
            for field in required_scene_fields:
                if field not in scene:
                    raise ValueError(f"Scene {i+1} missing required field: {field}")
    
    def save_story(self, story_data: Dict, filename: Optional[str] = None) -> Path:
        """
        Save story to JSON file
        
        Args:
            story_data: Story dictionary
            filename: Optional custom filename. If None, generates from title and timestamp
            
        Returns:
            Path to saved file
        """
        if filename is None:
            # Generate filename from title and timestamp
            title_slug = story_data.get("title", "story").lower()
            title_slug = "".join(c if c.isalnum() else "_" for c in title_slug)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{title_slug}_{timestamp}.json"
        
        filepath = config.STORIES_DIR / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Story saved to: {filepath}")
        return filepath
    
    def load_story(self, filepath: Path) -> Dict:
        """
        Load story from JSON file
        
        Args:
            filepath: Path to story JSON file
            
        Returns:
            Story dictionary
        """
        with open(filepath, "r", encoding="utf-8") as f:
            story_data = json.load(f)
        
        self._validate_story(story_data)
        return story_data


def main():
    """Test the story generator"""
    # Example usage
    generator = StoryGenerator()
    
    test_prompt = "A brave astronaut discovers a mysterious alien civilization on a distant planet"
    story = generator.generate_story(test_prompt, num_scenes=5)
    
    print(json.dumps(story, indent=2, ensure_ascii=False))
    
    # Save the story
    filepath = generator.save_story(story)
    print(f"\nStory saved to: {filepath}")


if __name__ == "__main__":
    main()

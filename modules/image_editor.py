"""
Image Editor Module
Allows modification and regeneration of scene images
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image

import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEditor:
    """Edit and regenerate scene images"""
    
    def __init__(self, image_generator):
        """
        Initialize image editor
        
        Args:
            image_generator: ImageGenerator instance for regeneration
        """
        self.image_gen = image_generator
    
    def preview_images(self, image_paths: List[Path]) -> None:
        """
        Display image paths for review
        
        Args:
            image_paths: List of image file paths
        """
        print(f"\n{'='*80}")
        print("🖼️  생성된 이미지")
        print(f"{'='*80}")
        
        for i, img_path in enumerate(image_paths, 1):
            try:
                img = Image.open(img_path)
                print(f"{i}. {img_path.name}")
                print(f"   크기: {img.size[0]}x{img.size[1]}")
                print(f"   경로: {img_path}")
            except Exception as e:
                print(f"{i}. {img_path.name} (오류: {e})")
        
        print(f"{'='*80}\n")
    
    def regenerate_image(
        self,
        scene: Dict,
        output_path: Path,
        new_prompt: Optional[str] = None,
        style_guide: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Path:
        """
        Regenerate a specific scene image
        
        Args:
            scene: Scene dictionary
            output_path: Path to save regenerated image
            new_prompt: Optional new prompt (if None, uses scene's prompt)
            style_guide: Optional style guide
            seed: Optional seed for reproducibility
            
        Returns:
            Path to regenerated image
        """
        if new_prompt:
            # Temporarily modify scene prompt
            original_prompt = scene.get('image_prompt', '')
            scene['image_prompt'] = new_prompt
            logger.info(f"Using new prompt: {new_prompt[:50]}...")
        
        try:
            # Regenerate image
            logger.info(f"Regenerating image for scene {scene.get('scene_number')}")
            self.image_gen.generate_image_from_scene(
                scene,
                output_path,
                seed=seed,
                style_guide=style_guide
            )
            
            # Restore original prompt if modified
            if new_prompt:
                scene['image_prompt'] = original_prompt
            
            return output_path
            
        except Exception as e:
            # Restore original prompt on error
            if new_prompt:
                scene['image_prompt'] = original_prompt
            raise
    
    def replace_image(
        self,
        scene_number: int,
        image_paths: List[Path],
        custom_image_path: Path
    ) -> List[Path]:
        """
        Replace a scene image with a custom image
        
        Args:
            scene_number: Scene number to replace (1-based)
            image_paths: Current list of image paths
            custom_image_path: Path to custom image
            
        Returns:
            Updated list of image paths
        """
        if scene_number < 1 or scene_number > len(image_paths):
            raise ValueError(f"Invalid scene number: {scene_number}")
        
        index = scene_number - 1
        
        # Load and validate custom image
        try:
            custom_img = Image.open(custom_image_path)
            
            # Resize if necessary
            target_size = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
            if custom_img.size != target_size:
                logger.info(f"Resizing custom image from {custom_img.size} to {target_size}")
                custom_img = custom_img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save to scene path
            custom_img.save(image_paths[index])
            logger.info(f"Replaced scene {scene_number} image with custom image")
            
        except Exception as e:
            logger.error(f"Failed to replace image: {e}")
            raise
        
        return image_paths
    
    def interactive_image_review(
        self,
        scenes: List[Dict],
        image_paths: List[Path],
        style_guide: Optional[str] = None,
        seed: Optional[int] = None
    ) -> List[Path]:
        """
        Interactive review and modification of generated images
        
        Args:
            scenes: List of scene dictionaries
            image_paths: List of generated image paths
            style_guide: Style guide used for generation
            seed: Base seed used
            
        Returns:
            Final list of image paths (possibly modified)
        """
        while True:
            self.preview_images(image_paths)
            
            print("이미지 수정 옵션:")
            print("  1. 계속 진행 (수정 없음)")
            print("  2. 특정 이미지 재생성")
            print("  3. 모든 이미지 재생성")
            print("  4. 특정 이미지 교체 (파일에서)")
            
            choice = input("\n선택 (1-4) [1]: ").strip()
            
            if not choice or choice == '1':
                return image_paths
            
            elif choice == '2':
                # Regenerate specific image
                scene_num_str = input("재생성할 씬 번호: ").strip()
                try:
                    scene_num = int(scene_num_str)
                    if 1 <= scene_num <= len(scenes):
                        index = scene_num - 1
                        
                        # Ask for new prompt or use different seed
                        use_new_prompt = input("새 프롬프트를 사용하시겠습니까? (y/N): ").strip().lower()
                        new_prompt = None
                        if use_new_prompt in ['y', 'yes']:
                            new_prompt = input("새 프롬프트 입력: ").strip()
                        
                        # Use different seed
                        new_seed = seed + 1000 if seed else None
                        
                        # Regenerate
                        print(f"\n🔄 씬 {scene_num} 이미지 재생성 중...")
                        self.regenerate_image(
                            scenes[index],
                            image_paths[index],
                            new_prompt=new_prompt,
                            style_guide=style_guide,
                            seed=new_seed
                        )
                        print(f"✅ 씬 {scene_num} 이미지 재생성 완료")
                    else:
                        print(f"❌ 1-{len(scenes)} 사이의 숫자를 입력하세요.")
                except ValueError:
                    print("❌ 유효한 숫자를 입력하세요.")
            
            elif choice == '3':
                # Regenerate all images
                confirm = input("모든 이미지를 재생성하시겠습니까? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print("\n🔄 모든 이미지 재생성 중...")
                    for i, (scene, img_path) in enumerate(zip(scenes, image_paths)):
                        new_seed = seed + 100 + i if seed else None
                        self.regenerate_image(
                            scene,
                            img_path,
                            style_guide=style_guide,
                            seed=new_seed
                        )
                        print(f"✅ 씬 {i+1}/{len(scenes)} 완료")
                    print("✅ 모든 이미지 재생성 완료")
            
            elif choice == '4':
                # Replace with custom image
                scene_num_str = input("교체할 씬 번호: ").strip()
                custom_path_str = input("이미지 파일 경로: ").strip()
                
                try:
                    scene_num = int(scene_num_str)
                    custom_path = Path(custom_path_str)
                    
                    if not custom_path.exists():
                        print(f"❌ 파일을 찾을 수 없습니다: {custom_path}")
                        continue
                    
                    self.replace_image(scene_num, image_paths, custom_path)
                    print(f"✅ 씬 {scene_num} 이미지 교체 완료")
                    
                except ValueError:
                    print("❌ 유효한 씬 번호를 입력하세요.")
                except Exception as e:
                    print(f"❌ 오류: {e}")


def main():
    """Test image editor"""
    print("Image Editor 테스트")
    # 테스트 코드는 실제 구현 시 추가


if __name__ == "__main__":
    main()

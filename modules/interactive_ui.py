"""
Interactive UI Helper Module
Provides functions for interactive user input
"""
import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """
    Get user input with optional default value
    
    Args:
        prompt: Prompt message to display
        default: Default value if user presses enter
        
    Returns:
        User input string
    """
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def get_choice(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """
    Get user choice from a list of options
    
    Args:
        prompt: Question to ask
        choices: List of valid choices
        default: Default choice
        
    Returns:
        Selected choice
    """
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " (기본값)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")
    
    while True:
        if default:
            response = input(f"선택 (1-{len(choices)}) [{choices.index(default)+1}]: ").strip()
            if not response:
                return default
        else:
            response = input(f"선택 (1-{len(choices)}): ").strip()
        
        try:
            index = int(response) - 1
            if 0 <= index < len(choices):
                return choices[index]
            else:
                print(f"❌ 1-{len(choices)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """
    Get yes/no confirmation from user
    
    Args:
        prompt: Question to ask
        default: Default value (True for yes, False for no)
        
    Returns:
        Boolean response
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} ({default_str}): ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes', '예', 'ㅇ']


def confirm_step(step_name: str, details: str = "") -> str:
    """
    Confirm step completion and ask for next action
    
    Args:
        step_name: Name of completed step
        details: Additional details about the step
        
    Returns:
        Action: 'continue', 'retry', or 'quit'
    """
    print(f"\n{'='*80}")
    print(f"✅ {step_name} 완료")
    if details:
        print(f"   {details}")
    print(f"{'='*80}")
    
    print("\n다음 작업:")
    print("  1. 계속 진행")
    print("  2. 이 단계 재시도")
    print("  3. 종료")
    
    while True:
        response = input("선택 (1-3) [1]: ").strip()
        
        if not response or response == '1':
            return 'continue'
        elif response == '2':
            return 'retry'
        elif response == '3':
            return 'quit'
        else:
            print("❌ 1, 2, 또는 3을 입력하세요.")


def display_story(story: Dict[str, Any]) -> None:
    """
    Display story details in a readable format
    
    Args:
        story: Story dictionary
    """
    print(f"\n{'='*80}")
    print(f"📖 생성된 스토리")
    print(f"{'='*80}")
    print(f"제목: {story.get('title', 'Untitled')}")
    if 'theme' in story:
        print(f"테마: {story.get('theme')}")
    print(f"씬 개수: {len(story.get('scenes', []))}")
    print()
    
    for scene in story.get('scenes', []):
        print(f"씬 {scene.get('scene_number')}:")
        print(f"  이미지: {scene.get('image_prompt', '')[:80]}...")
        print(f"  내레이션: {scene.get('narration', '')[:80]}...")
        print(f"  지속 시간: {scene.get('duration')}초")
        print()


def display_images_info(image_paths: List) -> None:
    """
    Display information about generated images
    
    Args:
        image_paths: List of image file paths
    """
    print(f"\n{'='*80}")
    print(f"🎨 생성된 이미지")
    print(f"{'='*80}")
    for i, path in enumerate(image_paths, 1):
        print(f"{i}. {path}")
    print()


def get_image_modification_choice() -> Dict[str, Any]:
    """
    Ask user if they want to modify any images
    
    Returns:
        Dictionary with modification choices
    """
    print("\n이미지를 수정하시겠습니까?")
    print("  1. 아니오, 계속 진행")
    print("  2. 특정 이미지 재생성")
    print("  3. 모든 이미지 재생성")
    
    while True:
        response = input("선택 (1-3) [1]: ").strip()
        
        if not response or response == '1':
            return {'action': 'continue'}
        elif response == '2':
            scene_num = input("재생성할 씬 번호: ").strip()
            try:
                return {'action': 'regenerate_one', 'scene_number': int(scene_num)}
            except ValueError:
                print("❌ 유효한 숫자를 입력하세요.")
        elif response == '3':
            return {'action': 'regenerate_all'}
        else:
            print("❌ 1, 2, 또는 3을 입력하세요.")


def interactive_setup() -> Dict[str, Any]:
    """
    Interactive mode for setting up video generation options
    
    Returns:
        Dictionary with user choices
    """
    print("\n" + "="*80)
    print("🎬 Text-to-Video Generator - 대화형 모드")
    print("="*80)
    
    # Get prompt
    prompt = get_user_input("\n📝 비디오 주제를 입력하세요")
    while not prompt:
        print("❌ 주제는 필수입니다.")
        prompt = get_user_input("📝 비디오 주제를 입력하세요")
    
    # Get number of scenes
    num_scenes = get_choice(
        "\n🎬 씬 개수를 선택하세요",
        ["3", "6", "9"],
        default="3"
    )
    
    # Get image model
    image_models = {
        "Stable Diffusion 로컬 (안정적)": "local",
        "FLUX.1-dev 로컬 (가장 정확, 느림)": "flux",
        "Nano Banana (Gemini API, 유료)": "nano-banana"
    }
    
    model_display = list(image_models.keys())
    selected_display = get_choice(
        "\n🎨 이미지 생성 모델을 선택하세요",
        model_display,
        default=model_display[0]
    )
    image_model = image_models[selected_display]
    
    # Get TTS language
    tts_languages = {
        "한국어": "ko",
        "English": "en",
        "日本語": "ja",
        "中文": "zh-CN"
    }
    
    lang_display = list(tts_languages.keys())
    selected_lang = get_choice(
        "\n🎤 TTS 언어를 선택하세요",
        lang_display,
        default=lang_display[0]
    )
    tts_lang = tts_languages[selected_lang]
    
    # Get image style (optional)
    print("\n🎨 이미지 스타일을 입력하세요 (선택사항)")
    print("   예시: traditional Korean art, anime style, watercolor painting")
    image_style = get_user_input("스타일 (엔터 = 기본값 사용)", default=None)
    if image_style == "":
        image_style = None
    
    # Get resize mode
    resize_modes = {
        "Padding (비율 유지 + 여백 추가)": "padding",
        "Crop (비율 유지 + 넘치는 부분 자르기)": "crop",
        "Stretch (비율 무시, 찌그러질 수 있음)": "stretch"
    }
    
    resize_display = list(resize_modes.keys())
    selected_resize = get_choice(
        "\n📐 이미지 비율 보존 방법을 선택하세요",
        resize_display,
        default=resize_display[0]
    )
    resize_mode = resize_modes[selected_resize]
    
    # Get subtitle option
    add_subtitles = get_yes_no("\n📝 자막을 추가하시겠습니까?", default=True)
    
    # Get seed (optional)
    use_seed = get_yes_no("\n🎲 재현 가능한 결과를 위해 시드를 사용하시겠습니까?", default=False)
    seed = None
    if use_seed:
        seed_input = get_user_input("시드 값 (숫자)", default="42")
        try:
            seed = int(seed_input)
        except ValueError:
            print("❌ 유효하지 않은 시드, 랜덤으로 생성됩니다.")
            seed = None
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 설정 요약")
    print(f"{'='*80}")
    print(f"주제: {prompt}")
    print(f"씬 개수: {num_scenes}")
    print(f"이미지 모델: {selected_display}")
    print(f"이미지 비율: {selected_resize}")
    print(f"TTS 언어: {selected_lang}")
    if image_style:
        print(f"이미지 스타일: {image_style}")
    print(f"자막: {'예' if add_subtitles else '아니오'}")
    if seed:
        print(f"시드: {seed}")
    print(f"{'='*80}")
    
    if not get_yes_no("\n이 설정으로 시작하시겠습니까?", default=True):
        print("❌ 취소되었습니다.")
        exit(0)
    
    return {
        'prompt': prompt,
        'num_scenes': int(num_scenes),
        'image_model': image_model,
        'tts_lang': tts_lang,
        'image_style': image_style,
        'resize_mode': resize_mode,
        'add_subtitles': add_subtitles,
        'seed': seed
    }


def main():
    """Test interactive UI"""
    result = interactive_setup()
    print("\n설정 완료:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

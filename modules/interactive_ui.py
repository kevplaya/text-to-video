"""
Interactive UI Helper Module
Provides functions for interactive user input

Enhanced with:
- Reference image path input
- Video AI backend selection
- Domain mode selection + domain-specific parameters
- Location lock
- Aspect ratio selection
- Enhanced story display (scene_action, scene_background, motion_prompt)
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
                print(f"1-{len(choices)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")


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
    print(f"  {step_name} 완료")
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
            print("1, 2, 또는 3을 입력하세요.")


def display_story(story: Dict[str, Any]) -> None:
    """
    Display story details in a readable format.
    Shows scene_action, scene_background, motion_prompt if available.
    """
    print(f"\n{'='*80}")
    print(f"  생성된 스토리")
    print(f"{'='*80}")
    print(f"제목: {story.get('title', 'Untitled')}")
    if 'theme' in story:
        print(f"테마: {story.get('theme')}")
    print(f"씬 개수: {len(story.get('scenes', []))}")
    print()

    for scene in story.get('scenes', []):
        print(f"--- 씬 {scene.get('scene_number')} ---")

        # New structured fields
        scene_action = scene.get('scene_action', '')
        scene_background = scene.get('scene_background', '')
        motion_prompt = scene.get('motion_prompt', '')

        if scene_action or scene_background:
            if scene_action:
                print(f"  액션: {scene_action[:100]}")
            if scene_background:
                print(f"  배경: {scene_background[:100]}")
            if motion_prompt:
                print(f"  모션: {motion_prompt[:80]}")
        else:
            # Legacy single image_prompt
            print(f"  이미지: {scene.get('image_prompt', '')[:100]}")

        narration = scene.get('narration', '')
        if narration:
            print(f"  내레이션: {narration[:100]}{'...' if len(narration) > 100 else ''}")
        print(f"  지속 시간: {scene.get('duration')}초")
        print()


def display_images_info(image_paths: List) -> None:
    """
    Display information about generated images
    """
    print(f"\n{'='*80}")
    print(f"  생성된 이미지")
    print(f"{'='*80}")
    for i, path in enumerate(image_paths, 1):
        print(f"{i}. {path}")
    print()


def get_image_modification_choice() -> Dict[str, Any]:
    """
    Ask user if they want to modify any images
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
                print("유효한 숫자를 입력하세요.")
        elif response == '3':
            return {'action': 'regenerate_all'}
        else:
            print("1, 2, 또는 3을 입력하세요.")


def _get_domain_params(domain: str) -> Dict[str, Any]:
    """
    Collect domain-specific parameters interactively.

    Returns:
        Dictionary with domain-specific parameter values
    """
    params: Dict[str, Any] = {}

    if domain == "animal-shorts":
        params['role_situation'] = get_user_input(
            "  동물 역할/상황 (예: 전통시장 상인, 카페 사장)")
        params['location'] = get_user_input(
            "  장소 (예: 시장 앞, 카페)", default="")
        params['tone'] = get_user_input(
            "  분위기 (예: 유머, 따뜻한, 감동)", default="유머")

    elif domain == "cooking":
        params['menu'] = get_user_input("  요리 메뉴 (예: 김치볶음밥)")
        params['cooking_mood'] = get_user_input(
            "  요리 분위기 (예: 혼밥, 캠핑)", default="")
        params['location'] = get_user_input(
            "  장소 (예: 야외 캠핑장, 집 주방)", default="")

    elif domain == "lookbook":
        params['style_concept'] = get_user_input(
            "  스타일 컨셉 (예: Y2K, 미니멀, 스트릿)")
        params['mood'] = get_user_input(
            "  분위기 (예: 도시적, 자연, 빈티지)", default="")
        params['location'] = get_user_input(
            "  장소 (예: 서울 거리, 스튜디오)", default="")

    elif domain == "situation-drama":
        params['role_situation'] = get_user_input(
            "  상황 설명 (예: 직장 상사와 갈등)")
        params['location'] = get_user_input(
            "  장소 (예: 사무실, 카페)", default="")
        params['character_position'] = get_user_input(
            "  캐릭터 위치 (예: 중앙, 왼쪽)", default="")
        params['mood'] = get_user_input(
            "  분위기 (예: 긴장감, 유머)", default="")

    # Remove empty strings
    return {k: v for k, v in params.items() if v}


def interactive_setup() -> Dict[str, Any]:
    """
    Interactive mode for setting up video generation options.
    Includes new options: reference image, Video AI, domain, location lock, aspect ratio.

    Returns:
        Dictionary with user choices
    """
    print("\n" + "="*80)
    print("  Text-to-Video Generator - 대화형 모드")
    print("="*80)

    # Get prompt
    prompt = get_user_input("\n비디오 주제를 입력하세요")
    while not prompt:
        print("주제는 필수입니다.")
        prompt = get_user_input("비디오 주제를 입력하세요")

    # Get number of scenes
    num_scenes = get_choice(
        "씬 개수를 선택하세요",
        ["3", "6", "9"],
        default="3"
    )

    # --- Domain mode ---
    domain_modes = {
        "일반 (범용)": "general",
        "동물 쇼츠": "animal-shorts",
        "쿠킹": "cooking",
        "룩북": "lookbook",
        "상황극": "situation-drama",
    }
    domain_display = list(domain_modes.keys())
    selected_domain_display = get_choice(
        "도메인 모드를 선택하세요",
        domain_display,
        default=domain_display[0],
    )
    domain = domain_modes[selected_domain_display]

    # Collect domain-specific params
    domain_params: Dict[str, Any] = {}
    if domain != "general":
        print(f"\n--- {selected_domain_display} 파라미터 ---")
        domain_params = _get_domain_params(domain)

    # --- Image model ---
    image_models = {
        "Stable Diffusion 로컬 (안정적)": "local",
        "FLUX.1-dev 로컬 (가장 정확, 느림)": "flux",
        "Nano Banana (Gemini API, 유료)": "nano-banana",
        "Grok (xAI, 유료)": "grok",
    }
    model_display = list(image_models.keys())
    selected_display = get_choice(
        "이미지 생성 모델을 선택하세요",
        model_display,
        default=model_display[0]
    )
    image_model = image_models[selected_display]

    # --- Reference image (Character DNA) ---
    reference_image = None
    use_ref = get_yes_no("\n참조 이미지로 캐릭터 DNA를 추출하시겠습니까?", default=False)
    if use_ref:
        reference_image = get_user_input("  참조 이미지 경로")
        if reference_image:
            from pathlib import Path
            if not Path(reference_image).exists():
                print(f"  경고: 파일을 찾을 수 없습니다 — {reference_image}")
                reference_image = None

    # --- Video AI ---
    video_ai_options = {
        "사용하지 않음": "none",
        "Grok (xAI)": "grok",
        "Veo (Google)": "veo",
        "Kling (Kuaishou)": "kling",
    }
    video_ai_display = list(video_ai_options.keys())
    selected_video_ai = get_choice(
        "Video AI 백엔드를 선택하세요",
        video_ai_display,
        default=video_ai_display[0],
    )
    video_ai = video_ai_options[selected_video_ai]

    # --- Aspect ratio ---
    aspect_ratios = {
        "9:16 (세로, YouTube Shorts)": "9:16",
        "16:9 (가로, 일반 영상)": "16:9",
        "1:1 (정사각형)": "1:1",
    }
    ar_display = list(aspect_ratios.keys())
    selected_ar = get_choice(
        "화면비를 선택하세요",
        ar_display,
        default=ar_display[0],
    )
    aspect_ratio = aspect_ratios[selected_ar]

    # --- Location lock ---
    location_lock = None
    use_location_lock = get_yes_no("\n장소를 고정하시겠습니까? (모든 씬 동일 배경)", default=False)
    if use_location_lock:
        location_lock = get_user_input("  고정할 장소 (예: 사무실, 카페 내부)")

    # --- TTS language ---
    tts_languages = {
        "한국어": "ko",
        "English": "en",
        "日本語": "ja",
        "中文": "zh-CN"
    }
    lang_display = list(tts_languages.keys())
    selected_lang = get_choice(
        "TTS 언어를 선택하세요",
        lang_display,
        default=lang_display[0]
    )
    tts_lang = tts_languages[selected_lang]

    # --- Image style ---
    print("\n이미지 스타일을 입력하세요 (선택사항)")
    print("   예시: traditional Korean art, anime style, watercolor painting")
    image_style = get_user_input("스타일 (엔터 = 기본값 사용)", default=None)
    if image_style == "":
        image_style = None

    # --- Resize mode ---
    resize_modes = {
        "Padding (비율 유지 + 여백 추가)": "padding",
        "Crop (비율 유지 + 넘치는 부분 자르기)": "crop",
        "Stretch (비율 무시, 찌그러질 수 있음)": "stretch"
    }
    resize_display = list(resize_modes.keys())
    selected_resize = get_choice(
        "이미지 비율 보존 방법을 선택하세요",
        resize_display,
        default=resize_display[0]
    )
    resize_mode = resize_modes[selected_resize]

    # --- Subtitles ---
    add_subtitles = get_yes_no("\n자막을 추가하시겠습니까?", default=True)

    # --- Seed ---
    use_seed = get_yes_no("\n재현 가능한 결과를 위해 시드를 사용하시겠습니까?", default=False)
    seed = None
    if use_seed:
        seed_input = get_user_input("시드 값 (숫자)", default="42")
        try:
            seed = int(seed_input)
        except ValueError:
            print("유효하지 않은 시드, 랜덤으로 생성됩니다.")
            seed = None

    # --- Summary ---
    print(f"\n{'='*80}")
    print("  설정 요약")
    print(f"{'='*80}")
    print(f"주제: {prompt}")
    print(f"씬 개수: {num_scenes}")
    print(f"도메인: {selected_domain_display}")
    if domain_params:
        for k, v in domain_params.items():
            print(f"  {k}: {v}")
    print(f"이미지 모델: {selected_display}")
    if reference_image:
        print(f"참조 이미지: {reference_image}")
    print(f"Video AI: {selected_video_ai}")
    print(f"화면비: {selected_ar}")
    if location_lock:
        print(f"장소 고정: {location_lock}")
    print(f"이미지 비율: {selected_resize}")
    print(f"TTS 언어: {selected_lang}")
    if image_style:
        print(f"이미지 스타일: {image_style}")
    print(f"자막: {'예' if add_subtitles else '아니오'}")
    if seed:
        print(f"시드: {seed}")
    print(f"{'='*80}")

    if not get_yes_no("\n이 설정으로 시작하시겠습니까?", default=True):
        print("취소되었습니다.")
        exit(0)

    result = {
        'prompt': prompt,
        'num_scenes': int(num_scenes),
        'image_model': image_model,
        'tts_lang': tts_lang,
        'image_style': image_style,
        'resize_mode': resize_mode,
        'add_subtitles': add_subtitles,
        'seed': seed,
        'domain': domain,
        'video_ai': video_ai,
        'aspect_ratio': aspect_ratio,
        'reference_image': reference_image,
        'location_lock': location_lock,
    }

    # Merge domain-specific params
    result.update(domain_params)

    return result


def main():
    """Test interactive UI"""
    result = interactive_setup()
    print("\n설정 완료:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

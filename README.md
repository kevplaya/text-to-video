# Text-to-Video Generator

AI 기반으로 텍스트 프롬프트에서 비디오를 생성합니다. Google Gemini로 스토리를 만들고, Stable Diffusion으로 이미지를 생성하며, TTS로 음성을 추가해 최종 비디오를 만듭니다.

## 주요 기능

- Google Gemini API를 사용한 스토리 생성
- Stable Diffusion 기반 이미지 생성 (Hugging Face, FLUX.1-dev, Imagen 지원)
- Google TTS 또는 ElevenLabs TTS로 음성 생성
- 이미지에 자막 추가 (선택)
- MoviePy로 비디오 합성

## 설치

### 필수 요구사항

- Python 3.8 이상
- ffmpeg

### 설치 단계

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
```

`.env` 파일에 API 키를 입력하세요:

```env
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_huggingface_token  # Stable Diffusion 사용시
ELEVENLABS_API_KEY=your_elevenlabs_key  # 선택사항
```

#### API 키 발급

- **Google Gemini**: https://makersuite.google.com/app/apikey
- **HuggingFace**: https://huggingface.co/settings/tokens
- **ElevenLabs** (선택): https://elevenlabs.io/

## 사용법

### 기본 사용

```bash
python main.py "호랑이의 모험"
```

### 옵션

```bash
# 씬 개수 지정
python main.py "로봇의 하루" --scenes 8

# 이미지 모델 선택
python main.py "마법의 숲" --image-model huggingface

# 자막 비활성화
python main.py "바다 속 탐험" --no-subtitles

# 재현 가능한 결과
python main.py "미래 도시" --seed 42

# 출력 파일명 지정
python main.py "드래곤 이야기" --output-name dragon_tale
```

### 이미지 모델 옵션

- `huggingface`: Stable Diffusion (무료, HF Inference API)
- `flux`: FLUX.1-dev (로컬, 고품질, GPU 필요)
- `imagen`: Google Imagen (유료)
- `nano-banana`: Gemini Image (유료)

## 프로젝트 구조

```
text-to-video/
├── main.py                    # CLI 진입점
├── config.py                  # 설정 관리
├── requirements.txt           # 의존성
├── modules/                   # 핵심 모듈
│   ├── story_generator.py     # 스토리 생성
│   ├── image_generator.py     # 이미지 생성
│   ├── tts_generator.py       # TTS
│   ├── subtitle_overlay.py    # 자막 추가
│   └── video_composer.py      # 비디오 합성
└── output/                    # 생성 파일
    ├── stories/
    ├── images/
    ├── audio/
    └── videos/
```

## 설정 수정

`config.py`에서 비디오 설정을 변경할 수 있습니다:

```python
# 유튜브 쇼츠 기준
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
DEFAULT_SCENE_DURATION = 5

IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920

SUBTITLE_FONT_SIZE = 60
SUBTITLE_POSITION = "bottom"
```

## 문제 해결

### GPU 메모리 부족

`config.py`에서 이미지 해상도를 낮추세요:

```python
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
```

### MoviePy 오류

ffmpeg 설치:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

### Stable Diffusion 다운로드 느림

첫 실행 시 모델(약 5GB)이 다운로드됩니다. 한 번만 다운로드되며 이후엔 캐시를 사용합니다.

## 성능

6개 씬 기준 생성 시간 벤치마크:

- **GPU 사용**: 5-7분
- **CPU 사용**: 25-45분

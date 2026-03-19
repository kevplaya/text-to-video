# Text-to-Video Generator

AI 기반으로 텍스트 프롬프트에서 YouTube Shorts 비디오를 생성하는 CLI 도구입니다. 8단계 파이프라인으로 스토리 생성부터 최종 비디오 합성까지 자동화합니다.

## 주요 기능

- **8단계 AI 파이프라인**: DNA 추출 → 도메인 프롬프트 → 스토리 → 이미지 → Video AI → 자막 → TTS → 비디오 합성
- **6-beat 서사 아크**: Hook → Situation → Problem → Reaction → Escalation → Punchline 구조의 스토리 생성
- **캐릭터 DNA 추출**: GPT-4o Vision으로 참조 이미지에서 캐릭터 특성 분석 → 씬 간 일관성 유지
- **다중 이미지 백엔드**: Stable Diffusion, FLUX.1-dev, Imagen, Nano Banana (Gemini), Grok (xAI)
- **Video AI 지원**: Grok, Veo (Google), Kling으로 씬별 AI 비디오 클립 생성
- **도메인별 템플릿**: 동물 쇼츠, 쿠킹, 룩북, 상황극 특화 프롬프트
- **다중 TTS**: Gemini TTS, ElevenLabs (타이밍 자막 지원), gTTS
- **혼합 비디오 합성**: 정적 이미지 + AI 비디오 클립을 씬별로 조합
- **대화형 모드**: 단계별 확인/재시도/종료 선택

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
GOOGLE_API_KEY=your_google_api_key          # 필수
HF_TOKEN=your_huggingface_token             # Stable Diffusion 사용시
ELEVENLABS_API_KEY=your_elevenlabs_key      # 선택사항
GPT_API_KEY=your_openai_api_key             # 캐릭터 DNA 추출시
GROK_API_KEY=your_xai_api_key              # Grok 이미지/비디오 사용시
KLING_API_KEY=your_kling_access_key         # Kling 비디오 사용시
KLING_API_SECRET=your_kling_secret_key      # Kling 비디오 사용시
```

#### API 키 발급

- **Google Gemini**: https://makersuite.google.com/app/apikey
- **HuggingFace**: https://huggingface.co/settings/tokens
- **OpenAI (GPT-4o)**: https://platform.openai.com/api-keys
- **xAI (Grok)**: https://console.x.ai/
- **Kling**: https://platform.klingai.com/
- **ElevenLabs**: https://elevenlabs.io/

## 사용법

### 기본 사용

```bash
python main.py "호랑이의 모험"
python main.py --interactive                # 대화형 모드
```

### CLI 옵션

```bash
# 씬 개수 및 언어 지정
python main.py "로봇의 하루" --scenes 6 --tts-lang ko

# 이미지 모델 선택
python main.py "마법의 숲" --image-model grok
python main.py "마법의 숲" --image-model nano-banana

# 캐릭터 DNA + 참조 이미지
python main.py "캐릭터 이야기" --reference-image ./character.jpg --image-model grok

# Video AI로 씬별 비디오 클립 생성
python main.py "모험 이야기" --video-ai grok --scenes 3
python main.py "모험 이야기" --video-ai veo
python main.py "모험 이야기" --video-ai kling

# 도메인별 특화 모드
python main.py "고양이" --domain animal-shorts --role-situation "전통시장 상인" --location "시장 앞"
python main.py "요리" --domain cooking --menu "김치볶음밥" --cooking-mood "혼밥"
python main.py "패션" --domain lookbook --style-concept "Y2K" --mood "도시적"
python main.py "직장" --domain situation-drama --role-situation "상사와 갈등" --location "사무실"

# 장소 고정 (모든 씬 동일 배경)
python main.py "인터뷰" --location-lock "카페 내부"

# 화면비 선택
python main.py "풍경" --aspect-ratio 16:9

# 기타 옵션
python main.py "미래 도시" --seed 42 --no-subtitles --output-name my_video
python main.py "이야기" --tts elevenlabs --voice-id "voice_id_here"
```

### 이미지 모델

| 모델 | 플래그 | 설명 |
|------|--------|------|
| Stable Diffusion | `--image-model local` | 로컬, GPU 필요 |
| HuggingFace API | `--image-model huggingface` | 무료, 느림 |
| FLUX.1-dev | `--image-model flux` | 최고 품질, GPU 필요 |
| Google Imagen | `--image-model imagen` | 유료, 클라우드 |
| Nano Banana | `--image-model nano-banana` | Gemini 이미지, 유료 |
| Grok | `--image-model grok` | xAI, 유료, 참조 이미지 지원 |

### Video AI 백엔드

| 백엔드 | 플래그 | 설명 |
|--------|--------|------|
| Grok | `--video-ai grok` | xAI, 이미지→비디오 |
| Veo | `--video-ai veo` | Google, Gemini API 기반 |
| Kling | `--video-ai kling` | Kuaishou, JWT 인증 |

## 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    8-Step Pipeline                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 캐릭터 DNA 추출 (선택)                               │
│     └─ GPT-4o Vision → {image_type, character_dna,       │
│        style_lock, negative_prompt}                      │
│                                                          │
│  2. 도메인 프롬프트 컴파일 (선택)                         │
│     └─ animal-shorts / cooking / lookbook /               │
│        situation-drama → Gemini JSON                     │
│                                                          │
│  3. 스토리 생성                                           │
│     └─ Gemini API → 6-beat 서사 아크                     │
│     └─ scene_action + scene_background + motion_prompt   │
│     └─ 모델 폴백 체인                                    │
│                                                          │
│  4. 이미지 생성                                           │
│     └─ 구조화 프롬프트: reference → preservation →        │
│        DNA → style → scene → quality → avoid             │
│     └─ 실패 시 프롬프트 재작성 → 재시도                   │
│                                                          │
│  5. Video AI 생성 (선택)                                  │
│     └─ 씬별 이미지→비디오 (Grok/Veo/Kling)               │
│     └─ video_prompt_builder로 모션 프롬프트 조합           │
│                                                          │
│  6. 자막 오버레이                                         │
│     └─ PIL 텍스트 렌더링                                  │
│                                                          │
│  7. 오디오 생성                                           │
│     └─ Gemini TTS → ElevenLabs (타이밍) → gTTS           │
│                                                          │
│  8. 비디오 합성                                           │
│     └─ MoviePy: ImageClip + VideoFileClip 혼합           │
│     └─ 타이밍 자막 동기화                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 프로젝트 구조

```
text-to-video/
├── main.py                              # CLI 진입점 (8단계 파이프라인)
├── config.py                            # 설정 관리 (API 키, 모델 ID, 경로)
├── requirements.txt                     # 의존성
├── .env.example                         # 환경 변수 템플릿
├── CLAUDE.md                            # Claude Code 가이드
│
├── modules/                             # 핵심 모듈
│   ├── exceptions.py                    # 커스텀 예외 클래스
│   │
│   ├── gemini_client.py                 # Gemini API 클라이언트 (스레드 안전)
│   ├── gpt_client.py                    # GPT-4o Vision (캐릭터 DNA 추출)
│   ├── grok_client.py                   # xAI Grok (이미지 + 비디오)
│   ├── veo_client.py                    # Google Veo (비디오)
│   ├── kling_client.py                  # Kling AI (비디오, JWT 인증)
│   ├── elevenlabs_client.py             # ElevenLabs TTS (타이밍 자막)
│   │
│   ├── story_generator.py              # 스토리 생성 (6-beat 서사 아크)
│   ├── image_generator.py              # 이미지 생성 (다중 백엔드)
│   ├── tts_generator.py                # TTS (Gemini/ElevenLabs/gTTS)
│   ├── subtitle_overlay.py             # 자막 오버레이
│   ├── video_composer.py               # 비디오 합성 (혼합 씬)
│   ├── video_ai_generator.py           # Video AI 오케스트레이터
│   ├── video_prompt_builder.py         # 비디오 프롬프트 조합
│   │
│   ├── prompt_constants.py             # 프롬프트 상수
│   ├── prompt_utils.py                 # 프롬프트 유틸리티
│   │
│   ├── animal_shorts_prompt.py         # 동물 쇼츠 도메인 컴파일러
│   ├── cooking_prompt.py               # 쿠킹 도메인 컴파일러
│   ├── lookbook_prompt.py              # 룩북 도메인 컴파일러
│   ├── situation_drama_prompt.py       # 상황극 도메인 컴파일러
│   │
│   ├── interactive_ui.py              # 대화형 UI
│   └── image_editor.py                # 이미지 편집/리뷰
│
└── output/                             # 생성 파일
    └── video_YYYYMMDD_HHMMSS/
        ├── story.json
        ├── images/
        ├── audio/
        ├── videos_ai/
        └── final_video.mp4
```

## 핵심 설계 원칙

- **Graceful degradation**: 각 기능은 선택적. API 키 미설정 시 해당 기능을 건너뜀. `GOOGLE_API_KEY`만으로 기본 동작 보장.
- **하위 호환**: 기존 CLI 명령어 그대로 동작. `image_prompt` 필드 유지.
- **모델 폴백 체인**: primary 모델 → retryable 에러(503, rate_limit, quota_exceeded) 시 fallback 모델로 자동 전환.
- **프롬프트 재작성**: 이미지 생성 실패 시 Gemini로 프롬프트를 단순화하여 1회 재시도.
- **캐릭터 보존**: scene_action + scene_background 분리로 씬 간 캐릭터 아이덴티티 드리프트 방지.

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

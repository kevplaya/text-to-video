# 🎬 Interactive Mode Guide

## 새로운 기능

### ✅ 1. 대화형 모드
- 옵션을 기억할 필요 없이 질문에 답변하면서 비디오 생성
- 초보자 친화적

### ✅ 2. 단계별 확인
- 각 단계 완료 후 결과 확인
- 만족스럽지 않으면 재시도 가능

### ✅ 3. 이미지 수정
- 생성된 이미지 리뷰
- 특정 이미지만 재생성
- 커스텀 이미지로 교체

### ✅ 4. 프로젝트별 폴더
- 실행마다 독립된 폴더에 모든 결과물 저장
- 깔끔한 정리

### ✅ 5. 한글 자막 지원
- AppleSDGothicNeo 폰트로 한글 완벽 지원

### ✅ 6. 유튜브 쇼츠 크기
- 1080x1920 세로 영상 (9:16 비율)

### ✅ 7. TTS 옵션
- 언어 선택 (한국어, 영어, 일본어 등)
- 목소리 ID 선택 (ElevenLabs)

### ✅ 8. 이미지 스타일
- 커스텀 스타일 가이드 지정

---

## 🚀 사용 방법

### 대화형 모드 (추천!)

```bash
python main.py --interactive
# 또는
python main.py -i
```

**진행 과정**:

1. **주제 입력**
   ```
   📝 비디오 주제를 입력하세요: 호랑이형님의 모험
   ```

2. **씬 개수 선택**
   ```
   🎬 씬 개수를 선택하세요
     1. 3 (기본값)
     2. 6
     3. 9
   선택 (1-3) [1]: 1
   ```

3. **이미지 모델 선택**
   ```
   🎨 이미지 생성 모델을 선택하세요
     1. Stable Diffusion 로컬 (안정적) (기본값)
     2. FLUX.1-dev 로컬 (가장 정확, 느림)
     3. Nano Banana (Gemini API, 유료)
   선택 (1-3) [1]: 1
   ```

4. **TTS 언어 선택**
   ```
   🎤 TTS 언어를 선택하세요
     1. 한국어 (기본값)
     2. English
     3. 日本語
     4. 中文
   선택 (1-4) [1]: 1
   ```

5. **이미지 스타일** (선택사항)
   ```
   🎨 이미지 스타일을 입력하세요 (선택사항)
      예시: traditional Korean art, anime style, watercolor painting
   스타일 (엔터 = 기본값 사용): traditional Korean art, vibrant colors
   ```

6. **자막 여부**
   ```
   📝 자막을 추가하시겠습니까? (Y/n): y
   ```

7. **시드 사용** (선택사항)
   ```
   🎲 재현 가능한 결과를 위해 시드를 사용하시겠습니까? (y/N): n
   ```

8. **설정 확인 및 시작**
   ```
   ================================================================================
   📋 설정 요약
   ================================================================================
   주제: 호랑이형님의 모험
   씬 개수: 3
   이미지 모델: Stable Diffusion 로컬 (안정적)
   TTS 언어: 한국어
   이미지 스타일: traditional Korean art, vibrant colors
   자막: 예
   ================================================================================
   
   이 설정으로 시작하시겠습니까? (Y/n): y
   ```

---

## 단계별 확인

### Step 1: 스토리 생성

```
📖 Step 1/6: Generating story...
   ✓ Story saved to: output/video_XXX/story.json
   ✓ Title: 호랑이형님의 모험
   ✓ Scenes: 3

================================================================================
✅ 스토리 생성 완료
   제목: 호랑이형님의 모험
================================================================================

다음 작업:
  1. 계속 진행
  2. 이 단계 재시도
  3. 종료
선택 (1-3) [1]: 1
```

### Step 2: 이미지 생성 & 수정

```
🎨 Step 2/6: Generating images...
   Generating images: 100%|███████| 3/3 [05:30<00:00]
   ✓ Generated 3 images

================================================================================
🖼️  생성된 이미지
================================================================================
1. scene_01.png
   크기: 1080x1920
   경로: output/video_XXX/images/scene_01.png
...

이미지 수정 옵션:
  1. 계속 진행 (수정 없음)
  2. 특정 이미지 재생성
  3. 모든 이미지 재생성
  4. 특정 이미지 교체 (파일에서)

선택 (1-4) [1]: 2
재생성할 씬 번호: 1
새 프롬프트를 사용하시겠습니까? (y/N): n

🔄 씬 1 이미지 재생성 중...
✅ 씬 1 이미지 재생성 완료
```

### Step 3-5: 나머지 단계

각 단계마다 동일한 확인 프로세스

---

## CLI 모드 (빠른 실행)

```bash
# 기본 실행
python main.py "호랑이형님의 모험" --scenes 3

# 영어 + 커스텀 스타일
python main.py "Space Adventure" --tts-lang en --image-style "sci-fi, cyberpunk"

# 모든 옵션 사용
python main.py "이야기" \
  --scenes 6 \
  --image-model local \
  --image-style "watercolor painting" \
  --tts-lang ko \
  --seed 42 \
  --output-name my_project
```

---

## 출력 폴더 구조

```
output/
└── video_20260205_123456/      # 또는 커스텀 이름
    ├── story.json
    ├── images/
    │   ├── scene_01.png
    │   ├── scene_02.png
    │   ├── scene_03.png
    │   └── subtitled/
    │       ├── scene_01.png
    │       ├── scene_02.png
    │       └── scene_03.png
    ├── audio/
    │   └── narration.mp3
    └── final_video.mp4
```

---

## 🎨 이미지 스타일 예시

| 스타일 | 설명 |
|--------|------|
| `traditional Korean art, vibrant colors` | 한국 전통 미술 |
| `anime style, Studio Ghibli` | 지브리 애니메이션 |
| `watercolor painting, soft colors` | 수채화 |
| `cyberpunk, neon lights, futuristic` | 사이버펑크 |
| `oil painting, impressionist` | 인상파 유화 |
| `photorealistic, 8k, detailed` | 사실적 |

---

## 📝 주의사항

### Interactive 모드
- 터미널에서 직접 실행 필요 (백그라운드 실행 불가)
- 각 단계에서 충분히 확인 후 진행

### 이미지 모델
- **local**: 로컬 Stable Diffusion (무료, 안정적)
- **flux**: FLUX.1-dev (가장 정확하지만 메모리 많이 필요)
- **nano-banana**: Gemini API (유료 플랜 필요)

### 성능
- 로컬 모델은 첫 실행 시 다운로드 필요 (~5GB)
- CPU: 씬당 15-20분
- GPU: 씬당 30-60초

---

**Happy Video Creating! 🎥✨**

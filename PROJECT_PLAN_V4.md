# CLAP2Diffusion V4: Hybrid Hierarchical-Gated Architecture

## 프로젝트 개요

CLAP2Diffusion V4는 텍스트와 오디오를 창의적으로 조합하여 풍부한 이미지를 생성하는 멀티모달 생성 모델 핵심 혁신은 **Hierarchical Audio Decomposition**과 **Gated Cross-Attention**을 결합한 하이브리드 접근법

### 주요 목표
- 텍스트가 제공하는 객체 정보와 오디오가 제공하는 환경/활동 정보를 창의적으로 조합
- 5,000개의 제한된 AudioCaps 데이터로 효과적인 학습
- 텍스트와 오디오가 불일치하는 경우에도 의미있는 이미지 생성
- SonicDiffusion의 정교한 제어 + 우리의 의미적 분해 결합

## 핵심 차별화 포인트

### vs SonicDiffusion
| 항목 | SonicDiffusion | CLAP2Diffusion V4 (Hybrid) |
|------|---------------|-------------------|
| 아키텍처 | Gated Cross-Attention + 4-layer MLP | Hierarchical Decomposition + Gated Attention |
| 정보 융합 | 텍스트와 오디오를 별도 경로로 처리 | 계층적 분해 후 gated fusion |
| 오디오 표현 | Flat 77 tokens | Hierarchical (foreground/background/ambience) |
| 학습 복잡도 | 많은 adapter layers | Targeted hierarchical adapters |
| 역할 분담 | 텍스트와 오디오가 독립적 조건 | 텍스트=객체, 오디오=계층적 환경 |

## 아키텍처 설계 (Hybrid)

### 전체 구조
```
Input Layer:
├── Text Input 
│   └── CLIP Encoder (frozen)
│       └── 77 tokens (텍스트 조건)
│
└── Audio Input
    └── CLAP Encoder (frozen)
        └── Hierarchical Decomposition Module
            ├── Foreground (5 tokens): 주요 소리/활동
            ├── Background (3 tokens): 배경 소리
            └── Ambience (2 tokens): 환경 분위기
            └── Audio Projection (4-layer MLP)
                └── 77 audio tokens

UNet with Gated Hierarchical Attention:
├── Down Blocks
│   └── Gated Audio Attention (foreground-focused)
├── Mid Block
│   └── Gated Audio Attention (full hierarchy)
└── Up Blocks
    └── Gated Audio Attention (ambience-focused)

Generation:
└── Denoised Image Output
```

### 핵심 모듈

#### 1. Hierarchical Audio Decomposition + Projection
```python
class HybridAudioEncoder(nn.Module):
    def __init__(self):
        # Step 1: Hierarchical Decomposition
        self.foreground_proj = nn.Linear(512, 768 * 5)
        self.background_proj = nn.Linear(512, 768 * 3)
        self.ambience_proj = nn.Linear(512, 768 * 2)
        self.hierarchy_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Step 2: 4-layer MLP Projection (like SonicDiffusion)
        self.audio_projector = nn.ModuleList([
            TransformerBlock(768, 8, 4),  # 4 transformer layers
            TransformerBlock(768, 8, 4),
            TransformerBlock(768, 8, 4),
            TransformerBlock(768, 8, 4)
        ])
```

#### 2. Gated Hierarchical Attention
```python
class GatedHierarchicalAttention(nn.Module):
    def __init__(self, hierarchy_level="full"):
        # hierarchy_level: "foreground", "background", "ambience", "full"
        self.gate = nn.Parameter(torch.zeros(1))
        self.hierarchy_level = hierarchy_level
        
    def forward(self, hidden_states, audio_hierarchy):
        # Select hierarchy based on UNet level
        if self.hierarchy_level == "foreground":
            audio_context = audio_hierarchy["foreground"]
        elif self.hierarchy_level == "ambience":
            audio_context = audio_hierarchy["ambience"]
        else:
            audio_context = audio_hierarchy["full"]
        
        # Gated fusion
        gate = torch.sigmoid(self.gate)
        output = hidden_states + gate * cross_attention(hidden_states, audio_context)
```

#### 3. Creative Composition Strategy
오디오와 텍스트의 관계에 따른 처리:
- **일치 (Matching)**: "개" + [개 짖는 소리] → 강화된 개 이미지
- **보완 (Complementary)**: "사람" + [카페 소음] → 카페에 있는 사람
- **창의적 (Creative)**: "고양이" + [비 소리] → 비 오는 날 창가의 고양이
- **모순 (Contradictory)**: "자동차" + [바다 소리] → 해변가의 자동차

## 데이터 전략

### AudioCaps Dataset
- **총 샘플**: 5,002 audio-image-caption triplets
- **특징**: 복잡한 다중 소리 캡션
  - 예: "dogs barking and people talking"
  - 예: "a female is giving a speech as cars pass by"

### 캡션 파싱 전략
```python
def parse_caption_hierarchy(caption):
    # 키워드 기반 계층 추출
    # "while", "as" → 동시 발생 이벤트
    # "and" → 다중 이벤트
    # "with", "in" → 환경 정보
    return {
        'primary': main_event,
        'secondary': background_event,
        'context': environment
    }
```

### 학습 데이터 처리
- **No Curriculum Learning**: 전체 5,000개 동시 사용
- **No Fallback**: 모든 경우에 오디오 활용 (품질이 낮아도 사용)
- **캡션 길이 기반 정렬**: 선택사항으로 간단→복잡 순서

## 학습 계획

### Frozen Components
- CLAP Audio Encoder ✓
- CLIP Text Encoder ✓
- Stable Diffusion UNet (base layers) ✓

### Trainable Parameters (Hybrid)
| 모듈 | 파라미터 수 | 설명 |
|------|------------|------|
| Hierarchical Decomposition | ~1.2M | 3개 projection layers |
| 4-layer Audio Projector | ~2.5M | Transformer blocks (SonicDiffusion style) |
| Gated Adapters | ~800K | UNet adapter layers |
| Gate Parameters | ~100 | 레벨별 gate parameters |
| Hierarchy Weights | 3 | 계층별 중요도 |
| LoRA (optional) | ~500K | UNet 미세조정용 |
| **Total** | **~5.0M** | 여전히 효율적 |

### Training Schedule
```
Stage 1: Projection Alignment (3,000 steps)
- Learning rate: 5e-5
- Batch size: 8
- Objective: CLAP → CLIP space alignment

Stage 2: Weight Optimization (2,000 steps)  
- Learning rate: 1e-5
- Batch size: 8
- Objective: Hierarchy weights 최적화

Total: 5,000 steps (~4시간 예상)
```

## 구현 파일 구조

```
CLAP2Diffusion/
├── PROJECT_PLAN_V4.md (이 파일)
├── src/
│   ├── models/
│   │   ├── v3_backup/              # V3 백업
│   │   ├── hierarchical_audio_v4.py # 핵심 모듈
│   │   └── compositional_fusion_v4.py
│   ├── data/
│   │   └── audiocaps_hierarchical.py
│   └── utils/
│       └── caption_parser.py
├── scripts/
│   ├── train_v4.py
│   └── evaluate_v4.py
└── configs/
    └── v4_config.json
```

## 평가 메트릭

### 정량적 평가
- **FID Score**: 생성 이미지 품질
- **CLIP Score**: 텍스트-이미지 정렬
- **Audio-Visual Alignment**: 오디오-이미지 일치도

### 정성적 평가
| 케이스 | 평가 기준 | 예시 |
|--------|----------|------|
| 일치 | 정보 강화 정도 | "개" + 개소리 → 더 명확한 개 |
| 보완 | 환경 반영 정확도 | "사람" + 카페 → 카페 환경 표현 |
| 창의적 | 조합의 자연스러움 | "고양이" + 비 → 비 오는 날 고양이 |
| 모순 | 타협의 합리성 | "차" + 바다 → 해변 주차장 |

## 예상 결과

### 성공 시나리오
1. **환경 정보 보완**: 텍스트만으로는 부족한 배경/분위기 추가
2. **창의적 장면 구성**: 서로 다른 모달리티 정보의 의미있는 조합
3. **복잡한 장면 이해**: 다중 소리원을 계층적으로 반영

### 한계점
1. **매우 복잡한 오디오**: 3개 이상 소리원은 부분 정보만 활용
2. **추상적 소리**: 구체적 의미가 없는 소리는 분위기로만 반영
3. **완벽한 정렬 어려움**: 창의적 조합 중심, 정확한 매칭은 제한적

## 리스크 관리

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|-----------|
| CLIP/CLAP 공간 불일치 | 높음 | 중간 | Linear projection으로 alignment |
| 5,000개 데이터 부족 | 중간 | 높음 | Frozen models 활용, minimal parameters |
| 텍스트/오디오 충돌 | 높음 | 낮음 | Creative composition 전략 |
| 학습 불안정 | 낮음 | 중간 | Small LR, gradient clipping |
| 복잡한 오디오 처리 실패 | 중간 | 낮음 | Hierarchical decomposition |

## SonicDiffusion에서 통합한 기능들

### 1. Core Features
- **Gated Cross-Attention**: Learnable gate parameters로 오디오 영향력 제어
- **Parallel Audio Path**: audio_context를 별도 파라미터로 전달
- **Selective Block Activation**: use_adapter_list=[False, True, True]
- **Multi-scale Convolutions**: 다양한 temporal scale 캡처
- **Domain-specific Gates**: 도메인별 gate 파라미터 저장/로드
- **f_multiplier**: 실시간 오디오 영향력 조절

### 2. 우리만의 혁신
- **Hierarchical Decomposition**: 오디오를 의미적 계층으로 분해
- **Level-specific Hierarchy**: UNet 레벨별 다른 hierarchy 적용
  - Down: Foreground (객체/액션)
  - Mid: Full hierarchy (전체)
  - Up: Ambience (환경/분위기)
- **Compositional Strategy**: 텍스트-오디오 관계 기반 adaptive fusion

## 개발 일정

| 단계 | 작업 | 예상 시간 |
|------|------|-----------|
| 1 | V3 백업 및 환경 설정 | ✓ 완료 |
| 2 | PROJECT_PLAN_V4.md 작성 | ✓ 완료 |
| 3 | HierarchicalAudioDecomposition 구현 | 2시간 |
| 4 | AudioCaps 데이터로더 구현 | 2시간 |
| 5 | 학습 스크립트 작성 | 2시간 |
| 6 | 초기 학습 및 디버깅 | 4시간 |
| 7 | 평가 스크립트 작성 | 2시간 |
| 8 | 전체 학습 (5k steps) | 4시간 |
| 9 | 평가 및 분석 | 2시간 |
| 10 | 문서화 및 정리 | 2시간 |
| **Total** | | **~20시간** |

## 성공 기준

### 최소 목표
- Baseline (text-only) 대비 FID score 개선
- 환경 정보가 이미지에 반영됨
- 5가지 이상 창의적 조합 예시 생성

### 이상적 목표
- FID < 30
- CLIP score > 0.25
- 사용자 평가에서 70% 이상 "개선됨" 응답
- 논문 발표 가능한 수준의 결과

## 참고 문헌

1. SonicDiffusion: Audio-Driven Image Generation and Editing (2024)
2. AudioCaps: Generating Captions for Audios in The Wild (2019)
3. CLAP: Learning Audio Concepts from Natural Language Supervision (2022)
4. Stable Diffusion: High-Resolution Image Synthesis (2022)

---

*Last Updated: 2024*
*Version: 4.0*
*Author: CLAP2Diffusion Team*
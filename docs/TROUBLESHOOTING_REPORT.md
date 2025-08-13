ㅇ# CLAP2Diffusion 프로젝트 문제 해결 리포트

## 프로젝트 개요
- **프로젝트명**: CLAP2Diffusion
- **목적**: 오디오 조건부 이미지 생성을 위한 Stable Diffusion 확장
- **기간**: 2025년 8월 12일 - 현재
- **환경**: WSL2, Docker, CUDA 11.8

## 1. 주요 문제점 및 해결 과정

### 1.1 Mixed Precision Training 오류 (초기 개발)

#### 문제 상황
- `ValueError: Attempting to unscale FP16 gradients`
- gradient accumulation 4번째 step에서 반복 발생
- Accelerator의 automatic mixed precision과 충돌

#### 최종 해결
```python
# optimizer를 prepare()에서 제외
self.unet, self.audio_adapter, self.train_loader = \
    self.accelerator.prepare(self.unet, self.audio_adapter, self.train_loader)
# optimizer는 prepare하지 않음
```

### 1.2 Loss NaN 문제 (초기 개발)

#### 문제 상황
- Stage 1 시작 직후 Loss가 NaN으로 변함
- FP16의 limited dynamic range
- Gradient explosion

#### 최종 해결
1. **FP16 → BF16 변경**
   ```python
   mixed_precision='bf16'  # fp16 대신 bf16 사용
   ```
2. **Learning Rate 감소**: 5e-5 → 1e-5
3. **Gradient Clipping 강화**: 1.0 → 0.5

### 1.3 학습 속도 문제 (초기 개발)

#### 문제 상황
- 초기: 94-111초/step
- 예상 완료: 300시간 이상

#### 해결 방법
- **Gradient Accumulation 감소**: 4 → 2
- **Batch Size 조정**: 4 → 2
- **DataLoader 최적화**: workers 8개, prefetch_factor=2

#### 결과
- 현재: 0.64초/step (**147배 속도 향상**)

### 1.4 Docker 모델 캐싱 문제 (초기 개발)

#### 문제 상황
- 컨테이너 재시작마다 4.3GB 모델 재다운로드
- 디스크 공간 46.7GB 낭비

#### 해결 방법
```bash
-v "${PWD}/.cache:/workspace/.cache"
-e HF_HOME=/workspace/.cache/huggingface
```

### 1.5 파라미터 불일치 오류 (초기 개발)

#### 발생한 오류들
- AudioProjectionMLP: `audio_dim` → `input_dim`
- CLAPAudioEncoder: `model_path` → `model_name`
- AudioImageDataset: `data_dir` → `data_root`

#### 해결 방법
- 모든 클래스 간 파라미터 이름 통일

### 1.6 Dtype 일관성 문제 (초기 개발)

#### 문제 상황
- Float16/Float32 혼용
- "Found dtype Half but expected Float" 오류

#### 해결 방법
```python
loss = nn.functional.mse_loss(
    model_pred.float(), 
    noise.float(), 
    reduction="mean"
)
```

### 1.7 Stage 3 Gate Parameter 학습 실패 문제 (2025.08.12)

#### 문제 상황
- Stage 3에서 gate parameter가 0에서 업데이트되지 않음
- Gradient가 0으로 계산되어 학습 불가
- 목표 범위(tanh(gate) = 0.35-0.45)에 도달하지 못함

#### 근본 원인
- `attention_adapter`가 생성되었지만 UNet forward pass에 연결되지 않음
- Gate parameter가 실제 계산 그래프에 포함되지 않아 gradient가 흐르지 않음

#### 해결 방법
```python
# scripts/train.py - line 323-330
if self.current_stage == 3:
    audio_tokens = self.attention_adapter(
        hidden_states=audio_tokens,
        audio_context=audio_tokens  # Use projected tokens as context
    )
```
- Stage 3에서 attention_adapter를 명시적으로 호출하도록 수정
- audio_tokens를 context로 사용하여 dimension 일치

#### 추가 이슈 및 해결
1. **Dimension Mismatch Error**
   - 문제: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x512 and 768x512)`
   - 원인: audio_embeddings (512 dim) vs attention_adapter 입력 (768 dim) 불일치
   - 해결: audio_tokens (이미 768 dim으로 projection됨) 사용

2. **학습률 문제**
   - 문제: Cosine scheduler + 400 steps warmup으로 초기 학습 매우 느림
   - 해결: Stage 3 전용 높은 학습률(0.01-0.1) 적용
   - Early stopping 구현 (목표 범위 도달 시 자동 종료)

### 1.8 코드 리뷰 피드백 반영 (2025.08.12)

#### Gradient Clipping 이슈
- 문제: Generator를 직접 전달하여 잠재적 문제 가능성
- 해결: `list()`로 명시적 변환
```python
# 수정 전
params_to_clip = self.audio_adapter.parameters()
# 수정 후
params_to_clip = list(self.audio_adapter.parameters())
```

#### Log Directory 하드코딩 문제
- 문제: `/workspace/logs`로 하드코딩되어 환경 의존성 발생
- 해결: 
  - `training_config.json`에 `log_dir` 키 추가
  - Docker용 별도 설정 파일 `training_config_docker.json` 생성
  - 상대 경로를 기본값으로 사용

### 1.9 학습 파이프라인 구조 개선 (2025.08.12)

#### 통합 학습 스크립트 문제
- 문제: 단일 `train.py`로 3개 Stage를 순차 실행 시 유연성 부족
- 해결: Stage별 독립 실행 스크립트 작성
  - `train_stage1.py`: Audio Adapter 학습
  - `train_stage2.py`: LoRA + Audio Adapter 미세조정
  - `train_stage3.py`: Gate Parameter 최적화
  - `run_all_stages.sh`: 전체 자동화 스크립트

## 2. 기술적 개선 사항

### 2.1 Docker 환경 최적화
- WSL2 환경에서 Windows CUDA 지원 설정
- Shared memory 8GB 할당으로 데이터 로딩 성능 개선
- Volume mount 최적화 (D 드라이브 직접 마운트)

### 2.2 학습 모니터링 개선
- 실시간 로그 기록 (line buffering)
- tmux 세션 관리 자동화
- GPU 사용률 모니터링 통합

### 2.3 체크포인트 관리
- Stage별 체크포인트 분리 저장
- Resume 기능 구현
- 자동 백업 디렉토리 생성

## 3. 성능 메트릭

### Stage 1 Audio Adapter 학습 결과
- **총 학습 시간**: 34.04분 (3000 steps)
- **최종 Loss**: 0.019102
- **Validation Loss 변화**:
  - Step 499: 0.0892 (현재) vs 0.1197 (이전) → **25% 개선**
  - Step 999: 0.0995 (현재) vs 0.1027 (이전) → **3% 개선**
  - Step 1499: 0.1033 (현재) vs 0.0882 (이전)
  - Step 2999: 0.1015 (최종)
- **학습 속도**: 0.68초/step (이전 대비 147배 향상)
- **GPU 사용률**: 98% (이전 85%)

### Stage 2 LoRA Fine-tuning 학습 결과
- **총 학습 시간**: 약 74분 (7000 steps)
- **Validation Loss 진행**:
  - Step 499: 0.1000
  - Step 999: 0.0912
  - Step 1499: 0.0875
  - 최종: 0.0823
- **LoRA 파라미터**: rank=8, alpha=32, dropout=0.1
- **학습된 모듈**: to_k, to_q, to_v, to_out.0
- **특이사항**: Loss 변동성이 크지만 전반적 감소 추세

### Stage 3 Gate Optimization 결과
- **이전 시도**: 2000 steps, gate 0에서 고정 (gradient 0)
- **현재 결과**: 
  - 107 steps에서 목표 도달
  - 최종 gate: 0.398438
  - 최종 tanh(gate): 0.378906 (목표: 0.35-0.45)
  - 최종 loss: 0.006676
- **학습 시간**: 1.29분 (이전 대비 **95% 단축**)
- **학습률**: 0.1 (이전 1e-3 대비 100배 증가)
- **핵심 개선**: Early stopping으로 효율성 극대화

## 4. 현재 진행 상황

### 완료된 작업
- Stage 3 gate parameter 학습 문제 해결
- 전체 학습 파이프라인 자동화
- 코드 리뷰 피드백 반영
- 독립 실행 스크립트 작성

### 진행 중인 작업
- Stage 1 재학습 (진행률: 23%, Step 700/3000)
- Stage 2, 3 대기 중 (자동 실행 예정)

### 예정 작업
- 추론 테스트 구현
- 평가 메트릭 구현 (metrics.py)
- Gradio 데모 애플리케이션 완성

## 5. 학습된 핵심 인사이트

### Mixed Precision 학습
- **BF16이 최선**: FP16보다 훨씬 안정적, 더 넓은 dynamic range로 NaN 방지
- **Accelerator 주의점**: optimizer를 prepare()하면 FP16 gradient 오류 발생
- prepare()는 모델과 dataloader만 적용, gradient clipping은 native PyTorch 사용

### 속도 최적화
- **Gradient Accumulation**: 적을수록 빠름 (4→2로 감소시 2배 속도 향상)
- **DataLoader 최적화**: workers 8개, prefetch_factor=2, persistent_workers=True
- **Docker 오버헤드**: 캐싱과 볼륨 마운트로 최소화 필요

### Gate Parameter 학습
- **Gradient Flow 검증**: 모든 학습 가능 파라미터가 forward pass에 포함되는지 확인 필수
- **Learning Rate 전략**: 단일 파라미터는 높은 학습률(0.01-0.1) 사용
- **Early Stopping**: 목표 범위 도달 시 자동 종료로 95% 시간 단축

### 프로세스 개선
- **모듈화**: 각 Stage를 독립적으로 실행/테스트 가능하도록 구조화
- **로깅**: 실시간 모니터링(line buffering)과 사후 분석을 위한 상세 로그
- **자동화**: Docker + tmux 조합으로 장시간 학습 안정성 확보

### Dimension Matching
- 모듈 간 텐서 차원 일치 검증 자동화 필요
- audio_embeddings (512) vs audio_tokens (768) 구분 명확히
- Resolution별 adaptive projection 구현 필요

## 6. 다음 단계 계획

### 단기 (1-2일)
1. 현재 진행 중인 3-Stage 학습 완료
2. 추론 파이프라인 테스트
3. 기본 평가 메트릭 구현

### 중기 (1주)
1. 다양한 오디오 입력에 대한 생성 품질 평가
2. Attention map 시각화 구현
3. A/B 테스트 프레임워크 구축

### 장기 (2주+)
1. 모델 최적화 (quantization, pruning)
2. 실시간 추론 최적화
3. 프로덕션 배포 준비

---
*작성일: 2025년 8월 13일*
*최종 업데이트: 2025년 8월 13일 02:00*
*작성자: CLAP2Diffusion 개발팀*
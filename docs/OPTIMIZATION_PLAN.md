# CLAP2Diffusion 최적화 계획

## 📋 핵심 원칙
1. **기존 코드 보존** - 작동하는 코드는 건드리지 않기
2. **점진적 적용** - 한 번에 하나씩 테스트
3. **Fallback 준비** - 문제 발생 시 즉시 원복 가능
4. **검증 우선** - 성능 향상보다 안정성 우선

## 🚨 이전 문제점 기반 주의사항

### 1. Dtype 일관성 (최우선)
```python
# ❌ 문제를 일으킨 코드
model_pred = unet(latents, timesteps, encoder_hidden_states)  # Mixed dtype

# ✅ 안전한 접근
model_pred = unet(latents.to(dtype), timesteps, encoder_hidden_states.to(dtype))
loss = F.mse_loss(model_pred.float(), target.float())  # Always float for loss
```

### 2. Dimension Matching
```python
# ❌ 문제: 512 vs 768 차원 불일치
audio_embeddings = clap_encoder(audio)  # 512 dim
attention_input = audio_embeddings  # Error!

# ✅ 해결: 명시적 projection
audio_embeddings = clap_encoder(audio)  # 512 dim
audio_tokens = audio_adapter(audio_embeddings)  # 512 -> 768 dim
attention_input = audio_tokens  # OK
```

### 3. Accelerator 호환성
```python
# ❌ 문제: Accelerator와 optimizer prepare 충돌
optimizer = accelerator.prepare(optimizer)  # FP16 gradient error

# ✅ 해결: optimizer는 prepare하지 않음
model, dataloader = accelerator.prepare(model, dataloader)
# optimizer는 별도 관리
```

## 📝 단계별 최적화 계획

### Phase 1: 안전한 Data Pipeline 개선 (위험도: 낮음)
**목표**: I/O 병목 해결, 학습 속도 20% 향상

#### 1.1 DataLoader 최적화 (기존 dataset.py 유지)
```python
# src/data/dataset_wrapper.py (새 파일)
class DatasetWrapper:
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.cache = {}  # Simple caching
    
    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]
```

#### 1.2 안전한 Augmentation 추가
```python
# src/data/safe_augmentation.py
class SafeAugmentation:
    def __init__(self, enabled=False):  # 기본값 OFF
        self.enabled = enabled
    
    def apply(self, audio, image):
        if not self.enabled:
            return audio, image
        # 간단한 augmentation만
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])  # Horizontal flip only
        return audio, image
```

### Phase 2: 학습 안정성 개선 (위험도: 중간)
**목표**: NaN 방지, 학습 안정화

#### 2.1 Gradient Monitoring
```python
# src/utils/gradient_monitor.py
class GradientMonitor:
    def check_gradients(self, model, step):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 100 or torch.isnan(param.grad).any():
                    print(f"Warning at step {step}: {name} grad_norm={grad_norm}")
                    return False
        return True
```

#### 2.2 Safe Mixed Precision
```python
# BF16 강제 (FP16보다 안정적)
class SafeMixedPrecision:
    def __init__(self):
        self.dtype = torch.bfloat16
        self.scaler = None  # BF16은 scaler 불필요
    
    def forward(self, model, inputs):
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            return model(**inputs)
```

### Phase 3: 메모리 최적화 (위험도: 중간)
**목표**: 메모리 40% 절약, 배치 크기 2배

#### 3.1 Selective Gradient Checkpointing
```python
# 안정적인 레이어만 checkpointing
def setup_gradient_checkpointing(unet):
    # Down blocks만 (안정적)
    for down_block in unet.down_blocks:
        down_block.gradient_checkpointing = True
    # Mid, Up blocks는 그대로 (Stage 3 문제 방지)
```

#### 3.2 Memory-Efficient Optimizer
```python
# 8-bit Adam (bitsandbytes)
import bitsandbytes as bnb

def create_optimizer(params, lr, use_8bit=False):
    if use_8bit:
        # Stage 1,2만 적용 (Stage 3는 제외)
        return bnb.optim.Adam8bit(params, lr=lr)
    else:
        return torch.optim.Adam(params, lr=lr)
```

### Phase 4: Stage별 최적화 (위험도: 높음)

#### 4.1 Stage 1 최적화
```python
# Safe Stage 1 improvements
config_stage1 = {
    "learning_rate": 1e-4,  # 검증된 값
    "warmup_steps": 100,    # 짧게
    "gradient_clip": 0.5,   # 보수적
    "use_ema": False,       # Stage 1에선 불필요
}
```

#### 4.2 Stage 2 최적화
```python
# LoRA rank 동적 조정
def adaptive_lora_rank(step, initial_rank=8):
    if step < 1000:
        return 4  # 초반엔 작게
    elif step < 3000:
        return 8  # 중반
    else:
        return 16  # 후반엔 크게
```

#### 4.3 Stage 3 최적화 (특별 주의!)
```python
# Stage 3는 이미 최적화됨 - 건드리지 않기!
config_stage3 = {
    "learning_rate": 0.1,      # 검증된 값
    "optimizer": "Adam",        # SGD 시도 X
    "early_stopping": True,     # 필수
    "target_range": (0.35, 0.45),  # 검증된 범위
}
```

## 🔧 구현 전략

### 1. 새 학습 스크립트 생성
```bash
scripts/
├── train.py                 # 원본 유지
├── train_optimized_v1.py    # Phase 1 적용
├── train_optimized_v2.py    # Phase 1+2 적용
└── train_optimized_final.py # 전체 적용
```

### 2. 설정 파일 분리
```bash
configs/
├── training_config.json           # 원본 유지
├── training_config_safe.json      # 안전한 설정
└── training_config_optimized.json # 최적화 설정
```

### 3. 테스트 프로토콜
```python
# test_optimization.py
def test_optimization(config_path):
    # 1. 100 steps만 테스트
    # 2. Loss 모니터링
    # 3. NaN 체크
    # 4. 메모리 사용량 체크
    # 5. 속도 측정
    pass
```

## 📊 검증 메트릭

### 필수 체크 항목
- [ ] Loss가 감소하는가?
- [ ] NaN이 발생하지 않는가?
- [ ] Gate parameter가 학습되는가? (Stage 3)
- [ ] 메모리 오버플로우가 없는가?
- [ ] 이전보다 빠른가?

### 성능 목표
- 학습 속도: 20-30% 향상
- 메모리 사용: 30-40% 감소
- 학습 안정성: NaN 발생 0%
- 최종 품질: 기존 동등 이상

## ⚠️ 롤백 계획

### 문제 발생 시
1. 즉시 원본 `train.py` 사용
2. 기존 `training_config.json` 복원
3. 체크포인트에서 재개
4. 문제점 문서화

### 백업 전략
```bash
# 작동하는 버전 백업
cp scripts/train.py scripts/train_stable_backup.py
cp configs/training_config.json configs/training_config_stable_backup.json
```

## 🚀 실행 순서

### Week 1: 안전한 개선
1. DataLoader 캐싱 (Phase 1.1)
2. 간단한 Augmentation (Phase 1.2)
3. 테스트 및 검증

### Week 2: 중간 위험 개선
1. Gradient Monitoring (Phase 2.1)
2. BF16 최적화 (Phase 2.2)
3. Memory 최적화 (Phase 3)

### Week 3: 고위험 개선
1. Stage별 세부 튜닝 (Phase 4)
2. 전체 통합 테스트
3. 최종 벤치마크

## 📝 체크리스트

### 시작 전
- [x] 현재 작동하는 코드 백업
- [x] 트러블슈팅 문서 검토
- [ ] 테스트 환경 준비

### 각 Phase 후
- [ ] 성능 측정
- [ ] 안정성 테스트
- [ ] 문제점 기록
- [ ] 다음 단계 결정

---
*작성일: 2025년 8월 13일*
*기반: TROUBLESHOOTING_REPORT.md의 모든 문제 해결 경험*
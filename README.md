# CLAP2Diffusion

Audio-guided image generation using CLAP and Stable Diffusion.

## Project Overview

CLAP2Diffusion enables audio-conditioned image generation by integrating CLAP (Contrastive Language-Audio Pre-training) audio encoders with Stable Diffusion. The model learns to generate images that correspond to input audio through a novel audio adapter architecture.

## Architecture

- **Base Model**: Stable Diffusion v1.5
- **Audio Encoder**: LAION CLAP (larger_clap_music_and_speech)
- **Audio Adapter**: Custom MLP projection (512d → 768d × 8 tokens)
- **Attention Adapter**: Gated cross-attention with learnable gate parameter
- **Training Strategy**: 3-stage progressive training

## Project Structure

```
CLAP2Diffusion/
├── src/                    # Core source code
│   ├── models/            # Model implementations
│   ├── data/              # Dataset and data processing
│   └── utils/             # Utility functions
├── scripts/               # Training and evaluation scripts
│   ├── train_*_optimized.py  # Optimized training scripts
│   └── run_all_stages.py     # Full pipeline runner
├── configs/               # Configuration files
├── data/vggsound/        # Dataset (audio + frames)
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
└── docs/                 # Documentation

```

## Dataset

Currently using VGGSound dataset:
- **Total pairs**: 1,549 audio-image pairs
- **Balanced subset**: 664 pairs (for training)
- **Classes**: 8 sound categories (ocean_waves, dog_barking, thunder, fire, engine, rain, helicopter, siren)

## Training

### 3-Stage Training Pipeline

1. **Stage 1**: Audio Adapter Training (3k steps)
   - Train only audio projection MLP
   - Freeze all other components
   
2. **Stage 2**: LoRA + Adapter Fine-tuning (7k steps)
   - Train LoRA parameters + audio adapter
   - Rank-8 LoRA on attention layers
   
3. **Stage 3**: Gate Optimization (2k steps)
   - Fine-tune gate parameter only
   - Optimize audio-visual balance

### Quick Start

```bash
# Train all stages with balanced dataset
python scripts/run_all_stages.py

# Or train individual stages
python scripts/train_stage1_optimized.py --use_balanced
python scripts/train_stage2_optimized.py --use_balanced
python scripts/train_stage3_optimized.py --use_balanced
```

## Requirements

See `requirements.txt` for dependencies. Main requirements:
- PyTorch 2.1+
- Transformers 4.35+
- Diffusers 0.23+
- Accelerate 0.24+

## Development Branches

See `BRANCH_STRATEGY.md` for multi-branch development plan:
- `feature/code-improvements`: Current baseline implementation
- `feature/sonic-improvements`: SonicDiffusion-inspired enhancements (planned)
- `feature/cutting-edge`: 2024-2025 state-of-the-art techniques (planned)

## Status

**Current**: Training Stage 1 with class-weighted loss on balanced dataset

## License

Research project - see individual component licenses.

## Contact

For issues and contributions, please open an issue on the repository.
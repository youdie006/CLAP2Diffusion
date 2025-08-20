# CLAP2Diffusion

Audio to Image generation using CLAP + Stable Diffusion

## V4 Hybrid Architecture (Current)
- Hierarchical audio decomposition (foreground/background/ambience)
- Optimized weighted fusion approach
- Memory-efficient training with LoRA
- ~20 seconds per training step

## Key Features
- Text-audio creative composition
- 5,002 AudioCaps training pairs
- Minimal trainable parameters (~5M)
- Compatible with Stable Diffusion v1.5

## Training Progress
- Stage 1: Audio-text alignment (3,000 steps)
- Stage 2: Hierarchical optimization (2,000 steps)
- Currently training V4 model

## Docs
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Status
V4 Hybrid model in active training
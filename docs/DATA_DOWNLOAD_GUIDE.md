# VGGSound Dataset Download Guide

This guide helps you download the VGGSound dataset for CLAP2Diffusion training.

## Prerequisites

### 1. Create Conda Environment
```bash
# Create new conda environment
conda create -n clap_data python=3.10 -y

# Activate environment
conda activate clap_data
```

### 2. Install Required Packages
```bash
# Install download tools
pip install yt-dlp tqdm

# Install ffmpeg (choose one method)
# Option A: Via conda (recommended)
conda install -c conda-forge ffmpeg -y

# Option B: Via system package manager
# Ubuntu/Debian: sudo apt-get install ffmpeg
# MacOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### 3. Create requirements file
Save as `requirements_data.txt`:
```
yt-dlp>=2025.1.0
tqdm>=4.65.0
```

## Dataset Download

### Quick Start (100 samples for testing)
```bash
conda activate clap_data
python scripts/download_vggsound_real.py \
    --max_per_class 10 \
    --num_workers 2
```

### Small Dataset (1,000 samples, ~1.5GB)
```bash
python scripts/download_vggsound_real.py \
    --max_per_class 50 \
    --num_workers 4
```

### Medium Dataset (5,000 samples, ~7.5GB) - Recommended
```bash
python scripts/download_vggsound_real.py \
    --max_per_class 170 \
    --num_workers 4
```

### Large Dataset (20,000 samples, ~30GB)
```bash
python scripts/download_vggsound_real.py \
    --max_per_class 650 \
    --num_workers 8
```

### Download Only Broken Files
```bash
python scripts/download_vggsound_real.py \
    --fix_broken \
    --num_workers 4
```

## Download Options

```bash
python scripts/download_vggsound_real.py --help

Options:
  --output_dir         Output directory (default: data/vggsound)
  --csv_path          Path to VGGSound CSV (default: data/vggsound/vggsound.csv)
  --target_classes    Specific classes to download (e.g., thunder rain dog)
  --max_per_class     Maximum samples per class (default: 100)
  --num_workers       Parallel download workers (default: 4)
  --fix_broken        Only download broken/missing files
```

## Target Classes

Default classes for audio-conditioned generation:
- Natural sounds: thunder, ocean waves, rain, wind
- Animals: dog barking, cat meowing, bird chirping
- Human: applause, laughter, speech
- Mechanical: engine, siren, helicopter
- Actions: glass breaking, door closing, explosion

## Expected Download Times

| Dataset Size | Files | Storage | Time (4 workers) |
|-------------|-------|---------|------------------|
| Test | 100 | 150MB | 5-10 min |
| Small | 1,000 | 1.5GB | 30-45 min |
| Medium | 5,000 | 7.5GB | 2-3 hours |
| Large | 20,000 | 30GB | 8-10 hours |

## Troubleshooting

### 1. yt-dlp not found
```bash
pip install --upgrade yt-dlp
```

### 2. ffmpeg not found
```bash
# Check if installed
which ffmpeg

# If not found, install via conda
conda install -c conda-forge ffmpeg -y
```

### 3. Download failures
- Some YouTube videos may be unavailable
- The script will skip failed downloads and continue
- Run with `--fix_broken` to retry failed downloads

### 4. Slow downloads
- Reduce `--num_workers` if getting rate limited
- YouTube may throttle excessive parallel downloads

## Data Structure

After download, your data directory will look like:
```
data/vggsound/
├── audio/           # WAV files (48kHz, mono, 10 seconds)
│   ├── videoID1.wav
│   └── ...
├── frames/          # JPG images (middle frame)
│   ├── videoID1.jpg
│   └── ...
├── train_metadata.json   # Training split
├── val_metadata.json     # Validation split
├── test_metadata.json    # Test split
└── vggsound.csv         # Original metadata
```

## Metadata Format

Each metadata JSON contains:
```json
{
  "video_id": "abc123",
  "audio_path": "audio/abc123.wav",
  "image_path": "frames/abc123.jpg",
  "text": "A scene with dog barking sound",
  "label": "dog barking"
}
```

## Next Steps

After downloading data:
1. Verify data integrity: `python scripts/analyze_dataset.py`
2. Start training: `docker run ... python scripts/train.py`
3. Or train locally: `python scripts/train.py`

## Legal Notice

This script downloads content from YouTube. Please ensure:
- You comply with YouTube's Terms of Service
- Use the data only for research purposes
- Do not redistribute the downloaded content
- Respect copyright and content creators' rights

## Support

For issues or questions:
- Check existing issues on GitHub
- Ensure all dependencies are correctly installed
- Try with a smaller dataset first
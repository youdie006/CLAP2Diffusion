#!/usr/bin/env python3
"""
Download and cache all required models locally
"""

import os
# Set cache directory to local project
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import ClapModel, ClapProcessor
import torch

print("=" * 60)
print("Model Downloader for CLAP2Diffusion")
print("=" * 60)

# Create cache directories
cache_dir = "/workspace/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(f"{cache_dir}/transformers", exist_ok=True)
os.makedirs(f"{cache_dir}/diffusers", exist_ok=True)

print(f"\nCache directory: {cache_dir}")
print("-" * 60)

# Download Stable Diffusion
print("\n[1/3] Downloading Stable Diffusion v1.5...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    print("✓ Stable Diffusion v1.5 downloaded (3.44GB)")
except Exception as e:
    print(f"✗ Error downloading Stable Diffusion: {e}")

# Download CLAP model
print("\n[2/3] Downloading CLAP model...")
try:
    clap_model = ClapModel.from_pretrained(
        "laion/larger_clap_music_and_speech",
        cache_dir=cache_dir
    )
    print("✓ CLAP model downloaded (776MB)")
except Exception as e:
    print(f"✗ Error downloading CLAP model: {e}")

# Download CLAP processor
print("\n[3/3] Downloading CLAP processor...")
try:
    clap_processor = ClapProcessor.from_pretrained(
        "laion/larger_clap_music_and_speech",
        cache_dir=cache_dir
    )
    print("✓ CLAP processor downloaded")
except Exception as e:
    print(f"✗ Error downloading CLAP processor: {e}")

print("\n" + "=" * 60)
print("All models downloaded successfully!")
print(f"Total cache size will be approximately 4.3GB")
print("=" * 60)
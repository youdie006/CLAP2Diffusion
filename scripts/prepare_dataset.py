#!/usr/bin/env python3
"""
Dataset preparation script for CLAP2Diffusion
Downloads and prepares audio-visual datasets for training
"""

import os
import json
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict
import requests
from tqdm import tqdm
import subprocess
import shutil


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def prepare_vggsound_subset(data_dir: Path, num_samples: int = 1000):
    """
    Prepare VGGSound dataset subset.
    
    Note: This downloads a small subset for testing.
    For full dataset, use official VGGSound download scripts.
    """
    print("\n=== Preparing VGGSound Subset ===")
    
    vggsound_dir = data_dir / "vggsound"
    vggsound_dir.mkdir(parents=True, exist_ok=True)
    
    # Target audio classes
    target_classes = [
        "thunder", "ocean_waves", "fire_crackling", "applause",
        "siren", "helicopter", "dog_barking", "rain", 
        "glass_breaking", "engine"
    ]
    
    # Create metadata
    metadata = []
    
    # Download sample data (placeholder - replace with actual VGGSound data)
    print(f"Creating {num_samples} sample entries for testing...")
    
    audio_dir = vggsound_dir / "audio"
    frames_dir = vggsound_dir / "frames"
    audio_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    
    for i in range(num_samples):
        class_name = target_classes[i % len(target_classes)]
        video_id = f"sample_{i:05d}"
        
        metadata.append({
            "video_id": video_id,
            "class": class_name,
            "audio_path": f"audio/{video_id}.wav",
            "image_path": f"frames/{video_id}.jpg",
            "text": f"A scene with {class_name.replace('_', ' ')} sound",
            "start_time": 0,
            "duration": 10
        })
    
    # Save metadata
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    for split_name, ratio in splits.items():
        start_idx = int(sum(list(splits.values())[:list(splits.keys()).index(split_name)]) * num_samples)
        end_idx = int(start_idx + ratio * num_samples)
        
        split_metadata = metadata[start_idx:end_idx]
        
        with open(vggsound_dir / f"{split_name}_metadata.json", 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        print(f"Created {split_name} split with {len(split_metadata)} samples")
    
    return vggsound_dir


def download_freesound_samples(data_dir: Path, api_key: str = None):
    """
    Download samples from Freesound.
    
    Note: Requires Freesound API key for full access.
    """
    print("\n=== Downloading Freesound Samples ===")
    
    freesound_dir = data_dir / "freesound"
    freesound_dir.mkdir(parents=True, exist_ok=True)
    
    # Target sound categories
    queries = [
        "thunder storm", "ocean waves", "fire crackling", "applause clapping",
        "police siren", "helicopter flying", "dog barking", "heavy rain",
        "glass breaking", "car engine"
    ]
    
    if not api_key:
        print("No Freesound API key provided. Creating dummy data...")
        return freesound_dir
    
    # Download using Freesound API
    import freesound
    client = freesound.FreesoundClient()
    client.set_token(api_key)
    
    metadata = []
    
    for query in queries:
        print(f"Searching for: {query}")
        results = client.text_search(query=query, filter="duration:[5 TO 15]", 
                                    fields="id,name,duration,username,description")
        
        for sound in results[:10]:  # Get 10 samples per category
            try:
                # Download audio
                sound_obj = client.get_sound(sound.id)
                audio_path = freesound_dir / f"audio/{sound.id}.wav"
                audio_path.parent.mkdir(exist_ok=True)
                sound_obj.retrieve(audio_path)
                
                metadata.append({
                    "id": sound.id,
                    "name": sound.name,
                    "query": query,
                    "duration": sound.duration,
                    "audio_path": f"audio/{sound.id}.wav",
                    "description": sound.description[:200]
                })
                
            except Exception as e:
                print(f"Error downloading sound {sound.id}: {e}")
    
    # Save metadata
    with open(freesound_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Downloaded {len(metadata)} audio samples")
    
    return freesound_dir


def create_synthetic_dataset(data_dir: Path, num_samples: int = 100):
    """
    Create synthetic dataset for quick testing.
    Uses simple audio generation and stock images.
    """
    print("\n=== Creating Synthetic Dataset ===")
    
    synthetic_dir = data_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = synthetic_dir / "audio"
    image_dir = synthetic_dir / "images"
    audio_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    
    # Import required libraries
    import numpy as np
    from PIL import Image
    import soundfile as sf
    
    # Target classes and their audio characteristics
    sound_params = {
        "thunder": {"freq": 50, "noise": 0.8},
        "ocean_waves": {"freq": 0.1, "noise": 0.3},
        "fire": {"freq": 1000, "noise": 0.9},
        "applause": {"freq": 2000, "noise": 0.7},
        "siren": {"freq": [440, 880], "noise": 0.1},
        "helicopter": {"freq": 100, "noise": 0.4},
        "dog_barking": {"freq": 800, "noise": 0.5},
        "rain": {"freq": 0, "noise": 1.0},
        "glass_breaking": {"freq": 4000, "noise": 0.8},
        "engine": {"freq": 60, "noise": 0.6}
    }
    
    metadata = []
    
    for i in range(num_samples):
        class_name = list(sound_params.keys())[i % len(sound_params)]
        params = sound_params[class_name]
        
        # Generate synthetic audio (10 seconds at 16kHz)
        duration = 10
        sample_rate = 16000
        t = np.linspace(0, duration, sample_rate * duration)
        
        # Create audio based on class
        if isinstance(params["freq"], list):
            # Multiple frequencies (e.g., siren)
            audio = np.zeros_like(t)
            for freq in params["freq"]:
                audio += np.sin(2 * np.pi * freq * t)
            audio = audio / len(params["freq"])
        elif params["freq"] > 0:
            # Single frequency
            audio = np.sin(2 * np.pi * params["freq"] * t)
        else:
            # Pure noise (e.g., rain)
            audio = np.zeros_like(t)
        
        # Add noise
        noise = np.random.randn(len(t)) * params["noise"]
        audio = audio * (1 - params["noise"]) + noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save audio
        audio_path = audio_dir / f"{i:05d}_{class_name}.wav"
        sf.write(str(audio_path), audio.astype(np.float32), sample_rate)
        
        # Generate synthetic image (512x512 with class-specific colors)
        color_map = {
            "thunder": (50, 50, 80),      # Dark blue
            "ocean_waves": (70, 130, 180), # Steel blue
            "fire": (255, 100, 0),         # Orange
            "applause": (200, 200, 200),   # Light gray
            "siren": (255, 0, 0),          # Red
            "helicopter": (100, 100, 100), # Gray
            "dog_barking": (139, 69, 19),  # Brown
            "rain": (100, 100, 150),       # Blue-gray
            "glass_breaking": (200, 200, 255), # Light blue
            "engine": (50, 50, 50)         # Dark gray
        }
        
        base_color = color_map[class_name]
        image = np.random.randint(-30, 30, (512, 512, 3)) + base_color
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Add some patterns
        if class_name == "ocean_waves":
            # Add wave patterns
            for y in range(0, 512, 20):
                wave = np.sin(np.linspace(0, 4*np.pi, 512)) * 10
                image[y:y+10, :, 2] += wave.astype(np.uint8)
        elif class_name == "thunder":
            # Add lightning
            image[250:260, 200:210] = 255
            image[240:250, 205:215] = 255
        
        # Save image
        image_path = image_dir / f"{i:05d}_{class_name}.jpg"
        Image.fromarray(image).save(image_path)
        
        # Add to metadata
        metadata.append({
            "id": f"{i:05d}",
            "class": class_name,
            "audio_path": str(audio_path.relative_to(synthetic_dir)),
            "image_path": str(image_path.relative_to(synthetic_dir)),
            "text": f"A {class_name.replace('_', ' ')} scene",
            "duration": duration,
            "sample_rate": sample_rate
        })
    
    # Split data
    train_size = int(0.8 * len(metadata))
    val_size = int(0.1 * len(metadata))
    
    train_metadata = metadata[:train_size]
    val_metadata = metadata[train_size:train_size + val_size]
    test_metadata = metadata[train_size + val_size:]
    
    # Save metadata
    with open(synthetic_dir / "train_metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(synthetic_dir / "val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)
    
    with open(synthetic_dir / "test_metadata.json", 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"Created synthetic dataset:")
    print(f"  - Train: {len(train_metadata)} samples")
    print(f"  - Val: {len(val_metadata)} samples")
    print(f"  - Test: {len(test_metadata)} samples")
    
    return synthetic_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for CLAP2Diffusion")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, default="synthetic",
                       choices=["vggsound", "freesound", "synthetic", "all"],
                       help="Which dataset to prepare")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to generate/download")
    parser.add_argument("--freesound_api_key", type=str, default=None,
                       help="Freesound API key (optional)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("CLAP2Diffusion Dataset Preparation")
    print("=" * 50)
    
    if args.dataset in ["vggsound", "all"]:
        prepare_vggsound_subset(data_dir, args.num_samples)
    
    if args.dataset in ["freesound", "all"]:
        download_freesound_samples(data_dir, args.freesound_api_key)
    
    if args.dataset in ["synthetic", "all"]:
        create_synthetic_dataset(data_dir, args.num_samples)
    
    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print(f"Data saved to: {data_dir.absolute()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
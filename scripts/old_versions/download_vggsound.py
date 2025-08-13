#!/usr/bin/env python3
"""
VGGSound dataset download script for CLAP2Diffusion
Downloads audio-visual data from VGGSound dataset
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict
import subprocess
import concurrent.futures
from tqdm import tqdm
import time
import random


def download_youtube_video(video_id: str, start_time: float, output_dir: Path, quality: str = "bestaudio"):
    """Download audio/video from YouTube using yt-dlp."""
    audio_path = output_dir / "audio" / f"{video_id}.wav"
    video_path = output_dir / "frames" / f"{video_id}.mp4"
    
    # Skip if already downloaded
    if audio_path.exists() and video_path.exists():
        return True
    
    try:
        # Download audio
        audio_cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-f", quality,
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(audio_path),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--extract-audio",
            "--postprocessor-args", f"-ss {start_time} -t 10 -ar 48000"
        ]
        
        subprocess.run(audio_cmd, check=True, capture_output=True)
        
        # Download video frame at start_time
        video_cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-f", "best[height<=720]",
            "-o", str(video_path),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--postprocessor-args", f"-ss {start_time} -t 1 -vframes 1"
        ]
        
        subprocess.run(video_cmd, check=True, capture_output=True)
        
        # Extract single frame as jpg
        frame_path = output_dir / "frames" / f"{video_id}.jpg"
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path),
            "-y"
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Remove mp4 file
        video_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"Error downloading {video_id}: {e}")
        return False


def load_vggsound_metadata(csv_path: Path) -> List[Dict]:
    """Load VGGSound CSV metadata."""
    metadata = []
    
    # Target audio classes for CLAP2Diffusion
    target_classes = {
        "thunder": ["thunder", "thunderstorm"],
        "ocean_waves": ["waves", "ocean", "sea"],
        "fire": ["fire", "burning", "crackling"],
        "applause": ["applause", "clapping"],
        "siren": ["siren", "alarm"],
        "helicopter": ["helicopter"],
        "dog_barking": ["dog", "barking"],
        "rain": ["rain", "raining"],
        "glass_breaking": ["glass", "breaking", "shatter"],
        "engine": ["engine", "motor", "vehicle"]
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                video_id = row[0]
                start_time = float(row[1])
                label = row[2].lower()
                
                # Check if label matches target classes
                for target_class, keywords in target_classes.items():
                    if any(keyword in label for keyword in keywords):
                        metadata.append({
                            "video_id": video_id,
                            "start_time": start_time,
                            "label": label,
                            "target_class": target_class
                        })
                        break
    
    return metadata


def download_vggsound_subset(
    output_dir: Path,
    csv_path: Path = None,
    max_samples_per_class: int = 100,
    num_workers: int = 4
):
    """Download VGGSound subset for target classes."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    
    # Download VGGSound CSV if not provided
    if csv_path is None or not csv_path.exists():
        print("Downloading VGGSound metadata CSV...")
        csv_url = "https://github.com/hche11/VGGSound/raw/master/data/vggsound.csv"
        csv_path = output_dir / "vggsound.csv"
        
        subprocess.run([
            "wget", csv_url, "-O", str(csv_path)
        ], check=True)
    
    # Load and filter metadata
    print("Loading metadata...")
    all_metadata = load_vggsound_metadata(csv_path)
    
    # Sample evenly from each class
    sampled_metadata = []
    class_counts = {}
    
    for item in all_metadata:
        target_class = item["target_class"]
        if target_class not in class_counts:
            class_counts[target_class] = 0
        
        if class_counts[target_class] < max_samples_per_class:
            sampled_metadata.append(item)
            class_counts[target_class] += 1
    
    print(f"Found {len(sampled_metadata)} samples across {len(class_counts)} classes")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} samples")
    
    # Download samples
    print(f"\nDownloading samples using {num_workers} workers...")
    
    def download_wrapper(item):
        return download_youtube_video(
            item["video_id"],
            item["start_time"],
            output_dir
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_wrapper, item) for item in sampled_metadata]
        
        success_count = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                success_count += 1
    
    print(f"Successfully downloaded {success_count}/{len(sampled_metadata)} samples")
    
    # Create metadata JSON files
    create_metadata_splits(sampled_metadata, output_dir)


def create_metadata_splits(metadata: List[Dict], output_dir: Path):
    """Create train/val/test splits with metadata."""
    
    # Shuffle metadata
    random.shuffle(metadata)
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    total = len(metadata)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = metadata[:train_size]
    val_data = metadata[train_size:train_size + val_size]
    test_data = metadata[train_size + val_size:]
    
    # Format metadata for training
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        formatted_data = []
        for item in split_data:
            formatted_data.append({
                "video_id": item["video_id"],
                "audio_path": f"audio/{item['video_id']}.wav",
                "image_path": f"frames/{item['video_id']}.jpg",
                "text": f"A scene with {item['target_class'].replace('_', ' ')} sound",
                "class": item["target_class"],
                "original_label": item["label"]
            })
        
        output_path = output_dir / f"{split_name}_metadata.json"
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        
        print(f"Created {split_name} split with {len(formatted_data)} samples")


def main():
    parser = argparse.ArgumentParser(description="Download VGGSound dataset for CLAP2Diffusion")
    parser.add_argument("--output_dir", type=str, default="data/vggsound",
                        help="Output directory for downloaded data")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to VGGSound CSV file")
    parser.add_argument("--max_samples_per_class", type=int, default=100,
                        help="Maximum samples per audio class")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel download workers")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        subprocess.run(["yt-dlp", "--version"], check=True, capture_output=True)
    except:
        print("Error: yt-dlp not found. Install with: pip install yt-dlp")
        return
    
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except:
        print("Error: ffmpeg not found. Install ffmpeg first.")
        return
    
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path) if args.csv_path else None
    
    download_vggsound_subset(
        output_dir,
        csv_path,
        args.max_samples_per_class,
        args.num_workers
    )
    
    print("\nVGGSound download complete!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Download VGGSound dataset from YouTube using yt-dlp
"""

import os
import csv
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random
import time

def check_dependencies():
    """Check if required tools are installed"""
    import shutil
    
    tools = {
        'yt-dlp': 'pip install yt-dlp',
        'ffmpeg': 'conda install -c conda-forge ffmpeg -y'
    }
    
    for tool, install_cmd in tools.items():
        if shutil.which(tool):
            print(f"✓ {tool} found")
        else:
            print(f"✗ {tool} not found. Install with: {install_cmd}")
            return False
    return True

def download_youtube_clip(video_id, start_time, label, output_dir, duration=10):
    """Download 10-second clip from YouTube"""
    
    audio_path = output_dir / "audio" / f"{video_id}.wav"
    frame_path = output_dir / "frames" / f"{video_id}.jpg"
    
    # Skip if both files exist and are valid
    if audio_path.exists() and frame_path.exists():
        if audio_path.stat().st_size > 100000 and frame_path.stat().st_size > 2000:
            return True, f"Already exists: {video_id}"
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Download audio with yt-dlp
        audio_cmd = [
            "yt-dlp",
            url,
            "-f", "bestaudio",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(audio_path.with_suffix('.%(ext)s')),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "--postprocessor-args", f"ffmpeg:-ss {start_time} -t {duration} -ar 48000 -ac 1"
        ]
        
        result = subprocess.run(audio_cmd, capture_output=True, timeout=60)
        
        if not audio_path.exists() or audio_path.stat().st_size < 100000:
            return False, f"Audio download failed: {video_id}"
        
        # Download video and extract frame
        temp_video = output_dir / "temp" / f"{video_id}.mp4"
        temp_video.parent.mkdir(exist_ok=True)
        
        video_cmd = [
            "yt-dlp",
            url,
            "-f", "best[height<=480]",  # Limit resolution for faster download
            "-o", str(temp_video),
            "--no-playlist",
            "--quiet",
            "--no-warnings"
        ]
        
        subprocess.run(video_cmd, capture_output=True, timeout=60)
        
        if temp_video.exists():
            # Extract frame at start_time + 5 seconds (middle of clip)
            frame_time = start_time + 5
            frame_cmd = [
                "ffmpeg",
                "-i", str(temp_video),
                "-ss", str(frame_time),
                "-vframes", "1",
                "-q:v", "2",
                str(frame_path),
                "-y",
                "-loglevel", "error"
            ]
            subprocess.run(frame_cmd, check=True, timeout=30)
            
            # Clean up temp video
            temp_video.unlink()
        
        # Verify outputs
        if audio_path.exists() and frame_path.exists():
            if audio_path.stat().st_size > 100000 and frame_path.stat().st_size > 2000:
                return True, f"Success: {video_id} ({label})"
        
        return False, f"Invalid output: {video_id}"
        
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {video_id}"
    except Exception as e:
        return False, f"Error {video_id}: {str(e)}"

def load_vggsound_csv(csv_path):
    """Load and parse VGGSound CSV"""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4:
                data.append({
                    'video_id': row[0],
                    'start_time': int(row[1]),
                    'label': row[2],
                    'split': row[3]
                })
    return data

def download_vggsound(
    output_dir: Path,
    csv_path: Path,
    target_classes=None,
    max_per_class=100,
    num_workers=4,
    fix_broken_only=False
):
    """Download VGGSound dataset"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    (output_dir / "temp").mkdir(exist_ok=True)
    
    # Load metadata
    print("Loading VGGSound metadata...")
    all_data = load_vggsound_csv(csv_path)
    print(f"Total entries: {len(all_data)}")
    
    # Filter by target classes if specified
    if target_classes:
        filtered_data = [d for d in all_data if any(tc in d['label'].lower() for tc in target_classes)]
        print(f"Filtered to {len(filtered_data)} entries matching target classes")
    else:
        filtered_data = all_data
    
    # If fixing broken files only
    if fix_broken_only:
        to_download = []
        for item in filtered_data:
            audio_path = output_dir / "audio" / f"{item['video_id']}.wav"
            if not audio_path.exists() or audio_path.stat().st_size < 100000:
                to_download.append(item)
        print(f"Found {len(to_download)} broken/missing files to download")
    else:
        # Sample evenly from each class
        class_samples = {}
        for item in filtered_data:
            label = item['label']
            if label not in class_samples:
                class_samples[label] = []
            if len(class_samples[label]) < max_per_class:
                class_samples[label].append(item)
        
        to_download = []
        for label, samples in class_samples.items():
            to_download.extend(samples)
        
        print(f"Will download {len(to_download)} samples from {len(class_samples)} classes")
    
    # Download with progress
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                download_youtube_clip,
                item['video_id'],
                item['start_time'],
                item['label'],
                output_dir
            ): item
            for item in to_download
        }
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                item = futures[future]
                success, message = future.result()
                
                if success:
                    successful.append(item)
                else:
                    failed.append((item, message))
                
                pbar.update(1)
                pbar.set_postfix(success=len(successful), failed=len(failed))
    
    # Clean up temp directory
    temp_dir = output_dir / "temp"
    if temp_dir.exists():
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed and len(failed) < 20:
        print("\nFailed downloads:")
        for item, msg in failed[:10]:
            print(f"  {msg}")
    
    # Create metadata
    create_metadata(successful, output_dir)
    
    return successful

def create_metadata(items, output_dir):
    """Create train/val/test metadata"""
    
    # Group by split
    splits = {'train': [], 'val': [], 'test': []}
    
    for item in items:
        # Use original split for train/test, create val from train
        if item['split'] == 'test':
            splits['test'].append(item)
        else:
            splits['train'].append(item)
    
    # Create validation set from train (10%)
    random.shuffle(splits['train'])
    val_size = len(splits['train']) // 10
    splits['val'] = splits['train'][:val_size]
    splits['train'] = splits['train'][val_size:]
    
    # Save metadata
    for split_name, split_items in splits.items():
        metadata = []
        for item in split_items:
            metadata.append({
                'video_id': item['video_id'],
                'audio_path': f"audio/{item['video_id']}.wav",
                'image_path': f"frames/{item['video_id']}.jpg",
                'text': f"A scene with {item['label']} sound",
                'label': item['label']
            })
        
        metadata_path = output_dir / f"{split_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created {split_name} split: {len(metadata)} samples")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download VGGSound dataset")
    parser.add_argument("--output_dir", type=str, default="data/vggsound",
                        help="Output directory")
    parser.add_argument("--csv_path", type=str, default="data/vggsound/vggsound.csv",
                        help="Path to VGGSound CSV")
    parser.add_argument("--target_classes", nargs='+', default=None,
                        help="Target classes to download (e.g., thunder rain dog)")
    parser.add_argument("--max_per_class", type=int, default=100,
                        help="Maximum samples per class")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel downloads")
    parser.add_argument("--fix_broken", action="store_true",
                        help="Only download broken/missing files")
    
    args = parser.parse_args()
    
    if not check_dependencies():
        return
    
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        print("Download from: https://github.com/hche11/VGGSound/raw/master/data/vggsound.csv")
        return
    
    # Default target classes for CLAP2Diffusion
    if args.target_classes is None and not args.fix_broken:
        args.target_classes = [
            'thunder', 'ocean', 'waves', 'fire', 'applause',
            'siren', 'helicopter', 'dog', 'barking', 'rain',
            'glass', 'engine', 'car', 'explosion', 'gunshot'
        ]
        print(f"Using default target classes: {args.target_classes}")
    
    # Download
    successful = download_vggsound(
        output_dir,
        csv_path,
        args.target_classes,
        args.max_per_class,
        args.num_workers,
        args.fix_broken
    )
    
    print(f"\nDataset ready at: {output_dir}")
    print(f"Total usable samples: {len(successful)}")

if __name__ == "__main__":
    main()
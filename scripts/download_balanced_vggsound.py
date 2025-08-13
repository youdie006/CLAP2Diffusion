#!/usr/bin/env python3
"""
Download balanced VGGSound dataset
Target: ~1000 samples per class (download 1500 to account for failures)
"""

import os
import csv
import random
from pathlib import Path
from collections import defaultdict
import subprocess
import json

# Target classes and current counts
CURRENT_COUNTS = {
    'engine': 336,
    'rain': 208, 
    'fire': 124,
    'siren': 65,
    'dog_barking': 39,
    'ocean_waves': 22,
    'thunder': 0,
    'applause': 0,
    'helicopter': 0,
    'glass_breaking': 0
}

# VGGSound class mappings
CLASS_MAPPINGS = {
    'engine': ['engine accelerating', 'engine idling', 'car engine starting', 'engine knocking'],
    'rain': ['rain', 'rainstorm', 'rain on surface', 'thunderstorm'],
    'fire': ['fire crackling', 'fire alarm', 'campfire', 'bonfire'],
    'siren': ['siren', 'ambulance siren', 'police siren', 'fire truck siren'],
    'dog_barking': ['dog barking', 'dogs barking', 'dog howling', 'puppy barking'],
    'ocean_waves': ['ocean waves', 'waves crashing', 'sea waves', 'beach waves'],
    'thunder': ['thunder', 'thunderstorm', 'thunder rumbling', 'distant thunder'],
    'applause': ['applause', 'clapping', 'crowd applause', 'audience applause'],
    'helicopter': ['helicopter', 'helicopter flying', 'chopper', 'helicopter hovering'],
    'glass_breaking': ['glass breaking', 'glass shattering', 'window breaking', 'glass smashing']
}

TARGET_PER_CLASS = 1000  # Final target
DOWNLOAD_PER_CLASS = 1500  # Download extra to account for failures

def load_vggsound_csv():
    """Load VGGSound CSV metadata"""
    csv_path = Path("data/vggsound/vggsound.csv")
    
    if not csv_path.exists():
        print("Downloading VGGSound CSV...")
        subprocess.run([
            "wget", "-O", str(csv_path),
            "https://github.com/hche11/VGGSound/raw/master/data/vggsound.csv"
        ])
    
    videos_by_class = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if exists
        
        for row in reader:
            if len(row) >= 3:
                video_id = row[0]
                start_time = int(row[1])
                label = row[2].lower()
                
                # Map to our target classes
                for target_class, patterns in CLASS_MAPPINGS.items():
                    if any(pattern in label for pattern in patterns):
                        videos_by_class[target_class].append({
                            'video_id': video_id,
                            'start_time': start_time,
                            'label': label
                        })
                        break
    
    return videos_by_class

def download_class_samples(class_name, videos, output_dir, max_samples=1500):
    """Download samples for a specific class"""
    needed = max(0, DOWNLOAD_PER_CLASS - CURRENT_COUNTS.get(class_name, 0))
    
    if needed == 0:
        print(f"✓ {class_name}: Already have enough samples")
        return 0
    
    print(f"\n{'='*60}")
    print(f"Downloading {needed} samples for: {class_name}")
    print(f"Current: {CURRENT_COUNTS.get(class_name, 0)}, Target: {TARGET_PER_CLASS}")
    print(f"{'='*60}")
    
    # Shuffle for variety
    random.shuffle(videos)
    
    downloaded = 0
    failed = 0
    
    for video in videos[:needed + 100]:  # Try extra in case of failures
        if downloaded >= needed:
            break
            
        video_id = video['video_id']
        start_time = video['start_time']
        
        # Check if already exists
        audio_path = output_dir / "audio" / f"{video_id}.wav"
        frame_path = output_dir / "frames" / f"{video_id}.jpg"
        
        if audio_path.exists() and frame_path.exists():
            print(f"  ✓ Already exists: {video_id}")
            downloaded += 1
            continue
        
        # Download
        print(f"  Downloading {downloaded+1}/{needed}: {video_id}...", end=' ')
        
        # Audio download
        audio_cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"-ss {start_time} -t 10 -ar 48000",
            "-o", str(audio_path),
            "--quiet",
            "--no-warnings"
        ]
        
        # Frame download
        frame_time = start_time + 5  # Middle of 10-second clip
        frame_cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "--skip-download",
            "--write-thumbnail",
            "--convert-thumbnails", "jpg",
            "-o", str(output_dir / "frames" / f"{video_id}"),
            "--quiet",
            "--no-warnings"
        ]
        
        try:
            # Try with timeout
            result = subprocess.run(audio_cmd, timeout=30, capture_output=True)
            if result.returncode == 0 and audio_path.exists():
                # Try to get frame
                subprocess.run(frame_cmd, timeout=10, capture_output=True)
                
                # Alternative: extract frame from video
                if not frame_path.exists():
                    extract_cmd = [
                        "ffmpeg", "-i", f"https://www.youtube.com/watch?v={video_id}",
                        "-ss", str(frame_time),
                        "-frames:v", "1",
                        "-q:v", "2",
                        str(frame_path),
                        "-y"
                    ]
                    subprocess.run(extract_cmd, timeout=15, capture_output=True)
                
                if frame_path.exists():
                    print("✓")
                    downloaded += 1
                else:
                    print("✗ (no frame)")
                    audio_path.unlink()  # Remove audio if no frame
                    failed += 1
            else:
                print("✗ (download failed)")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print("✗ (timeout)")
            failed += 1
            # Clean up partial downloads
            if audio_path.exists():
                audio_path.unlink()
            if frame_path.exists():
                frame_path.unlink()
        except Exception as e:
            print(f"✗ ({str(e)[:20]})")
            failed += 1
            
        # Save progress every 10 downloads
        if downloaded % 10 == 0:
            print(f"  Progress: {downloaded}/{needed} downloaded, {failed} failed")
    
    print(f"\n✓ {class_name}: Downloaded {downloaded} new samples")
    return downloaded

def main():
    output_dir = Path("data/vggsound")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    
    # Load VGGSound metadata
    print("Loading VGGSound metadata...")
    videos_by_class = load_vggsound_csv()
    
    # Print available samples
    print("\nAvailable samples in VGGSound:")
    for class_name in CURRENT_COUNTS.keys():
        available = len(videos_by_class.get(class_name, []))
        current = CURRENT_COUNTS[class_name]
        needed = max(0, TARGET_PER_CLASS - current)
        print(f"  {class_name}: {available} available, {current} current, {needed} needed")
    
    # Download missing samples
    total_downloaded = 0
    
    # Priority: Download classes with 0 samples first
    priority_classes = [c for c, count in CURRENT_COUNTS.items() if count == 0]
    other_classes = [c for c, count in CURRENT_COUNTS.items() if count > 0 and count < TARGET_PER_CLASS]
    
    for class_name in priority_classes + other_classes:
        if class_name in videos_by_class:
            downloaded = download_class_samples(
                class_name,
                videos_by_class[class_name],
                output_dir
            )
            total_downloaded += downloaded
    
    print(f"\n{'='*60}")
    print(f"Total downloaded: {total_downloaded} new samples")
    print(f"{'='*60}")
    
    # Run cleanup and validation
    print("\nRunning cleanup and validation...")
    os.system("python scripts/cleanup_data.py")
    os.system("python scripts/validate_audio.py")
    
    # Regenerate metadata
    print("\nRegenerating metadata...")
    os.system("python scripts/create_vggsound_metadata.py")
    
    print("\n✓ Balanced dataset download complete!")

if __name__ == "__main__":
    main()
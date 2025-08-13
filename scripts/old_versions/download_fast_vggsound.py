#!/usr/bin/env python3
"""
Fast VGGSound downloader - audio only, no video/frame extraction
Much faster than full download
"""

import os
import csv
import json
import random
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

TARGET_CLASSES = ['rain', 'fire', 'siren', 'dog_barking', 'ocean_waves', 
                  'thunder', 'applause', 'helicopter', 'glass_breaking', 'engine']

def download_audio_and_frame(video_id, start_time, label, output_dir):
    """Download audio and extract single frame - optimized"""
    audio_path = output_dir / "audio" / f"{video_id}.wav"
    frame_path = output_dir / "frames" / f"{video_id}.jpg"
    
    # Skip if both exist
    if (audio_path.exists() and audio_path.stat().st_size > 100000 and 
        frame_path.exists() and frame_path.stat().st_size > 2000):
        return True, "exists"
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Download audio if needed
        if not audio_path.exists() or audio_path.stat().st_size < 100000:
            audio_cmd = [
                "yt-dlp",
                url,
                "-f", "bestaudio",
                "-x",
                "--audio-format", "wav",
                "-o", str(audio_path.with_suffix('.%(ext)s')),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "--socket-timeout", "10",
                "--retries", "1",
                "--postprocessor-args", f"ffmpeg:-ss {start_time} -t 10 -ar 48000"
            ]
            
            result = subprocess.run(audio_cmd, capture_output=True, timeout=30)
            
            if not audio_path.exists() or audio_path.stat().st_size < 100000:
                return False, "audio_failed"
        
        # Extract frame directly using ffmpeg (faster than downloading whole video)
        if not frame_path.exists() or frame_path.stat().st_size < 2000:
            frame_time = start_time + 5  # Middle of 10-second clip
            frame_cmd = [
                "ffmpeg",
                "-ss", str(frame_time),
                "-i", url,
                "-frames:v", "1",
                "-q:v", "2",
                str(frame_path),
                "-y",
                "-loglevel", "error"
            ]
            
            result = subprocess.run(frame_cmd, capture_output=True, timeout=20)
            
            if not frame_path.exists() or frame_path.stat().st_size < 2000:
                # No fallback - frame must be from exact timestamp
                # Clean up audio if no frame (need both for training)
                if audio_path.exists():
                    audio_path.unlink()
                return False, "frame_failed"
        
        return True, "success"
            
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:20]

def main():
    output_dir = Path("data/vggsound")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    
    # Load VGGSound CSV
    csv_path = Path("data/vggsound/vggsound.csv")
    if not csv_path.exists():
        print("Downloading VGGSound CSV...")
        subprocess.run([
            "wget", "-O", str(csv_path),
            "https://github.com/hche11/VGGSound/raw/master/data/vggsound.csv"
        ])
    
    # Parse CSV
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                video_id = row[0]
                start_time = int(row[1])
                label = row[2].lower()
                
                # Filter by target classes
                if any(tc in label for tc in TARGET_CLASSES):
                    samples.append({
                        'video_id': video_id,
                        'start_time': start_time,
                        'label': label
                    })
    
    print(f"Found {len(samples)} samples for target classes")
    
    # Shuffle and limit
    random.shuffle(samples)
    samples = samples[:5000]  # Limit to 5000 attempts
    
    # Download with multiple workers
    print("Starting fast parallel download (audio + frames)...")
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:  # 8 parallel workers
        futures = {
            executor.submit(
                download_audio_and_frame,
                s['video_id'],
                s['start_time'],
                s['label'],
                output_dir
            ): s for s in samples
        }
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                success, msg = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(success=successful, failed=failed)
    
    print(f"\nCompleted: {successful} successful, {failed} failed")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    
    # Count final files
    audio_files = list((output_dir / "audio").glob("*.wav"))
    valid_files = [f for f in audio_files if f.stat().st_size > 100000]
    print(f"\nTotal valid audio files: {len(valid_files)}")

if __name__ == "__main__":
    main()
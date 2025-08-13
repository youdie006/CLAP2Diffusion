#!/usr/bin/env python3
"""
Download broken VGGSound files from YouTube
Only downloads files that are currently broken (< 1KB)
"""

import os
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import csv

def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except:
        print("yt-dlp not found. Install with: pip install yt-dlp")
        return False

def load_vggsound_csv():
    """Load VGGSound CSV file"""
    csv_path = Path("data/vggsound/vggsound.csv")
    data = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            video_id = row[0]
            start_time = int(row[1])
            label = row[2]
            split = row[3]
            data[video_id] = {
                'start_time': start_time,
                'label': label,
                'split': split
            }
    
    return data

def find_broken_files():
    """Find all broken audio files (< 1KB)"""
    broken = []
    audio_dir = Path("data/vggsound/audio")
    
    for wav_file in audio_dir.glob("*.wav"):
        if wav_file.stat().st_size < 1000:  # Less than 1KB
            video_id = wav_file.stem
            broken.append(video_id)
    
    return broken

def download_audio(video_id, start_time, output_dir, duration=10):
    """Download audio from YouTube"""
    output_path = output_dir / f"{video_id}.wav"
    
    # Skip if already exists and is valid
    if output_path.exists() and output_path.stat().st_size > 100000:
        return True, f"Already exists: {video_id}"
    
    try:
        # Download and extract audio with yt-dlp
        cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "-f", "bestaudio",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", str(output_path.with_suffix('.%(ext)s')),
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            # Post-processing to extract 10 seconds from start_time
            "--postprocessor-args", f"ffmpeg:-ss {start_time} -t {duration} -ar 48000 -ac 1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        
        if output_path.exists() and output_path.stat().st_size > 100000:
            return True, f"Success: {video_id}"
        else:
            return False, f"Failed: {video_id} - File too small"
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {video_id}"
    except Exception as e:
        return False, f"Error: {video_id} - {str(e)}"

def download_broken_files(max_workers=4, limit=None):
    """Download all broken files"""
    
    if not check_yt_dlp():
        return
    
    print("Loading VGGSound metadata...")
    vggsound_data = load_vggsound_csv()
    
    print("Finding broken files...")
    broken_ids = find_broken_files()
    print(f"Found {len(broken_ids)} broken files")
    
    if limit:
        broken_ids = broken_ids[:limit]
        print(f"Limiting to {limit} files")
    
    # Filter to only files in metadata
    to_download = []
    for video_id in broken_ids:
        if video_id in vggsound_data:
            to_download.append((video_id, vggsound_data[video_id]['start_time']))
    
    print(f"Will download {len(to_download)} files")
    
    output_dir = Path("data/vggsound/audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar
    success_count = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_audio, vid, start, output_dir): vid
            for vid, start in to_download
        }
        
        with tqdm(total=len(to_download), desc="Downloading") as pbar:
            for future in as_completed(futures):
                video_id = futures[future]
                success, message = future.result()
                
                if success:
                    success_count += 1
                else:
                    failed.append(message)
                
                pbar.update(1)
                pbar.set_postfix(success=success_count, failed=len(failed))
    
    print(f"\n{'='*60}")
    print(f"Download complete:")
    print(f"  Success: {success_count}/{len(to_download)}")
    print(f"  Failed: {len(failed)}")
    
    if failed and len(failed) < 20:
        print("\nFailed downloads:")
        for msg in failed[:10]:
            print(f"  {msg}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download broken VGGSound files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel downloads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to download")
    parser.add_argument("--test", action="store_true", help="Test with 5 files only")
    args = parser.parse_args()
    
    if args.test:
        args.limit = 5
    
    download_broken_files(max_workers=args.workers, limit=args.limit)

if __name__ == "__main__":
    main()
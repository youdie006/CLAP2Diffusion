#!/usr/bin/env python3
"""
Download sample of Flickr-SoundNet dataset
Downloads a manageable subset for CLAP2Diffusion training
"""

import os
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import time

def download_video(url, video_id, output_dir, timeout=60):
    """Download video and extract audio/frame"""
    
    audio_path = output_dir / "audio" / f"{video_id}.wav"
    frame_path = output_dir / "frames" / f"{video_id}.jpg"
    
    # Skip if already exists
    if audio_path.exists() and frame_path.exists():
        return True, f"Already exists: {video_id}"
    
    try:
        # Download video temporarily
        temp_video = output_dir / "temp" / f"{video_id}.mp4"
        temp_video.parent.mkdir(exist_ok=True)
        
        # Download with wget (more reliable than ffmpeg for Flickr)
        wget_cmd = [
            "wget", "-q", "-O", str(temp_video),
            "--timeout=30", "--tries=2",
            url
        ]
        
        result = subprocess.run(wget_cmd, capture_output=True, timeout=timeout)
        
        if not temp_video.exists() or temp_video.stat().st_size < 1000:
            return False, f"Download failed: {video_id}"
        
        # Extract audio (16kHz for CLAP)
        audio_cmd = [
            "ffmpeg", "-i", str(temp_video),
            "-vn", "-ar", "16000", "-ac", "1",
            "-f", "wav", str(audio_path),
            "-y", "-loglevel", "error"
        ]
        subprocess.run(audio_cmd, check=True, timeout=30)
        
        # Extract middle frame
        duration_cmd = [
            "ffprobe", "-i", str(temp_video),
            "-show_entries", "format=duration",
            "-v", "quiet", "-of", "csv=p=0"
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        
        try:
            duration = float(duration_result.stdout.strip())
            middle_time = duration / 2
        except:
            middle_time = 1.0  # Default to 1 second
        
        # Extract frame at middle time
        frame_cmd = [
            "ffmpeg", "-i", str(temp_video),
            "-ss", str(middle_time),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path),
            "-y", "-loglevel", "error"
        ]
        subprocess.run(frame_cmd, check=True, timeout=30)
        
        # Clean up temp video
        temp_video.unlink()
        
        # Verify outputs
        if audio_path.exists() and frame_path.exists():
            if audio_path.stat().st_size > 10000 and frame_path.stat().st_size > 1000:
                return True, f"Success: {video_id}"
        
        return False, f"Invalid output: {video_id}"
        
    except subprocess.TimeoutExpired:
        return False, f"Timeout: {video_id}"
    except Exception as e:
        return False, f"Error {video_id}: {str(e)}"
    finally:
        # Clean up temp file if exists
        if temp_video.exists():
            temp_video.unlink()

def download_flickr_sample(
    output_dir: Path,
    num_samples: int = 1000,
    num_workers: int = 4,
    urls_file: str = "flickr_urls.txt"
):
    """Download sample of Flickr-SoundNet dataset"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    (output_dir / "temp").mkdir(exist_ok=True)
    
    print(f"Loading URLs from {urls_file}...")
    
    # Load and sample URLs
    with open(urls_file, 'r') as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    print(f"Total URLs available: {len(all_urls)}")
    
    # Random sample
    if num_samples < len(all_urls):
        selected_urls = random.sample(all_urls, num_samples)
    else:
        selected_urls = all_urls
    
    print(f"Will download {len(selected_urls)} samples")
    
    # Extract video IDs from URLs
    download_tasks = []
    for url in selected_urls:
        # Extract video ID from URL
        # Format: http://www.flickr.com/videos/USER_ID/VIDEO_ID/play/...
        parts = url.split('/')
        try:
            video_idx = parts.index('videos') + 2
            video_id = parts[video_idx]
            download_tasks.append((url, video_id))
        except:
            continue
    
    print(f"Prepared {len(download_tasks)} download tasks")
    
    # Download with progress bar
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(download_video, url, vid, output_dir): (url, vid)
            for url, vid in download_tasks
        }
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                url, vid = futures[future]
                success, message = future.result()
                
                if success:
                    successful.append(vid)
                else:
                    failed.append((vid, message))
                
                pbar.update(1)
                pbar.set_postfix(success=len(successful), failed=len(failed))
    
    # Clean up temp directory
    temp_dir = output_dir / "temp"
    if temp_dir.exists():
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()
    
    print(f"\nDownload complete!")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if failed and len(failed) < 20:
        print("\nFailed downloads:")
        for vid, msg in failed[:10]:
            print(f"  {msg}")
    
    # Create metadata splits
    create_metadata(successful, output_dir)
    
    return successful

def create_metadata(video_ids, output_dir):
    """Create train/val/test metadata"""
    
    random.shuffle(video_ids)
    
    # Split 80/10/10
    n = len(video_ids)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    
    train_ids = video_ids[:train_size]
    val_ids = video_ids[train_size:train_size + val_size]
    test_ids = video_ids[train_size + val_size:]
    
    # Create metadata for each split
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        metadata = []
        for vid in split_ids:
            metadata.append({
                "video_id": vid,
                "audio_path": f"audio/{vid}.wav",
                "image_path": f"frames/{vid}.jpg",
                "text": "",  # No text annotations available
                "source": "flickr"
            })
        
        metadata_path = output_dir / f"{split_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created {split_name} split: {len(metadata)} samples")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download Flickr-SoundNet sample")
    parser.add_argument("--output_dir", type=str, default="data/flickr_soundnet",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to download")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel downloads")
    parser.add_argument("--urls_file", type=str, default="flickr_urls.txt",
                        help="Path to URLs file")
    
    args = parser.parse_args()
    
    # Check dependencies
    for cmd in ["wget", "ffmpeg", "ffprobe"]:
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True)
        except:
            print(f"Error: {cmd} not found. Please install it first.")
            return
    
    output_dir = Path(args.output_dir)
    
    # Download samples
    successful = download_flickr_sample(
        output_dir,
        args.num_samples,
        args.num_workers,
        args.urls_file
    )
    
    print(f"\nDataset ready at: {output_dir}")
    print(f"Total usable samples: {len(successful)}")
    
    # Estimate size
    if successful:
        total_audio = sum((output_dir / "audio" / f"{vid}.wav").stat().st_size 
                         for vid in successful if (output_dir / "audio" / f"{vid}.wav").exists())
        total_frames = sum((output_dir / "frames" / f"{vid}.jpg").stat().st_size 
                          for vid in successful if (output_dir / "frames" / f"{vid}.jpg").exists())
        
        print(f"Total size: {(total_audio + total_frames) / (1024**3):.2f} GB")
        print(f"  Audio: {total_audio / (1024**3):.2f} GB")
        print(f"  Frames: {total_frames / (1024**2):.2f} MB")

if __name__ == "__main__":
    main()
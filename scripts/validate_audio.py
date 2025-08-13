#!/usr/bin/env python3
"""
Validate audio files - check if they can be loaded
"""

import os
from pathlib import Path
import shutil
import wave
import contextlib

def validate_wav_files():
    data_dir = Path("data/vggsound")
    audio_dir = data_dir / "audio"
    frames_dir = data_dir / "frames"
    
    # Create backup for corrupted files
    backup_dir = data_dir / "backup_corrupted"
    backup_audio = backup_dir / "audio"
    backup_frames = backup_dir / "frames"
    backup_audio.mkdir(parents=True, exist_ok=True)
    backup_frames.mkdir(parents=True, exist_ok=True)
    
    valid_files = []
    corrupted_files = []
    small_files = []
    
    print("Validating audio files...")
    
    for audio_file in audio_dir.glob("*.wav"):
        file_size = audio_file.stat().st_size
        
        # Check if file is too small (likely corrupted)
        if file_size < 1000:  # Less than 1KB
            small_files.append((audio_file.name, file_size))
            corrupted_files.append(audio_file.stem)
            continue
        
        # Try to open with wave module
        try:
            with contextlib.closing(wave.open(str(audio_file), 'r')) as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / float(rate)
                
                # Check if duration is reasonable (at least 1 second)
                if duration < 1.0:
                    print(f"  Too short ({duration:.2f}s): {audio_file.name}")
                    corrupted_files.append(audio_file.stem)
                else:
                    valid_files.append(audio_file.stem)
                    
        except Exception as e:
            print(f"  Error reading {audio_file.name}: {str(e)}")
            corrupted_files.append(audio_file.stem)
    
    print(f"\nValidation Results:")
    print(f"Valid files: {len(valid_files)}")
    print(f"Corrupted/invalid files: {len(corrupted_files)}")
    
    if small_files:
        print(f"\nSmall files (likely incomplete downloads):")
        for name, size in small_files[:10]:
            print(f"  - {name}: {size} bytes")
        if len(small_files) > 10:
            print(f"  ... and {len(small_files) - 10} more")
    
    # Move corrupted files and their images
    if corrupted_files:
        print(f"\nMoving {len(corrupted_files)} corrupted audio-image pairs to backup...")
        for stem in corrupted_files:
            # Move audio
            audio_file = audio_dir / f"{stem}.wav"
            if audio_file.exists():
                shutil.move(str(audio_file), str(backup_audio / audio_file.name))
            
            # Move image
            image_file = frames_dir / f"{stem}.jpg"
            if image_file.exists():
                shutil.move(str(image_file), str(backup_frames / image_file.name))
    
    # Final count
    final_wav_count = len(list(audio_dir.glob("*.wav")))
    final_jpg_count = len(list(frames_dir.glob("*.jpg")))
    
    print(f"\n=== Final Clean Dataset ===")
    print(f"Audio files (valid wav): {final_wav_count}")
    print(f"Image files (jpg): {final_jpg_count}")
    print(f"Complete pairs: {min(final_wav_count, final_jpg_count)}")
    
    return final_wav_count, final_jpg_count

if __name__ == "__main__":
    validate_wav_files()
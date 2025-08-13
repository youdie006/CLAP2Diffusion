#!/usr/bin/env python3
"""
Clean up VGGSound data - keep only wav/jpg pairs
"""

import os
from pathlib import Path
import shutil

def cleanup_vggsound_data():
    data_dir = Path("data/vggsound")
    audio_dir = data_dir / "audio"
    frames_dir = data_dir / "frames"
    
    # Create backup directory
    backup_dir = data_dir / "backup_non_wav"
    backup_audio = backup_dir / "audio"
    backup_frames = backup_dir / "frames"
    backup_audio.mkdir(parents=True, exist_ok=True)
    backup_frames.mkdir(parents=True, exist_ok=True)
    
    # Get all wav files (without extension)
    wav_files = set()
    for audio_file in audio_dir.glob("*.wav"):
        wav_files.add(audio_file.stem)
    
    print(f"Found {len(wav_files)} wav files")
    
    # Move non-wav audio files to backup
    moved_audio = 0
    for audio_file in audio_dir.iterdir():
        if audio_file.suffix != '.wav' and audio_file.is_file():
            shutil.move(str(audio_file), str(backup_audio / audio_file.name))
            moved_audio += 1
    
    print(f"Moved {moved_audio} non-wav audio files to backup")
    
    # Move images without corresponding wav files to backup
    moved_frames = 0
    kept_frames = 0
    for frame_file in frames_dir.glob("*.jpg"):
        if frame_file.stem not in wav_files:
            shutil.move(str(frame_file), str(backup_frames / frame_file.name))
            moved_frames += 1
        else:
            kept_frames += 1
    
    print(f"Moved {moved_frames} orphan image files to backup")
    print(f"Kept {kept_frames} matching image files")
    
    # Find wav files without images
    orphan_wavs = []
    for wav_stem in wav_files:
        if not (frames_dir / f"{wav_stem}.jpg").exists():
            orphan_wavs.append(wav_stem)
    
    if orphan_wavs:
        print(f"\nWarning: {len(orphan_wavs)} wav files without matching images:")
        for wav in orphan_wavs[:10]:
            print(f"  - {wav}.wav")
        if len(orphan_wavs) > 10:
            print(f"  ... and {len(orphan_wavs) - 10} more")
        
        # Move orphan wav files to backup
        for wav_stem in orphan_wavs:
            wav_file = audio_dir / f"{wav_stem}.wav"
            if wav_file.exists():
                shutil.move(str(wav_file), str(backup_audio / wav_file.name))
        print(f"Moved {len(orphan_wavs)} orphan wav files to backup")
    
    # Final count
    final_wav_count = len(list(audio_dir.glob("*.wav")))
    final_jpg_count = len(list(frames_dir.glob("*.jpg")))
    
    print(f"\n=== Final Status ===")
    print(f"Audio files (wav): {final_wav_count}")
    print(f"Image files (jpg): {final_jpg_count}")
    print(f"Matched pairs: {min(final_wav_count, final_jpg_count)}")
    print(f"Backup location: {backup_dir}")
    
    return final_wav_count, final_jpg_count

if __name__ == "__main__":
    cleanup_vggsound_data()
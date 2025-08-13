#!/usr/bin/env python3
"""
Check actual data status (not from metadata)
"""
from pathlib import Path
import json

audio_dir = Path("data/vggsound/audio")
frames_dir = Path("data/vggsound/frames")

# Count files
audio_files = list(audio_dir.glob("*.wav"))
frame_files = list(frames_dir.glob("*.jpg"))

print("="*60)
print("ACTUAL DATA STATUS")
print("="*60)

print(f"\n📊 File Counts:")
print(f"  Audio files (WAV): {len(audio_files)}")
print(f"  Frame files (JPG): {len(frame_files)}")

# Check valid sizes
valid_audio = [f for f in audio_files if f.stat().st_size > 100000]
valid_frames = [f for f in frame_files if f.stat().st_size > 2000]
small_audio = [f for f in audio_files if f.stat().st_size <= 100000]
small_frames = [f for f in frame_files if f.stat().st_size <= 2000]

print(f"\n✅ Valid Files (by size):")
print(f"  Valid audio (>100KB): {len(valid_audio)}")
print(f"  Valid frames (>2KB): {len(valid_frames)}")
print(f"  Small/broken audio: {len(small_audio)}")
print(f"  Small/broken frames: {len(small_frames)}")

# Check pairs
audio_ids = {f.stem for f in audio_files}
frame_ids = {f.stem for f in frame_files}
complete_pairs = audio_ids & frame_ids
audio_only = audio_ids - frame_ids
frame_only = frame_ids - audio_ids

print(f"\n🔗 Pairing Status:")
print(f"  Complete pairs: {len(complete_pairs)}")
print(f"  Audio only (no frame): {len(audio_only)}")
print(f"  Frame only (no audio): {len(frame_only)}")

# Valid pairs (both files are good size)
valid_audio_ids = {f.stem for f in valid_audio}
valid_frame_ids = {f.stem for f in valid_frames}
valid_pairs = valid_audio_ids & valid_frame_ids

print(f"\n✨ Valid Complete Pairs:")
print(f"  Both files valid size: {len(valid_pairs)}")

# Load metadata to check classes
metadata_files = ["train_metadata.json", "val_metadata.json", "test_metadata.json"]
all_metadata = []
for mf in metadata_files:
    mf_path = Path(f"data/vggsound/{mf}")
    if mf_path.exists():
        with open(mf_path, 'r') as f:
            all_metadata.extend(json.load(f))

if all_metadata:
    # Class distribution
    from collections import Counter
    metadata_classes = Counter(item['class'] for item in all_metadata)
    
    # Check actual files by class
    metadata_lookup = {item['video_id']: item['class'] for item in all_metadata}
    actual_classes = Counter()
    
    for video_id in valid_pairs:
        if video_id in metadata_lookup:
            actual_classes[metadata_lookup[video_id]] += 1
    
    print(f"\n📈 Class Distribution (valid pairs in metadata):")
    for cls in sorted(actual_classes.keys()):
        print(f"  {cls:15} {actual_classes[cls]:4} samples")
    
    # Files not in metadata
    new_files = valid_pairs - set(metadata_lookup.keys())
    if new_files:
        print(f"\n⚠️ Valid pairs NOT in metadata: {len(new_files)}")
        print("  (Need to regenerate metadata)")

print(f"\n{'='*60}")
print(f"SUMMARY: {len(valid_pairs)} valid pairs ready for training")
print(f"{'='*60}")
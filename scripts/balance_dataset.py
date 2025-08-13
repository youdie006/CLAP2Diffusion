#!/usr/bin/env python3
"""
Balance dataset by:
1. Downsampling majority classes
2. Finding more samples for minority classes
"""
from pathlib import Path
import json
import random
import shutil
from collections import Counter

def balance_existing_data():
    """Balance by limiting each class to max 100 samples"""
    
    audio_dir = Path("data/vggsound/audio")
    frames_dir = Path("data/vggsound/frames")
    
    # Load all metadata
    metadata_files = ["train_metadata.json", "val_metadata.json", "test_metadata.json"]
    all_metadata = []
    for mf in metadata_files:
        mf_path = Path(f"data/vggsound/{mf}")
        if mf_path.exists():
            with open(mf_path, 'r') as f:
                all_metadata.extend(json.load(f))
    
    # Group by class
    class_samples = {}
    for item in all_metadata:
        cls = item['class']
        if cls not in class_samples:
            class_samples[cls] = []
        
        # Check if files exist and are valid
        audio_path = audio_dir / f"{item['video_id']}.wav"
        frame_path = frames_dir / f"{item['video_id']}.jpg"
        
        if (audio_path.exists() and audio_path.stat().st_size > 100000 and
            frame_path.exists() and frame_path.stat().st_size > 2000):
            class_samples[cls].append(item)
    
    print("Current class distribution:")
    for cls, samples in sorted(class_samples.items()):
        print(f"  {cls:15} {len(samples):4} samples")
    
    # Balance to max 100 per class
    MAX_PER_CLASS = 100
    balanced_metadata = []
    
    print(f"\nBalancing to max {MAX_PER_CLASS} per class:")
    for cls, samples in class_samples.items():
        if len(samples) > MAX_PER_CLASS:
            # Random sample
            selected = random.sample(samples, MAX_PER_CLASS)
            print(f"  {cls}: {len(samples)} -> {MAX_PER_CLASS} (downsampled)")
        else:
            selected = samples
            print(f"  {cls}: {len(samples)} (kept all)")
        
        balanced_metadata.extend(selected)
    
    # Split into train/val/test (80/10/10)
    random.shuffle(balanced_metadata)
    total = len(balanced_metadata)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = balanced_metadata[:train_size]
    val_data = balanced_metadata[train_size:train_size + val_size]
    test_data = balanced_metadata[train_size + val_size:]
    
    # Save balanced metadata
    with open('data/vggsound/train_metadata_balanced.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    with open('data/vggsound/val_metadata_balanced.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    with open('data/vggsound/test_metadata_balanced.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nBalanced dataset created:")
    print(f"  Total samples: {total}")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Show final class distribution
    final_counter = Counter(item['class'] for item in balanced_metadata)
    print(f"\nFinal class distribution:")
    for cls, count in final_counter.most_common():
        print(f"  {cls:15} {count:4} samples ({count/total*100:.1f}%)")
    
    return balanced_metadata

def check_missing_classes():
    """Check what classes are missing"""
    target_classes = ['thunder', 'applause', 'helicopter', 'glass_breaking']
    
    print("\n" + "="*60)
    print("MISSING CLASSES TO DOWNLOAD:")
    print("="*60)
    
    for cls in target_classes:
        print(f"  - {cls}: Need to download samples")
    
    print("\nSuggestion: Create a focused download script for these classes")

if __name__ == "__main__":
    balanced = balance_existing_data()
    check_missing_classes()
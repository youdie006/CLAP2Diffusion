"""
Validate and split AudioCaps dataset
- Ensure exactly 5000 matched audio-frame pairs
- Verify file integrity
- Create train/val/test splits
"""

import os
import json
import random
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import soundfile as sf
from PIL import Image
from tqdm import tqdm

def validate_and_split_audiocaps(
    data_root: str = "/mnt/d/MyProject/CLAP2Diffusion/data/audiocaps",
    target_samples: int = 5000,
    train_ratio: float = 0.8,  # 4000 samples
    val_ratio: float = 0.1,     # 500 samples  
    test_ratio: float = 0.1,    # 500 samples
    seed: int = 42
):
    """
    Validate AudioCaps data and create clean splits
    """
    
    random.seed(seed)
    
    print("=" * 60)
    print("AudioCaps Data Validation and Splitting")
    print("=" * 60)
    
    audio_dir = Path(data_root) / "audio"
    frames_dir = Path(data_root) / "frames"
    captions_path = Path(data_root) / "captions.json"
    
    # Load captions
    print("\n1. Loading captions...")
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)
    print(f"   Loaded {len(captions_data)} captions")
    
    # Get all files
    print("\n2. Scanning files...")
    audio_files = {f.stem: f for f in audio_dir.glob("*.wav")}
    frame_files = {f.stem: f for f in frames_dir.glob("*.jpg")}
    
    print(f"   Found {len(audio_files)} audio files")
    print(f"   Found {len(frame_files)} frame files")
    
    # Find valid matched pairs
    print("\n3. Finding matched pairs...")
    matched_ids = set(audio_files.keys()) & set(frame_files.keys())
    print(f"   Initial matched pairs: {len(matched_ids)}")
    
    # Validate each pair
    print("\n4. Validating file integrity...")
    valid_pairs = []
    invalid_pairs = []
    
    for sample_id in tqdm(matched_ids, desc="Validating pairs"):
        audio_path = audio_files[sample_id]
        frame_path = frame_files[sample_id]
        
        try:
            # Check audio file
            info = sf.info(audio_path)
            if info.duration < 1.0:  # Skip very short audio
                invalid_pairs.append((sample_id, "Audio too short"))
                continue
            
            # Check image file
            img = Image.open(frame_path)
            if img.size[0] < 128 or img.size[1] < 128:  # Skip tiny images
                invalid_pairs.append((sample_id, "Image too small"))
                continue
            
            # Valid pair
            valid_pairs.append({
                "id": sample_id,
                "audio_path": str(audio_path.relative_to(data_root)),
                "frame_path": str(frame_path.relative_to(data_root)),
                "audio_duration": info.duration,
                "audio_samplerate": info.samplerate,
                "image_size": img.size,
                "caption": captions_data.get(sample_id, "Audio-visual content")
            })
            
        except Exception as e:
            invalid_pairs.append((sample_id, str(e)))
    
    print(f"\n   Valid pairs: {len(valid_pairs)}")
    print(f"   Invalid pairs: {len(invalid_pairs)}")
    
    if len(invalid_pairs) > 0:
        print("\n   Sample of invalid pairs:")
        for sample_id, reason in invalid_pairs[:5]:
            print(f"     {sample_id}: {reason}")
    
    # Select exactly target_samples
    if len(valid_pairs) < target_samples:
        print(f"\n⚠️  Warning: Only {len(valid_pairs)} valid pairs found (target: {target_samples})")
        target_samples = len(valid_pairs)
    else:
        print(f"\n5. Selecting {target_samples} samples from {len(valid_pairs)} valid pairs...")
        valid_pairs = random.sample(valid_pairs, target_samples)
    
    # Analyze captions for better categorization
    print("\n6. Analyzing captions for categorization...")
    
    # Enhanced keyword lists for better categorization
    category_keywords = {
        "speech": ["speak", "talk", "voice", "say", "man", "woman", "child", "person", "people", 
                  "speech", "conversation", "laugh", "cry", "scream", "shout", "whisper"],
        "music": ["music", "song", "instrument", "play", "sing", "melody", "rhythm", "beat",
                 "piano", "guitar", "drum", "violin", "orchestra", "band"],
        "nature": ["wind", "water", "rain", "thunder", "ocean", "wave", "bird", "chirp", 
                  "forest", "tree", "river", "storm", "nature", "weather"],
        "urban": ["car", "vehicle", "traffic", "street", "city", "siren", "horn", "engine",
                 "motor", "drive", "road", "bus", "truck", "motorcycle", "train"],
        "mechanical": ["machine", "engine", "motor", "drill", "saw", "tool", "mechanical",
                      "click", "beep", "alarm", "bell", "typing", "keyboard"],
        "animal": ["dog", "cat", "animal", "bark", "meow", "roar", "chirp", "bird",
                  "duck", "cow", "horse", "pig", "sheep", "goat", "chicken"],
        "ambient": ["background", "room", "silence", "quiet", "hum", "buzz", "white noise"]
    }
    
    # Categorize each sample
    for sample in valid_pairs:
        caption = sample["caption"].lower()
        
        # Score each category
        category_scores = {}
        for cat, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in caption)
            if score > 0:
                category_scores[cat] = score
        
        # Assign category with highest score
        if category_scores:
            sample["category"] = max(category_scores, key=category_scores.get)
        else:
            sample["category"] = "general"
        
        # Assign hierarchy based on category
        hierarchy_map = {
            "speech": {"primary": "foreground", "weights": [0.8, 0.15, 0.05]},
            "music": {"primary": "background", "weights": [0.3, 0.5, 0.2]},
            "nature": {"primary": "ambience", "weights": [0.1, 0.3, 0.6]},
            "urban": {"primary": "foreground", "weights": [0.6, 0.3, 0.1]},
            "mechanical": {"primary": "foreground", "weights": [0.7, 0.2, 0.1]},
            "animal": {"primary": "foreground", "weights": [0.6, 0.3, 0.1]},
            "ambient": {"primary": "ambience", "weights": [0.1, 0.2, 0.7]},
            "general": {"primary": "background", "weights": [0.33, 0.34, 0.33]}
        }
        
        hierarchy = hierarchy_map[sample["category"]]
        sample["hierarchy"] = hierarchy
    
    # Create splits
    print("\n7. Creating train/val/test splits...")
    
    # Shuffle for random split
    random.shuffle(valid_pairs)
    
    train_size = int(target_samples * train_ratio)  # 4000
    val_size = int(target_samples * val_ratio)      # 500
    test_size = target_samples - train_size - val_size  # 500
    
    train_samples = valid_pairs[:train_size]
    val_samples = valid_pairs[train_size:train_size + val_size]
    test_samples = valid_pairs[train_size + val_size:]
    
    # Add split labels
    for sample in train_samples:
        sample["split"] = "train"
    for sample in val_samples:
        sample["split"] = "val"
    for sample in test_samples:
        sample["split"] = "test"
    
    # Combine all samples
    all_samples = train_samples + val_samples + test_samples
    
    # Calculate statistics
    print("\n8. Computing statistics...")
    
    stats = {
        "total_samples": len(all_samples),
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples)
        },
        "categories": {},
        "hierarchy_distribution": {"foreground": 0, "background": 0, "ambience": 0}
    }
    
    # Category distribution per split
    for split in ["train", "val", "test"]:
        split_samples = [s for s in all_samples if s["split"] == split]
        for cat in category_keywords.keys():
            cat_count = sum(1 for s in split_samples if s["category"] == cat)
            if cat not in stats["categories"]:
                stats["categories"][cat] = {}
            stats["categories"][cat][split] = cat_count
    
    # Add general category if exists
    general_count = sum(1 for s in all_samples if s["category"] == "general")
    if general_count > 0:
        stats["categories"]["general"] = {
            "train": sum(1 for s in train_samples if s["category"] == "general"),
            "val": sum(1 for s in val_samples if s["category"] == "general"),
            "test": sum(1 for s in test_samples if s["category"] == "general")
        }
    
    # Hierarchy distribution
    for sample in all_samples:
        stats["hierarchy_distribution"][sample["hierarchy"]["primary"]] += 1
    
    # Create final metadata
    metadata = {
        "dataset": "AudioCaps",
        "version": "v4_validated",
        "total_samples": len(all_samples),
        "target_samples": target_samples,
        "samples": all_samples,
        "statistics": stats,
        "validation_info": {
            "initial_pairs": len(matched_ids),
            "valid_pairs": len(valid_pairs),
            "invalid_pairs": len(invalid_pairs),
            "selected_samples": len(all_samples)
        }
    }
    
    # Save metadata
    output_path = Path(data_root) / "metadata_validated.json"
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation and Splitting Complete!")
    print("=" * 60)
    
    print(f"\nFinal Dataset:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Train: {len(train_samples)} ({len(train_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Val: {len(val_samples)} ({len(val_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Test: {len(test_samples)} ({len(test_samples)/len(all_samples)*100:.1f}%)")
    
    print(f"\nCategory Distribution:")
    for cat, splits in stats["categories"].items():
        total = sum(splits.values())
        if total > 0:
            print(f"  {cat:12s}: {total:4d} (train:{splits.get('train',0):4d}, val:{splits.get('val',0):3d}, test:{splits.get('test',0):3d})")
    
    print(f"\nHierarchy Distribution:")
    for level, count in stats["hierarchy_distribution"].items():
        print(f"  {level:12s}: {count:4d} ({count/len(all_samples)*100:.1f}%)")
    
    print(f"\nOutput saved to: {output_path}")
    
    # Create split lists for easy access
    print("\n9. Creating split lists...")
    splits_dir = Path(data_root) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    for split_name, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_file = splits_dir / f"{split_name}_ids.txt"
        with open(split_file, 'w') as f:
            for sample in split_samples:
                f.write(f"{sample['id']}\n")
        print(f"  Created {split_file}")
    
    return metadata


if __name__ == "__main__":
    metadata = validate_and_split_audiocaps()
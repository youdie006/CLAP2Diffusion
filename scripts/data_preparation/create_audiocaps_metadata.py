"""
Create metadata for AudioCaps dataset
Organized script for data preparation
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

def create_audiocaps_metadata(
    data_root: str = "/mnt/d/MyProject/CLAP2Diffusion/data/audiocaps",
    output_path: str = None
):
    """
    Create metadata JSON for AudioCaps dataset with hierarchical categories
    """
    
    if output_path is None:
        output_path = os.path.join(data_root, "metadata.json")
    
    audio_dir = os.path.join(data_root, "audio")
    frames_dir = os.path.join(data_root, "frames")
    captions_path = os.path.join(data_root, "captions.json")
    
    print("=" * 50)
    print("AudioCaps Metadata Generation")
    print("=" * 50)
    
    # Load captions if available
    captions_data = {}
    if os.path.exists(captions_path):
        print(f"Loading captions from {captions_path}")
        with open(captions_path, 'r') as f:
            captions_data = json.load(f)
            print(f"  Loaded {len(captions_data)} captions")
    
    # Get all audio and frame files
    print("\nScanning files...")
    audio_files = set([f.replace('.wav', '') for f in os.listdir(audio_dir) if f.endswith('.wav')])
    frame_files = set([f.replace('.jpg', '') for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    print(f"  Found {len(audio_files)} audio files")
    print(f"  Found {len(frame_files)} frame files")
    
    # Find matching pairs
    matched_ids = list(audio_files.intersection(frame_files))
    print(f"  Matched pairs: {len(matched_ids)}")
    
    # Create metadata entries
    metadata = {
        "dataset": "AudioCaps",
        "version": "v4_hybrid",
        "total_samples": len(matched_ids),
        "samples": []
    }
    
    # Hierarchical categories for AudioCaps (V4 specific)
    hierarchy_categories = {
        "speech": {
            "weights": [0.8, 0.15, 0.05],  # foreground, background, ambience
            "primary": "foreground",
            "keywords": ["speech", "talk", "voice", "speak", "say", "person", "man", "woman", "child"]
        },
        "music": {
            "weights": [0.3, 0.5, 0.2],
            "primary": "background",
            "keywords": ["music", "song", "instrument", "play", "sing", "melody", "rhythm", "beat"]
        },
        "nature": {
            "weights": [0.1, 0.3, 0.6],
            "primary": "ambience",
            "keywords": ["nature", "wind", "water", "rain", "bird", "thunder", "ocean", "forest"]
        },
        "urban": {
            "weights": [0.6, 0.3, 0.1],
            "primary": "foreground",
            "keywords": ["city", "traffic", "car", "street", "urban", "siren", "horn", "engine"]
        },
        "mechanical": {
            "weights": [0.7, 0.2, 0.1],
            "primary": "foreground", 
            "keywords": ["machine", "engine", "motor", "mechanical", "drill", "saw", "tool"]
        },
        "animal": {
            "weights": [0.6, 0.3, 0.1],
            "primary": "foreground",
            "keywords": ["dog", "cat", "animal", "bark", "meow", "roar", "chirp"]
        }
    }
    
    # Composition types with weights for balanced distribution
    composition_distribution = {
        "matching": 0.4,      # 40% - Direct audio-text alignment
        "complementary": 0.3,  # 30% - Related but not exact
        "creative": 0.2,      # 20% - Creative interpretation
        "contradictory": 0.1  # 10% - Minimal alignment
    }
    
    print("\nProcessing samples...")
    for idx, sample_id in enumerate(tqdm(matched_ids, desc="Creating metadata")):
        # Get caption if available
        caption = "Audio-visual sample"
        if isinstance(captions_data, dict):
            if sample_id in captions_data:
                caption = captions_data[sample_id]
            elif 'captions' in captions_data and sample_id in captions_data['captions']:
                caption = captions_data['captions'][sample_id]
        
        # Determine category based on keywords in caption
        category = "general"
        max_score = 0
        
        for cat, info in hierarchy_categories.items():
            score = sum(1 for kw in info["keywords"] if kw in caption.lower())
            if score > max_score:
                max_score = score
                category = cat
        
        # Assign composition type based on distribution
        comp_rand = random.random()
        cumulative = 0
        composition_type = "matching"
        for comp_type, weight in composition_distribution.items():
            cumulative += weight
            if comp_rand < cumulative:
                composition_type = comp_type
                break
        
        # Create hierarchical structure for V4
        hierarchy_info = hierarchy_categories.get(category, hierarchy_categories["nature"])
        
        # Create sample entry
        sample = {
            "id": sample_id,
            "audio_path": f"audio/{sample_id}.wav",
            "frame_path": f"frames/{sample_id}.jpg",
            "caption": caption,
            "category": category,
            "hierarchy": {
                "primary": hierarchy_info["primary"],
                "weights": hierarchy_info["weights"],
                "levels": ["foreground", "background", "ambience"]
            },
            "composition_type": composition_type,
            "f_multiplier": {
                "matching": 0.8,
                "complementary": 0.6,
                "creative": 0.4,
                "contradictory": 0.2
            }[composition_type],
            "duration": 10.0,  # AudioCaps clips are typically 10 seconds
            "sample_rate": 44100
        }
        
        metadata["samples"].append(sample)
    
    # Split into train/val/test (80/10/10)
    print("\nCreating data splits...")
    random.shuffle(metadata["samples"])
    total = len(metadata["samples"])
    train_split = int(total * 0.8)
    val_split = int(total * 0.1)
    
    # Add split information
    for i, sample in enumerate(metadata["samples"]):
        if i < train_split:
            sample["split"] = "train"
        elif i < train_split + val_split:
            sample["split"] = "val"
        else:
            sample["split"] = "test"
    
    # Calculate detailed statistics
    metadata["statistics"] = {
        "total_samples": total,
        "splits": {
            "train": train_split,
            "val": val_split,
            "test": total - train_split - val_split
        },
        "categories": {},
        "composition_types": {},
        "hierarchy_distribution": {
            "foreground": 0,
            "background": 0,
            "ambience": 0
        }
    }
    
    # Count categories and composition types
    for sample in metadata["samples"]:
        # Category stats
        cat = sample["category"]
        if cat not in metadata["statistics"]["categories"]:
            metadata["statistics"]["categories"][cat] = {"total": 0, "train": 0, "val": 0, "test": 0}
        metadata["statistics"]["categories"][cat]["total"] += 1
        metadata["statistics"]["categories"][cat][sample["split"]] += 1
        
        # Composition type stats
        comp = sample["composition_type"]
        if comp not in metadata["statistics"]["composition_types"]:
            metadata["statistics"]["composition_types"][comp] = 0
        metadata["statistics"]["composition_types"][comp] += 1
        
        # Hierarchy stats
        metadata["statistics"]["hierarchy_distribution"][sample["hierarchy"]["primary"]] += 1
    
    # Save metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Metadata Generation Complete!")
    print("=" * 50)
    print(f"Output: {output_path}")
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Train: {train_split} ({train_split/total*100:.1f}%)")
    print(f"  Val: {val_split} ({val_split/total*100:.1f}%)")
    print(f"  Test: {total - train_split - val_split} ({(total - train_split - val_split)/total*100:.1f}%)")
    
    print(f"\nCategory Distribution:")
    for cat, stats in metadata["statistics"]["categories"].items():
        print(f"  {cat}: {stats['total']} samples")
    
    print(f"\nComposition Type Distribution:")
    for comp, count in metadata["statistics"]["composition_types"].items():
        print(f"  {comp}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nHierarchy Primary Distribution:")
    for level, count in metadata["statistics"]["hierarchy_distribution"].items():
        print(f"  {level}: {count} ({count/total*100:.1f}%)")
    
    return metadata


if __name__ == "__main__":
    # Run metadata generation
    metadata = create_audiocaps_metadata()
    
    # Optionally create a backup
    backup_path = "/mnt/d/MyProject/CLAP2Diffusion/data/audiocaps/metadata_backup.json"
    print(f"\nCreating backup at {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(metadata, f, indent=2)
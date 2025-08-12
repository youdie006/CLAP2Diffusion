#!/usr/bin/env python3
"""
Create metadata JSON files from downloaded VGGSound data
"""

import json
import os
from pathlib import Path
import random
import csv

def load_vggsound_csv(csv_path):
    """Load VGGSound CSV and create label mapping"""
    label_map = {}
    
    # Target classes mapping
    target_classes = {
        "thunder": ["thunder", "thunderstorm"],
        "ocean_waves": ["waves", "ocean", "sea", "surf"],
        "fire": ["fire", "burning", "crackling", "flame"],
        "applause": ["applause", "clapping", "cheering"],
        "siren": ["siren", "alarm", "ambulance", "police"],
        "helicopter": ["helicopter", "chopper"],
        "dog_barking": ["dog", "barking", "bark"],
        "rain": ["rain", "raining", "rainfall"],
        "glass_breaking": ["glass", "breaking", "shatter", "smash"],
        "engine": ["engine", "motor", "vehicle", "car", "motorcycle"]
    }
    
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    video_id = row[0]
                    label = row[2].lower()
                    
                    # Map to target class
                    for target_class, keywords in target_classes.items():
                        if any(keyword in label for keyword in keywords):
                            label_map[video_id] = target_class
                            break
                    
                    # Default mapping if no match
                    if video_id not in label_map:
                        label_map[video_id] = "other"
    
    return label_map

def create_metadata():
    """Create metadata from downloaded files"""
    
    vggsound_dir = Path('data/vggsound')
    audio_dir = vggsound_dir / 'audio'
    frames_dir = vggsound_dir / 'frames'
    csv_path = vggsound_dir / 'vggsound.csv'
    
    # Load label mapping
    label_map = load_vggsound_csv(csv_path)
    
    # Find matched audio-image pairs
    audio_files = list(audio_dir.glob('*.wav'))
    matched_files = []
    
    for audio_file in audio_files:
        video_id = audio_file.stem
        frame_file = frames_dir / f'{video_id}.jpg'
        
        if frame_file.exists():
            matched_files.append({
                'video_id': video_id,
                'audio_path': f'audio/{video_id}.wav',
                'image_path': f'frames/{video_id}.jpg',
                'class': label_map.get(video_id, 'other'),
                'text': f"A scene with {label_map.get(video_id, 'ambient').replace('_', ' ')} sound"
            })
    
    print(f'Found {len(matched_files)} matched audio-image pairs')
    
    # Count classes
    class_counts = {}
    for item in matched_files:
        cls = item['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print('Class distribution:')
    for cls, count in sorted(class_counts.items()):
        print(f'  {cls}: {count} samples')
    
    # Shuffle and split data
    random.shuffle(matched_files)
    
    total = len(matched_files)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = matched_files[:train_size]
    val_data = matched_files[train_size:train_size + val_size]
    test_data = matched_files[train_size + val_size:]
    
    # Save metadata
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        output_path = vggsound_dir / f'{split_name}_metadata.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f'Created {split_name} split with {len(split_data)} samples')
    
    print('Metadata creation complete!')

if __name__ == '__main__':
    create_metadata()
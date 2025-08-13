#!/usr/bin/env python3
"""Find valid audio samples for each class"""

import json
import os
from pathlib import Path

# Load metadata
with open('data/vggsound/train_metadata.json', 'r') as f:
    metadata = json.load(f)

# Target classes
target_classes = ['thunder', 'ocean_waves', 'dog_barking', 'fire', 'rain', 
                  'applause', 'helicopter', 'siren', 'engine']

# Find valid samples
valid_samples = {}
for cls in target_classes:
    for item in metadata:
        if item['class'] == cls:
            audio_file = Path('data/vggsound') / item['audio_path']
            if audio_file.exists() and audio_file.stat().st_size > 100000:  # > 100KB
                valid_samples[cls] = {
                    'audio': str(audio_file),
                    'size': f"{audio_file.stat().st_size / 1024 / 1024:.1f}MB",
                    'video_id': item['video_id']
                }
                break

# Print results
print("Valid samples found:")
print("-" * 50)
for cls, info in valid_samples.items():
    print(f"{cls:15} : {info['video_id']:15} ({info['size']})")
    
print("\nMissing classes:", [c for c in target_classes if c not in valid_samples])
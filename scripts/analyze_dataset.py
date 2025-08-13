#!/usr/bin/env python3
"""Analyze VGGSound dataset issues"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_dataset():
    # Load metadata
    with open('data/vggsound/train_metadata.json', 'r') as f:
        train_data = json.load(f)
    with open('data/vggsound/val_metadata.json', 'r') as f:
        val_data = json.load(f)
    with open('data/vggsound/test_metadata.json', 'r') as f:
        test_data = json.load(f)
    
    all_data = train_data + val_data + test_data
    
    # Analyze files
    stats = {
        'total': len(all_data),
        'by_class': defaultdict(lambda: {'total': 0, 'valid': 0, 'broken': 0, 'missing': 0}),
        'broken_files': [],
        'valid_files': [],
        'missing_files': [],
        'file_sizes': defaultdict(int)
    }
    
    print("Analyzing dataset...")
    print("-" * 60)
    
    for item in all_data:
        cls = item['class']
        audio_path = Path('data/vggsound') / item['audio_path']
        image_path = Path('data/vggsound') / item['image_path']
        
        stats['by_class'][cls]['total'] += 1
        
        # Check audio file
        if audio_path.exists():
            size = audio_path.stat().st_size
            if size < 1000:  # Less than 1KB - likely broken
                stats['by_class'][cls]['broken'] += 1
                stats['broken_files'].append({
                    'path': str(audio_path),
                    'size': size,
                    'class': cls,
                    'video_id': item['video_id']
                })
                stats['file_sizes'][size] += 1
            else:
                stats['by_class'][cls]['valid'] += 1
                stats['valid_files'].append({
                    'path': str(audio_path),
                    'size': size,
                    'class': cls,
                    'video_id': item['video_id']
                })
        else:
            stats['by_class'][cls]['missing'] += 1
            stats['missing_files'].append(str(audio_path))
    
    # Print results
    print(f"Total samples in metadata: {stats['total']}")
    print(f"Valid audio files: {len(stats['valid_files'])}")
    print(f"Broken audio files: {len(stats['broken_files'])}")
    print(f"Missing audio files: {len(stats['missing_files'])}")
    print()
    
    # File size distribution for broken files
    print("Broken file sizes:")
    for size, count in sorted(stats['file_sizes'].items()):
        if size < 1000:
            print(f"  {size} bytes: {count} files")
    print()
    
    # Class-wise analysis
    print("Class-wise breakdown:")
    print(f"{'Class':<15} {'Total':<8} {'Valid':<8} {'Broken':<8} {'Missing':<8} {'Valid%':<8}")
    print("-" * 60)
    
    for cls in sorted(stats['by_class'].keys()):
        info = stats['by_class'][cls]
        valid_pct = (info['valid'] / info['total'] * 100) if info['total'] > 0 else 0
        print(f"{cls:<15} {info['total']:<8} {info['valid']:<8} {info['broken']:<8} {info['missing']:<8} {valid_pct:<8.1f}%")
    
    # Identify classes with worst corruption
    print("\nMost affected classes (by broken file %):")
    affected = []
    for cls, info in stats['by_class'].items():
        if info['total'] > 0:
            broken_pct = info['broken'] / info['total'] * 100
            affected.append((cls, broken_pct, info['broken'], info['total']))
    
    for cls, pct, broken, total in sorted(affected, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cls}: {pct:.1f}% broken ({broken}/{total})")
    
    # Check training set specifically
    print("\nTraining set analysis:")
    train_valid = sum(1 for item in train_data 
                     if (Path('data/vggsound') / item['audio_path']).exists() 
                     and (Path('data/vggsound') / item['audio_path']).stat().st_size > 1000)
    train_broken = sum(1 for item in train_data 
                      if (Path('data/vggsound') / item['audio_path']).exists() 
                      and (Path('data/vggsound') / item['audio_path']).stat().st_size <= 1000)
    
    print(f"  Total: {len(train_data)}")
    print(f"  Valid: {train_valid} ({train_valid/len(train_data)*100:.1f}%)")
    print(f"  Broken: {train_broken} ({train_broken/len(train_data)*100:.1f}%)")
    
    # Sample broken files
    print("\nSample broken files (all same size?):")
    for bf in stats['broken_files'][:5]:
        print(f"  {bf['video_id']}: {bf['size']} bytes ({bf['class']})")
    
    return stats

if __name__ == "__main__":
    stats = analyze_dataset()
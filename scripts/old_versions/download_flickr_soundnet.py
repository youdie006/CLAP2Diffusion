#!/usr/bin/env python3
"""
Download Flickr-SoundNet dataset for CLAP2Diffusion
Dataset contains 144k audio-image pairs from Flickr videos
"""

import os
import json
import argparse
from pathlib import Path
import subprocess
import requests
from tqdm import tqdm
import hashlib
import tarfile
import zipfile

def download_file(url, dest_path, chunk_size=8192):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path

def download_flickr_soundnet(output_dir: Path):
    """Download Flickr-SoundNet dataset"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (from official sources)
    # Note: These are example URLs - actual URLs need to be obtained from dataset authors
    dataset_info = {
        "metadata": {
            "url": "https://github.com/cvondrick/soundnet/raw/master/flickr_soundnet.txt",
            "desc": "Flickr-SoundNet metadata"
        },
        "features": {
            "url": "http://data.csail.mit.edu/soundnet/flickr_soundnet_features.tar.gz",
            "desc": "Pre-extracted audio features",
            "size": "5GB"
        }
    }
    
    print("=" * 60)
    print("Flickr-SoundNet Dataset Download")
    print("=" * 60)
    print("This dataset contains 144k audio-image pairs from Flickr")
    print()
    
    # Step 1: Download metadata
    print("Step 1: Downloading metadata...")
    metadata_path = output_dir / "flickr_soundnet_metadata.txt"
    
    try:
        download_file(dataset_info["metadata"]["url"], metadata_path)
    except:
        print("Note: Direct download links may require registration.")
        print("Please visit: https://github.com/cvondrick/soundnet")
        print("And follow instructions to get dataset access.")
        return
    
    # Step 2: Parse metadata and prepare download list
    print("\nStep 2: Parsing metadata...")
    
    audio_urls = []
    image_urls = []
    
    with open(metadata_path, 'r') as f:
        for line in f:
            # Parse Flickr video URLs
            # Format: flickr_video_id, audio_url, image_url
            parts = line.strip().split(',')
            if len(parts) >= 2:
                video_id = parts[0]
                # Construct Flickr URLs
                audio_urls.append({
                    "id": video_id,
                    "url": f"https://www.flickr.com/video_download.gne?id={video_id}"
                })
                # Image frame extraction needed from video
    
    print(f"Found {len(audio_urls)} entries in metadata")
    
    # Step 3: Create directory structure
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    
    # Step 4: Download samples (limited for testing)
    print("\nStep 3: Downloading samples...")
    print("Note: Full dataset download requires:")
    print("- Flickr API access for videos")
    print("- FFmpeg for audio/frame extraction")
    print("- ~50GB storage space")
    
    # Create download instructions
    instructions = {
        "dataset": "Flickr-SoundNet",
        "total_samples": 144000,
        "source": "Flickr videos with natural sound",
        "download_steps": [
            "1. Register for Flickr API access",
            "2. Use flickr_video_ids to download videos",
            "3. Extract audio tracks (16kHz WAV)",
            "4. Extract middle frame as image",
            "5. Create train/val/test splits (80/10/10)"
        ],
        "audio_format": {
            "sample_rate": 16000,
            "channels": 1,
            "duration": "variable (2-10 seconds)",
            "format": "wav"
        },
        "image_format": {
            "size": "256x256 or original",
            "format": "jpg"
        }
    }
    
    with open(output_dir / "download_instructions.json", 'w') as f:
        json.dump(instructions, f, indent=2)
    
    print("\nDownload instructions saved to:", output_dir / "download_instructions.json")
    
    # Alternative: Use pre-processed version if available
    print("\n" + "=" * 60)
    print("Alternative: Pre-processed Dataset")
    print("=" * 60)
    print("For faster setup, consider using pre-processed versions:")
    print("1. SoundNet pre-trained features")
    print("2. AudioSet with synchronized frames")
    print("3. VGGSound (with proper YouTube downloads)")
    
    return output_dir

def create_flickr_loader(data_dir: Path):
    """Create data loader for Flickr-SoundNet"""
    
    loader_code = '''#!/usr/bin/env python3
"""
Flickr-SoundNet data loader for CLAP2Diffusion
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from PIL import Image
import numpy as np

class FlickrSoundNetDataset(Dataset):
    """Dataset for Flickr-SoundNet audio-image pairs"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 audio_duration: float = 4.0,
                 sample_rate: int = 16000,
                 image_size: int = 256):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.image_size = image_size
        
        # Load metadata
        metadata_path = self.data_dir / f"{split}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} {split} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_dir / item["audio_path"]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Crop/pad to fixed duration
        target_length = int(self.audio_duration * self.sample_rate)
        if waveform.shape[1] > target_length:
            # Random crop
            start = random.randint(0, waveform.shape[1] - target_length)
            waveform = waveform[:, start:start + target_length]
        else:
            # Pad with zeros
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Load image
        image_path = self.data_dir / item["image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # Resize image
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return {
            "audio": waveform.squeeze(0),  # (num_samples,)
            "image": image,  # (3, H, W)
            "text": item.get("caption", ""),
            "id": item.get("id", f"sample_{idx}")
        }

def create_dataloader(data_dir: str, 
                     batch_size: int = 32,
                     num_workers: int = 4):
    """Create data loaders for all splits"""
    
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = FlickrSoundNetDataset(data_dir, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders
'''
    
    loader_path = data_dir / "flickr_soundnet_loader.py"
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    print(f"Data loader created: {loader_path}")
    return loader_path

def main():
    parser = argparse.ArgumentParser(description="Download Flickr-SoundNet dataset")
    parser.add_argument("--output_dir", type=str, default="data/flickr_soundnet",
                        help="Output directory for dataset")
    parser.add_argument("--create_loader", action="store_true",
                        help="Create PyTorch data loader")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Download dataset
    download_flickr_soundnet(output_dir)
    
    # Create data loader
    if args.create_loader:
        create_flickr_loader(output_dir)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Follow download_instructions.json to get full dataset")
    print("2. Or use alternative pre-processed datasets")
    print("3. Update training config to use new dataset path")
    print("4. Run training with: python scripts/train.py --config configs/training_config.json")

if __name__ == "__main__":
    main()
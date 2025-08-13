"""
Optimized dataset with caching and efficient loading
"""

import json
import random
from pathlib import Path
from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
import torchaudio
from PIL import Image
import numpy as np
from functools import lru_cache
import hashlib


class OptimizedAudioImageDataset(Dataset):
    """Optimized dataset with caching and prefetching"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        audio_duration: float = 10.0,
        sample_rate: int = 48000,
        image_size: int = 512,
        use_augmentation: bool = True,
        cache_audio: bool = True,
        cache_size: int = 1000
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.cache_audio = cache_audio
        
        # Load metadata
        metadata_path = self.data_dir / f"{split}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter valid entries
        self.valid_indices = []
        for idx, item in enumerate(self.metadata):
            audio_path = self.data_dir / item['audio_path']
            image_path = self.data_dir / item['image_path']
            if audio_path.exists() and image_path.exists():
                self.valid_indices.append(idx)
        
        print(f"Loaded {len(self.valid_indices)} valid {split} samples")
        
        # Initialize augmentation
        if self.use_augmentation:
            from src.data.augmentation import AudioAugmentation, ImageAugmentation
            self.audio_aug = AudioAugmentation()
            self.image_aug = ImageAugmentation()
        
        # Setup caching
        if self.cache_audio:
            self._setup_cache(cache_size)
    
    def _setup_cache(self, cache_size):
        """Setup LRU cache for audio loading"""
        @lru_cache(maxsize=cache_size)
        def _load_audio_cached(path_str):
            waveform, sr = torchaudio.load(path_str)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform
        
        self._load_audio_cached = _load_audio_cached
    
    def _load_audio(self, audio_path):
        """Load and process audio with caching"""
        if self.cache_audio:
            waveform = self._load_audio_cached(str(audio_path))
        else:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
        
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Crop or pad to fixed duration
        target_length = int(self.audio_duration * self.sample_rate)
        if waveform.shape[1] > target_length:
            # Random crop for training, center crop for val/test
            if self.split == 'train':
                start = random.randint(0, waveform.shape[1] - target_length)
            else:
                start = (waveform.shape[1] - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        else:
            # Pad with zeros
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform.squeeze(0)  # Return 1D tensor
    
    def _load_image(self, image_path):
        """Load and process image efficiently"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize with LANCZOS for better quality
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual index
        actual_idx = self.valid_indices[idx]
        item = self.metadata[actual_idx]
        
        # Load audio and image
        audio_path = self.data_dir / item['audio_path']
        image_path = self.data_dir / item['image_path']
        
        audio = self._load_audio(audio_path)
        image = self._load_image(image_path)
        
        # Apply augmentation if training
        if self.use_augmentation:
            audio = self.audio_aug(audio, self.sample_rate)
            image = self.image_aug(image)
        
        # Get text description
        text = item.get('text', '')
        if not text and 'label' in item:
            text = f"A scene with {item['label']} sound"
        
        return {
            'audio': audio,
            'image': image,
            'text': text,
            'id': item.get('video_id', f'sample_{idx}'),
            'class': item.get('class', 'unknown')  # Add class label for weighted loss
        }


class DataPrefetcher:
    """Prefetch data to GPU for faster training"""
    
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            # Move data to GPU asynchronously
            self.next_batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in self.next_batch.items()
            }
    
    def __iter__(self):
        return self
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        
        if batch is None:
            raise StopIteration
        
        self.preload()
        return batch


def create_optimized_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_to_gpu: bool = True,
    device: str = 'cuda'
):
    """Create optimized dataloaders with prefetching"""
    
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = OptimizedAudioImageDataset(
            data_dir=data_dir,
            split=split,
            use_augmentation=(split == 'train')
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if split == 'train' else 1,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=(split == 'train'),
            persistent_workers=True if num_workers > 0 else False
        )
        
        # Add GPU prefetching for training
        if prefetch_to_gpu and split == 'train' and torch.cuda.is_available():
            loader = DataPrefetcher(loader, device)
        
        loaders[split] = loader
    
    return loaders
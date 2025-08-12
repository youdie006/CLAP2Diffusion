"""
Audio-Image Dataset for CLAP2Diffusion Training
Supports VGGSound, AudioSet, and custom datasets
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import librosa
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import webdataset as wds
from torchvision import transforms


class AudioImageDataset(Dataset):
    """Dataset for audio-image pairs with text descriptions."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        audio_sample_rate: int = 48000,
        audio_duration: float = 10.0,
        image_size: int = 512,
        augment: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing data
            split: Dataset split (train/val/test)
            audio_sample_rate: Audio sampling rate
            audio_duration: Audio clip duration in seconds
            image_size: Target image size
            augment: Whether to apply augmentations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.image_size = image_size
        self.augment = augment and split == "train"
        
        # Load metadata
        metadata_path = self.data_root / f"{split}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Setup transforms
        self.image_transform = self._get_image_transform()
        
        # Target audio classes (10 categories)
        self.target_classes = [
            "thunder", "ocean_waves", "fire", "applause", "siren",
            "helicopter", "dog_barking", "rain", "glass_breaking", "engine"
        ]
        
        # Filter data by target classes if specified
        if self.target_classes:
            self.metadata = [
                item for item in self.metadata
                if any(cls in item.get('class', '').lower() for cls in self.target_classes)
            ]
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _get_image_transform(self):
        """Get image transformation pipeline."""
        transform_list = []
        
        if self.augment:
            transform_list.extend([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
            ])
        else:
            transform_list.extend([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        # Load audio
        waveform, sr = librosa.load(
            audio_path,
            sr=self.audio_sample_rate,
            duration=self.audio_duration,
            mono=True
        )
        
        # Pad or trim to exact duration
        target_length = int(self.audio_sample_rate * self.audio_duration)
        if len(waveform) < target_length:
            # Pad with zeros
            waveform = np.pad(
                waveform,
                (0, target_length - len(waveform)),
                mode='constant'
            )
        else:
            # Trim or randomly crop
            if self.augment:
                start = random.randint(0, len(waveform) - target_length)
                waveform = waveform[start:start + target_length]
            else:
                waveform = waveform[:target_length]
        
        # Audio augmentation
        if self.augment:
            # Random gain
            if random.random() < 0.5:
                gain = random.uniform(0.8, 1.2)
                waveform = waveform * gain
            
            # Add noise
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.01, waveform.shape)
                waveform = waveform + noise
        
        return waveform.astype(np.float32)
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image
        """
        image = Image.open(image_path).convert('RGB')
        return image
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with audio, image, and text data
        """
        item = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_root / item['audio_path']
        audio = self.load_audio(str(audio_path))
        audio = torch.from_numpy(audio)
        
        # Load image
        image_path = self.data_root / item['image_path']
        image = self.load_image(str(image_path))
        image = self.image_transform(image)
        
        # Get text description
        text = item.get('text', item.get('class', ''))
        
        # Additional metadata
        metadata = {
            'class': item.get('class', ''),
            'video_id': item.get('video_id', ''),
            'timestamp': item.get('timestamp', 0)
        }
        
        return {
            'audio': audio,
            'image': image,
            'text': text,
            'metadata': metadata
        }


class WebDatasetLoader:
    """WebDataset loader for large-scale training."""
    
    def __init__(
        self,
        urls: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        audio_sample_rate: int = 48000,
        audio_duration: float = 10.0,
        image_size: int = 512,
        shuffle_buffer: int = 1000
    ):
        """
        Initialize WebDataset loader.
        
        Args:
            urls: List of tar file URLs or paths
            batch_size: Batch size
            num_workers: Number of data loading workers
            audio_sample_rate: Audio sampling rate
            audio_duration: Audio clip duration
            image_size: Target image size
            shuffle_buffer: Size of shuffle buffer
        """
        self.urls = urls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        
        # Setup transforms
        self.image_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def preprocess_audio(self, audio_bytes):
        """Preprocess audio from bytes."""
        import io
        
        # Load audio from bytes
        waveform, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=self.audio_sample_rate,
            duration=self.audio_duration,
            mono=True
        )
        
        # Pad or trim
        target_length = int(self.audio_sample_rate * self.audio_duration)
        if len(waveform) < target_length:
            waveform = np.pad(
                waveform,
                (0, target_length - len(waveform)),
                mode='constant'
            )
        else:
            waveform = waveform[:target_length]
        
        return torch.from_numpy(waveform.astype(np.float32))
    
    def preprocess_image(self, image_bytes):
        """Preprocess image from bytes."""
        import io
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.image_transform(image)
    
    def create_dataloader(self):
        """Create WebDataset dataloader."""
        dataset = (
            wds.WebDataset(self.urls)
            .shuffle(self.shuffle_buffer)
            .decode("pil")
            .to_tuple("audio", "jpg", "json")
            .map_tuple(
                self.preprocess_audio,
                self.preprocess_image,
                lambda x: x
            )
            .batched(self.batch_size)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return dataloader


def create_dummy_dataset(
    output_dir: str,
    num_samples: int = 100,
    split: str = "train"
):
    """
    Create dummy dataset for testing.
    
    Args:
        output_dir: Output directory
        num_samples: Number of samples to create
        split: Dataset split name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    audio_dir = output_dir / "audio"
    image_dir = output_dir / "images"
    audio_dir.mkdir(exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    
    # Audio classes
    classes = [
        "thunder", "ocean_waves", "fire", "applause", "siren",
        "helicopter", "dog_barking", "rain", "glass_breaking", "engine"
    ]
    
    metadata = []
    
    for i in range(num_samples):
        # Generate dummy audio (white noise)
        audio = np.random.randn(48000 * 10).astype(np.float32) * 0.1
        audio_path = audio_dir / f"{split}_{i:05d}.wav"
        
        import soundfile as sf
        sf.write(str(audio_path), audio, 48000)
        
        # Generate dummy image (random colors)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_path = image_dir / f"{split}_{i:05d}.jpg"
        Image.fromarray(image).save(image_path)
        
        # Create metadata
        class_name = random.choice(classes)
        metadata.append({
            'audio_path': f"audio/{split}_{i:05d}.wav",
            'image_path': f"images/{split}_{i:05d}.jpg",
            'text': f"A photo with the sound of {class_name}",
            'class': class_name,
            'video_id': f"dummy_{i:05d}",
            'timestamp': 0
        })
    
    # Save metadata
    metadata_path = output_dir / f"{split}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created dummy dataset with {num_samples} samples at {output_dir}")


if __name__ == "__main__":
    # Test dataset creation
    create_dummy_dataset("./data/dummy", num_samples=10, split="train")
    create_dummy_dataset("./data/dummy", num_samples=5, split="val")
    
    # Test dataset loading
    dataset = AudioImageDataset("./data/dummy", split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Text: {sample['text']}")
"""
Safe augmentation with minimal risk
Only simple, tested transformations
"""

import torch
import random
from typing import Tuple


class SafeAugmentation:
    """Conservative augmentation to avoid training instability"""
    
    def __init__(self, enabled=False, flip_prob=0.5):
        """
        Args:
            enabled: Whether to apply augmentation (default: False for safety)
            flip_prob: Probability of horizontal flip
        """
        self.enabled = enabled
        self.flip_prob = flip_prob
        
    def apply(self, audio: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply safe augmentations
        
        Args:
            audio: Audio tensor
            image: Image tensor [C, H, W]
            
        Returns:
            Augmented audio and image tensors
        """
        if not self.enabled:
            return audio, image
        
        # Only horizontal flip for images (safest augmentation)
        if random.random() < self.flip_prob:
            image = torch.flip(image, dims=[-1])  # Flip width dimension
        
        # No audio augmentation in safe mode
        # Audio augmentation can cause dimension/dtype issues
        
        return audio, image
    
    def __call__(self, audio: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply(audio, image)
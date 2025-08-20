"""
Hierarchical Audio Decomposition Module for CLAP2Diffusion V4
Decomposes audio features into foreground, background, and ambience layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class HierarchicalAudioDecomposition(nn.Module):
    """
    Decomposes CLAP audio features into hierarchical semantic tokens:
    - Foreground: Primary sounds/activities (5 tokens)
    - Background: Secondary/environmental sounds (3 tokens)  
    - Ambience: Overall atmosphere/mood (2 tokens)
    
    Total: 10 audio tokens to be combined with 67 text tokens = 77 tokens
    """
    
    def __init__(
        self,
        audio_dim: int = 512,      # CLAP audio feature dimension (512 after projection)
        text_dim: int = 768,        # CLIP text token dimension
        num_foreground: int = 5,    # Tokens for main sounds
        num_background: int = 3,    # Tokens for background sounds
        num_ambience: int = 2,      # Tokens for atmosphere
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_foreground = num_foreground
        self.num_background = num_background
        self.num_ambience = num_ambience
        self.total_tokens = num_foreground + num_background + num_ambience
        
        # Projection layers for each hierarchy level
        # Each projects CLAP features (512d) to multiple CLIP-compatible tokens (768d each)
        self.foreground_proj = nn.Sequential(
            nn.Linear(audio_dim, text_dim * 2),  # 512 -> 1536
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 2, text_dim * num_foreground)  # 1536 -> 3840
        )
        
        self.background_proj = nn.Sequential(
            nn.Linear(audio_dim, text_dim),  # 512 -> 768
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim, text_dim * num_background)  # 768 -> 2304
        )
        
        self.ambience_proj = nn.Sequential(
            nn.Linear(audio_dim, text_dim // 2),  # 512 -> 384
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, text_dim * num_ambience)  # 384 -> 1536
        )
        
        # Learnable hierarchy weights (importance of each level)
        # Initialize with decreasing importance: foreground > background > ambience
        self.hierarchy_weights = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        )
        
        # Layer normalization for output tokens
        self.layer_norm = nn.LayerNorm(text_dim)
        
        # Optional: Cross-hierarchy attention for inter-level relationships
        self.cross_hierarchy_attn = CrossHierarchyAttention(
            text_dim, num_heads=4, dropout=dropout
        )
        
    def forward(
        self, 
        audio_features: torch.Tensor,
        return_hierarchy: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to decompose audio into hierarchical tokens
        
        Args:
            audio_features: CLAP audio embeddings [batch_size, audio_dim]
            return_hierarchy: If True, return separate hierarchy components
            
        Returns:
            audio_tokens: Hierarchical audio tokens [batch_size, total_tokens, text_dim]
            hierarchy_dict (optional): Dict with separate hierarchy components
        """
        batch_size = audio_features.shape[0]
        
        # Project to each hierarchy level
        foreground = self.foreground_proj(audio_features)  # [B, text_dim * num_foreground]
        background = self.background_proj(audio_features)  # [B, text_dim * num_background]
        ambience = self.ambience_proj(audio_features)      # [B, text_dim * num_ambience]
        
        # Reshape to token sequences
        foreground = foreground.view(batch_size, self.num_foreground, self.text_dim)
        background = background.view(batch_size, self.num_background, self.text_dim)
        ambience = ambience.view(batch_size, self.num_ambience, self.text_dim)
        
        # Apply hierarchy weights
        weights = F.softmax(self.hierarchy_weights, dim=0)
        foreground = foreground * weights[0]
        background = background * weights[1]
        ambience = ambience * weights[2]
        
        # Concatenate all hierarchy levels
        audio_tokens = torch.cat([foreground, background, ambience], dim=1)  # [B, 10, 768]
        
        # Apply cross-hierarchy attention to model relationships
        audio_tokens = self.cross_hierarchy_attn(audio_tokens)
        
        # Layer normalization
        audio_tokens = self.layer_norm(audio_tokens)
        
        if return_hierarchy:
            hierarchy_dict = {
                'foreground': foreground,
                'background': background,
                'ambience': ambience,
                'weights': weights,
                'combined': audio_tokens
            }
            return audio_tokens, hierarchy_dict
        
        return audio_tokens
    
    def get_hierarchy_info(self) -> Dict[str, torch.Tensor]:
        """Get current hierarchy weights and configuration"""
        weights = F.softmax(self.hierarchy_weights, dim=0)
        return {
            'weights': weights.detach(),
            'num_tokens': {
                'foreground': self.num_foreground,
                'background': self.num_background,
                'ambience': self.num_ambience,
                'total': self.total_tokens
            }
        }


class CrossHierarchyAttention(nn.Module):
    """
    Self-attention across hierarchy levels to model inter-level relationships
    E.g., how foreground sounds relate to background context
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention across audio tokens
        
        Args:
            x: Input tokens [batch_size, num_tokens, dim]
            
        Returns:
            Output tokens with cross-hierarchy relationships [batch_size, num_tokens, dim]
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is [B, num_heads, N, head_dim]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class HierarchicalAudioEncoder(nn.Module):
    """
    Complete audio encoding pipeline with CLAP + Hierarchical Decomposition
    """
    
    def __init__(
        self,
        clap_model_name: str = "laion/larger_clap_music_and_speech",
        freeze_clap: bool = True,
        **decomposition_kwargs
    ):
        super().__init__()
        
        # Load CLAP encoder and feature extractor
        from transformers import ClapModel, ClapFeatureExtractor
        self.clap = ClapModel.from_pretrained(clap_model_name)
        self.feature_extractor = ClapFeatureExtractor.from_pretrained(clap_model_name)
        
        if freeze_clap:
            for param in self.clap.parameters():
                param.requires_grad = False
        
        # Hierarchical decomposition module
        self.decomposer = HierarchicalAudioDecomposition(**decomposition_kwargs)
        
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int = 48000,  # CLAP requires 48kHz
        return_hierarchy: bool = False
    ) -> torch.Tensor:
        """
        Encode audio to hierarchical tokens
        
        Args:
            audio: Raw audio waveform [batch_size, audio_length] or [audio_length]
            sample_rate: Audio sample rate
            return_hierarchy: If True, return hierarchy components
            
        Returns:
            Hierarchical audio tokens [batch_size, 10, 768]
        """
        # Ensure audio has correct shape for CLAP
        # CLAP expects [batch_size, audio_length]
        if audio.dim() == 1:
            # Single audio sample without batch dimension
            audio = audio.unsqueeze(0)  # [1, audio_length]
        elif audio.dim() == 2:
            # Already [batch_size, audio_length]
            pass
        elif audio.dim() == 3:
            # Could be [batch_size, channels, audio_length] or other format
            # Try to squeeze or select first channel
            if audio.shape[1] == 1:
                audio = audio.squeeze(1)  # [batch_size, 1, audio_length] -> [batch_size, audio_length]
            else:
                # Take first channel or mean across channels
                audio = audio.mean(dim=1)  # [batch_size, channels, audio_length] -> [batch_size, audio_length]
        else:
            raise ValueError(f"Unexpected audio shape: {audio.shape}. Expected [batch_size, audio_length] or [audio_length]")
        
        # Ensure audio is float32 (CLAP requirement)
        audio = audio.to(torch.float32)
        
        # Get CLAP audio embeddings using the proper method
        with torch.no_grad() if self.clap.training == False else torch.enable_grad():
            # Convert raw audio to features using feature extractor
            # Feature extractor expects numpy array or list
            audio_numpy = audio.cpu().numpy()
            
            # Process through feature extractor
            # Always use 48kHz as CLAP was trained with this rate
            inputs = self.feature_extractor(
                audio_numpy, 
                sampling_rate=48000,  # Force 48kHz
                return_tensors="pt"
            )
            
            # Move to same device as model
            inputs = {k: v.to(audio.device) for k, v in inputs.items()}
            
            # Get audio features through CLAP model
            # Use the complete model pipeline for proper feature extraction
            audio_outputs = self.clap.audio_model(**inputs)
            audio_features = self.clap.audio_projection(audio_outputs.pooler_output)
        
        # Decompose into hierarchical tokens
        if return_hierarchy:
            # Return both hierarchical tokens and raw CLAP features
            tokens, hierarchy = self.decomposer(audio_features, return_hierarchy=True)
            # Add raw CLAP features (512d) to hierarchy for adapter use
            hierarchy['clap_features'] = audio_features
            return tokens, hierarchy
        else:
            # Just return tokens for backward compatibility
            return self.decomposer(audio_features, return_hierarchy=False)


if __name__ == "__main__":
    # Test the module
    print("Testing Hierarchical Audio Decomposition Module...")
    
    # Create module
    decomposer = HierarchicalAudioDecomposition()
    print(f"Module created with {sum(p.numel() for p in decomposer.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    audio_features = torch.randn(batch_size, 512)  # Mock CLAP features
    
    # Get audio tokens
    audio_tokens = decomposer(audio_features)
    print(f"Output shape: {audio_tokens.shape}")  # Should be [4, 10, 768]
    
    # Get hierarchy info
    hierarchy_info = decomposer.get_hierarchy_info()
    print(f"Hierarchy weights: {hierarchy_info['weights']}")
    print(f"Token distribution: {hierarchy_info['num_tokens']}")
    
    # Test with hierarchy return
    audio_tokens, hierarchy = decomposer(audio_features, return_hierarchy=True)
    print(f"Foreground shape: {hierarchy['foreground'].shape}")  # [4, 5, 768]
    print(f"Background shape: {hierarchy['background'].shape}")  # [4, 3, 768]
    print(f"Ambience shape: {hierarchy['ambience'].shape}")      # [4, 2, 768]
    
    print("Hierarchical Audio Decomposition Module test passed!")
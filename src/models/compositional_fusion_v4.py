"""
Compositional Fusion Module for CLAP2Diffusion V4
Combines text tokens (objects) with audio tokens (environment/context) creatively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class CompositionalFusion(nn.Module):
    """
    Fuses text tokens (67) and audio tokens (10) into unified conditioning (77 tokens)
    Implements creative composition strategies for different text-audio relationships
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        num_text_tokens: int = 67,
        num_audio_tokens: int = 10,
        fusion_strategy: str = "role_based",  # role_based, attention, or linear
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.num_text_tokens = num_text_tokens
        self.num_audio_tokens = num_audio_tokens
        self.total_tokens = 77  # Standard CLIP token count
        self.fusion_strategy = fusion_strategy
        
        # Fusion strategy modules
        if fusion_strategy == "role_based":
            # Role-based fusion: text=objects, audio=context
            self.role_gate = nn.Parameter(torch.ones(1))  # Balance between modalities
            self.context_modulator = ContextModulator(text_dim, dropout)
            
        elif fusion_strategy == "attention":
            # Attention-based fusion
            self.fusion_attn = FusionAttention(
                text_dim, num_heads=8, dropout=dropout
            )
            
        elif fusion_strategy == "linear":
            # Simple linear blending
            self.fusion_weight = nn.Parameter(torch.tensor(0.7))  # Text weight
            
        # Position embeddings for combined sequence
        self.position_embed = nn.Parameter(
            torch.randn(1, self.total_tokens, text_dim) * 0.02
        )
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(text_dim)
        
        # Composition classifier (determines relationship type)
        self.composition_classifier = CompositionClassifier(text_dim)
        
    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_hierarchy: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Fuse text and audio tokens based on their relationship
        
        Args:
            text_tokens: Text embeddings [batch_size, 67, 768]
            audio_tokens: Audio embeddings [batch_size, 10, 768]
            audio_hierarchy: Optional hierarchy info from audio decomposition
            
        Returns:
            fused_tokens: Combined conditioning [batch_size, 77, 768]
            fusion_info: Dictionary with fusion details
        """
        batch_size = text_tokens.shape[0]
        
        # Classify composition type (matching, complementary, creative, contradictory)
        composition_type = self.composition_classifier(text_tokens, audio_tokens)
        
        # Apply fusion based on strategy
        if self.fusion_strategy == "role_based":
            fused = self._role_based_fusion(
                text_tokens, audio_tokens, composition_type, audio_hierarchy
            )
        elif self.fusion_strategy == "attention":
            fused = self._attention_fusion(text_tokens, audio_tokens)
        else:  # linear
            fused = self._linear_fusion(text_tokens, audio_tokens)
        
        # Add position embeddings
        fused = fused + self.position_embed
        
        # Layer normalization
        fused = self.layer_norm(fused)
        
        # Prepare fusion info
        fusion_info = {
            'composition_type': composition_type,
            'strategy': self.fusion_strategy,
            'audio_hierarchy': audio_hierarchy
        }
        
        return fused, fusion_info
    
    def _role_based_fusion(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        composition_type: torch.Tensor,
        audio_hierarchy: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Role-based fusion: Text provides WHAT, Audio provides WHERE/HOW/MOOD
        """
        batch_size = text_tokens.shape[0]
        
        # Modulate audio based on composition type
        audio_modulated = self.context_modulator(
            audio_tokens, composition_type, audio_hierarchy
        )
        
        # Adaptive gating based on composition type
        gate = torch.sigmoid(self.role_gate)
        
        # For matching: strengthen both
        # For complementary: balance
        # For creative: emphasize audio context
        # For contradictory: reduce conflict
        
        composition_weights = self._get_composition_weights(composition_type)
        
        # Weighted combination
        text_weighted = text_tokens * (1 - gate * composition_weights[:, :1])
        audio_weighted = audio_modulated * (gate * composition_weights[:, 1:2])
        
        # Concatenate: [text_tokens, audio_tokens]
        fused = torch.cat([text_weighted, audio_weighted], dim=1)
        
        return fused
    
    def _attention_fusion(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Attention-based fusion with cross-modal interactions
        """
        # Concatenate tokens
        combined = torch.cat([text_tokens, audio_tokens], dim=1)
        
        # Apply fusion attention
        fused = self.fusion_attn(combined, text_tokens.shape[1])
        
        return fused
    
    def _linear_fusion(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple linear blending of modalities
        """
        weight = torch.sigmoid(self.fusion_weight)
        
        # Weighted average for overlapping positions
        text_weighted = text_tokens * weight
        
        # Concatenate
        fused = torch.cat([text_weighted, audio_tokens * (1 - weight)], dim=1)
        
        return fused
    
    def _get_composition_weights(
        self,
        composition_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Get fusion weights based on composition type
        
        Composition types:
        0: Matching - both modalities reinforce
        1: Complementary - balanced contribution
        2: Creative - audio adds new context
        3: Contradictory - careful blending
        """
        batch_size = composition_type.shape[0]
        weights = torch.zeros(batch_size, 2, device=composition_type.device)
        
        # Define weight patterns for each type
        weight_patterns = {
            0: [0.8, 0.8],  # Matching: both strong
            1: [0.6, 0.6],  # Complementary: balanced
            2: [0.4, 0.8],  # Creative: emphasize audio
            3: [0.7, 0.3],  # Contradictory: reduce audio
        }
        
        for i in range(batch_size):
            comp_type = composition_type[i].item()
            weights[i] = torch.tensor(weight_patterns.get(comp_type, [0.5, 0.5]))
        
        return weights


class ContextModulator(nn.Module):
    """
    Modulates audio context based on composition type and hierarchy
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.modulation_net = nn.Sequential(
            nn.Linear(dim + 4, dim),  # +4 for composition type encoding
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
    def forward(
        self,
        audio_tokens: torch.Tensor,
        composition_type: torch.Tensor,
        audio_hierarchy: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Modulate audio tokens based on context
        """
        batch_size, num_tokens, dim = audio_tokens.shape
        
        # One-hot encode composition type
        comp_onehot = F.one_hot(composition_type, num_classes=4).float()
        comp_expanded = comp_onehot.unsqueeze(1).expand(-1, num_tokens, -1)
        
        # Concatenate with audio tokens
        audio_with_comp = torch.cat([audio_tokens, comp_expanded], dim=-1)
        
        # Apply modulation
        modulated = self.modulation_net(audio_with_comp)
        
        # Residual connection
        output = audio_tokens + modulated
        
        # If hierarchy info available, apply hierarchy-specific modulation
        if audio_hierarchy is not None:
            # Emphasize different levels based on composition type
            # E.g., for "creative", emphasize ambience more
            weights = audio_hierarchy.get('weights', torch.ones(3))
            # Apply differential weighting (simplified for now)
            output = output * (1 + 0.1 * weights.mean())
        
        return output


class FusionAttention(nn.Module):
    """
    Cross-modal attention for text-audio fusion
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
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
        
        # Modality-specific attention masks
        self.register_buffer('modality_mask', None)
        
    def forward(
        self,
        x: torch.Tensor,
        text_length: int
    ) -> torch.Tensor:
        """
        Apply cross-modal attention
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Optional: Apply modality-aware masking
        # (e.g., allow text to attend to audio but not vice versa)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CompositionClassifier(nn.Module):
    """
    Classifies the relationship between text and audio
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        self.text_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 4)  # 4 composition types
        )
        
    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify composition type
        
        Returns:
            Composition type indices [batch_size]
            0: Matching
            1: Complementary
            2: Creative
            3: Contradictory
        """
        # Pool tokens to get global features
        text_pooled = self.text_pool(text_tokens.transpose(1, 2)).squeeze(-1)
        audio_pooled = self.audio_pool(audio_tokens.transpose(1, 2)).squeeze(-1)
        
        # Concatenate and classify
        combined = torch.cat([text_pooled, audio_pooled], dim=-1)
        logits = self.classifier(combined)
        
        # Return argmax as composition type
        return torch.argmax(logits, dim=-1)


if __name__ == "__main__":
    # Test the module
    print("Testing Compositional Fusion Module...")
    
    # Create module
    fusion = CompositionalFusion(fusion_strategy="role_based")
    print(f"Module created with {sum(p.numel() for p in fusion.parameters())} parameters")
    
    # Test inputs
    batch_size = 4
    text_tokens = torch.randn(batch_size, 67, 768)
    audio_tokens = torch.randn(batch_size, 10, 768)
    
    # Test fusion
    fused_tokens, fusion_info = fusion(text_tokens, audio_tokens)
    print(f"Fused shape: {fused_tokens.shape}")  # Should be [4, 77, 768]
    print(f"Composition types: {fusion_info['composition_type']}")
    
    # Test different strategies
    for strategy in ["role_based", "attention", "linear"]:
        fusion_module = CompositionalFusion(fusion_strategy=strategy)
        output, _ = fusion_module(text_tokens, audio_tokens)
        print(f"{strategy} fusion output shape: {output.shape}")
    
    print("Compositional Fusion Module test passed!")
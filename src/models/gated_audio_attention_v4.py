"""
Gated Audio Attention Module for CLAP2Diffusion V4 Hybrid Architecture
Combines SonicDiffusion's gated attention with our hierarchical decomposition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from einops import rearrange, repeat


class GatedAudioAttention(nn.Module):
    """
    Gated Cross-Attention for audio conditioning
    Inspired by SonicDiffusion but adapted for hierarchical audio features
    
    Key features:
    - Learnable gate parameter for audio influence control
    - Hierarchy-aware attention (foreground/background/ambience)
    - Residual connections for stable training
    """
    
    def __init__(
        self,
        query_dim: int = 768,
        context_dim: int = 768,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        hierarchy_level: str = "full",  # "foreground", "background", "ambience", "full"
        use_gate: bool = True,
        initial_gate_value: float = 0.0
    ):
        super().__init__()
        
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.hierarchy_level = hierarchy_level
        self.use_gate = use_gate
        
        # Q, K, V projections for cross-attention
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable gate parameter (like SonicDiffusion)
        if use_gate:
            self.gate = nn.Parameter(torch.tensor(initial_gate_value))
        else:
            self.register_buffer('gate', torch.tensor(1.0))
        
        # Hierarchy-specific weights
        self.hierarchy_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # [fore, back, amb]
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_context: torch.Tensor,
        audio_hierarchy: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        f_multiplier: float = 1.0
    ) -> torch.Tensor:
        """
        Apply gated audio attention
        
        Args:
            hidden_states: UNet hidden states [batch, seq_len, dim]
            audio_context: Audio features [batch, audio_tokens, dim]
            audio_hierarchy: Optional dict with hierarchical audio features
            attention_mask: Optional attention mask
            f_multiplier: Dynamic multiplier for audio influence
            
        Returns:
            Output with audio conditioning applied [batch, seq_len, dim]
        """
        batch_size = hidden_states.shape[0]
        
        # Store residual for gated addition
        residual = hidden_states
        
        # Normalize input
        hidden_states = self.norm(hidden_states)
        
        # Select audio context based on hierarchy level
        if audio_hierarchy is not None and self.hierarchy_level != "full":
            audio_context = self._select_hierarchy(audio_context, audio_hierarchy)
        
        # Compute Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(audio_context)
        v = self.to_v(audio_context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Output projection
        out = self.to_out(out)
        
        # Apply gate with f_multiplier
        gate_value = torch.sigmoid(self.gate) * f_multiplier
        
        # Gated residual connection
        output = residual + gate_value * out
        
        return output
    
    def _select_hierarchy(
        self,
        audio_context: torch.Tensor,
        audio_hierarchy: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Select specific hierarchy level from audio features
        """
        if self.hierarchy_level == "foreground" and "foreground" in audio_hierarchy:
            return audio_hierarchy["foreground"]
        elif self.hierarchy_level == "background" and "background" in audio_hierarchy:
            return audio_hierarchy["background"]
        elif self.hierarchy_level == "ambience" and "ambience" in audio_hierarchy:
            return audio_hierarchy["ambience"]
        else:
            return audio_context
    
    def get_gate_value(self) -> float:
        """Get current gate value"""
        return torch.sigmoid(self.gate).item()
    
    def set_gate_value(self, value: float):
        """Set gate value (for domain-specific loading)"""
        with torch.no_grad():
            # Inverse sigmoid to set the parameter correctly
            value_tensor = torch.tensor(value, dtype=torch.float32)
            self.gate.data = torch.logit(value_tensor)


class HierarchicalGatedAdapter(nn.Module):
    """
    Complete adapter module combining hierarchical decomposition with gated attention
    This replaces the standard cross-attention in specific UNet blocks
    """
    
    def __init__(
        self,
        in_channels: int,
        audio_dim: int = 512,
        num_audio_tokens: int = 77,  # Standard CLIP token count
        hierarchy_config: Dict[str, int] = None,
        dropout: float = 0.1,
        use_4layer_projection: bool = True  # Use SonicDiffusion-style projection
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.audio_dim = audio_dim
        self.num_audio_tokens = num_audio_tokens
        
        # Default hierarchy configuration
        if hierarchy_config is None:
            hierarchy_config = {
                "foreground": 5,
                "background": 3,
                "ambience": 2
            }
        self.hierarchy_config = hierarchy_config
        
        # Audio projection (SonicDiffusion style if enabled)
        if use_4layer_projection:
            self.audio_projection = AudioProjector4Layer(
                audio_dim=audio_dim,
                output_tokens=num_audio_tokens,
                dropout=dropout
            )
        else:
            # Simple linear projection: audio_dim (512) -> text_dim (768) * num_tokens
            self.audio_projection = nn.Linear(audio_dim, 768 * num_audio_tokens)
        
        # Gated attention module
        # context_dim should be 768 (text_dim) since audio tokens are projected to that dimension
        self.gated_attention = GatedAudioAttention(
            query_dim=in_channels,
            context_dim=768,  # CLIP text dimension, not audio_dim
            dropout=dropout
        )
        
        # Hierarchy-specific projections
        # Project from audio_dim (512) to hidden_dim (768) * num_tokens for each hierarchy level
        self.hierarchy_projections = nn.ModuleDict({
            "foreground": nn.Linear(audio_dim, 768 * hierarchy_config["foreground"]),  # 512 -> 768*5
            "background": nn.Linear(audio_dim, 768 * hierarchy_config["background"]),  # 512 -> 768*3
            "ambience": nn.Linear(audio_dim, 768 * hierarchy_config["ambience"])      # 512 -> 768*2
        })
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_features: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        f_multiplier: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply hierarchical gated audio conditioning
        
        Returns:
            Conditioned hidden states and attention info
        """
        batch_size = hidden_states.shape[0]
        
        # Project audio features to tokens
        if hasattr(self.audio_projection, 'forward'):
            # AudioProjector4Layer outputs [batch, num_tokens, 768]
            audio_tokens = self.audio_projection(audio_features)
        else:
            # Simple linear projection needs reshaping
            audio_tokens = self.audio_projection(audio_features)
            # Reshape to [batch, num_tokens, 768] (text_dim, not audio_dim)
            audio_tokens = audio_tokens.view(batch_size, self.num_audio_tokens, 768)
        
        # Generate hierarchical representations
        audio_hierarchy = {}
        for level, proj in self.hierarchy_projections.items():
            level_features = proj(audio_features)
            num_tokens = self.hierarchy_config[level]
            # Reshape to [batch, num_tokens, 768] (text_dim, not audio_dim)
            audio_hierarchy[level] = level_features.view(batch_size, num_tokens, 768)
        
        # Apply gated attention
        output = self.gated_attention(
            hidden_states,
            audio_tokens,
            audio_hierarchy=audio_hierarchy,
            f_multiplier=f_multiplier
        )
        
        # Prepare info dict
        info = {
            "gate_value": self.gated_attention.get_gate_value(),
            "hierarchy_used": self.gated_attention.hierarchy_level,
            "f_multiplier": f_multiplier
        }
        
        return output, info


class AudioProjector4Layer(nn.Module):
    """
    4-layer transformer-based audio projector (SonicDiffusion style)
    Uses self-attention and residual connections for rich audio representations
    """
    
    def __init__(
        self,
        audio_dim: int = 512,  # CLAP output dimension (768 for larger_clap_music_and_speech)
        hidden_dim: int = 768,  # CLIP text dimension
        output_tokens: int = 77,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_tokens = output_tokens
        
        # Initial projection: audio_dim -> hidden_dim * output_tokens
        self.input_projection = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim * output_tokens // 4),
            nn.GELU(),
            nn.Linear(hidden_dim * output_tokens // 4, hidden_dim * output_tokens // 2),
            nn.GELU(),
            nn.Linear(hidden_dim * output_tokens // 2, hidden_dim * output_tokens),
            nn.Dropout(dropout)
        )
        
        # Transformer blocks with self-attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to token sequence
        
        Args:
            audio_features: [batch, audio_dim]
            
        Returns:
            Projected tokens [batch, num_tokens, hidden_dim]
        """
        batch_size = audio_features.shape[0]
        
        # Ensure correct input shape
        if audio_features.dim() == 3:
            audio_features = audio_features.squeeze(1)  # [B, 1, D] -> [B, D]
        
        # Initial projection
        x = self.input_projection(audio_features)  # [B, hidden_dim * num_tokens]
        
        # Reshape to token sequence
        x = x.view(batch_size, self.output_tokens, self.hidden_dim)  # [B, tokens, hidden_dim]
        
        # Apply transformer blocks with residual connections
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output


class TransformerBlock(nn.Module):
    """
    Basic transformer block with self-attention and FFN
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with residual connections
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + x
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


if __name__ == "__main__":
    # Test the modules
    print("Testing Gated Audio Attention Module...")
    
    # Test GatedAudioAttention
    gated_attn = GatedAudioAttention(hierarchy_level="foreground")
    hidden = torch.randn(2, 64, 768)  # [batch, seq_len, dim]
    audio = torch.randn(2, 10, 768)   # [batch, audio_tokens, dim]
    
    output = gated_attn(hidden, audio)
    print(f"GatedAudioAttention output shape: {output.shape}")
    print(f"Gate value: {gated_attn.get_gate_value():.4f}")
    
    # Test HierarchicalGatedAdapter
    adapter = HierarchicalGatedAdapter(in_channels=768)
    hidden = torch.randn(2, 64, 768)
    audio_feat = torch.randn(2, 512)  # Fixed: CLAP outputs 512 dimensions
    
    output, info = adapter(hidden, audio_feat)
    print(f"\nHierarchicalGatedAdapter output shape: {output.shape}")
    print(f"Info: {info}")
    
    # Test AudioProjector4Layer
    projector = AudioProjector4Layer()
    audio_input = torch.randn(2, 512)  # CLAP dimension
    projected = projector(audio_input)
    print(f"\nAudioProjector4Layer output shape: {projected.shape}")
    
    print("\nAll tests passed!")